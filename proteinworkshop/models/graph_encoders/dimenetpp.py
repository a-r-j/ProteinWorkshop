from typing import Callable, Set, Union

import torch
import torch_scatter
from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch
from torch_geometric.nn.models.dimenet import DimeNetPlusPlus, triplets

from proteinworkshop.models.graph_encoders.components.blocks import (
    DimeNetEmbeddingBlock,
)
from proteinworkshop.models.utils import get_activations
from proteinworkshop.types import EncoderOutput


class DimeNetPPModel(DimeNetPlusPlus):
    def __init__(
        self,
        hidden_channels: int = 128,
        out_dim: int = 1,
        num_layers: int = 4,
        int_emb_size: int = 64,
        basis_emb_size: int = 8,
        out_emb_channels: int = 256,
        num_spherical: int = 7,
        num_radial: int = 6,
        cutoff: float = 10,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = "swish",
        readout: str = "add",
    ):
        """Initializes an instance of the DimeNetPPModel model with the
        provided parameters.

        .. note::
            The `act` parameter can be either a string representing a built-in
            activation function, or a callable object that serves as a custom
            activation function.

        :param hidden_channels: Number of channels in the hidden layers
            (default: ``128``)
        :type hidden_channels: int
        :param out_dim: Output dimension of the model (default: ``1``)
        :type out_dim: int
        :param num_layers: Number of layers in the model (default: ``4``)
        :type num_layers: int
        :param int_emb_size: Embedding size for interaction features
            (default: ``64``)
        :type int_emb_size: int
        :param basis_emb_size: Embedding size for basis functions
            (default: ``8``)
        :type basis_emb_size: int
        :param out_emb_channels: Number of channels in the output embeddings
            (default: ``256``)
        :type out_emb_channels: int
        :param num_spherical: Number of spherical harmonics (default: ``7``)
        :type num_spherical: int
        :param num_radial: Number of radial basis functions (default: ``6``)
        :type num_radial: int
        :param cutoff: Cutoff distance for interactions (default: ``10``)
        :type cutoff: float
        :param max_num_neighbors: Maximum number of neighboring atoms to
            consider (default: ``32``)
        :type max_num_neighbors: int
        :param envelope_exponent: Exponent of the envelope function
            (default: ``5``)
        :type envelope_exponent: int
        :param num_before_skip: Number of layers before the skip connections
            (default: ``1``)
        :type num_before_skip: int
        :param num_after_skip: Number of layers after the skip connections
            (default: ``2``)
        :type num_after_skip: int
        :param num_output_layers: Number of output layers (default: ``3``)
        :type num_output_layers: int
        :param act: Activation function to use, defaults to ``"swish"``
        :type act: Union[str, Callable], optional
        :param readout: Global pooling method to be used (default: ``"add"``)
        :type readout: str
        """
        super().__init__(
            hidden_channels,
            out_dim,
            num_layers,
            int_emb_size,
            basis_emb_size,
            out_emb_channels,
            num_spherical,
            num_radial,
            cutoff,
            max_num_neighbors,
            envelope_exponent,
            num_before_skip,
            num_after_skip,
            num_output_layers,
            act,
        )
        self.readout = readout
        # Override embedding block.
        self.emb = DimeNetEmbeddingBlock(
            num_radial, hidden_channels, get_activations(act)
        )

    @property
    def required_batch_attributes(self) -> Set[str]:
        """Required batch attributes for this encoder.

        - ``x``: Node features (shape: :math:`(n, d)`)
        - ``pos``: Node positions (shape: :math:`(n, 3)`)
        - ``edge_index``: Edge indices (shape: :math:`(2, e)`)
        - ``batch``: Batch indices (shape: :math:`(n,)`)

        :return: _description_
        :rtype: Set[str]
        """
        return {"pos", "edge_index", "x", "batch"}

    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Implements the forward pass of the DimeNet++ encoder.
        
        Returns the node embedding and graph embedding in a dictionary.

        :param batch: Batch of data to encode.
        :type batch: Union[Batch, ProteinBatch]
        :return: Dictionary of node and graph embeddings. Contains
            ``node_embedding`` and ``graph_embedding`` fields. The node
            embedding is of shape :math:`(|V|, d)` and the graph embedding is
            of shape :math:`(n, d)`, where :math:`|V|` is the number of nodes
            and :math:`n` is the number of graphs in the batch and :math:`d` is
            the dimension of the embeddings.
        :rtype: EncoderOutput
        """
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            batch.edge_index, num_nodes=batch.x.size(0)
        )

        # Calculate distances.
        dist = (batch.pos[i] - batch.pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = batch.pos[idx_i]
        pos_ji, pos_ki = batch.pos[idx_j] - pos_i, batch.pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(batch.x, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=batch.pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(
            self.interaction_blocks, self.output_blocks[1:]
        ):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i)

        return EncoderOutput(
            {
                "node_embedding": P,
                "graph_embedding": P.sum(dim=0)
                if batch is None
                else torch_scatter.scatter(
                    P, batch.batch, dim=0, reduce=self.readout
                ),
            }
        )


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils
    from graphein.protein.tensor.data import get_random_batch

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(
        root / "configs" / "encoder" / "dimenet_plus_plus.yaml"
    )
    encoder = hydra.utils.instantiate(cfg.dimenet_plus_plus)
    print(encoder)
    batch = get_random_batch(2)
    batch.edges("knn_8", cache="edge_index")
    batch.pos = batch.coords[:, 1, :]
    batch.scalar_node_features = batch.residue_type
    out = encoder.forward(batch)
    print(out)
