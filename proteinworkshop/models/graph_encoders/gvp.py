from typing import Set, Union

import torch
import torch.nn.functional as F
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import jaxtyped
from torch_geometric.data import Batch

import proteinworkshop.models.graph_encoders.layers.gvp as gvp
from proteinworkshop.models.graph_encoders.components import blocks
from proteinworkshop.models.utils import get_aggregation
from proteinworkshop.types import EncoderOutput


class GVPGNNModel(torch.nn.Module):
    def __init__(
        self,
        s_dim: int = 128,
        v_dim: int = 16,
        s_dim_edge: int = 32,
        v_dim_edge: int = 1,
        r_max: float = 10.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        num_layers: int = 5,
        pool: str = "sum",
        residual: bool = True,
    ):
        """
        Initializes an instance of the GVPGNNModel class with the provided
        parameters.

        :param s_dim: Dimension of the node state embeddings (default: ``128``)
        :type s_dim: int
        :param v_dim: Dimension of the node vector embeddings (default: ``16``)
        :type v_dim: int
        :param s_dim_edge: Dimension of the edge state embeddings
            (default: ``32``)
        :type s_dim_edge: int
        :param v_dim_edge: Dimension of the edge vector embeddings
            (default: ``1``)
        :type v_dim_edge: int
        :param r_max: Maximum distance for Bessel basis functions
            (default: ``10.0``)
        :type r_max: float
        :param num_bessel: Number of Bessel basis functions (default: ``8``)
        :type num_bessel: int
        :param num_polynomial_cutoff: Number of polynomial cutoff basis
            functions (default: ``5``)
        :type num_polynomial_cutoff: int
        :param num_layers: Number of layers in the model (default: ``5``)
        :type num_layers: int
        :param pool: Global pooling method to be used
            (default: ``"sum"``)
        :type pool: str
        :param residual: Whether to use residual connections
            (default: ``True``)
        :type residual: bool
        """
        super().__init__()
        _DEFAULT_V_DIM = (s_dim, v_dim)
        _DEFAULT_E_DIM = (s_dim_edge, v_dim_edge)
        self.r_max = r_max
        self.num_layers = num_layers
        activations = (F.relu, None)

        # Node embedding
        self.emb_in = torch.nn.LazyLinear(s_dim)
        self.W_v = torch.nn.Sequential(
            gvp.LayerNorm((s_dim, 0)),
            gvp.GVP(
                (s_dim, 0),
                _DEFAULT_V_DIM,
                activations=(None, None),
                vector_gate=True,
            ),
        )
        # Edge embedding
        self.radial_embedding = blocks.RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        self.W_e = torch.nn.Sequential(
            gvp.LayerNorm((self.radial_embedding.out_dim, 1)),
            gvp.GVP(
                (self.radial_embedding.out_dim, 1),
                _DEFAULT_E_DIM,
                activations=(None, None),
                vector_gate=True,
            ),
        )
        # Stack of GNN layers
        self.layers = torch.nn.ModuleList(
            gvp.GVPConvLayer(
                _DEFAULT_V_DIM,
                _DEFAULT_E_DIM,
                activations=activations,
                vector_gate=True,
                residual=residual,
            )
            for _ in range(num_layers)
        )
        # Output GVP
        self.W_out = torch.nn.Sequential(
            gvp.LayerNorm(_DEFAULT_V_DIM),
            gvp.GVP(
                _DEFAULT_V_DIM,
                (s_dim, 0),
                activations=activations,
                vector_gate=True,
            ),
        )
        # Global pooling/readout function
        self.readout = get_aggregation(pool)

    @property
    def required_batch_attributes(self) -> Set[str]:
        """Required batch attributes for this encoder.

        - ``edge_index`` (shape ``[2, num_edges]``)
        - ``pos`` (shape ``[num_nodes, 3]``)
        - ``x`` (shape ``[num_nodes, num_node_features]``)
        - ``batch`` (shape ``[num_nodes]``)

        :return: _description_
        :rtype: Set[str]
        """
        return {"edge_index", "pos", "x", "batch"}

    @jaxtyped(typechecker=typechecker)
    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Implements the forward pass of the GVP-GNN encoder.

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
        # Edge features
        vectors = (
            batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]
        )  # [n_edges, 3]
        lengths = torch.linalg.norm(
            vectors, dim=-1, keepdim=True
        )  # [n_edges, 1]

        h_V = self.emb_in(batch.x)
        h_E = (
            self.radial_embedding(lengths),
            torch.nan_to_num(torch.div(vectors, lengths)).unsqueeze_(-2),
        )

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.layers:
            h_V = layer(h_V, batch.edge_index, h_E)

        out = self.W_out(h_V)

        return EncoderOutput(
            {
                "node_embedding": out,
                "graph_embedding": self.readout(
                    out, batch.batch
                ),  # (n, d) -> (batch_size, d)
                # "pos": pos  # TODO it is possible to output pos with GVP if needed
            }
        )


if __name__ == "__main__":
    import hydra
    import omegaconf

    from proteinworkshop import constants

    cfg = omegaconf.OmegaConf.load(
        constants.SRC_PATH / "config" / "encoder" / "gvp.yaml"
    )
    enc = hydra.utils.instantiate(cfg)
    print(enc)
