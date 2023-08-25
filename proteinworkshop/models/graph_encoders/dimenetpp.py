from math import sqrt
from typing import Callable, Set, Union

import torch
import torch_scatter
from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch
from torch_geometric.nn.models.dimenet import DimeNetPlusPlus, triplets

from proteinworkshop.models.graph_encoders.components.blocks import DimeNetEmbeddingBlock
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
        self.emb = DimeNetEmbeddingBlock(num_radial, hidden_channels, get_activations(act))

    @property
    def required_batch_attributes(self) -> Set[str]:
        return {"pos", "edge_index", "x", "batch"}

    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
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

        return EncoderOutput({
            "node_embedding": P,
            "graph_embedding": P.sum(dim=0)
            if batch is None
            else torch_scatter.scatter(P, batch.batch, dim=0, reduce=self.readout),
        })


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
