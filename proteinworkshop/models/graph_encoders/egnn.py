from typing import Set

import torch
import torch.nn as nn

from proteinworkshop.models.graph_encoders.layers.egnn import EGNNLayer
from proteinworkshop.models.utils import get_aggregation
from proteinworkshop.types import EncoderOutput


class EGNNModel(nn.Module):
    def __init__(
        self,
        num_layers: int = 5,
        emb_dim: int = 128,
        activation: str = "relu",
        norm: str = "layer",
        aggr: str = "sum",
        pool: str = "sum",
        residual: bool = True
    ):
        """E(n) Equivariant GNN model

        Args:
            num_layers: (int) - number of message passing layers
            emb_dim: (int) - hidden dimension
            in_dim: (int) - initial node feature dimension
            out_dim: (int) - output number of classes
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
            pool: (str) - global pooling function (sum/mean)
            residual: (bool) - whether to use residual connections
        """
        super().__init__()

        # Embedding lookup for initial node features
        self.emb_in = torch.nn.LazyLinear(emb_dim)

        # Stack of GNN layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(EGNNLayer(emb_dim, activation, norm, aggr))

        # Global pooling/readout function
        self.pool = get_aggregation(pool)

        self.residual = residual

    @property
    def required_batch_attributes(self) -> Set[str]:
        return {"x", "pos", "edge_index", "batch"}

    def forward(self, batch) -> EncoderOutput:
        h = self.emb_in(batch.x)  # (n,) -> (n, d)
        pos = batch.pos  # (n, 3)

        for conv in self.convs:
            # Message passing layer
            h_update, pos_update = conv(h, pos, batch.edge_index)

            # Update node features (n, d) -> (n, d)
            h = h + h_update if self.residual else h_update

            # Update node coordinates (no residual) (n, 3) -> (n, 3)
            pos = pos_update

        return EncoderOutput({
            "node_embedding": h,
            "graph_embedding": self.pool(h, batch.batch),  # (n, d) -> (batch_size, d)
            "pos": pos # Position
        })


if __name__ == "__main__":
    import hydra
    import omegaconf

    from proteinworkshop import constants

    cfg = omegaconf.OmegaConf.load(constants.PROJECT_PATH / "configs" / "encoder" / "egnn.yaml")
    enc = hydra.utils.instantiate(cfg)
    print(enc)



