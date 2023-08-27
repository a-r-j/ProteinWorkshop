import torch
import torch_scatter
from graphein.protein.tensor.types import CoordTensor, EdgeTensor
from torch.nn import Linear, Sequential
from torch_geometric.nn import MessagePassing

from proteinworkshop.models.utils import get_activations
from proteinworkshop.types import NodeFeatureTensor


class EGNNLayer(MessagePassing):
    def __init__(
        self,
        emb_dim: int,
        activation: str = "relu",
        norm: str = "layer",
        aggr: str = "add",
    ):
        """E(n) Equivariant GNN Layer

        Paper: E(n) Equivariant Graph Neural Networks, Satorras et al.

        Args:
            emb_dim: (int) - hidden dimension `d`
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.activation = get_activations(activation)
        self.norm = {
            "layer": torch.nn.LayerNorm,
            "batch": torch.nn.BatchNorm1d,
        }[norm]

        # MLP `\psi_h` for computing messages `m_ij`
        self.mlp_msg = Sequential(
            Linear(2 * emb_dim + 1, emb_dim),
            self.norm(emb_dim),
            self.activation,
            Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
        )
        # MLP `\psi_x` for computing messages `\overrightarrow{m}_ij`
        self.mlp_pos = Sequential(
            Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            Linear(emb_dim, 1),
        )
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        self.mlp_upd = Sequential(
            Linear(2 * emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
        )

    def forward(
        self, h: NodeFeatureTensor, pos: CoordTensor, edge_index: EdgeTensor
    ):
        """
        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (e, 2) - pairs of edges (i, j)
        Returns:
            out: [(n, d),(n,3)] - updated node features
        """
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_i, h_j, pos_i, pos_j):
        # Compute messages
        pos_diff = pos_i - pos_j
        dists = torch.norm(pos_diff, dim=-1).unsqueeze(1)
        msg = torch.cat([h_i, h_j, dists], dim=-1)
        msg = self.mlp_msg(msg)
        # Scale magnitude of displacement vector
        pos_diff = pos_diff * self.mlp_pos(msg)
        # TODO some papers divide pos_diff by (dists + 1) to stabilise model.
        # TODO lucidrains implementation clamps pos_diff between some [-n, +n], also for stability.
        return msg, pos_diff

    def aggregate(self, inputs, index):
        msgs, pos_diffs = inputs
        # Aggregate messages
        msg_aggr = torch_scatter.scatter(
            msgs, index, dim=self.node_dim, reduce=self.aggr
        )
        # Aggregate displacement vectors
        pos_aggr = torch_scatter.scatter(
            pos_diffs, index, dim=self.node_dim, reduce="mean"
        )
        return msg_aggr, pos_aggr

    def update(self, aggr_out, h, pos):
        msg_aggr, pos_aggr = aggr_out
        upd_out = self.mlp_upd(torch.cat([h, msg_aggr], dim=-1))
        upd_pos = pos + pos_aggr
        return upd_out, upd_pos

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"
