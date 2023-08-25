from typing import Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as log
from torch_scatter import scatter_add

from proteinworkshop.models.graph_encoders.layers import gear_net
from proteinworkshop.models.utils import get_aggregation
from proteinworkshop.types import EncoderOutput


class GearNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_relation: int,
        num_layers: int,
        emb_dim: int,
        short_cut: bool,
        concat_hidden: bool,
        batch_norm: bool,
        num_angle_bin: Optional[int],
        activation: str = "relu",
        pool: str = "sum",
    ) -> None:
        super().__init__()
        # Base parameters
        self.num_relation = num_relation
        self.input_dim = input_dim
        # Edge message passing layers
        # If not None, this enables Edge Message passing
        self.num_angle_bin = num_angle_bin
        self.edge_input_dim = self.get_num_edge_features()
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        n_hid = [emb_dim] * num_layers

        self.dims = [self.input_dim] + n_hid
        self.activations = [getattr(F, activation) for _ in n_hid]
        self.batch_norm = batch_norm

        # Initialise Node layers
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                gear_net.GeometricRelationalGraphConv(
                    input_dim=self.dims[i],
                    output_dim=self.dims[i + 1],
                    num_relation=self.num_relation,
                    edge_input_dim=self.edge_input_dim,  # None,
                    batch_norm=batch_norm,
                    activation=self.activations[i],
                )
            )

        if self.num_angle_bin:
            log.info("Using Edge Message Passing")
            self.edge_input_dim = self.get_num_edge_features()
            self.edge_dims = [self.edge_input_dim] + self.dims[:-1]
            self.spatial_line_graph = gear_net.SpatialLineGraph(self.num_angle_bin)
            self.edge_layers = nn.ModuleList()
            for i in range(len(self.edge_dims) - 1):
                self.edge_layers.append(
                    gear_net.GeometricRelationalGraphConv(
                        self.edge_dims[i],
                        self.edge_dims[i + 1],
                        self.num_angle_bin,
                        None,
                        batch_norm=self.batch_norm,
                        activation=self.activations[i],
                    )
                )
        # Batch Norm
        if self.batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))

        # Readout
        self.readout = get_aggregation(pool)

    @property
    def required_batch_attributes(self) -> Set[str]:
        return {"x", "edge_index", "edge_type", "edge_attr", "num_nodes", "batch"}

    def forward(self, graph) -> EncoderOutput:
        """
        Compute the node representations and the graph representation(s).
        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        graph.edge_weight = torch.ones(
            graph.edge_index.shape[1], dtype=torch.float, device=graph.x.device
        )
        layer_input = graph.x
        graph.edge_index = torch.cat([graph.edge_index, graph.edge_type])
        graph.edge_feature = self.gear_net_edge_features(graph)
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            line_graph.edge_weight = torch.ones(
                line_graph.edge_index.shape[1], dtype=torch.float, device=graph.x.device
            )
            edge_input = line_graph.x.float()

        for i in range(len(self.layers)):
            hidden = self.layers[i](graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_input)
                edge_weight = graph.edge_weight.unsqueeze(-1)
                # node_out = graph.edge_index[:, 1] * self.num_relation + graph.edge_index[:, 2]
                node_out = graph.edge_index[1, :] * self.num_relation + graph.edge_index[2, :]
                update = scatter_add(
                    edge_hidden * edge_weight,
                    node_out,
                    dim=0,
                    dim_size=graph.num_nodes * self.num_relation,
                )
                update = update.view(graph.num_nodes, self.num_relation * edge_hidden.shape[1])
                update = self.layers[i].linear(update)
                update = self.layers[i].activation(update)
                hidden = hidden + update
                edge_input = edge_hidden
            if self.batch_norm:
                hidden = self.batch_norms[i](hidden)
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        return EncoderOutput({
            "node_embedding": node_feature,
            "graph_embedding": self.readout(node_feature, graph.batch),
        })

    def get_num_edge_features(self) -> int:
        seq_dist = 1
        dist = 1
        return self.input_dim * 2 + self.num_relation + seq_dist + dist

    def gear_net_edge_features(self, b):
        u = b.x[b.edge_index[0]]
        v = b.x[b.edge_index[1]]
        edge_type = F.one_hot(b.edge_type, self.num_relation)[0]
        dists = torch.pairwise_distance(b.pos[b.edge_index[0]], b.pos[b.edge_index[1]]).unsqueeze(1)
        seq_dist = torch.abs(b.edge_index[0] - b.edge_index[1]).unsqueeze(1)
        return torch.cat([u, v, edge_type, seq_dist, dists], dim=1)


if __name__ == "__main__":
    import hydra
    import omegaconf

    from proteinworkshop import constants

    cfg = omegaconf.OmegaConf.load(constants.PROJECT_PATH / "configs" / "encoder" / "gear_net.yaml")
    enc = hydra.utils.instantiate(cfg)
    print(enc)

