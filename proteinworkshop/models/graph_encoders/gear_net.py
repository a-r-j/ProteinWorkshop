from typing import Optional, Set, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from graphein.protein.tensor.data import ProteinBatch
from loguru import logger as log
from torch_geometric.data import Batch
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
        """Initializes an instance of the GearNet model.

        :param input_dim: Dimension of the input node features
        :type input_dim: int
        :param num_relation: Number of edge types
        :type num_relation: int
        :param num_layers: Number of layers in the model
        :type num_layers: int
        :param emb_dim: Dimension of the node embeddings
        :type emb_dim: int
        :param short_cut: Whether to use short cut connections
        :type short_cut: bool
        :param concat_hidden: Whether to concatenate hidden representations
        :type concat_hidden: bool
        :param batch_norm: Whether to use batch norm
        :type batch_norm: bool
        :param num_angle_bin: Number of angle bins for edge message passing.
            If ``None``, edge message passing is not disabled.
        :type num_angle_bin: Optional[int]
        :param activation: Activation function to use, defaults to "relu"
        :type activation: str, optional
        :param pool: Pooling operation to use, defaults to "sum"
        :type pool: str, optional
        """
        super().__init__()
        # Base parameters
        self.num_relation = num_relation
        self.input_dim = input_dim
        # Edge message passing layers
        # If not None, this enables Edge Message passing
        self.num_angle_bin = num_angle_bin
        self.edge_input_dim = self._get_num_edge_features()
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
            self.edge_input_dim = self._get_num_edge_features()
            self.edge_dims = [self.edge_input_dim] + self.dims[:-1]
            self.spatial_line_graph = gear_net.SpatialLineGraph(
                self.num_angle_bin
            )
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
        """Required batch attributes for this encoder.

        - ``x`` Positions (shape ``[num_nodes, 3]``)
        - ``edge_index`` Edge indices (shape ``[2, num_edges]``)
        - ``edge_type`` Edge types (shape ``[num_edges]``)
        - ``edge_attr`` Edge attributes (shape ``[num_edges, num_edge_features]``)
        - ``num_nodes`` Number of nodes (int)
        - ``batch`` Batch indices (shape ``[num_nodes]``)

        :return: Set of required batch attributes.
        :rtype: Set[str]
        """
        return {
            "x",
            "pos",
            "edge_index",
            "edge_type",
            "edge_attr",
            "num_nodes",
            "batch",
        }

    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Implements the forward pass of the GearNet encoder.

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
        hiddens = []
        batch.edge_weight = torch.ones(
            batch.edge_index.shape[1], dtype=torch.float, device=batch.x.device
        )
        layer_input = batch.x
        batch.edge_index = torch.cat([batch.edge_index, batch.edge_type])
        batch.edge_feature = self.gear_net_edge_features(batch)
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(batch)
            line_graph.edge_weight = torch.ones(
                line_graph.edge_index.shape[1],
                dtype=torch.float,
                device=batch.x.device,
            )
            edge_input = line_graph.x.float()

        for i in range(len(self.layers)):
            hidden = self.layers[i](batch, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_input)
                edge_weight = batch.edge_weight.unsqueeze(-1)
                # node_out = graph.edge_index[:, 1] * self.num_relation + graph.edge_index[:, 2]
                node_out = (
                    batch.edge_index[1, :] * self.num_relation
                    + batch.edge_index[2, :]
                )
                update = scatter_add(
                    edge_hidden * edge_weight,
                    node_out,
                    dim=0,
                    dim_size=batch.num_nodes * self.num_relation,
                )
                update = update.view(
                    batch.num_nodes, self.num_relation * edge_hidden.shape[1]
                )
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

        return EncoderOutput(
            {
                "node_embedding": node_feature,
                "graph_embedding": self.readout(node_feature, batch.batch),
            }
        )

    def _get_num_edge_features(self) -> int:
        """Compute the number of edge features."""
        seq_dist = 1
        dist = 1
        return self.input_dim * 2 + self.num_relation + seq_dist + dist

    def gear_net_edge_features(
        self, b: Union[Batch, ProteinBatch]
    ) -> torch.Tensor:
        """Compute edge features for the gear net encoder.

        - Concatenate node features of the two nodes in each edge
        - Concatenate the edge type
        - Compute the distance between the two nodes in each edge
        - Compute the sequence distance between the two nodes in each edge

        :param b: Batch of data to encode.
        :type b: Union[Batch, ProteinBatch]
        :return: Edge features
        :rtype: torch.Tensor
        """
        u = b.x[b.edge_index[0]]
        v = b.x[b.edge_index[1]]
        edge_type = F.one_hot(b.edge_type, self.num_relation)[0]
        dists = torch.pairwise_distance(
            b.pos[b.edge_index[0]], b.pos[b.edge_index[1]]
        ).unsqueeze(1)
        seq_dist = torch.abs(b.edge_index[0] - b.edge_index[1]).unsqueeze(1)
        return torch.cat([u, v, edge_type, seq_dist, dists], dim=1)


if __name__ == "__main__":
    import hydra
    import omegaconf

    from proteinworkshop import constants

    cfg = omegaconf.OmegaConf.load(
        constants.SRC_PATH / "config" / "encoder" / "gear_net.yaml"
    )
    enc = hydra.utils.instantiate(cfg)
    print(enc)
