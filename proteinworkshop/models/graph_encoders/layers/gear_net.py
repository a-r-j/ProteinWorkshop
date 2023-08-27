import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_scatter import scatter_add
from torchdrug import data, utils


class MessagePassingBase(nn.Module):
    """
    Base module for message passing.
    Any custom message passing module should be derived from this class.
    """

    gradient_checkpoint = False

    def message(self, graph, input):
        """
        Compute edge messages for the graph.
        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`
        Returns:
            Tensor: edge messages of shape :math:`(|E|, ...)`
        """
        raise NotImplementedError

    def aggregate(self, graph, message):
        """
        Aggregate edge messages to nodes.
        Parameters:
            graph (Graph): graph(s)
            message (Tensor): edge messages of shape :math:`(|E|, ...)`
        Returns:
            Tensor: node updates of shape :math:`(|V|, ...)`
        """
        raise NotImplementedError

    def message_and_aggregate(self, graph, input):
        """
        Fused computation of message and aggregation over the graph.
        This may provide better time or memory complexity than separate calls of
        :meth:`message <MessagePassingBase.message>` and :meth:`aggregate <MessagePassingBase.aggregate>`.
        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`
        Returns:
            Tensor: node updates of shape :math:`(|V|, ...)`
        """
        message = self.message(graph, input)
        return self.aggregate(graph, message)

    def _message_and_aggregate(self, *tensors):
        graph = data.Graph.from_tensors(tensors[:-1])
        input = tensors[-1]
        return self.message_and_aggregate(graph, input)

    def combine(self, input, update):
        """
        Combine node input and node update.
        Parameters:
            input (Tensor): node representations of shape :math:`(|V|, ...)`
            update (Tensor): node updates of shape :math:`(|V|, ...)`
        """
        raise NotImplementedError

    def forward(self, graph, input):
        """
        Perform message passing over the graph(s).
        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`
        """
        update = self.message_and_aggregate(graph, input)
        return self.combine(input, update)


class RelationalGraphConv(MessagePassingBase):
    """
    Relational graph convolution operator from `Modeling Relational Data with Graph Convolutional Networks`_.
    .. _Modeling Relational Data with Graph Convolutional Networks:
        https://arxiv.org/pdf/1703.06103.pdf
    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    eps = 1e-10

    def __init__(
        self,
        input_dim,
        output_dim,
        num_relation,
        edge_input_dim=None,
        batch_norm=False,
        activation="relu",
    ):
        super(RelationalGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.edge_input_dim = edge_input_dim
        self.batch_norm = nn.BatchNorm1d(output_dim) if batch_norm else None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        # self.self_loop = nn.Linear(input_dim, output_dim)
        self.self_loop = nn.LazyLinear(output_dim)
        # self.linear = nn.Linear(num_relation * input_dim, output_dim)
        self.linear = nn.LazyLinear(output_dim)
        self.edge_linear = nn.LazyLinear(input_dim) if edge_input_dim else None

    def message(self, graph, input):
        node_in = graph.edge_index[:, 0]
        message = input[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        return message

    def aggregate(self, graph, message):
        assert graph.num_relation == self.num_relation

        node_out = (
            graph.edge_index[:, 1] * self.num_relation + graph.edge_index[:, 2]
        )
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(
            message * edge_weight,
            node_out,
            dim=0,
            dim_size=graph.num_nodes * self.num_relation,
        ) / (
            scatter_add(
                edge_weight,
                node_out,
                dim=0,
                dim_size=graph.num_nodes * self.num_relation,
            )
            + self.eps
        )
        return update.view(graph.num_nodes, self.num_relation * self.input_dim)

    def message_and_aggregate(self, graph, input):
        assert graph.num_relation == self.num_relation

        node_in, node_out, relation = graph.edge_index
        node_out = node_out * self.num_relation + relation
        degree_out = scatter_add(
            graph.edge_weight,
            node_out,
            dim_size=graph.num_nodes * graph.num_relation,
        )
        edge_weight = graph.edge_weight / degree_out[node_out]
        adjacency = utils.sparse_coo_tensor(
            torch.stack([node_in, node_out]),
            edge_weight,
            (graph.num_nodes, graph.num_nodes * graph.num_relation),
        )
        # update = torch.sparse.mm(adjacency.t(), input)
        adjacency = adjacency.t().to_sparse_csr()
        # update = adjacency.t() @ input
        update = adjacency @ input
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = self.edge_linear(edge_input)
            edge_weight = edge_weight.unsqueeze(-1)
            edge_update = scatter_add(
                edge_input * edge_weight,
                node_out,
                dim=0,
                dim_size=graph.num_nodes * graph.num_relation,
            )
            update += edge_update

        return update.view(graph.num_nodes, self.num_relation * self.input_dim)

    def combine(self, input, update):
        output = self.linear(update) + self.self_loop(input)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class GeometricRelationalGraphConv(RelationalGraphConv):
    """
    Geometry-aware relational graph convolution operator from
    `Protein Representation Learning by Geometric Structure Pretraining`_.
    .. _Protein Representation Learning by Geometric Structure Pretraining:
        https://arxiv.org/pdf/2203.06125.pdf
    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_relation: int,
        edge_input_dim: Optional[int] = None,
        batch_norm: bool = False,
        activation: str = "relu",
    ):
        super(GeometricRelationalGraphConv, self).__init__(
            input_dim,
            output_dim,
            num_relation,
            edge_input_dim,
            batch_norm,
            activation,
        )

    def aggregate(self, graph, message):
        assert graph.num_relation == self.num_relation

        node_out = (
            graph.edge_index[:, 1] * self.num_relation + graph.edge_index[:, 2]
        )
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(
            message * edge_weight,
            node_out,
            dim=0,
            dim_size=graph.num_nodes * self.num_relation,
        )
        update = update.view(
            graph.num_nodes, self.num_relation * self.input_dim
        )

        return update

    def message_and_aggregate(self, graph, input):
        assert graph.num_relation == self.num_relation

        node_in, node_out, relation = graph.edge_index
        node_out = node_out * self.num_relation + relation

        adjacency = utils.sparse_coo_tensor(
            torch.stack([node_in, node_out]),
            graph.edge_weight,
            (graph.num_nodes, graph.num_nodes * graph.num_relation),
        )
        # update = torch.sparse.mm(adjacency.t(), input)
        adjacency = adjacency.t().to_sparse_csr()
        # update = adjacency.t() @ input
        update = adjacency @ input
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = self.edge_linear(edge_input)
            edge_weight = graph.edge_weight.unsqueeze(-1)
            edge_update = scatter_add(
                edge_input * edge_weight,
                node_out,
                dim=0,
                dim_size=graph.num_nodes * graph.num_relation,
            )
            update += edge_update

        return update.view(graph.num_nodes, self.num_relation * self.input_dim)


class SpatialLineGraph(nn.Module):
    """
    Spatial line graph construction module from `Protein Representation Learning by Geometric Structure Pretraining`_.

    .. _Protein Representation Learning by Geometric Structure Pretraining:
        https://arxiv.org/pdf/2203.06125.pdf

    Parameters:
        num_angle_bin (int, optional): number of bins to discretize angles between edges
    """

    def __init__(self, num_angle_bin=8):
        super(SpatialLineGraph, self).__init__()
        self.num_angle_bin = num_angle_bin

    def forward(self, graph):
        """
        Generate the spatial line graph of the input graph.
        The edge types are decided by the angles between two adjacent edges in the input graph.

        Parameters:
            graph (PackedGraph): :math:`n` graph(s)

        Returns:
            graph (PackedGraph): the spatial line graph
        """
        line_graph = get_line_graph(graph)
        node_in, node_out = graph.edge_index[:2]
        edge_in, edge_out = line_graph.edge_index  # .t()

        # compute the angle ijk
        node_i = node_out[edge_out]
        node_j = node_in[edge_out]
        node_k = node_in[edge_in]
        vector1 = graph.pos[node_i] - graph.pos[node_j]
        vector2 = graph.pos[node_k] - graph.pos[node_j]
        x = (vector1 * vector2).sum(dim=-1)
        y = torch.cross(vector1, vector2).norm(dim=-1)
        angle = torch.atan2(y, x)
        relation = (angle / math.pi * self.num_angle_bin).long()
        edge_list = torch.cat(
            [line_graph.edge_index, relation.unsqueeze(0)], dim=0
        )  # .t()

        return type(line_graph)(
            edge_index=edge_list,
            num_nodes=line_graph.num_nodes,
            # offsets=line_graph._offsets,
            num_edges=line_graph.num_edges,
            num_relation=self.num_angle_bin,
            x=line_graph.x,
            # meta_dict=line_graph.meta_dict,
            # **line_graph.data_dict,
        )


def get_line_graph(graph: Batch):
    """
    Construct a line graph of this graph.
    The node feature of the line graph is inherited from the edge feature of the original graph.
    In the line graph, each node corresponds to an edge in the original graph.
    For a pair of edges (a, b) and (b, c) that share the same intermediate node in the original graph,
    there is a directed edge (a, b) -> (b, c) in the line graph.
    Returns:
        Graph
    """
    node_in, node_out = graph.edge_index[:2]
    edge_index = torch.arange(graph.edge_index.shape[1], device=graph.x.device)
    edge_in = edge_index[node_out.argsort()]
    edge_out = edge_index[node_in.argsort()]

    degree_in = node_in.bincount(minlength=graph.num_nodes)
    degree_out = node_out.bincount(minlength=graph.num_nodes)
    size = degree_out * degree_in
    starts = (size.cumsum(0) - size).repeat_interleave(size)
    range = torch.arange(size.sum(), device=graph.x.device)
    # each node u has degree_out[u] * degree_in[u] local edges
    local_index = range - starts
    local_inner_size = degree_in.repeat_interleave(size)
    edge_in_offset = (degree_out.cumsum(0) - degree_out).repeat_interleave(
        size
    )
    edge_out_offset = (degree_in.cumsum(0) - degree_in).repeat_interleave(size)
    edge_in_index = (
        torch.div(local_index, local_inner_size, rounding_mode="floor")
        + edge_in_offset
    )
    edge_out_index = local_index % local_inner_size + edge_out_offset

    edge_in = edge_in[edge_in_index]
    edge_out = edge_out[edge_out_index]
    edge_list = torch.stack([edge_in, edge_out], dim=-1)
    x = getattr(graph, "edge_feature", None)
    num_nodes = graph.edge_index.shape[1]
    num_edge = size.sum()

    return Batch(
        edge_index=edge_list.t(), num_nodes=num_nodes, num_edge=num_edge, x=x
    )
