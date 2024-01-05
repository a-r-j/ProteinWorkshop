from typing import List, Set, Union

import torch.nn as nn
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import jaxtyped
from torch_geometric.data import Batch
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GCNConv,
    RGATConv,
    RGCNConv,
    Sequential,
    TransformerConv,
)
from torch_geometric.nn.conv import MessagePassing

from proteinworkshop.models.utils import get_activations, get_aggregation
from proteinworkshop.types import ActivationType, EncoderOutput, GNNLayerType


def get_gnn_layer(layer: GNNLayerType) -> MessagePassing:
    """Returns a GNN layer.

    :param layer: Name of the layer.
    :type layer: GNNLayerType
    :return: A GNN layer.
    :rtype: MessagePassing
    """
    if layer == "GCN":
        return GCNConv  # type: ignore
    elif layer == "GATv2":
        return GATv2Conv  # type: ignore
    elif layer == "GAT":
        return GATConv  # type: ignore
    elif layer == "GRAPH_TRANSFORMER":
        return TransformerConv  # type: ignore
    elif layer == "RGCN":
        return RGCNConv  # type: ignore
    elif layer == "RGAT":
        return RGATConv  # type: ignore
    else:
        raise ValueError(f"Unknown layer: {layer}")


class GNNModel(nn.Module):
    def __init__(
        self,
        model_name: GNNLayerType,
        layer_types: List[GNNLayerType],
        n_hid: List[int],
        activations: List[ActivationType],
        dropout: float,
        readout: str,
        edge_types: int = 1,
        edge_weight: bool = False,
        edge_features: bool = False,
    ):
        """Initializes a GNN encoder with the provided parameters.

        :param model_name: Name of the GNN layer to use.
        :type model_name: GNNLayerType
        :param layer_types: List of GNN layer types.
        :type layer_types: List[GNNLayerType]
        :param n_hid: List of hidden dimensions. Must be the same length as
            ``layer_types``.
        :type n_hid: List[int]
        :param activations: List of activation functions. Must be the same
            length as ``layer_types``.
        :type activations: List[ActivationType]
        :param dropout: Amount of dropout to apply.
        :type dropout: float
        :param readout: Readout function to use.
        :type readout: str
        :param edge_types: Number of edge types, defaults to ``1``.
        :type edge_types: int, optional
        :param edge_weight: Whether to use edge weights, defaults to ``False``.
        :type edge_weight: bool, optional
        :param edge_features: Whether to use edge features, defaults to
            ``False``.
        :type edge_features: bool, optional
        """
        super().__init__()
        self.model_name = model_name
        self.layer_types = layer_types
        self.n_hid = n_hid
        self.activations = activations
        self.dropout = dropout

        self.edge_types = edge_types  # TODO in config validation
        self.edge_weight = edge_weight  # TODO in config validation
        self.edge_features = edge_features  # TODO in config validation

        self.layers = self._build_gnn_encoder()
        self.readout = get_aggregation(readout)

    def _build_gnn_encoder(self) -> nn.Sequential:
        """Builds a GNN encoder."""
        assert len(self.n_hid) == len(
            self.activations
        ), f"Hidden dims {len(self.n_hid)} and activations {len(self.activations)} do not match"

        io_str = "x, edge_index -> x"
        input_str = "x, edge_index"
        if self.edge_weight:  # Todo should probably control this elsewhere
            io_str = "x, edge_index, edge_weight -> x"
            input_str = "x, edge_index, edge_weight"
        elif self.edge_features:
            io_str = "x, edge_index, edge_attr -> x"
            input_str = "x, edge_index, edge_attr"

        # First layer
        gnn_layers = [
            (
                get_gnn_layer(self.model_name)(-1, self.n_hid[0]),
                io_str,
            )
        ]
        gnn_layers.append(get_activations(self.activations[0]))
        gnn_layers.append(nn.Dropout(self.dropout))

        for i, dim in enumerate(self.n_hid):
            if i < len(self.n_hid) - 2:
                gnn_layers.append(
                    (
                        get_gnn_layer(self.layer_types[i])(
                            dim, self.n_hid[i + 1]
                        ),
                        io_str,
                    )
                )
                gnn_layers.append(get_activations(self.activations[i + 1]))
                gnn_layers.append(nn.Dropout(self.dropout))

        gnn_layers.append(
            (
                get_gnn_layer(self.model_name)(self.n_hid[-1], self.n_hid[-1]),
                io_str,
            )
        )
        gnn_layers.append(get_activations(self.activations[-1]))
        return Sequential(input_str, gnn_layers)

    @property
    def required_batch_attributes(self) -> Set[str]:
        """Required batch attributes for this encoder.

        If edge_weight is ``True``, then the required batch attributes are:
        * ``x``, ``edge_index``, ``edge_weight``, ``batch``

        If edge_types is ``True``, then the required batch attributes are:
        * ``x``, ``edge_index``, ``edge_attr``, ``batch``

        Otherwise, the required batch attributes are:
        * ``x``, ``edge_index``, ``batch``

        :return: Set of required attributes
        :rtype: Set[str]
        """

        if self.edge_weight:
            return {"x", "edge_index", "edge_weight", "batch"}
        elif self.edge_types:
            return {"x", "edge_index", "edge_attr", "batch"}
        else:
            return {"x", "edge_index", "batch"}

    @jaxtyped(typechecker=typechecker)
    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Implements the forward pass of the GNN encoder.
        
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
        if self.edge_weight:
            x, edge_index, edge_weight = (
                batch.x,
                batch.edge_index,
                batch.edge_weight,
            )
            emb = self.layers(x, edge_index, edge_weight)
        elif self.edge_features:
            x, edge_index, edge_attr = (
                batch.x,
                batch.edge_index,
                batch.edge_attr,
            )
            emb = self.layers(x, edge_index, edge_attr)
        else:
            x, edge_index = batch.x, batch.edge_index
            emb = self.layers(x, edge_index)

        return EncoderOutput(
            {
                "node_embedding": emb,
                "graph_embedding": self.readout(emb, batch.batch),
            }
        )


if __name__ == "__main__":
    import omegaconf
    from hydra.utils import instantiate

    from proteinworkshop import constants

    config_path = constants.SRC_PATH / "config" / "encoder" / "gcn.yaml"
    cfg = omegaconf.OmegaConf.load(config_path)
    enc = instantiate(cfg)
    print(enc)
