"""Linear Decoders"""
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from loguru import logger

from proteinworkshop.models.utils import get_activations
from proteinworkshop.types import ActivationType


class LinearSkipBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: List[int],
        activations: List[ActivationType],
        out_dim: int,
        dropout: float = 0.0,
        skip: Literal["sum", "concat"] = "sum",
    ):
        """
        Initialise MLP with skip connections.

        :param hidden_dim: List of hidden dimensions
        :type hidden_dim: List[int]
        :param activations: List of activation functions
        :type activations: List[ActivationType]
        :param out_dim: Dimension of output
        :type out_dim: int
        :param dropout: Amount of dropout to apply, defaults to 0.0
        :type dropout: float, optional
        :param skip: Type of skip connection to use, defaults to "sum"
        :type skip: Literal["sum", "concat"], optional
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.activation_fns = activations
        self.dropout = dropout
        self._build_layers()
        self.skip = skip

    def _build_layers(self):
        """
        Build MLP layers and instantiate activation functions and dropout
        layers.
        """
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # Iterate over hidden dims
        # N.B We use lazy layers to avoid having to figure out the appropriate
        # input dimensions given the skip connections
        for i in range(len(self.hidden_dim)):
            self.layers.append(nn.LazyLinear(out_features=self.hidden_dim[i]))
            self.activations.append(get_activations(self.activation_fns[i]))
            self.dropout_layers.append(nn.Dropout(self.dropout))

        self.layers.append(nn.LazyLinear(self.out_dim))
        self.activations.append(get_activations(self.activation_fns[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Implements the forward pass of the MLP decoder with skip connections.

        :param x: Input tensor
        :type x: torch.Tensor
        :return: Output tensor
        :rtype: torch.Tensor
        """
        for i, layer in enumerate(self.layers):
            prev = x
            if i == len(self.layers) - 1:
                # No dropout on final layer
                return self.activations[i](layer(x))
            x = self.dropout_layers[i](self.activations[i](layer(x)))
            if self.skip == "concat":
                x = torch.cat([x, prev], dim=-1)
            elif self.skip == "sum":
                x = x + prev


class MLPDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: List[int],
        out_dim: int,
        activations: List[ActivationType],
        dropout: float,
        skip: bool = True,
        input: Optional[str] = None,
    ):
        """Initialise MLP decoder.

        :param hidden_dim: List of hidden dimensions
        :type hidden_dim: List[int]
        :param out_dim: Dimension of output
        :type out_dim: int
        :param activations: List of activation functions
        :type activations: List[ActivationType]
        :param dropout: Amount of dropout to apply
        :type dropout: float
        :param skip: Whether to use skip connections, defaults to ``True``
        :type skip: bool, optional
        :param input: Name of the encoder output to use as input
            (e.g. ``node_embedding`` or ``graph_embedding``),
            defaults to ``None``
        :type input: Optional[str], optional
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.activations = activations
        self.dropout = dropout
        self.input = input

        assert (
            len(self.activations) == len(self.hidden_dim) + 1
        ), f"Decoder activations {self.activations} and dims {self.hidden_dim} of incorrect length."

        if skip in {"sum", "concat"}:
            logger.info("Using skip connection in decoder.")
            self.layers = LinearSkipBlock(
                self.hidden_dim, self.activations, out_dim, dropout, skip
            )
        else:
            # First layer
            decoder_layers = nn.ModuleList([nn.LazyLinear(self.hidden_dim[0])])
            decoder_layers.append(get_activations(self.activations[0]))
            decoder_layers.append(nn.Dropout(self.dropout))

            # Iterate over remaining layers
            for i, _ in enumerate(self.hidden_dim):
                if i < len(self.hidden_dim) - 1:
                    decoder_layers.append(
                        nn.LazyLinear(self.hidden_dim[i + 1])
                    )
                    decoder_layers.append(
                        get_activations(self.activations[i + 1])
                    )
                    decoder_layers.append(nn.Dropout(self.dropout))

            # Last layer
            decoder_layers.append(nn.LazyLinear(out_dim))
            decoder_layers.append(get_activations(self.activations[-1]))
            self.layers = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MLP decoder.

        :param x: Input tensor
        :type x: torch.Tensor
        :return: Output tensor
        :rtype: torch.Tensor
        """
        return self.layers(x)


class PositionDecoder(nn.Module):
    def __init__(
        self,
        num_message_layers: int,
        message_hidden_dim: int,
        message_activation: ActivationType,
        message_dropout: float,
        message_skip: bool,
        num_distance_layers: int,
        distance_hidden_dim: int,
        distance_activation: ActivationType,
        distance_dropout: float,
        distance_skip: bool,
        aggr: str = "sum",
    ):
        """Implement MLP decoder for equivariant position prediction.

        :param num_message_layers: Number of message passing layers
        :type num_message_layers: int
        :param message_hidden_dim: Dimension of hidden layers in message MLP
        :type message_hidden_dim: int
        :param message_activation: Activation function to use in message MLP
        :type message_activation: ActivationType
        :param message_dropout: Amount of dropout to apply in message MLP
        :type message_dropout: float
        :param message_skip: Whether to use skip connections in message MLP
        :type message_skip: bool
        :param num_distance_layers: Number of distance MLP layers
        :type num_distance_layers: int
        :param distance_hidden_dim: Hidden dimension of distance MLP
        :type distance_hidden_dim: int
        :param distance_activation: Activation function to use in distance MLP
        :type distance_activation: ActivationType
        :param distance_dropout: Dropout to apply in distance MLP
        :type distance_dropout: float
        :param distance_skip: Whether to use skip connections in distance MLP
        :type distance_skip: bool
        :param aggr: Aggregation function to use in message passing,
            defaults to "sum"
        :type aggr: str, optional
        """

        super().__init__()
        self.aggr = aggr

        self.message_mlp = MLPDecoder(
            hidden_dim=[message_hidden_dim] * num_message_layers,
            activations=[message_activation] * num_message_layers + ["none"],
            dropout=message_dropout,
            skip=message_skip,
            out_dim=1,
        )

        self.distance_mlp = MLPDecoder(
            hidden_dim=[distance_hidden_dim] * num_distance_layers,
            activations=[distance_activation] * num_distance_layers + ["none"],
            skip=distance_skip,
            dropout=distance_dropout,
            out_dim=1,
        )
        self.requires_pos = True

    def forward(
        self,
        edge_index: torch.Tensor,
        scalar_features: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Implement forward pass of MLP decoder for equivariant position prediction.

        :param edge_index: Tensor of edge indices
        :type edge_index: torch.Tensor
        :param scalar_features: Tensor of scalar features ``(N x D)``
        :type scalar_features: torch.Tensor
        :param pos: Tensor of positions ``(N x 3)``
        :type pos: torch.Tensor
        :return: Tensor of predicted positions ``(N x 3)``
        :rtype: torch.Tensor
        """
        dists = torch.pairwise_distance(
            pos[edge_index[0]], pos[edge_index[1]]
        ).unsqueeze(-1)
        vecs = F.normalize(pos[edge_index[0]] - pos[edge_index[1]], dim=-1)

        dists = self.distance_mlp(dists)
        message_input = torch.cat(
            [
                scalar_features[edge_index[0]],
                scalar_features[edge_index[1]],
                dists,
            ],
            dim=-1,
        )
        message = self.message_mlp(message_input)

        x = message * vecs

        return torch_scatter.scatter(x, edge_index[1], dim=0, reduce=self.aggr)
