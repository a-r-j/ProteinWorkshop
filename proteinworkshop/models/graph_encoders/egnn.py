from typing import Set, Union

import torch
import torch.nn as nn
from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch

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
        residual: bool = True,
    ):
        """E(n) Equivariant GNN model

        Instantiates an instance of the EGNNModel class with the provided
        parameters.

        :param num_layers: Number of message passing layers, defaults to ``5``
        :type num_layers: int, optional
        :param emb_dim: Dimension of the node embeddings, defaults to ``128``
        :type emb_dim: int, optional
        :param activation: Activation function to use, defaults to ``"relu"``
        :type activation: str, optional
        :param norm: Normalisation layer to use, defaults to ``"layer"``
        :type norm: str, optional
        :param aggr: Aggregation function to use, defaults to ``"sum"``
        :type aggr: str, optional
        :param pool: Pooling operation to use, defaults to ``"sum"``
        :type pool: str, optional
        :param residual: Whether to use residual connections, defaults to
            ``True``
        :type residual: bool, optional
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
        """Required batch attributes for this encoder.

        - ``x``: Node features (shape: :math:`(n, d)`)
        - ``pos``: Node positions (shape: :math:`(n, 3)`)
        - ``edge_index``: Edge indices (shape: :math:`(2, e)`)
        - ``batch``: Batch indices (shape: :math:`(n,)`)

        :return: Set of required batch attributes
        :rtype: Set[str]
        """
        return {"x", "pos", "edge_index", "batch"}

    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Performs a forward pass of the EGNN model.

        Returns the node embedding and graph embedding in a dictionary with
        fields ``node_embedding`` and ``graph_embedding``. The node embedding
        is of shape :math:`(n, d)` and the graph embedding is of shape
        :math:`(batch_size, d)`, where :math:`n` is the number of nodes and
        :math:`d` is the dimension of the embeddings.

        :param batch: Batch of data to encode
        :type batch: Union[Batch, ProteinBatch]
        :return: Dictionary of node and graph embeddings
        :rtype: EncoderOutput
        """
        h = self.emb_in(batch.x)  # (n,) -> (n, d)
        pos = batch.pos  # (n, 3)

        for conv in self.convs:
            # Message passing layer
            h_update, pos_update = conv(h, pos, batch.edge_index)

            # Update node features (n, d) -> (n, d)
            h = h + h_update if self.residual else h_update

            # Update node coordinates (no residual) (n, 3) -> (n, 3)
            pos = pos_update

        return EncoderOutput(
            {
                "node_embedding": h,
                "graph_embedding": self.pool(
                    h, batch.batch
                ),  # (n, d) -> (batch_size, d)
                "pos": pos,  # Position
            }
        )


if __name__ == "__main__":
    import hydra
    import omegaconf

    from proteinworkshop import constants

    cfg = omegaconf.OmegaConf.load(
        constants.PROJECT_PATH / "configs" / "encoder" / "egnn.yaml"
    )
    enc = hydra.utils.instantiate(cfg)
    print(enc)
