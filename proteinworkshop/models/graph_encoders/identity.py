from typing import Set, Union

import torch.nn as nn
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import jaxtyped
from torch_geometric.data import Batch

from proteinworkshop.models.utils import get_aggregation
from proteinworkshop.types import EncoderOutput


class IdentityModel(nn.Module):
    """Identity encoder for debugging purposes."""

    def __init__(self, readout: str = "sum"):
        """Initializes an instance of the IdentityModel class.

        :param readout: Readout to use for graph embedding, defaults to "sum"
        :type readout: str, optional
        """
        super().__init__()
        self.readout = get_aggregation(readout)

    @property
    def required_batch_attributes(self) -> Set[str]:
        """Set of required batch attributes for this encoder.

        Requires: ``x`` and ``batch``

        :return: Set of required batch attributes.
        :rtype: Set[str]
        """
        return {"x", "batch"}

    @jaxtyped(typechecker=typechecker)
    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Implements the forward pass of the IdentityModel encoder.

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
        output = {
            "node_embedding": batch.x,
            "graph_embedding": self.readout(batch.x, batch.batch),
        }
        return EncoderOutput(output)
