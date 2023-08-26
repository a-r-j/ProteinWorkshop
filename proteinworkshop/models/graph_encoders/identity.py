from typing import Set, Union

import torch.nn as nn
from beartype import beartype
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import jaxtyped
from proteinworkshop.models.utils import get_aggregation
from proteinworkshop.types import EncoderOutput
from torch_geometric.data import Batch


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

    @jaxtyped
    @beartype
    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Implements the forward pass of the IdentityModel class.

        Returns the node embedding and graph embedding in a dictionary.

        :param batch: Batch of data to encode.
        :type batch: Union[Batch, ProteinBatch]
        :return: Dictionary of node and graph embeddings.
        :rtype: EncoderOutput
        """
        output = {
            "node_embedding": batch.x,
            "graph_embedding": self.readout(batch.x, batch.batch),
        }
        return EncoderOutput(output)
