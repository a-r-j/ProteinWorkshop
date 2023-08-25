from typing import Set, Union

import torch.nn as nn
from beartype import beartype
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import jaxtyped
from proteinworkshop.models.utils import get_aggregation
from proteinworkshop.types import EncoderOutput
from torch_geometric.data import Batch


class IdentityModel(nn.Module):
    """Identity encoder for debuggin purposes."""
    def __init__(self, readout: str = "sum"):
        super().__init__()
        self.readout = get_aggregation(readout)

    @property
    def required_batch_attributes(self) -> Set[str]:
        return {"x", "batch"}

    @jaxtyped
    @beartype
    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        output = {
            "node_embedding": batch.x,
            "graph_embedding": self.readout(batch.x, batch.batch),
        }
        return EncoderOutput(output)
