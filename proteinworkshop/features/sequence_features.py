"""Sequence features for protein data objects."""
from typing import Union

import torch
import torch.nn.functional as F
from beartype import beartype as typechecker
from torch_geometric.data import Batch, Data


@typechecker
def amino_acid_one_hot(
    x: Union[Batch, Data], num_classes: int = 23
) -> torch.Tensor:
    """Returns one-hot encoding of amino acid sequence.

    :param x: Protein data object containing a ``residue_type`` attribute.
    :type x: Union[Batch, Data]
    :param num_classes: Number of classes to encode, defaults to 23
    :type num_classes: int, optional
    :returns: One-hot encoding of amino acid sequence
    :rtype: torch.Tensor
    """
    return F.one_hot(x.residue_type, num_classes=num_classes).float()
