"""Implements sequence denoising task."""
import copy
from typing import Literal, Set, Union

import torch
from beartype import beartype as typechecker
from graphein.protein.tensor.data import Protein
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class SequenceNoiseTransform(BaseTransform):
    def __init__(
        self,
        corruption_rate: float,
        corruption_strategy: Literal["mutate", "mask"],
    ):
        """Corrupts the sequence of a protein by randomly flipping residues to
        another type or masking them.

        .. note::

             - The data object this is called on must have a ``residue_type``
             attribute. See: :py:meth:`required_attributes`
             - The original sequence is stored as
             ``data.residue_type_uncorrupted``.
             - The indices of the corrupted residues are stored as
             ``data.sequence_corruption_mask``.


        :param corruption_rate: Fraction of residues to corrupt.
        :type corruption_rate: float
        :param corruption_strategy: Strategy to use for corruption. Either:
            ``mutate`` or ``mask``.
        :type corruption_strategy: Literal[mutate, mask]
        """
        self.corruption_rate = corruption_rate
        self.corruption_strategy = corruption_strategy

    @property
    def required_attributes(self) -> Set[str]:
        return {"residue_type"}

    @typechecker
    def __call__(self, x: Union[Data, Protein]) -> Union[Data, Protein]:
        x.residue_type_uncorrupted = copy.deepcopy(x.residue_type)
        # Get indices of residues to corrupt
        indices = torch.randint(
            0,
            x.residue_type.shape[0],
            (int(x.residue_type.shape[0] * self.corruption_rate),),
            device=x.residue_type.device,
        ).long()

        # Apply corruption
        if self.corruption_strategy == "mutate":
            # Set indices to random residue type
            x.residue_type[indices] = torch.randint(
                0,
                23,  # TODO: probably best to not hardcode this
                (indices.shape[0],),
                device=x.residue_type.device,
            )
        elif self.corruption_strategy == "mask":
            # Set indices to 23 -> "UNK"
            x.residue_type[
                indices
            ] = 23  # TODO: probably best to not hardcode this
        else:
            raise NotImplementedError(
                f"Corruption strategy: {self.corruption_strategy} not supported."
            )
        # Get indices of applied corruptions
        index = torch.zeros(x.residue_type.shape[0])
        index[indices] = 1
        x.sequence_corruption_mask = index.bool()

        return x

    def __repr__(self) -> str:
        return f"{self.__class__}(corruption_strategy: {self.corruption_strategy} corruption_rate: {self.corruption_rate})"


if __name__ == "__main__":
    from graphein.protein.tensor.data import get_random_protein

    p = get_random_protein()

    orig_residues = p.residue_type

    task = SequenceNoiseTransform(
        corruption_rate=0.99, corruption_strategy="mutate"
    )

    p = task(p)

    print(p.residue_type)
    print(p.residue_type_uncorrupted)

    task = SequenceNoiseTransform(
        corruption_rate=0.99, corruption_strategy="mask"
    )

    p = task(p)
    print(p.residue_type)
    print(p.residue_type_uncorrupted)
