"""Implements a transform for corrupting the Cartesian coordinates of a protein structure."""
import copy
from typing import Literal, Set, Union

import torch
from beartype import beartype as typechecker
from graphein.protein.tensor.data import Protein
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class StructuralNoiseTransform(BaseTransform):
    """Adds noise to the coordinates of a protein structure.

    Sets the following attributes on the protein data object:

    - ``coords_uncorrupted``: The original coordinates of the protein.
    - ``noise``: The noise added to the coordinates.
    - ``coords``: The original coordinates + noise.

    :param corruption_rate: Magnitude of corruption to apply to the coordinates.
    :type corruption_rate: float
    :param corruption_strategy: Noise strategy to use for corruption.
    :type corruption_strategy: Literal["uniform", "gaussian"]
    """

    def __init__(
        self,
        corruption_rate: float,
        corruption_strategy: Literal["uniform", "gaussian"],
    ):
        self.corruption_rate = corruption_rate
        self.corruption_strategy = corruption_strategy

    @property
    def required_attributes(self) -> Set[str]:
        return {"coords"}

    @typechecker
    def __call__(self, x: Union[Data, Protein]) -> Union[Data, Protein]:
        """Adds noise to the coordinates of a protein structure.

        :param x: Protein data object
        :type x: Union[Data, Protein]
        :raises ValueError: If the corruption strategy is not supported.
        :return: Protein data object with corrupted coordinates.
        :rtype: Union[Data, Protein]
        """
        x.coords_uncorrupted = copy.deepcopy(x.coords)

        with torch.no_grad():
            if self.corruption_strategy == "uniform":
                noise = torch.rand_like(x.coords, device=x.coords.device)
                noise = (noise - 0.5) * 2 * self.corruption_rate
            elif self.corruption_strategy == "gaussian":
                noise = (
                    torch.randn_like(x.coords, device=x.coords.device)
                    * self.corruption_rate
                )
            else:
                raise ValueError(
                    f"Corruption strategy: {self.corruption_strategy} not supported."
                )

        pad_indices = torch.where(x.coords == 1e-5)
        x.noise = noise
        x.coords += noise
        x.coords[pad_indices] = 1e-5
        return x

    def __repr__(self) -> str:
        return f"{self.__class__}(corruption_strategy: {self.corruption_strategy} corruption_rate: {self.corruption_rate})"


if __name__ == "__main__":
    from graphein.protein.tensor.data import get_random_protein

    p = get_random_protein()
    task = StructuralNoiseTransform(
        corruption_rate=5, corruption_strategy="uniform"
    )

    def rmsd(x, y):
        return torch.sqrt(torch.mean((x - y) ** 2))

    p = task(p)

    print(rmsd(p.coords, p.coords_uncorrupted))
