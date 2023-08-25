import pytest
import torch
from graphein.protein.tensor.data import get_random_protein
from proteinworkshop.tasks.structural_denoising import StructuralNoiseTransform


def test_instantiate_transform():
    p = get_random_protein()
    task = StructuralNoiseTransform(corrution_rate=5, corruption_strategy="uniform")

    def rmsd(x, y):
        return torch.sqrt(torch.mean((x - y) ** 2))

    p = task(p)

    print(rmsd(p.coords, p.coords_uncorrupted))
