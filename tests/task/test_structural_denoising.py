import pytest
import torch
from graphein.protein.tensor.data import get_random_protein
from proteinworkshop.tasks.structural_denoising import StructuralNoiseTransform


def test_structure_noise_transform():
    p = get_random_protein()
    task = StructuralNoiseTransform(corruption_rate=5, corruption_strategy="uniform")

    def rmsd(x, y):
        return torch.sqrt(torch.mean((x - y) ** 2))

    p = task(p)

    assert rmsd(p.coords, p.coords_uncorrupted) > 1
    assert rmsd(p.coords_uncorrupted, p.coords) < 5
