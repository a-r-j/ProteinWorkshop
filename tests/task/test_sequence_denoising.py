from graphein.protein.tensor.data import get_random_protein
import copy
from proteinworkshop.tasks.sequence_denoising import SequenceNoiseTransform


def test_sequence_noise_transform_mutate():
    p = get_random_protein()

    orig_residues = copy.deepcopy(p.residue_type)

    task = SequenceNoiseTransform(corruption_rate=0.99, corruption_strategy="mutate")

    p = task(p)

    assert (p.residue_type != orig_residues).sum() > 0
    assert (p.residue_type_uncorrupted != p.residue_type).sum() > 0
    assert (
        p.residue_type_uncorrupted == orig_residues
    ).sum() == p.residue_type_uncorrupted.shape[0]


def test_sequence_noise_transform_mask():
    p = get_random_protein()

    orig_residues = copy.deepcopy(p.residue_type)
    task = SequenceNoiseTransform(corruption_rate=0.99, corruption_strategy="mask")

    p = task(p)

    assert (p.residue_type != orig_residues).sum() > 0
    assert (p.residue_type_uncorrupted != p.residue_type).sum() > 0
    assert (
        p.residue_type_uncorrupted == orig_residues
    ).sum() == p.residue_type_uncorrupted.shape[0]
