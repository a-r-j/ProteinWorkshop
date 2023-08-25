from graphein.protein.tensor.data import get_random_protein
from proteinworkshop.tasks.sequence_denoising import SequenceNoiseTransform


def test_instantiate_transform():
    p = get_random_protein()

    orig_residues = p.residue_type

    task = SequenceNoiseTransform(corruption_rate=0.99, corruption_strategy="mutate")

    p = task(p)

    print(p.residue_type)
    print(p.residue_type_uncorrupted)

    task = SequenceNoiseTransform(corruption_rate=0.99, corruption_strategy="mask")

    p = task(p)
    print(p.residue_type)
    print(p.residue_type_uncorrupted)
