"""Implementation of a transform to remove residues with missing CA atoms."""
import numpy as np
import torch
from torch_geometric import transforms as T


class RemoveMissingCa(T.BaseTransform):
    """Removes residues with missing CA atoms from a protein structure."""

    def __init__(self, fill_value: float = 1e-5, ca_idx: int = 1) -> None:
        """Initialise the transform.

        :param fill_value: Value used to denote missing atoms in the
            ``Protein`` data object. Defaults to ``1e-5``.
        :type fill_value: float, optional
        :param ca_idx: Index of the CA atom (in dimension 1) in the coords
            attribute of the Protein data object. By default this is 1, as the
            coords attribute is of shape ``(N, 37, 3)`` where ``N`` is the
            number of residues.
        :type ca_idx: int, optional
        """
        self.fill_value = fill_value
        self.ca_idx = ca_idx

    def __call__(self, data):
        """Remove residues with missing CA atoms from a protein structure.

        :param data: Protein data object.
        :type data: Protein
        :return: Protein data object with missing residues removed.
        :rtype: Protein
        """
        # Check for missing CA atoms
        # If there are no missing CA atoms, return the data
        mask = data.coords[:, self.ca_idx, 0] != self.fill_value
        if torch.all(mask):
            return data

        data.coords = data.coords[mask]
        data.residue_type = data.residue_type[mask]
        data.residues = np.array(data.residues)[mask]
        data.residue_id = np.array(data.residue_id)[mask]
        data.chains = data.chains[mask]

        if data.x is not None:
            data.x = data.x[mask]

        if hasattr(data, "amino_acid_one_hot"):
            data.amino_acid_one_hot = data.amino_acid_one_hot[mask]

        if hasattr(data, "seq_pos"):
            data.seq_pos = data.seq_pos[mask]
        return data


if __name__ == "__main__":
    from graphein.protein.tensor.data import get_random_protein

    a = get_random_protein()
    print(a)

    t = RemoveMissingCa()
    print(t(a))

    a = torch.load(
        "../../../protein-workshop/data/FoldClassification/processed/d2vzsa2.pt"
    )
    print(a)

    t = RemoveMissingCa()
    print(t(a))
