from typing import Dict, Union

import numpy as np
import scipy.spatial as spatial
import torch
from graphein.protein.tensor.data import Protein
from torch_geometric import transforms as T
from torch_geometric.data import Data

from proteinworkshop.features.representation import get_full_atom_coords


class BindingSiteTransform(T.BaseTransform):
    def __init__(self, radius: float = 3.5, ca_only: bool = True) -> None:
        """Extracts Protein-Protein interaction sites from a protein structure.

        .. note::

            The chains to be kept as inputs must be specified as
            ``data.graph_y``. This is typically set in the dataloader.

        :param radius: Maximum distance between chains to be considered as
            interacting, defaults to 3.5 angstrom
        :type radius: float, optional
        :param ca_only: Whether to use only the alpha carbon atoms for
            determining interactions
        :type ca_only: bool, optional
        """
        self.radius = radius
        self.fill_value = 1e-5
        self.ca_only = ca_only
        charstr: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.chain_map: Dict[str, int] = {
            charstr[i]: i for i in range(len(charstr))
        }

    def __call__(self, data: Union[Protein, Data]):
        # Map the chain labels to integers
        target_chains = []

        chain_strs = [res.split(":")[0] for res in data.residue_id]
        chain_strs = list(np.unique(chain_strs))

        for chain in data.graph_y:
            target_chains.append(chain_strs.index(chain))

        target_chains = torch.tensor(target_chains)
        target_indices = torch.where(torch.isin(data.chains, target_chains))[0]

        # Create a mask for the target chains
        mask = torch.zeros(data.coords.shape[0], dtype=torch.bool)
        mask[target_indices] = True

        # Extract the target chains and the other chains
        target_struct = data.coords[mask]
        other_chains = data.coords[~mask]

        N_TARGET_RESIDUES = target_struct.shape[0]

        # Unwrap the coordinates
        other_chains = other_chains.reshape(-1, 3)
        # Remove any rows with 1e-5
        other_chains = other_chains[
            ~torch.all(other_chains == self.fill_value, dim=1)
        ]

        # Create a KDTree
        # If Ca only, we only see if the interacting chains are within the
        # threshold distance of Ca atoms on the input chains
        if self.ca_only:
            kd_tree = spatial.KDTree(target_struct[:, 1, :])
        else:
            # If we are not using CA only, we need to flatten the coordinates
            # And keep track of the atom->residue mapping
            coords, res_idx, _ = get_full_atom_coords(target_struct)
            kd_tree = spatial.KDTree(coords)

        indices = kd_tree.query_ball_point(other_chains, self.radius)
        indices = [item for sublist in indices for item in sublist]
        indices = torch.tensor(indices, dtype=torch.long)

        # If not CA only, we need to map the atom indices back to residues
        if not self.ca_only:
            indices = torch.unique(res_idx[indices])

        label = torch.zeros(N_TARGET_RESIDUES)
        label[indices] = 1
        data.node_y = label.long()

        # Delete the graph label containing the chains to avoid the potential
        # to incorrectly use them as label
        del data.graph_y
        # Subset the data to only the target chains
        data.coords = target_struct
        data.residue_type = data.residue_type[mask]
        data.residues = np.array(data.residues)[mask]
        data.residue_id = np.array(data.residue_id)[mask]
        data.chains = data.chains[mask]

        if data.x is not None:
            data.x = data.x[mask]

        if hasattr(data, "seq_pos"):
            data.seq_pos = data.seq_pos[mask]

        if hasattr(data, "amino_acid_one_hot"):
            data.amino_acid_one_hot = data.amino_acid_one_hot[mask]

        return data


if __name__ == "__main__":
    from graphein.protein.tensor.data import get_random_protein

    a = get_random_protein()
    a.graph_label = "A"

    t = BindingSiteTransform(radius=4, ca_only=False)
    out = t(a)
    print(out)
    print(out.node_y.sum())
