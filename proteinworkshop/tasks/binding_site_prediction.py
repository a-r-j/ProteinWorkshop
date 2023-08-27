from typing import List, Set, Union

import torch
from graphein.protein.tensor.data import Protein
from scipy import spatial
from torch_geometric import transforms
from torch_geometric.data import Data

from proteinworkshop.features.representation import get_full_atom_coords


class BindingSiteTransform(transforms.BaseTransform):
    """
    Extracts binding site labels for a given set of HETATMs.

    This transform builds a KDTree from the protein coordinates.
    Atoms belonging to HETATMs (specified by the ``hetatms`` arg at
    initialization) are then queried against the KDTree to obtain indices of
    residues within ``threshold`` distance of the HETATM.

    These indices are used to assign node labels to the protein graph.
    If ``multilabel`` is set to ``True``, then each binding HETATM will be
    assigned a separate label (i.e. whether residue
    :math:`i` is proximal to HETATM :math:`j` is given by:
    :math:`\hat{y}_{ij} \in \mathbb{R}^{|V| \times |H|}`).
    Otherwise, the labels will be assigned as a single label
    (i.e. is residue :math:`i` proximal to any HETATM :math:`\hat{y} \in
    \mathbb{R}^{|V|}`). proximal to any HETATM).

    If ``ca_only`` is set to ``True``, then only the alpha carbon atoms will be
    used to determine proximity. If ``ca_only`` is set to ``False``, then all
    atoms will be used to determine proximity. I.e. if any atom in a residue is
    within ``threshold`` distance of a HETATM, then the residue will be labeled
    accordingly.

    .. warning::
        This transform requires that the ``data.coords`` and ``data.hetatms``
        fields to be set on the input Data/Batch. See:
        :py:meth:`required_attributes`
    """

    def __init__(
        self,
        hetatms: List[str],
        threshold: float,
        ca_only: bool = False,
        multilabel: bool = True,
    ) -> None:
        """Initializes the BindingSiteTransform.

        :param hetatms: List of HETATM names to use for labeling.
        :type hetatms: List[str]
        :param threshold: Threshold distance for determining proximity in
            angstroms.
        :type threshold: float
        :param ca_only: Whether to use only the alpha carbon atoms for
            assigning proximity labels, defaults to ``False``.
        :type ca_only: bool, optional
        :param multilabel: Whether to assign multilabel labels,
            defaults to ``True``
        :type multilabel: bool, optional
        """
        self.hetatms = hetatms
        self.threshold = threshold
        self.ca_only = ca_only
        self.multilabel = multilabel
        self.num_classes = len(hetatms) if multilabel else 1

    @property
    def required_attributes(self) -> Set[str]:
        """Returns the required batch attributes that this transform requires.

        I.e. ``data.coords`` and ``data.hetatms`` must be set.

        :return: Set of required attributes
        :rtype: Set[str]
        """
        return {"coords", "hetatms"}

    def __call__(self, data: Union[Data, Protein]) -> Union[Data, Protein]:
        # Create a KDTree
        # If Ca only, we only see if the hetatms are within the
        # threshold distance of Ca atoms on the input structure
        if self.ca_only:
            kd_tree = spatial.KDTree(data.coords[:, 1, :])
        else:
            # If we are not using CA only, we need to flatten the coordinates
            # And keep track of the atom->residue mapping
            coords, res_idx, _ = get_full_atom_coords(data.coords)
            kd_tree = spatial.KDTree(coords)

        if self.multilabel:
            label = torch.zeros((data.coords.shape[0], self.num_classes))
        else:
            label = torch.zeros(data.coords.shape[0])

        for hetatm_idx, hetatm in enumerate(self.hetatms):
            try:
                indices = kd_tree.query_ball_point(
                    data.hetatms[0][hetatm].numpy(), r=self.threshold, p=2.0
                )
                indices = [item for sublist in indices for item in sublist]
                indices = torch.tensor(indices, dtype=torch.long)

                # If not CA only, we need to map the atom indices back to residues
                if not self.ca_only:
                    indices = torch.unique(res_idx[indices])

                if self.multilabel:
                    label[indices, hetatm_idx] = 1
                else:
                    label[indices] = 1
                setattr(data, "node_y", label)
            except KeyError:
                continue

        return data

    def __repr__(self) -> str:
        return f"{self.__class__}(hetatms: {self.hetatms}, threshold: {self.threshold})"


if __name__ == "__main__":
    import graphein

    graphein.verbose(False)
    from graphein.protein.tensor.io import protein_to_pyg

    pdb_code = "3eiy"

    p = protein_to_pyg(pdb_code=pdb_code, store_het=True)
    print(p)
    print(p.hetatms)

    transform = BindingSiteTransform(
        hetatms=["HOH", "POP", "SO4", "PEG"], threshold=7.0, ca_only=True
    )
    print(transform)

    p = transform(p)
    print(p)
    print(p.node_y)
