from typing import Set, Union

from graphein.protein.tensor.angles import dihedrals
from graphein.protein.tensor.data import Protein
from torch_geometric import transforms as T
from torch_geometric.data import Data


class BackboneDihedralPredictionTransform(T.BaseTransform):
    """
    Transform to store backbone dihedral angles as attributes on proteins.

    This is used for setting the labels in a SSL context,
    not for featurisation.

    Sets dihedrals as an attribute of the Batch object (i.e.
    ``batch.dihedrals``). This is retrieved in
    :py:meth:`proteinworkshop.models.base.BaseModel.get_labels` for supervision.

    .. seealso::
        :py:meth:`proteinworkshop.models.base.BaseModel.get_labels`
    """

    def __init__(self):
        pass

    @property
    def required_attributes(self) -> Set[str]:
        """Required batch attributes for this transform.

         - ``coords`` are required for computing dihedrals. This is a tensor of
            shape :math:`(N, 37, 3)` where :math:`N` is the number of residues,
            37 is the number of unique atoms in PDBs, and 3 is the x, y, z
            position of each atom.

        :return: Set of required attributes
        :rtype: Set[str]
        """
        return {"coords"}

    def __call__(self, x: Union[Protein, Data]) -> Union[Protein, Data]:
        """
        Compute backbone dihedral angles for a protein.

        Sets dihedrals as an attribute of the Batch object. This is retrieved
        in :py:meth:`proteinworkshop.models.base.BaseModel.get_labels`.

        :param x: Protein data object
        :type x: Union[Protein, Data]
        :return: Protein data object with dihedrals as an attribute
        :rtype: Union[Protein, Data]
        """
        x.dihedrals = dihedrals(x.coords, rad=True, embed=True)
        return x


# For debugging:
if __name__ == "__main__":
    import graphein

    graphein.verbose(False)
    from graphein.protein.tensor.io import protein_to_pyg

    pdb_code = "3eiy"
    p = protein_to_pyg(pdb_code=pdb_code, store_het=True)

    transform = BackboneDihedralPredictionTransform()
    print(transform)

    p = transform(p)
    print(p)
    print(p.dihedrals)
