"""Implementation of the Torsional Noise Transform."""
import copy
from typing import Union

import torch
from graphein.protein.tensor import pnerf
from graphein.protein.tensor.angles import angle_to_unit_circle, dihedrals
from graphein.protein.tensor.data import Protein
from torch_geometric import transforms as T
from torch_geometric.data import Data


class TorsionalNoiseTransform(T.BaseTransform):
    """
    Adds noise to the torsional angles of a protein.

    Cartesian coordinates are re-computed from the noisy dihedral angles
    using the pNeRF algorithm.

    The true dihedral angles are stored as an attribute on the protein object:
    ``batch.true_dihedrals``.

    .. warning::
        This will subset the data to only include the backbone atoms
        (N, Ca, C). The backbone oxygen can be placed with:
        ``graphein.protein.tensor.reconstruction.place_fourth_coord``.

        This will break, for example, sidechain torsion angle computation for
        the first few chi angles that are partially defined by backbone atoms.
    """

    def __init__(
        self,
        corruption_strategy: str = "gaussian",
        corruption_rate: float = 0.1,
    ):
        """Adds noise to the torsional angles of a protein.

        :param corruption_strategy: Type of noise distribution to use (gaussian
            or uniform), defaults to ``"gaussian"``.
        :type corruption_strategy: str, optional
        :param corruption_rate: Amount to scale noise by, defaults to 0.1
        :type corruption_rate: float, optional
        """
        self.corruption_strategy = corruption_strategy
        self.corruption_rate = corruption_rate

    def __call__(self, batch: Union[Data, Protein]) -> Union[Data, Protein]:
        # Compute uncorrupted dihedrals and store
        dihedral_angles = dihedrals(
            batch.coords, None, rad=True, embed=False
        )  # N_nodes x 3 (phi, psi, omega)
        batch.true_dihedrals = angle_to_unit_circle(
            copy.deepcopy(dihedral_angles)
        )  # N_nodes x 6

        # Compute noise (sample lengths)
        # Given a geodesic length s, the angle theta (in radians) can be
        # calculated as theta = s / r, where r is the radius of the circle.
        # Since the circle is a unit circle, r = 1, so theta = s.
        if self.corruption_strategy == "gaussian":
            noise = (
                torch.randn((dihedral_angles.shape[0], 3))
                * self.corruption_rate
            )
        elif self.corruption_strategy == "uniform":
            # Uniform noise in the range [-noise_std, noise_std]
            noise = (
                (torch.rand((dihedral_angles.shape[0], 3)) - 0.5)
                * 2
                * self.corruption_rate
            )
        else:
            raise ValueError(
                f"Unknown noise scheme: {self.corruption_strategy}. Must be 'gaussian' or 'uniform'"
            )

        # Store Cos/Sin transformed noised
        batch.torsional_noise = angle_to_unit_circle(noise)  # N_nodes x 6

        # The noised angle nu is the sum of the initial angle and the angle
        # corresponding to the geodesic length,
        # i.e., nu = phi/psi/omega + theta.
        noised_angles = dihedral_angles + noise  # N_nodes x 3

        # Embed noised angles onto the unit circle.
        # I.e. theta = [cos(theta), sin(theta)]
        noised_angles = angle_to_unit_circle(noised_angles)  # N_nodes x 6

        # Recompute coordinates from noisy dihedral angles using pNeRF
        # Uses uncorrupted bond lengths and angles
        noised_coords = pnerf.reconstruct_dihedrals(
            noised_angles, batch.coords
        )
        batch.coords = noised_coords.reshape(-1, 3, 3)
        return batch

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(Corruption Strategy: {self.corruption_strategy}, Corruption Rate: {self.corruption_rate})"


if __name__ == "__main__":
    import hydra
    import omegaconf
    from graphein.protein.tensor.data import get_random_protein

    config = omegaconf.OmegaConf.load(
        "../proteinworkshop/config/transforms/torsional_denoising.yaml"
    )

    a = get_random_protein()
    b = copy.deepcopy(a)

    t = hydra.utils.instantiate(config.torsional_denoising)
    print(t)

    out = t(a)

    def rmsd(a, b):
        return torch.sqrt(torch.mean((a - b) ** 2))

    print(rmsd(out.coords[:, 1, :], b.coords[:, 1, :]))

    out.plot_structure(["N", "CA", "C"])
