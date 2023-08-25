
import hydra
import omegaconf
import torch
import copy
from graphein.protein.tensor.data import get_random_protein


def test_instantiate_transform():

    config = omegaconf.OmegaConf.load(
        "../../configs/transforms/torsional_denoising.yaml"
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