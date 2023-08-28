import copy

import hydra
import omegaconf
import torch
from graphein.protein.tensor.data import get_random_protein

from proteinworkshop import constants


def test_instantiate_transform():
    config_path = (
        constants.HYDRA_CONFIG_PATH
        / "transforms"
        / "torsional_denoising.yaml"
    )

    config = omegaconf.OmegaConf.load(config_path)

    a = get_random_protein()
    b = copy.deepcopy(a)

    t = hydra.utils.instantiate(config.torsional_denoising)

    out = t(a)

    def rmsd(a, b):
        return torch.sqrt(torch.mean((a - b) ** 2))

    assert rmsd(out.coords, b.coords[:, :3, :]) > 3
