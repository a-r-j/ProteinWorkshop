import hydra
import omegaconf
import pytest
import torch.nn as nn
from proteinworkshop import constants


def test_instantiate_gear_net():
    cfg = omegaconf.OmegaConf.load(
        constants.PROJECT_PATH / "configs" / "encoder" / "gear_net.yaml"
    )
    enc = hydra.utils.instantiate(cfg)

    assert enc is not None
    assert isinstance(enc, nn.Module)


def test_encoder_forward_pass():
    cfg = omegaconf.OmegaConf.load(
        constants.PROJECT_PATH / "configs" / "encoder" / "gear_net.yaml"
    )
    enc = hydra.utils.instantiate(cfg)
