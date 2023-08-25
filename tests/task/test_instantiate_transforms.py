import os

import omegaconf
import pytest
from hydra.utils import instantiate
from torch_geometric.transforms import BaseTransform

from proteinworkshop import constants

TRANSFORM_CONFIG_DIR = constants.PROJECT_PATH / "configs" / "transforms"
TRANSFORMS = os.listdir(TRANSFORM_CONFIG_DIR)


def test_instantiate_encoders():
    for t in TRANSFORMS:
        config_path = TRANSFORM_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)
        transform = instantiate(cfg)

        if t == "none.yaml":
            continue
        else:
            assert transform, f"Transform {t} not instantiated!"


def test_transform_call(example_batch):
    for t in TRANSFORMS:
        config_path = TRANSFORM_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)
        transform = instantiate(cfg)

        if t == "none.yaml":
            continue
        else:
            out = transform(example_batch)
            assert out
