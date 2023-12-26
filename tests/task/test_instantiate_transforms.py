import os

import omegaconf
from hydra.utils import instantiate

from proteinworkshop import constants

TRANSFORM_CONFIG_DIR = constants.HYDRA_CONFIG_PATH / "transforms"
TRANSFORMS = os.listdir(TRANSFORM_CONFIG_DIR)


def test_instantiate_transforms():
    for t in TRANSFORMS:
        config_path = TRANSFORM_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)

        if t == "multihot_label_encoding.yaml":
            cfg.multihot_label_encoding.num_classes = 2
        transform = instantiate(cfg)

        if t == "none.yaml":
            continue
        else:
            assert transform, f"Transform {t} not instantiated!"


def test_transform_call(example_batch):
    for t in TRANSFORMS:
        config_path = TRANSFORM_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)

        if t == "multihot_label_encoding.yaml":
            cfg.multihot_label_encoding.num_classes = 2

        transform = instantiate(cfg)

        t = t.removesuffix(".yaml")

        if t == "ppi_site_prediction":
            continue
        elif t == "binding_site_prediction":
            continue
        elif t == "default":
            continue
        elif t == "none":
            continue

        transform = transform[t]
        out = transform(example_batch)
        assert out
