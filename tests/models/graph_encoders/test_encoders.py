import copy
import os
from typing import List

import omegaconf
import pytest
from hydra.utils import instantiate

from proteinworkshop import constants, register_custom_omegaconf_resolvers

ENCODERS: List[str] = [
    "gvp",
    "gcn",
    "schnet",
    "dimenet_plus_plus",
    "egnn",
    "gcpnet",
    "gear_net",
    "gear_net_edge",
    "identity",
]

FEATURES = os.listdir(constants.PROJECT_PATH / "configs" / "features")

register_custom_omegaconf_resolvers()


def test_instantiate_encoders():
    for encoder in ENCODERS:
        config_path = (
            constants.PROJECT_PATH / "configs" / "encoder" / f"{encoder}.yaml"
        )
        cfg = omegaconf.OmegaConf.create()
        cfg.encoder = omegaconf.OmegaConf.load(config_path)
        cfg.features = omegaconf.OmegaConf.load(
            constants.PROJECT_PATH / "configs" / "features" / "ca_bb.yaml"
        )
        cfg.task = omegaconf.OmegaConf.load(
            constants.PROJECT_PATH
            / "configs"
            / "task"
            / "ppi_site_prediction.yaml"
        )

        enc = instantiate(cfg.encoder)

        assert enc, f"Encoder {encoder} not instantiated!"


@pytest.mark.skip(reason="Too slow for GitHub Actions. Works locally.")
def test_encoder_forward_pass(example_batch):
    for encoder in ENCODERS:
        for feature in FEATURES:
            encoder_config_path = (
                constants.PROJECT_PATH
                / "configs"
                / "encoder"
                / f"{encoder}.yaml"
            )
            feature_config_path = (
                constants.PROJECT_PATH / "configs" / "features" / feature
            )

            cfg = omegaconf.OmegaConf.create()
            cfg.encoder = omegaconf.OmegaConf.load(encoder_config_path)
            cfg.features = omegaconf.OmegaConf.load(feature_config_path)
            cfg.task = omegaconf.OmegaConf.load(
                constants.PROJECT_PATH
                / "configs"
                / "task"
                / "ppi_site_prediction.yaml"
            )

            enc = instantiate(cfg.encoder)
            featuriser = instantiate(cfg.features)

            batch = featuriser(copy.copy(example_batch))
            print(example_batch)
            print(enc)
            out = enc(batch)
            assert out
            assert isinstance(out, dict)
            assert "node_embedding" in out
            assert "graph_embedding" in out
