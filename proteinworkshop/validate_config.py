"""Main module to load and train the model. This should be the program entry point."""
import argparse
import logging

import hydra
from omegaconf import DictConfig

from proteinworkshop import constants, utils
from proteinworkshop.configs import config

logging.getLogger("graphein").setLevel(logging.WARNING)


# Load hydra config from yaml filses and command line arguments.
def _main(cfg: DictConfig) -> None:
    """Load and validate the hydra config."""
    utils.extras(cfg)
    cfg = config.validate_config(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="template")
    args = parser.parse_args()

    decorator = hydra.main(
        version_base="1.3",
        config_path=str(constants.HYDRA_CONFIG_PATH),
        config_name=args.config,
    )

    decorator(_main)()  # pylint: disable=no-value-for-parameter
