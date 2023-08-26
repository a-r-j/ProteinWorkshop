"""Contains utility functions for instantiating callbacks from Hydra config."""
from typing import List

import hydra
from lightning.pytorch.callbacks import Callback
from loguru import logger
from omegaconf import DictConfig


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """
    Instantiates callbacks from Hydra config.

    :param callbacks_cfg: Hydra config for callbacks
    :type callbacks_cfg: DictConfig
    :raises TypeError: If callbacks config is not a DictConfig
    :return: List of instantiated callbacks
    :rtype: List[Callback]
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        logger.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            logger.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks
