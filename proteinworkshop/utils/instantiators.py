"""Contains utility functions for instantiating callbacks from Hydra config."""
from typing import List

import hydra
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from loguru import logger as log
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
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks

def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: Hydra config for loggers
    :type logger_cfg: DictConfig
    :raises TypeError: If logger config is not a DictConfig
    :return: List of instantiated loggers
    :rtype: List[Logger]
    """

    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger
