"""Contains functions for instantiating loggers from Hydra config."""
from typing import List

import hydra
from lightning.pytorch.loggers import Logger
from loguru import logger as log
from omegaconf import DictConfig


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
