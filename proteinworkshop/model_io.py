"""Utilities for loading models."""
import collections
import os
import pathlib
from typing import Any, Dict, Tuple, Union

import omegaconf
import torch
from loguru import logger

import wandb
from proteinworkshop.models.base import BenchMarkModel


def load_model_from_checkpoint(
    ckpt_path: Union[str, os.PathLike],
    load_state: bool = False,
    load_weights: bool = True,
) -> Tuple[BenchMarkModel, omegaconf.DictConfig]:
    """Loads a model from a checkpoint.

    :param ckpt_path: Path to the checkpoint.
    :type ckpt_path: Union[str, os.PathLike]
    :param load_weights: Whether to load the weights from the checkpoint.
    :type load_weights: bool
    :return: The model.
    """
    config = load_config_from_checkpoint(ckpt_path)
    model = BenchMarkModel(config)

    if load_state:
        return model.load_from_checkpoint(ckpt_path), config

    if load_weights:
        ckpt = torch.load(ckpt_path)
        return load_weights(model, ckpt["state_dict"]), config
    else:
        return model, config


def load_config_from_checkpoint(
    ckpt_path: Union[str, os.PathLike]
) -> omegaconf.DictConfig:
    """Loads a config from a checkpoint.

    :param ckpt_path: Path to the checkpoint.
    :type ckpt_path: Union[str, os.PathLike]
    :return: The config.
    """
    ckpt = torch.load(ckpt_path)
    config = ckpt["hyper_parameters"]["cfg"]
    config = omegaconf.OmegaConf.create(config)
    return config


def load_model_from_wandb(
    run_id: str,
    entity: str,
    project: str,
    version: str = "v0",
    load_state: bool = False,
    load_weights: bool = True,
) -> Dict[str, Any]:
    """Loads a model from wandb."""
    api = wandb.Api()
    run_name = f"{entity}/{project}/{run_id}"
    run = api.run(run_name)

    # Download config
    files = run.files()
    downloaded = False
    for file in files:
        if file.name == "config.yaml":
            file.download()
            logger.info(f"Download config for run {run_id}")
            downloaded = True

    if not downloaded:
        logger.error(f"Failed to download config for {run_name}")

    # TODO: Automatically select best model
    ckpt_ref = f"{entity}/{project}/model-{run_id}:{version}"
    artifact = api.artifact(ckpt_ref, type="model")
    artifact_dir = artifact.download()

    return load_model_from_checkpoint(
        pathlib.Path(artifact_dir) / "model.ckpt"
    )


def load_weights(
    model: BenchMarkModel, state_dict: Dict[str, Any]
) -> BenchMarkModel:
    """Load weights from a state dict into a model.

    :param model: The model.
    :type model: BenchMarkModel
    :param state_dict: The state dict containing weights.
    :type state_dict: Dict[str, Any]
    :return: The model with loaded weights.
    """
    encoder_weights = collections.OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("encoder"):
            encoder_weights[k.replace("encoder.", "")] = v
    err = model.encoder.load_state_dict(encoder_weights, strict=False)
    logger.warning(f"Error loading encoder weights: {err}")

    decoder_weights = collections.OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("decoder"):
            decoder_weights[k.replace("decoder.", "")] = v
    err = model.decoder.load_state_dict(decoder_weights, strict=False)
    logger.warning(f"Error loading decoder weights: {err}")

    logger.info("Freezing encoder!")
    for param in model.encoder.parameters():
        param.requires_grad = False

    logger.info("Freezing decoder!")
    for param in model.decoder.parameters():
        param.requires_grad = False

    return model
