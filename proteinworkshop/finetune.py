"""Entry point for finetuning a pretrained model."""
import collections
import copy
import sys
from typing import List

import graphein
import hydra
import lightning as L
import lovely_tensors as lt
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from loguru import logger as log
from omegaconf import DictConfig

from proteinworkshop import (
    constants,
    register_custom_omegaconf_resolvers,
    utils,
)
from proteinworkshop.configs import config
from proteinworkshop.models.base import BenchMarkModel

graphein.verbose(False)
lt.monkey_patch()


def finetune(cfg: DictConfig):
    assert cfg.ckpt_path, "No checkpoint path provided."

    L.seed_everything(cfg.seed)

    log.info("Instantiating datamodule:... ")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(
        cfg.dataset.datamodule
    )

    log.info("Instantiating model:... ")
    model: L.LightningModule = BenchMarkModel(cfg)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.callbacks.instantiate_callbacks(
        cfg.get("callbacks")
    )

    log.info("Instantiating loggers:... ")
    logger: List[Logger] = utils.loggers.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    # Initialize lazy layers for parameter counts
    # This is also required for the model to be able to load weights
    # Otherwise lazy layers will have their parameters reset
    # https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin
    log.info("Initializing lazy layers...")
    with torch.no_grad():
        datamodule.setup()  # type: ignore
        batch = next(iter(datamodule.val_dataloader()))
        log.info(f"Unfeaturized batch: {batch}")
        batch = model.featurise(batch)
        log.info(f"Featurized batch: {batch}")
        out = model.forward(batch)
        log.info(f"Model output: {out}")
        del batch, out

    # We only want to load weights
    if cfg.ckpt_path != "none":
        log.info(f"Loading weights from checkpoint {cfg.ckpt_path}...")
        state_dict = torch.load(cfg.ckpt_path)["state_dict"]

        if cfg.finetune.encoder.load_weights:
            encoder_weights = collections.OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("encoder"):
                    encoder_weights[k.replace("encoder.", "")] = v
            log.info(f"Loading encoder weights: {encoder_weights}")
            err = model.encoder.load_state_dict(encoder_weights, strict=False)
            log.warning(f"Error loading encoder weights: {err}")

        if cfg.finetune.decoder.load_weights:
            decoder_weights = collections.OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("decoder"):
                    decoder_weights[k.replace("decoder.", "")] = v
            log.info(f"Loading decoder weights: {decoder_weights}")
            err = model.decoder.load_state_dict(decoder_weights, strict=False)
            log.warning(f"Error loading decoder weights: {err}")

        if cfg.finetune.encoder.freeze:
            log.info("Freezing encoder!")
            for param in model.encoder.parameters():
                param.requires_grad = False

        if cfg.finetune.decoder.freeze:
            log.info("Freezing decoder!")
            for param in model.decoder.parameters():
                param.requires_grad = False
    else:
        log.info("No checkpoint path provided, skipping loading weights!")

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.logging_utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)  # type: ignore

    log.info("Starting finetuning!")
    trainer.fit(model=model, datamodule=datamodule)

    metric_dict = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        # Run test on all splits if using fold_classification dataset
        if (
            cfg.dataset.datamodule._target_
            == "proteinworkshop.datasets.fold_classification.FoldClassificationDataModule"
        ):
            splits = ["fold", "family", "superfamily"]
            wandb_logger = copy.deepcopy(trainer.logger)
            for split in splits:
                dataloader = datamodule.get_test_loader(split)
                trainer.logger = False
                results = trainer.test(
                    model=model, dataloaders=dataloader, ckpt_path="best"
                )[0]
                results = {f"{k}/{split}": v for k, v in results.items()}
                log.info(f"{split}: {results}")
                wandb_logger.log_metrics(results)
        else:
            trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    return metric_dict, object_dict


@hydra.main(
    version_base="1.3",
    config_path=str(constants.HYDRA_CONFIG_PATH),
    config_name="finetune",
)
def _main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)
    cfg = config.validate_config(cfg)
    finetune(cfg)


def _script_main(args):
    """
    Provides an entry point for the script dispatcher.

    Sets the sys.argv to the provided args and calls the main train function.
    """
    register_custom_omegaconf_resolvers()
    sys.argv = args
    _main()


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    _main()
