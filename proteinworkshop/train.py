"""
Main module to load and train the model. This should be the program entry
point.
"""
import copy
import sys
from typing import List, Optional

import graphein
import hydra
import lightning as L
import lovely_tensors as lt
import torch
import torch.nn as nn
import torch_geometric
from graphein.protein.tensor.dataloader import ProteinDataLoader
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


def _num_training_steps(
    train_dataset: ProteinDataLoader, trainer: L.Trainer
) -> int:
    """
    Returns total training steps inferred from datamodule and devices.

    :param train_dataset: Training dataloader
    :type train_dataset: ProteinDataLoader
    :param trainer: Lightning trainer
    :type trainer: L.Trainer
    :return: Total number of training steps
    :rtype: int
    """
    if trainer.max_steps != -1:
        return trainer.max_steps

    dataset_size = (
        trainer.limit_train_batches
        if trainer.limit_train_batches not in {0, 1}
        else len(train_dataset) * train_dataset.batch_size
    )

    log.info(f"Dataset size: {dataset_size}")

    num_devices = max(1, trainer.num_devices)
    effective_batch_size = (
        train_dataset.batch_size
        * trainer.accumulate_grad_batches
        * num_devices
    )
    return (dataset_size // effective_batch_size) * trainer.max_epochs


def train_model(
    cfg: DictConfig, encoder: Optional[nn.Module] = None
):  # sourcery skip: extract-method
    """
    Trains a model from a config.

    If ``encoder`` is provided, it is used instead of the one specified in the
    config.

    1. The datamodule is instantiated from ``cfg.dataset.datamodule``.
    2. The callbacks are instantiated from ``cfg.callbacks``.
    3. The logger is instantiated from ``cfg.logger``.
    4. The trainer is instantiated from ``cfg.trainer``.
    5. (Optional) If the config contains a scheduler, the number of training steps is
         inferred from the datamodule and devices and set in the scheduler.
    6. The model is instantiated from ``cfg.model``.
    7. The datamodule is setup and a dummy forward pass is run to initialise
    lazy layers for accurate parameter counts.
    8. Hyperparameters are logged to wandb if a logger is present.
    9. The model is compiled if ``cfg.compile`` is True.
    10. The model is trained if ``cfg.task_name`` is ``"train"``.
    11. The model is tested if ``cfg.test`` is ``True``.

    :param cfg: DictConfig containing the config for the experiment
    :type cfg: DictConfig
    :param encoder: Optional encoder to use instead of the one specified in
        the config
    :type encoder: Optional[nn.Module]
    """
    # set seed for random number generators in pytorch, numpy and python.random
    L.seed_everything(cfg.seed)

    log.info(
        f"Instantiating datamodule: <{cfg.dataset.datamodule._target_}..."
    )
    datamodule: L.LightningDataModule = hydra.utils.instantiate(
        cfg.dataset.datamodule
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.callbacks.instantiate_callbacks(
        cfg.get("callbacks")
    )

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.loggers.instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating trainer...")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    if cfg.get("scheduler"):
        if (
            cfg.scheduler.scheduler._target_
            == "flash.core.optimizers.LinearWarmupCosineAnnealingLR"
            and cfg.scheduler.interval == "step"
        ):
            datamodule.setup()  # type: ignore
            num_steps = _num_training_steps(
                datamodule.train_dataloader(), trainer
            )
            log.info(
                f"Setting number of training steps in scheduler to: {num_steps}"
            )
            cfg.scheduler.scheduler.warmup_epochs = (
                num_steps / trainer.max_epochs
            )
            cfg.scheduler.scheduler.max_epochs = num_steps
            log.info(cfg.scheduler)

    log.info("Instantiating model...")
    model: L.LightningModule = BenchMarkModel(cfg)

    if encoder is not None:
        log.info(f"Setting user-defined encoder {encoder}...")
        model.encoder = encoder

    log.info("Initializing lazy layers...")
    with torch.no_grad():
        datamodule.setup()  # type: ignore
        batch = next(iter(datamodule.val_dataloader()))
        log.info(f"Unfeaturized batch: {batch}")
        batch = model.featurise(batch)
        log.info(f"Featurized batch: {batch}")
        log.info(f"Example labels: {model.get_labels(batch)}")
        # Check batch has required attributes
        for attr in model.encoder.required_batch_attributes:  # type: ignore
            if not hasattr(batch, attr):
                raise AttributeError(
                    f"Batch {batch} does not have required attribute: {attr} ({model.encoder.required_batch_attributes})"
                )
        out = model(batch)
        log.info(f"Model output: {out}")
        del batch, out

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
        model = torch_geometric.compile(model, dynamic=True)

    if cfg.get("task_name") == "train":
        log.info("Starting training!")
        trainer.fit(
            model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
        )

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


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    version_base="1.3",
    config_path=str(constants.HYDRA_CONFIG_PATH),
    config_name="train",
)
def _main(cfg: DictConfig) -> None:
    """Load and validate the hydra config."""
    utils.extras(cfg)
    cfg = config.validate_config(cfg)
    train_model(cfg)


def _script_main(args: List[str]) -> None:
    """
    Provides an entry point for the script dispatcher.

    Sets the sys.argv to the provided args and calls the main train function.
    """
    sys.argv = args
    register_custom_omegaconf_resolvers()
    _main()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    register_custom_omegaconf_resolvers()
    _main()  # type: ignore
