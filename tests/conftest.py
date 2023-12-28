"""This file prepares config fixtures for other tests."""

from pathlib import Path

import pyrootutils
import pytest
from graphein.protein.tensor.data import ProteinBatch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from proteinworkshop.datasets.utils import create_example_batch


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training.

    :return: A DictConfig object containing a default Hydra configuration for
        training.
    """
    with initialize(
        version_base="1.3", config_path="../proteinworkshop/config/"
    ):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=["dataset=dummy", "logger=csv"],
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.env.paths.root_dir = str(
                pyrootutils.find_root(indicator=".project-root")
            )
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.5
            cfg.trainer.limit_val_batches = 0.5
            cfg.trainer.limit_test_batches = 0.5
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.dataset.datamodule.num_workers = 0
            cfg.dataset.datamodule.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.hydra.job.num = 0

    return cfg


@pytest.fixture(scope="package")
def cfg_finetune_global() -> DictConfig:
    """
    A pytest fixture for setting up a default Hydra DictConfig for evaluation.

    :return: A DictConfig containing a default Hydra configuration for evaluation.
    """
    with initialize(
        version_base="1.3", config_path="../proteinworkshop/config/"
    ):
        cfg = compose(
            config_name="finetune.yaml",
            return_hydra_config=True,
            overrides=["ckpt_path=.", "dataset=dummy", "logger=csv"],
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.env.paths.root_dir = str(
                pyrootutils.find_root(indicator=".project-root")
            )
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.5
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.dataset.datamodule.num_workers = 0
            cfg.dataset.datamodule.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None
            cfg.hydra.job.num = 0

    return cfg


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_train_global()` fixture,
    which accepts a temporary logging path `tmp_path` for generating a
    temporary logging path.

    This is called by each test which uses the `cfg_train` arg. Each test
    generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding
        to `tmp_path`.
    """
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.env.paths.output_dir = str(tmp_path)
        cfg.env.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_finetune(
    cfg_finetune_global: DictConfig, tmp_path: Path
) -> DictConfig:
    """
    A pytest fixture built on top of the `cfg_finetune_global()` fixture,
    which accepts a temporary logging path `tmp_path` for generating a
    temporary logging path.

    This is called by each test which uses the `cfg_finetune` arg. Each test
    generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding
        to `tmp_path`.
    """
    cfg = cfg_finetune_global.copy()

    with open_dict(cfg):
        cfg.env.paths.output_dir = str(tmp_path)
        cfg.env.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def example_batch() -> ProteinBatch:
    """Creates a random batch of proteins for testing"""
    return create_example_batch()
