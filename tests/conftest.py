"""This file prepares config fixtures for other tests."""

from pathlib import Path

import pyrootutils
import pytest
from graphein.protein.tensor.data import (ProteinBatch, get_random_batch,
                                          get_random_protein)
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict
import functools
import torch.nn.functional as F

from proteinworkshop.features.node_features import orientations
from proteinworkshop.features.utils import _normalize
from proteinworkshop.features.edge_features import pos_emb

@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training.

    :return: A DictConfig object containing a default Hydra configuration for
        training.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=["dataset=dummy"])

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
            cfg.logger = None

    return cfg


@pytest.fixture(scope="package")
def cfg_finetune_global() -> DictConfig:
    """
    A pytest fixture for setting up a default Hydra DictConfig for evaluation.

    :return: A DictConfig containing a default Hydra configuration for evaluation.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="finetune.yaml",
            return_hydra_config=True,
            overrides=["ckpt_path=.", "dataset=dummy"],
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
def cfg_finetune(cfg_finetune_global: DictConfig, tmp_path: Path) -> DictConfig:
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


@functools.lru_cache()
def _example_batch() -> ProteinBatch:
    proteins = []
    for _ in range(4):
        p = get_random_protein()
        p.x = p.residue_type
        proteins.append(p)

    batch = ProteinBatch.from_protein_list(proteins)

    batch.edges("knn_8", cache="edge_index")
    batch.edge_index = batch.edge_index.long()
    batch.pos = batch.coords[:, 1, :]
    batch.x = F.one_hot(batch.residue_type, num_classes=23).float()

    batch.x_vector_attr = orientations(batch.pos)
    batch.edge_attr = pos_emb(batch.edge_index, 9)
    batch.edge_vector_attr = _normalize(
        batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]
        )
    return batch

@pytest.fixture(scope="function")
def example_batch() -> ProteinBatch:
    """Creates a random batch of proteins for testing"""
    return _example_batch()
