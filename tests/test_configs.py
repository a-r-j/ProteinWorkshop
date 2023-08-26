import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from proteinworkshop.models.base import BenchMarkModel


def test_train_config(cfg_train: DictConfig) -> None:
    """
    Tests the training configuration provided by the `cfg_train` pytest
    fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.dataset
    assert cfg_train.encoder
    assert cfg_train.decoder
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.dataset.datamodule)
    hydra.utils.instantiate(cfg_train.encoder)
    hydra.utils.instantiate(cfg_train.trainer)

    model = BenchMarkModel(cfg_train)
    assert model


def test_finetune_config(cfg_finetune: DictConfig) -> None:
    """
    Tests the finetuning configuration provided by the `cfg_finetune` pytest
    fixture.

    :param cfg_train: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_finetune
    assert cfg_finetune.dataset
    assert cfg_finetune.encoder
    assert cfg_finetune.decoder
    assert cfg_finetune.trainer

    HydraConfig().set_config(cfg_finetune)

    hydra.utils.instantiate(cfg_finetune.dataset.datamodule)
    hydra.utils.instantiate(cfg_finetune.encoder)
    hydra.utils.instantiate(cfg_finetune.trainer)

    model = BenchMarkModel(cfg_finetune)
    assert model
