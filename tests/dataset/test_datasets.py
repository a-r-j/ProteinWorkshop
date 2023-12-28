import os

import omegaconf
from hydra.utils import instantiate
from lightning import LightningDataModule

from proteinworkshop import constants

DATASET_CONFIG_DIR = constants.HYDRA_CONFIG_PATH / "dataset"


def test_instantiate_datasets(tmp_path):
    """Tests we can instantiate all datasets."""
    for t in os.listdir(DATASET_CONFIG_DIR):
        config_path = DATASET_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)

        if "data_dir" in cfg.datamodule:
            cfg.datamodule.data_dir = tmp_path

        if "path" in cfg.datamodule:
            cfg.datamodule.path = tmp_path

        if "pdb_dir" in cfg.datamodule:
            cfg.datamodule.pdb_dir = tmp_path

        if "transforms" in cfg.datamodule:
            cfg.datamodule.transforms = None

        if "transform" in cfg.datamodule:
            cfg.datamodule.transform = None

        if (
            cfg.datamodule._target_
            == "graphein.ml.datasets.foldcomp_dataset.FoldCompLightningDataModule"
        ):
            continue

        if (
            cfg.datamodule._target_
            == "proteinworkshop.datasets.atom3d_datamodule.ATOM3DDataModule"
        ):
            continue

        dataset = instantiate(cfg.datamodule)

        assert dataset, f"Dataset {t} not instantiated!"
        assert isinstance(dataset, LightningDataModule)


def test_datasets_have_overwrite_attr(tmp_path):
    for t in os.listdir(DATASET_CONFIG_DIR):
        config_path = DATASET_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)

        if "data_dir" in cfg.datamodule:
            cfg.datamodule.data_dir = tmp_path

        if "path" in cfg.datamodule:
            cfg.datamodule.path = tmp_path

        if "pdb_dir" in cfg.datamodule:
            cfg.datamodule.pdb_dir = tmp_path

        if "transforms" in cfg.datamodule:
            cfg.datamodule.transforms = None

        if "transform" in cfg.datamodule:
            cfg.datamodule.transform = None

        if cfg.datamodule._target_ in {
            "graphein.ml.datasets.foldcomp_dataset.FoldCompLightningDataModule",
            "proteinworkshop.datasets.atom3d_datamodule.ATOM3DDataModule",
            "proteinworkshop.datasets.igfold.IgFoldDataModule",
        }:
            continue
        else:
            dm = instantiate(cfg.datamodule)
            assert hasattr(
                dm, "overwrite"
            ), f"Datamodules {dm} has no overwrite attribute"
