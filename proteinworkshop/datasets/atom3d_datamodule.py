# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import atom3d
import atom3d.datasets.datasets as da
import lightning as L
import torch
from beartype import beartype as typechecker
from graphein.protein.tensor.dataloader import ProteinDataLoader
from loguru import logger as log
from torch.utils.data import Dataset
from torch_geometric.loader import DynamicBatchSampler

from proteinworkshop.datasets.components import atom3d_dataset
from proteinworkshop.datasets.components.ppi_dataset import PPIDataset
from proteinworkshop.datasets.components.res_dataset import RESDataset
from proteinworkshop.datasets.components.sampler import (
    DistributedSamplerWrapper,
)

ITERABLE_DATASETS = ["PPI", "RES"]
SHARING_STRATEGY = "file_system"
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


def set_worker_sharing_strategy(worker_id: int):
    log.info(f"Setting multiprocessing sharing strategy: {SHARING_STRATEGY}")
    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


@typechecker
def get_data_path(
    dataset: str,
    lba_split: int = 30,
    ppi_split: str = "DIPS-split",
) -> str:
    data_paths = {
        "PSR": "PSR/splits/split-by-year/data/",
        "LBA": f"LBA/splits/split-by-sequence-identity-{lba_split}/data/",
        "PPI": f"PPI/splits/{ppi_split}/data/",
        "RES": "RES/splits/split-by-cath-topology/data/",
        "MSP": "MSP/splits/split-by-sequence-identity-30/data/",
    }

    if dataset not in data_paths:
        raise NotImplementedError(
            f"Dataset {dataset} is not implemented yet, please choose one of the following datasets: "
            f'{", ".join(list(data_paths.keys()))}'
        )

    return data_paths[dataset]


@typechecker
def get_test_data_path(
    dataset: str,
    lba_split: int = 30,
    ppi_split: str = "DIPS-split",
    test_phase: str = "test",
    use_dips_for_testing: bool = False,
) -> str:
    data_paths = {
        "PSR": f"PSR/splits/split-by-year/data/{test_phase}",
        "LBA": f"LBA/splits/split-by-sequence-identity-{lba_split}/data/{test_phase}",
        # default to testing PPI methods with DB5
        "PPI": f"PPI/splits/{ppi_split}/data/{test_phase}"
        if use_dips_for_testing
        else "PPI/raw/DB5/data/",
        "RES": f"RES/splits/split-by-cath-topology/data/{test_phase}",
        "MSP": f"MSP/splits/split-by-sequence-identity-30/data/{test_phase}",
    }

    if dataset not in data_paths:
        raise NotImplementedError(
            f"Test dataset {dataset} is not implemented yet, please choose one of the following test datasets: "
            f'{", ".join(list(data_paths.keys()))}'
        )

    return data_paths[dataset]


@typechecker
def get_task_split(
    task: str,
    lba_split: int = 30,
    ppi_split: str = "DIPS",
    res_split: str = "cath-topology",
    msp_split: int = 30,
) -> str:
    splits = {
        "PSR": "year",
        "LBA": f"sequence-identity-{lba_split}",
        "PPI": ppi_split,
        "RES": res_split,
        "MSP": f"sequence-identity-{msp_split}",
    }

    if task not in splits:
        raise NotImplementedError(
            f"Dataset {task} is not implemented yet, please choose one of the following datasets: "
            f'{", ".join(list(splits.keys()))}'
        )
    return splits[task]


class ATOM3DDataModule(L.LightningDataModule):
    """
    Adapted from https://github.com/sarpaykent/GBPNet

    A data wrapper for the ATOM3D package. It downloads any missing
    data files from Zenodo. Also applies transformations to the
    raw data to gather graph features.

    :param task: name of the task.
    :param data_dir: location where the data is stored for the tasks.
    :param lba_split: data split type for the LBA task (30 or 60).
    :param ppi_split: data split type for the PPI task (DIPS-split).
    :param max_units: maximum number of `unit` allowed in the input graphs.
    :param unit: component of graph topology to size limit with `max_units`.
    :param batch_size: mini-batch size.
    :param num_workers:  number of workers to be used for data loading.
    :param pin_memory: whether to reserve memory for faster data loading.
    """

    def __init__(
        self,
        task: str = "LBA",
        data_dir: str = os.path.join("data", "ATOM3D"),
        lba_split: int = 30,
        ppi_split: str = "DIPS-split",
        res_split: str = "cath-topology",
        max_units: int = 0,
        unit: str = "edge",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with `self.hparams` attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = {
            "PSR": atom3d_dataset.PSRTransform,
            "LBA": atom3d_dataset.LBATransform,
            "PPI": atom3d_dataset.BaseTransform,
            "RES": atom3d_dataset.BaseTransform,
            "MSP": atom3d_dataset.MSPTransform,
        }

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.train_phase, self.val_phase, self.test_phase = (
            "train",
            "val",
            "test",
        )

    def get_datasets(
        self, use_dips_for_testing: bool = False
    ) -> Tuple[da.LMDBDataset]:
        """Retrieve data from storage.

        Does not assign state (e.g., self.data_train = data_train).
        """
        relative_path = get_data_path(
            self.hparams.task, self.hparams.lba_split
        )
        relative_test_path = get_test_data_path(
            self.hparams.task,
            self.hparams.lba_split,
            self.hparams.ppi_split,
            self.test_phase,
            use_dips_for_testing=use_dips_for_testing,
        )
        full_path = os.path.join(self.hparams.data_dir, relative_path)
        full_test_path = os.path.join(
            self.hparams.data_dir, relative_test_path
        )
        full_raw_data_path = os.path.join(
            self.hparams.data_dir,
            self.hparams.task,
            "raw",
            self.hparams.task,
            "data",
        )

        transform = self.transforms[self.hparams.task]()
        custom_task_datasets = {"PPI": PPIDataset, "RES": RESDataset}
        dataset_class = (
            custom_task_datasets[self.hparams.task]
            if self.hparams.task in ITERABLE_DATASETS
            else partial(da.LMDBDataset, transform=transform)
        )

        if self.hparams.task in ["PPI"]:
            return (
                dataset_class(
                    full_path + self.train_phase, dataset_type="DIPS"
                ),
                dataset_class(full_path + self.val_phase, dataset_type="DIPS"),
                dataset_class(
                    full_test_path,
                    dataset_type="DIPS" if use_dips_for_testing else "DB5",
                ),
            )
        elif self.hparams.task in ["RES"]:
            return (
                dataset_class(
                    full_raw_data_path,
                    split_path=str(
                        Path(full_path).parent
                        / "indices"
                        / f"{self.train_phase}_indices.txt"
                    ),
                ),
                dataset_class(
                    full_raw_data_path,
                    split_path=str(
                        Path(full_path).parent
                        / "indices"
                        / f"{self.val_phase}_indices.txt"
                    ),
                ),
                dataset_class(
                    full_raw_data_path,
                    split_path=str(
                        Path(full_path).parent
                        / "indices"
                        / f"{self.test_phase}_indices.txt"
                    ),
                ),
            )
        else:
            return (
                dataset_class(full_path + self.train_phase),
                dataset_class(full_path + self.val_phase),
                dataset_class(full_test_path),
            )

    def prepare_data(self, use_dips_for_testing: bool = False):
        """Download data if needed.

        Do not use it to assign state (e.g., self.x = y).
        """
        relative_path = get_data_path(
            self.hparams.task, self.hparams.lba_split
        )
        relative_test_path = get_test_data_path(
            self.hparams.task,
            self.hparams.lba_split,
            self.hparams.ppi_split,
            self.test_phase,
            use_dips_for_testing=use_dips_for_testing,
        )

        full_path = os.path.join(self.hparams.data_dir, relative_path)
        full_test_path = os.path.join(
            self.hparams.data_dir, relative_test_path
        )
        full_task_path = os.path.join(self.hparams.data_dir, self.hparams.task)

        if not os.path.exists(full_path):
            # note: for downloading most ATOM3D datasets
            atom3d.datasets.download_dataset(
                self.hparams.task.split("_")[0],
                split=get_task_split(
                    self.hparams.task,
                    self.hparams.lba_split,
                    self.hparams.ppi_split,
                    self.hparams.res_split,
                ),
                out_path=os.path.join(
                    self.hparams.data_dir,
                    os.sep.join(relative_path.split("/")[:2]),
                ),
            )
        if not os.path.exists(full_test_path):
            # note: for downloading the PPI task's selected test dataset (e.g., DB5)
            atom3d.datasets.download_dataset(
                self.hparams.task.split("_")[0],
                split=None,
                out_path=os.path.join(
                    self.hparams.data_dir,
                    os.sep.join(relative_test_path.split("/")[:1]),
                ),
            )
        if self.hparams.task in ["RES"] and not os.path.exists(
            os.path.join(full_task_path, "raw")
        ):
            # note: for downloading the RES task's raw dataset files, which we need even when only using a specific split
            atom3d.datasets.download_dataset(
                self.hparams.task.split("_")[0],
                split=None,
                out_path=full_task_path,
            )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        Note: This method is called by Lightning with both `trainer.fit()` and `trainer.test()`.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            (
                self.data_train,
                self.data_val,
                self.data_test,
            ) = self.get_datasets()

    @typechecker
    def get_dataloader(
        self,
        dataset: Union[da.LMDBDataset, PPIDataset, RESDataset],
        batch_size: int = None,
        pin_memory: bool = False,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> ProteinDataLoader:
        if batch_size is None:
            batch_size = self.hparams.batch_size
        if pin_memory is None:
            pin_memory = self.hparams.pin_memory
        if self.hparams.max_units == 0:
            if (
                self.hparams.num_workers > 0
                and self.hparams.task not in ITERABLE_DATASETS
            ):
                return ProteinDataLoader(
                    dataset,
                    num_workers=self.hparams.num_workers,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    prefetch_factor=100,
                    worker_init_fn=set_worker_sharing_strategy,
                )
            return ProteinDataLoader(
                dataset,
                num_workers=self.hparams.num_workers,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        else:
            if torch.distributed.is_initialized():
                if self.hparams.task in ITERABLE_DATASETS:
                    # handle sharding for IterableDatasets separately
                    return ProteinDataLoader(
                        dataset,
                        num_workers=self.hparams.num_workers,
                        batch_size=batch_size,
                        pin_memory=pin_memory,
                    )
                return ProteinDataLoader(
                    dataset,
                    num_workers=self.hparams.num_workers,
                    batch_sampler=DistributedSamplerWrapper(
                        DynamicBatchSampler(
                            dataset,
                            max_num=self.hparams.max_units,
                            mode=self.hparams.unit,
                            shuffle=shuffle,
                        )
                    ),
                    pin_memory=pin_memory,
                    drop_last=drop_last,
                )
            else:
                return ProteinDataLoader(
                    dataset,
                    num_workers=self.hparams.num_workers,
                    batch_sampler=DynamicBatchSampler(
                        dataset,
                        max_num=self.hparams.max_units,
                        mode=self.hparams.unit,
                        shuffle=shuffle,
                    )
                    if self.hparams.task not in ITERABLE_DATASETS
                    else None,
                    pin_memory=pin_memory,
                    drop_last=drop_last,
                )

    def train_dataloader(self) -> ProteinDataLoader:
        return self.get_dataloader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.task not in ITERABLE_DATASETS,
            drop_last=self.hparams.task not in ITERABLE_DATASETS,
        )

    def val_dataloader(self) -> ProteinDataLoader:
        return self.get_dataloader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            drop_last=self.hparams.task in ["LBA", "PSR"],
        )

    def test_dataloader(self) -> ProteinDataLoader:
        return self.get_dataloader(
            self.data_test, batch_size=self.hparams.batch_size
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""

    def state_dict(self) -> Dict:
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    from proteinworkshop.constants import DATA_PATH

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    import pathlib

    # cfg = omegaconf.OmegaConf.load(root / "configs" / "dataset" / "atom3d_lba.yaml")
    # cfg.data_dir = str(root / "data" / "ATOM3D")
    # ds = hydra.utils.instantiate(cfg)
    # ds.prepare_data()
    # ds.setup()
    # dl = ds.val_dataloader()
    # for i in dl:
    #    print(i)
    #    break
    # cfg = omegaconf.OmegaConf.load(root / "configs" / "dataset" / "atom3d_psr.yaml")
    # cfg.datamodule.data_dir = pathlib.Path(DATA_PATH) / "ATOM3D"
    # ds = hydra.utils.instantiate(cfg.datamodule)
    # ds.prepare_data()
    # ds.setup()
    # dl = ds.val_dataloader()
    # for i in dl:
    #     print(i)
    #     break
    # cfg = omegaconf.OmegaConf.load(root / "configs" / "dataset" / "atom3d_ppi.yaml")
    # cfg.datamodule.data_dir = pathlib.Path(DATA_PATH) / "ATOM3D"
    # ds = hydra.utils.instantiate(cfg.datamodule)
    # ds.prepare_data()
    # ds.setup()
    # dl = ds.val_dataloader()
    # for i in dl:
    #     print(i)
    #     break
    # cfg = omegaconf.OmegaConf.load(root / "configs" / "dataset" / "atom3d_res.yaml")
    # cfg.datamodule.data_dir = pathlib.Path(DATA_PATH) / "ATOM3D"
    # ds = hydra.utils.instantiate(cfg.datamodule)
    # ds.prepare_data()
    # ds.setup()
    # dl = ds.val_dataloader()
    # for i in dl:
    #     print(i)
    #     break

    cfg = omegaconf.OmegaConf.load(
        root / "configs" / "dataset" / "atom3d_msp.yaml"
    )
    cfg.datamodule.data_dir = pathlib.Path(DATA_PATH) / "ATOM3D"

    ds = hydra.utils.instantiate(cfg.datamodule)
    ds.prepare_data()
    ds.setup()
    dl = ds.val_dataloader()
    for i in dl:
        print(i)
        break
