import functools
import json
import os
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional

import omegaconf
import wget
from graphein.protein.tensor.dataloader import ProteinDataLoader
from loguru import logger

from proteinworkshop.datasets.base import ProteinDataModule, ProteinDataset


class CATHDataModule(ProteinDataModule):
    """Data module for CATH dataset.

    :param path: Path to store data.
    :type path: str
    :param batch_size: Batch size for dataloaders.
    :type batch_size: int
    :param format: Format to load PDB files in.
    :type format: Literal["mmtf", "pdb"]
    :param pdb_dir: Path to directory containing PDB files.
    :type pdb_dir: str
    :param pin_memory: Whether to pin memory for dataloaders.
    :type pin_memory: bool
    :param in_memory: Whether to load the entire dataset into memory.
    :type in_memory: bool
    :param num_workers: Number of workers for dataloaders.
    :type num_workers: int
    :param dataset_fraction: Fraction of dataset to use.
    :type dataset_fraction: float
    :param transforms: List of transforms to apply to dataset.
    :type transforms: Optional[List[Callable]]
    :param overwrite: Whether to overwrite existing data.
        Defaults to ``False``.
    :type overwrite: bool
    """

    def __init__(
        self,
        path: str,
        batch_size: int,
        format: Literal["mmtf", "pdb"] = "mmtf",
        pdb_dir: Optional[str] = None,
        pin_memory: bool = True,
        in_memory: bool = False,
        num_workers: int = 16,
        dataset_fraction: float = 1.0,
        transforms: Optional[Iterable[Callable]] = None,
        overwrite: bool = False,
    ) -> None:
        super().__init__()

        self.data_dir = Path(path)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if transforms is not None:
            self.transform = self.compose_transforms(
                omegaconf.OmegaConf.to_container(transforms, resolve=True)
            )
        else:
            self.transform = None

        self.in_memory = in_memory
        self.overwrite = overwrite

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.format = format
        self.pdb_dir = pdb_dir

        self.dataset_fraction = dataset_fraction
        self.excluded_chains: List[str] = self.exclude_pdbs()

        self.prepare_data_per_node = False

    def download(self):
        """Downloads raw data from Ingraham et al."""
        self._download_chain_list()

    def parse_labels(self):
        """Not implemented for CATH dataset"""
        pass

    def exclude_pdbs(self):
        """Not implemented for CATH dataset"""
        return []

    def _download_chain_list(self):  # sourcery skip: move-assign
        URL = "http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set_splits.json"
        if not os.path.exists(self.data_dir / "chain_set_splits.json"):
            logger.info("Downloading dataset index file...")
            wget.download(URL, str(self.data_dir / "chain_set_splits.json"))
        else:
            logger.info("Found existing dataset index")

    @functools.lru_cache
    def parse_dataset(self) -> Dict[str, List[str]]:
        """Parses dataset index file

        Returns a dictionary with keys "train", "validation", and "test" and
        values as lists of PDB IDs.

        :return: Dictionary of PDB IDs
        :rtype: Dict[str, List[str]]
        """
        fpath = self.data_dir / "chain_set_splits.json"

        with open(fpath, "r") as file:
            data = json.load(file)

        self.train_pdbs = data["train"]
        logger.info(f"Found {len(self.train_pdbs)} chains in training set")
        logger.info("Removing obsolete PDBs from training set")
        self.train_pdbs = [
            pdb
            for pdb in self.train_pdbs
            if pdb[:4] not in self.obsolete_pdbs.keys()
        ]
        logger.info(f"{len(self.train_pdbs)} remaining training chains")

        logger.info(
            f"Sampling fraction {self.dataset_fraction} of training set"
        )
        fraction = int(self.dataset_fraction * len(self.train_pdbs))
        self.train_pdbs = random.sample(self.train_pdbs, fraction)

        self.val_pdbs = data["validation"]
        logger.info(f"Found {len(self.val_pdbs)} chains in validation set")
        logger.info("Removing obsolete PDBs from validation set")
        self.val_pdbs = [
            pdb
            for pdb in self.val_pdbs
            if pdb[:4] not in self.obsolete_pdbs.keys()
        ]
        logger.info(f"{len(self.val_pdbs)} remaining validation chains")

        self.test_pdbs = data["test"]
        logger.info(f"Found {len(self.test_pdbs)} chains in test set")
        logger.info("Removing obsolete PDBs from test set")
        self.test_pdbs = [
            pdb
            for pdb in self.test_pdbs
            if pdb[:4] not in self.obsolete_pdbs.keys()
        ]
        logger.info(f"{len(self.test_pdbs)} remaining test chains")
        return data

    def train_dataset(self) -> ProteinDataset:
        """Returns the training dataset.

        :return: Training dataset
        :rtype: ProteinDataset
        """
        if not hasattr(self, "train_pdbs"):
            self.parse_dataset()
        pdb_codes = [pdb.split(".")[0] for pdb in self.train_pdbs]
        chains = [pdb.split(".")[1] for pdb in self.train_pdbs]

        return ProteinDataset(
            root=str(self.data_dir),
            pdb_dir=self.pdb_dir,
            pdb_codes=pdb_codes,
            chains=chains,
            transform=self.transform,
            format=self.format,
            in_memory=self.in_memory,
            overwrite=self.overwrite,
        )

    def val_dataset(self) -> ProteinDataset:
        """Returns the validation dataset.

        :return: Validation dataset
        :rtype: ProteinDataset
        """
        if not hasattr(self, "val_pdbs"):
            self.parse_dataset()

        pdb_codes = [pdb.split(".")[0] for pdb in self.val_pdbs]
        chains = [pdb.split(".")[1] for pdb in self.val_pdbs]

        return ProteinDataset(
            root=str(self.data_dir),
            pdb_dir=self.pdb_dir,
            pdb_codes=pdb_codes,
            chains=chains,
            transform=self.transform,
            format=self.format,
            in_memory=self.in_memory,
            overwrite=self.overwrite,
        )

    def test_dataset(self) -> ProteinDataset:
        """Returns the test dataset.

        :return: Test dataset
        :rtype: ProteinDataset
        """
        if not hasattr(self, "test_pdbs"):
            self.parse_dataset()
        pdb_codes = [pdb.split(".")[0] for pdb in self.test_pdbs]
        chains = [pdb.split(".")[1] for pdb in self.test_pdbs]

        return ProteinDataset(
            root=str(self.data_dir),
            pdb_dir=self.pdb_dir,
            pdb_codes=pdb_codes,
            chains=chains,
            transform=self.transform,
            format=self.format,
            in_memory=self.in_memory,
            overwrite=self.overwrite,
        )

    def train_dataloader(self) -> ProteinDataLoader:
        """Returns the training dataloader.

        :return: Training dataloader
        :rtype: ProteinDataLoader
        """
        if not hasattr(self, "train_ds"):
            self.train_ds = self.train_dataset()
        return ProteinDataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> ProteinDataLoader:
        if not hasattr(self, "val_ds"):
            self.val_ds = self.val_dataset()
        return ProteinDataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> ProteinDataLoader:
        """Returns the test dataloader.

        :return: Test dataloader
        :rtype: ProteinDataLoader
        """
        if not hasattr(self, "test_ds"):
            self.test_ds = self.test_dataset()
        return ProteinDataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    import pathlib

    import hydra

    from proteinworkshop import constants

    cfg = omegaconf.OmegaConf.load(
        constants.SRC_PATH / "config" / "dataset" / "cath.yaml"
    )
    cfg.datamodule.path = pathlib.Path(constants.DATA_PATH) / "cath"  # type: ignore
    cfg.datamodule.pdb_dir = pathlib.Path(constants.DATA_PATH) / "pdb"  # type: ignore
    ds = hydra.utils.instantiate(cfg)
    print(ds)
    ds["datamodule"].val_dataloader()
    print(ds["datamodule"].val_ds[1])
