import os
import pathlib
from typing import Callable, List, Literal, Optional

import numpy as np
import omegaconf
import pandas as pd
import torch
import wget
from graphein.protein.tensor.dataloader import ProteinDataLoader
from loguru import logger as log
from sklearn.model_selection import train_test_split

from proteinworkshop.datasets.base import ProteinDataModule, ProteinDataset

CCPDB_DATASET_NAMES = Literal["metal", "ligands", "nucleotides", "nucliec"]


class CCPDBDataModule(ProteinDataModule):
    """Data module for CCPDB datasets.

    :param path: Path to store data.
    :type path: str
    :param pdb_dir: Path to directory containing structure files.
    :type pdb_dir: str
    :param name: Name of dataset to use.
    :type name: CCPDB_DATASET_NAMES
    :param batch_size: Batch size for dataloaders.
    :type batch_size: int
    :param num_workers: Number of workers for dataloaders.
    :type num_workers: int
    :param pin_memory: Whether to pin memory for dataloaders.
    :type pin_memory: bool
    :param in_memory: Whether to load dataset into memory, defaults to
        ``False``
    :type in_memory: bool, optional
    :param format: Format of the structure files, defaults to ``"mmtf"``.
    :type format: Literal[mmtf, pdb], optional
    :param obsolete_strategy: How to deal with obsolete PDBs,
        defaults to "drop"
    :type obsolete_strategy: str, optional
    :param split_strategy: How to split the data,
        defaults to ``"random"``
    :type split_strategy: Literal["random", 'stratified"], optional
    :param val_fraction: Fraction of the dataset to use for validation,
        defaults to ``0.1``
    :type val_fraction: float, optional
    :param test_fraction: Fraction of the dataset to use for testing,
        defaults to ``0.1``.
    :type test_fraction: float, optional
    :param transforms: List of transforms to apply to each example,
        defaults to ``None``.
    :type transforms: Optional[List[Callable]], optional
    :param overwrite: Whether to overwrite existing data, defaults to
        ``False``
    :type overwrite: bool, optional
    :raises ValueError: If train, val, and test fractions do not sum to 1.
    """

    def __init__(
        self,
        path: str,
        pdb_dir: str,
        name: CCPDB_DATASET_NAMES,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        in_memory=False,
        format: Literal["mmtf", "pdb"] = "mmtf",
        obsolete_strategy: str = "drop",
        split_strategy: Literal["random", "stratified"] = "random",
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        transforms: Optional[List[Callable]] = None,
        overwrite: bool = False,
    ):
        super().__init__()
        self.root = pathlib.Path(path)
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)
        self.pdb_dir = pdb_dir

        self.name = name
        self.DATASET_URL: str = f"https://raw.githubusercontent.com/a-r-j/graphein/master/datasets/proteins_{self.name}/PROTEINS_{self.name.upper()}.csv"
        self.DATASET_PATH = self.root / f"PROTEINS_{self.name.upper()}.csv"

        self.in_memory = in_memory

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.format = format
        self.split_strategy = split_strategy
        self.obsolete_strategy = obsolete_strategy
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.overwrite = overwrite

        if transforms is not None:
            self.transform = self.compose_transforms(
                omegaconf.OmegaConf.to_container(transforms, resolve=True)
            )
        else:
            self.transform = None

        if self.train_fraction + self.val_fraction + self.test_fraction != 1:
            raise ValueError("Train, val, and test fractions must sum to 1")

    def download(self):
        if not os.path.exists(self.DATASET_PATH):
            log.info(f"Downloading {self.name} dataset to {self.DATASET_PATH}")
            wget.download(self.DATASET_URL, str(self.root))
        else:
            log.info(f"Dataset found at {self.DATASET_PATH}")

        return super().download()

    def exclude_pdbs(self):
        pass

    def parse_dataset(self):
        df = pd.read_csv(self.DATASET_PATH, sep=",")
        log.info(f"Loaded {len(df)} examples from {self.DATASET_PATH}")

        # Drop obsolete PDBs
        log.info("Removing obsolete PDBs")
        if self.obsolete_strategy == "drop":
            df = df.loc[~df["PDB"].str.lower().isin(self.obsolete_pdbs.keys())]
            log.info(f"{len(df)} examples remain after removing obsolete PDBs")

        df["id"] = df["PDB"].str.lower() + "_" + df["chain"]

        # Graph labels
        labels = pd.factorize(df["interactor"])
        df["graph_labels"] = torch.tensor(labels[0])

        # Node labels
        df["node_labels"] = df.interacting_residues.apply(
            self._encode_node_label
        )

        # Split dataset
        stratify = (
            df["graph_labels"] if self.split_strategy == "stratified" else None
        )
        log.info(
            f"Splitting dataset into train ({self.train_fraction}), val ({self.val_fraction}), and test ({self.test_fraction}) sets."
        )
        train, val = train_test_split(
            df,
            test_size=self.val_fraction + self.test_fraction,
            stratify=stratify,
        )
        val, test = train_test_split(
            val,
            test_size=self.test_fraction
            / (self.val_fraction + self.test_fraction),
            stratify=stratify,
        )

        log.info(f"Train set: {len(train)} examples")
        log.info(f"Val set: {len(val)} examples")
        log.info(f"Test set: {len(test)} examples")
        self.train_data = train
        self.val_data = val
        self.test_data = test

    @staticmethod
    def _encode_node_label(label: str):
        labels = [1 if char == "+" else 0 for char in label]
        labels = np.array(labels)
        labels = torch.from_numpy(labels)
        return labels

    def parse_labels(self):
        pass  # Handled in dataset parsing

    def train_dataset(self) -> ProteinDataset:
        if not hasattr(self, "train_data"):
            self.parse_dataset()
        return ProteinDataset(
            root=str(self.root),
            pdb_dir=self.pdb_dir,
            pdb_codes=list(self.train_data["PDB"].values),
            chains=list(self.train_data["chain"].values),
            graph_labels=list(self.train_data["graph_labels"].values),
            node_labels=list(self.train_data["node_labels"].values),
            format=self.format,
            transform=self.transform,
            in_memory=self.in_memory,
            overwrite=self.overwrite,
        )

    def val_dataset(self) -> ProteinDataset:
        if not hasattr(self, "val_data"):
            self.parse_dataset()
        return ProteinDataset(
            root=str(self.root),
            pdb_dir=self.pdb_dir,
            pdb_codes=list(self.val_data["PDB"].values),
            chains=self.val_data["chain"].values,
            graph_labels=self.val_data["graph_labels"].values,
            node_labels=self.val_data["node_labels"].values,
            format=self.format,
            transform=self.transform,
            in_memory=self.in_memory,
            overwrite=self.overwrite,
        )

    def test_dataset(self) -> ProteinDataset:
        if not hasattr(self, "test_data"):
            self.parse_dataset()
        return ProteinDataset(
            root=str(self.root),
            pdb_dir=self.pdb_dir,
            pdb_codes=self.test_data["PDB"].values,
            chains=self.test_data["chain"].values,
            graph_labels=self.test_data["graph_labels"].values,
            node_labels=self.test_data["node_labels"].values,
            format=self.format,
            transform=self.transform,
            in_memory=self.in_memory,
            overwrite=self.overwrite,
        )

    def train_dataloader(self) -> ProteinDataLoader:
        return ProteinDataLoader(
            self.train_dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> ProteinDataLoader:
        return ProteinDataLoader(
            self.val_dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> ProteinDataLoader:
        return ProteinDataLoader(
            self.test_dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    from proteinworkshop import constants

    # path = constants.DATASETS_DIR / "proteins_cc_pdb"
    path = pathlib.Path(constants.DATA_PATH) / "ccpdb"
    pdb_dir = pathlib.Path(constants.DATA_PATH) / "pdb"
    name = "metal"
    batch_size = 32
    num_workers = 4
    pin_memory = True

    dataset = CCPDBDataModule(
        path, pdb_dir, name, batch_size, num_workers, pin_memory
    )
    dataset.download()
    # dataset.parse_dataset("train")
    dataset.train_dataset()
