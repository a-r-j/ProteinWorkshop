import os
import pathlib
import zipfile
from typing import Callable, Iterable, Literal, Optional

import graphein
import omegaconf
import pandas as pd
import torch
import wget
from graphein.protein.tensor.dataloader import ProteinDataLoader
from loguru import logger as log

from proteinworkshop.datasets.base import ProteinDataModule, ProteinDataset

graphein.verbose(False)


class DeepSeaProteinsDataModule(ProteinDataModule):
    """Data module for Deep Sea Proteins dataset.

    :param path: Path to store data.
    :type path: os.PathLike
    :param pdb_dir: Path to directory containing PDB files.
    :type pdb_dir: os.PathLike
    :param validation_fold: Name of validation fold to use.
    :type validation_fold: int
    :param batch_size: Batch size for dataloaders.
    :type batch_size: int
    :param in_memory: Whether to load the entire dataset into memory, defaults to False
    :type in_memory: bool, optional
    :param pin_memory: Whether to pin dataloader memory, defaults to True
    :type pin_memory: bool, optional
    :param num_workers: Number of dataloader workers, defaults to 16
    :type num_workers: int, optional
    :param obsolete_strategy: Strategy to deal with obsolete PDbs,
        defaults to "drop"
    :type obsolete_strategy: str, optional
    :param format: Format of the structure files, defaults to "mmtf"
    :type format: Literal[mmtf, pdb], optional
    :param transforms: Transforms to apply, defaults to None
    :type transforms: Optional[Iterable[Callable]], optional
    :param overwrite: Whether to overwrite existing data, defaults to
        ``False``
    :type overwrite: bool, optional
    """

    def __init__(
        self,
        path: os.PathLike,
        pdb_dir: os.PathLike,
        validation_fold: Literal["0", "1", "2", "3", "4"],
        batch_size: int,
        in_memory: bool = False,
        pin_memory: bool = True,
        num_workers: int = 16,
        obsolete_strategy: Literal["drop", "replace"] = "drop",
        format: Literal["mmtf", "pdb"] = "mmtf",
        transforms: Optional[Iterable[Callable]] = None,
        overwrite: bool = False,
    ):
        super().__init__()
        self.data_dir = pathlib.Path(path)
        if not os.path.exists(self.data_dir):
            log.info(f"Creating data directory: {self.data_dir}")
            os.makedirs(self.data_dir, exist_ok=True)

        self.validation_fold = validation_fold

        self.in_memory = in_memory

        self.batch_size = batch_size
        self.obsolete_strategy = obsolete_strategy
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.pdb_dir = pathlib.Path(pdb_dir)
        self.format = format
        self.overwrite = overwrite

        if transforms is not None:
            self.transform = self.compose_transforms(
                omegaconf.OmegaConf.to_container(transforms, resolve=True)
            )
        else:
            self.transform = None

        self.DATASET_URL = "https://www.zbh.uni-hamburg.de/forschung/amd/datasets/deep-sea-protein-structure/deep-sea-proteins-1.zip"
        self.ZIP_FNAME = "deep-sea-proteins-1.zip"

        self.DATA_FILES = {
            "decoy_subsets.tsv",
            "deep_sea_pdbs.tsv",
            "folds.tsv",
            "deep_sea_species.tsv",
            "protein_pairs.tsv",
        }

        # required in lightning >= 2.0
        self.prepare_data_per_node = True

    def download(self):
        """Downloads the Deep Sea Protein dataset to the data directory."""
        if not os.path.exists(self.data_dir / self.ZIP_FNAME):
            log.info(
                f"Downloading Deep Sea Protein dataset to {self.data_dir}"
            )
            wget.download(self.DATASET_URL, str(self.data_dir))
        else:
            log.info(
                f"Deep Sea Protein dataset already downloaded to {self.data_dir}"
            )

        if not all(
            os.path.exists(self.data_dir / fname) for fname in self.DATA_FILES
        ):
            log.info(f"Extracting {self.ZIP_FNAME} dataset to {self.data_dir}")
            with zipfile.ZipFile(
                self.data_dir / self.ZIP_FNAME, "r"
            ) as zip_ref:
                zip_ref.extractall(self.data_dir)
        else:
            log.info(
                f"Deep Sea Protein dataset already extracted to {self.data_dir}"
            )

    def parse_dataset(self, split: str) -> pd.DataFrame:
        """Parses the source raw dataset.

        :param split: Split to parse.
        :type split: str
        :return: Parsed dataset
        :rtype: pd.DataFrame
        """
        log.info(f"Parsing dataset from {self.data_dir / 'folds.tsv'}")
        df = pd.read_csv(self.data_dir / "folds.tsv", sep="\t")
        df[["pdb_code", "chain"]] = df["sample"].str.split("_", expand=True)
        df.pdb_code = df.pdb_code.str.lower()
        log.info(f"Dataset contains {len(df)} samples")
        # Check for obsolete PDBs

        if self.obsolete_strategy == "drop":
            log.info("Dropping obsolete PDBs from dataset")
            df = df.loc[~df.pdb_code.isin(self.obsolete_pdbs.keys())]
            log.info(
                f"Dataset contains {len(df)} samples after dropping obsolete PDBs"
            )
        else:
            raise NotImplementedError(
                f"Obsolete strategy {self.obsolete_strategy} not implemented."
            )

        if split == "train":
            df = df.loc[df.fold != "PM_group"]
            df = df.loc[df.fold != self.validation_fold]
            log.info(f"Train dataset contains {len(df)} samples")
            return df
        elif split == "validation":
            df = df.loc[df.fold == self.validation_fold]
            log.info(f"Validation dataset contains {len(df)} samples")
            return df
        elif split == "test":
            df = df.loc[df.fold == "PM_group"]
            log.info(f"Test dataset contains {len(df)} samples")
            return df
        else:
            raise ValueError(f"Split {split} not recognized")

    def parse_labels(self):
        """Not implemented. Labels handled in ``parse_dataset``."""
        # Labels are already in the dataset
        pass

    def exclude_pdbs(self):
        """Not implemented for Deep Sea Proteins dataset"""
        pass

    def _get_dataset(self, split: str) -> ProteinDataset:
        df = self.parse_dataset(split)
        return ProteinDataset(
            root=str(self.data_dir),
            pdb_codes=df.pdb_code.values.tolist(),
            pdb_dir=str(self.pdb_dir),
            chains=df.chain.values.tolist(),
            graph_labels=[torch.tensor(i) for i in df.label.values],
            format=self.format,
            transform=self.transform,
            in_memory=self.in_memory,
            overwrite=self.overwrite,
        )

    def train_dataset(self) -> ProteinDataset:
        """Returns the training dataset.

        :return: Training dataset
        :rtype: ProteinDataset
        """
        return self._get_dataset("train")

    def val_dataset(self) -> ProteinDataset:
        """Returns the validation dataset.

        :return: Validation dataset
        :rtype: ProteinDataset
        """
        return self._get_dataset("validation")

    def test_dataset(self) -> ProteinDataset:
        """Returns the test dataset.

        :return: Test dataset
        :rtype: ProteinDataset
        """
        return self._get_dataset("test")

    def train_dataloader(self) -> ProteinDataLoader:
        """Returns the training dataloader.

        :return: Training dataloader
        :rtype: ProteinDataLoader
        """
        return ProteinDataLoader(
            self.train_dataset(),
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> ProteinDataLoader:
        """Returns the validation dataloader.

        :return: Validation dataloader
        :rtype: ProteinDataLoader
        """
        return ProteinDataLoader(
            self.val_dataset(),
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> ProteinDataLoader:
        """Returns the test dataloader.

        :return: Test dataloader
        :rtype: ProteinDataLoader
        """
        return ProteinDataLoader(
            self.test_dataset(),
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=False,
        )


if __name__ == "__main__":
    import hydra

    from proteinworkshop import constants

    cfg = omegaconf.OmegaConf.load(
        constants.HYDRA_CONFIG_PATH / "dataset" / "deep_sea_proteins.yaml"
    )
    cfg.datamodule.path = str(
        pathlib.Path(constants.DATA_PATH) / "deep-sea-proteins"
    )
    cfg.datamodule.pdb_dir = str(pathlib.Path(constants.DATA_PATH) / "pdb")

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.download()
    dl = datamodule.test_dataloader()
    for i in dl:
        print(i)
