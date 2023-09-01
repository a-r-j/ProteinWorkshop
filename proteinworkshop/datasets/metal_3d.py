import os
import pathlib
from typing import Callable, Iterable, Literal, Optional, Union

import omegaconf
import pandas as pd
import wget
from graphein.protein.tensor.dataloader import ProteinDataLoader
from loguru import logger as log

from proteinworkshop.datasets.base import ProteinDataModule, ProteinDataset


class Metal3DDataModule(ProteinDataModule):
    def __init__(
        self,
        path: str,
        pdb_dir: Optional[Union[str, os.PathLike]] = None,
        format: Literal["mmtf", "pdb"] = "mmtf",
        in_memory: bool = False,
        transforms: Optional[Iterable[Callable]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        obsolete_strategy: str = "drop",  # Or replace
        overwrite: bool = False,
    ) -> None:
        super().__init__()

        self.root_dir = pathlib.Path(path)
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)

        self.pdb_dir = pathlib.Path(pdb_dir) if pdb_dir is not None else None

        self.format = format
        self.in_memory = in_memory
        self.overwrite = overwrite

        if transforms is not None:
            self.transform = self.compose_transforms(
                omegaconf.OmegaConf.to_container(transforms, resolve=True)
            )
        else:
            self.transform = None

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.obsolete_strategy = obsolete_strategy

        self.BASE_URL = "https://raw.githubusercontent.com/lcbc-epfl/metal-site-prediction/main/data/"

    def setup(self, stage: Optional[str] = None):
        self.download()

    def exclude_pdbs(self):
        pass

    def parse_labels(self):
        pass

    def download(self):
        if not os.path.exists(self.root_dir / "train.txt"):
            log.info(
                f"Downloading training data from {self.BASE_URL} to {self.root_dir}"
            )
            wget.download(
                f"{self.BASE_URL}train.txt", str(self.root_dir / "train.txt")
            )
        if not os.path.exists(self.root_dir / "val.txt"):
            log.info(
                f"Downloading training data from {self.BASE_URL} to {self.root_dir}"
            )
            wget.download(
                f"{self.BASE_URL}val.txt", str(self.root_dir / "val.txt")
            )
        if not os.path.exists(self.root_dir / "test.txt"):
            log.info(
                f"Downloading training data from {self.BASE_URL} to {self.root_dir}"
            )
            wget.download(
                f"{self.BASE_URL}test.txt", str(self.root_dir / "test.txt")
            )

    def parse_dataset(self, split: str) -> pd.DataFrame:
        df = pd.read_csv(self.root_dir / f"{split}.txt", header=None)
        df.columns = ["pdb"]
        df.pdb = df.pdb.str.lower()
        log.info(f"Found {len(df)} structures in {split} split")

        if self.obsolete_strategy == "drop":
            log.info("Removing obsolete PDBS")
            df = df.loc[~df["pdb"].isin(list(self.obsolete_pdbs.keys()))]
            log.info(
                f"Found {len(df)} structures in {split} split after removing obsolete PDBS"
            )
        else:
            raise NotImplementedError

        return df

    def _get_dataset(self, split: str) -> ProteinDataset:
        df = self.parse_dataset(split)
        return ProteinDataset(
            root=str(self.root_dir),
            pdb_dir=str(self.pdb_dir),
            pdb_codes=list(df["pdb"]),
            chains=["all"] * len(df),  # Get all chains
            transform=self.transform,
            format=self.format,
            in_memory=self.in_memory,
            store_het=True,
            overwrite=self.overwrite,
        )

    def train_dataset(self) -> ProteinDataset:
        return self._get_dataset("train")

    def val_dataset(self) -> ProteinDataset:
        return self._get_dataset("val")

    def test_dataset(self) -> ProteinDataset:
        return self._get_dataset("test")

    def train_dataloader(self) -> ProteinDataLoader:
        return ProteinDataLoader(
            self.train_dataset(),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> ProteinDataLoader:
        return ProteinDataLoader(
            self.val_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> ProteinDataLoader:
        return ProteinDataLoader(
            self.test_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
