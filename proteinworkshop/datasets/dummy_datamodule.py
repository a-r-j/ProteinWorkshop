import omegaconf
from typing import Optional, Union, Iterable, Callable
import os

from .base import ProteinDataset, ProteinDataModule
from graphein.protein.tensor.dataloader import ProteinDataLoader
import pathlib


class DummyDataModule(ProteinDataModule):
    def __init__(
        self,
        path: str,
        pdb_dir: Optional[str] = None,
        format: str = "mmtf",
        in_memory: bool = False,
        transforms: Optional[Iterable[Callable]] = None,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = True,
        obsolete_strategy: str = "drop",  # Or replace
    ) -> None:
        super().__init__()

        self.root_dir = pathlib.Path(path)
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)

        self.pdb_dir = pathlib.Path(pdb_dir) if pdb_dir is not None else None

        self.format = format
        self.in_memory = in_memory

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

    def setup(self, stage: Optional[str] = None):
        self.download()

    def exclude_pdbs(self):
        pass

    def parse_labels(self):
        pass

    def download(self):
        pass

    def parse_dataset(self, split: str):
        pass

    def _get_dataset(self, split: str) -> ProteinDataset:
        pdb_codes = ["3eiy", "4hhb"] * 16
        return ProteinDataset(
            root=str(self.root_dir),
            pdb_dir=str(self.pdb_dir),
            pdb_codes=pdb_codes,
            chains=["all"] * len(pdb_codes),  # Get all chains
            transform=self.transform,
            format=self.format,
            in_memory=self.in_memory,
            store_het=True,
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
