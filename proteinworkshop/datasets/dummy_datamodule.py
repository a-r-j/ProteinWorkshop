import os
import pathlib
from typing import Callable, Iterable, Literal, Optional

import omegaconf
from graphein.protein.tensor.dataloader import ProteinDataLoader

from .base import ProteinDataModule, ProteinDataset


class DummyDataModule(ProteinDataModule):
    def __init__(
        self,
        path: str,
        pdb_dir: Optional[str] = None,
        format: Literal["mmtf", "pdb"] = "mmtf",
        in_memory: bool = False,
        transforms: Optional[Iterable[Callable]] = None,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = True,
        obsolete_strategy: Literal["drop", "replace"] = "drop",  # Or replace
        overwrite: bool = True,
    ) -> None:
        """Data module for dummy dataset. Small dataset for testing purposes.

        :param path: Path to store data.
        :type path: str
        :param pdb_dir: Path to structures, defaults to None
        :type pdb_dir: Optional[str], optional
        :param format: Format of structures, defaults to "mmtf"
        :type format: Literal[&quot;mmtf&quot;, &quot;pdb&quot;], optional
        :param in_memory: Whether to store data in memory, defaults to False
        :type in_memory: bool, optional
        :param transforms: List of transforms to apply to data, defaults to None
        :type transforms: Optional[Iterable[Callable]], optional
        :param batch_size: Batch size for dataloader, defaults to 8
        :type batch_size: int, optional
        :param num_workers: Number of workers for dataloader, defaults to 0
        :type num_workers: int, optional
        :param pin_memory: Whether to pin memory for the dataloader, defaults to True
        :type pin_memory: bool, optional
        :param obsolete_strategy: How to handle obsolete PDBs, defaults to "drop"
        :type obsolete_strategy: Literal["drop", "replace"], optional
        :param overwrite: Whether to overwrite existing data, defaults to
            ``True``
        :type overwrite: bool, optional
        """
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

    def setup(self, stage: Optional[str] = None):
        """Download and prepare data for all splits."""
        self.download()

    def exclude_pdbs(self):
        """Not implemented. No PDBs to exlcude."""
        pass

    def parse_labels(self):
        """Not implemented. No labels."""
        pass

    def download(self):
        """Not implemented. No data to download"""
        pass

    def parse_dataset(self, split: str):
        """No dataset to parse."""
        pass

    def _get_dataset(self, split: str) -> ProteinDataset:
        """Returns a dummy dataset of 32 proteins."""
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
            overwrite=self.overwrite,
        )

    def train_dataset(self) -> ProteinDataset:
        """Returns a dummy dataset of 32 proteins.

        :returns: ProteinDataset -- Dummy dataset of 32 proteins.
        :rtype: ProteinDataset
        """
        return self._get_dataset("train")

    def val_dataset(self) -> ProteinDataset:
        """Returns a dummy dataset of 32 proteins.

        :return: ProteinDataset -- Dummy dataset of 32 proteins.
        :rtype: ProteinDataset
        """
        return self._get_dataset("val")

    def test_dataset(self) -> ProteinDataset:
        """Returns a dummy dataset of 32 proteins.

        :return: ProteinDataset -- Dummy dataset of 32 proteins.
        :rtype: ProteinDataset
        """
        return self._get_dataset("test")

    def train_dataloader(self) -> ProteinDataLoader:
        """Returns a dummy dataloader of 32 proteins.

        :return: ProteinDataLoader -- Dummy dataloader of 32 proteins.
        :rtype: ProteinDataLoader
        """
        return ProteinDataLoader(
            self.train_dataset(),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> ProteinDataLoader:
        """Returns a dummy dataloader of 32 proteins.

        :return: ProteinDataLoader -- Dummy dataloader of 32 proteins.
        :rtype: ProteinDataLoader
        """
        return ProteinDataLoader(
            self.val_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> ProteinDataLoader:
        """Returns a dummy dataloader of 32 proteins.

        :return: ProteinDataLoader -- Dummy dataloader of 32 proteins.
        :rtype: ProteinDataLoader
        """
        return ProteinDataLoader(
            self.test_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
