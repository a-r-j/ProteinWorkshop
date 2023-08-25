import os
import pathlib
from typing import Callable, List, Optional

import omegaconf
import torch
from graphein.protein.tensor.dataloader import ProteinDataLoader
from loguru import logger as log

try:
    from tdc.single_pred import Develop
except ImportError:
    log.warning("Dependency TDC not installed. Run: pip install PyTDC")

from proteinworkshop.datasets.base import ProteinDataModule, ProteinDataset


class AntibodyDevelopabilityDataModule(ProteinDataModule):
    def __init__(
        self,
        path: str,
        pdb_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        in_memory=False,
        format: str = "mmtf",
        obsolete_strategy: str = "drop",
        transforms: Optional[List[Callable]] = None,
    ) -> None:
        super().__init__()
        self.root = pathlib.Path(path)
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)
        self.pdb_dir = pdb_dir

        self.in_memory = in_memory

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.format = format
        self.obsolete_strategy = obsolete_strategy

        if transforms is not None:
            self.transform = self.compose_transforms(
                omegaconf.OmegaConf.to_container(transforms, resolve=True)
            )
        else:
            self.transform = None

    def exclude_pdbs(self):
        pass

    def parse_labels(self):
        pass

    def download(self):
        pass

    def parse_dataset(self):
        data = Develop("SAbDab_Chen", path=self.root)
        data = data.get_split()

        for split in ["train", "valid", "test"]:
            split_data = data[split]
            log.info(f"Found {len(split_data)} {split} samples.")
            if self.obsolete_strategy == "drop":
                split_data = split_data.loc[
                    ~split_data["Antibody_ID"]
                    .str.lower()
                    .isin(self.obsolete_pdbs.keys())
                ]
                log.info(
                    f"Found {len(split_data)} {split} samples after removing obsolete structures."
                )
                setattr(self, f"{split}_data", split_data)
            elif self.obsolete_strategy == "replace":
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Obsolete strategy {self.obsolete_strategy} not recognised."
                )

    def _get_dataset(self, split: str) -> ProteinDataset:
        if not hasattr(self, f"{split}_data"):
            self.parse_dataset()
        data = getattr(self, f"{split}_data")
        graph_labels = [torch.tensor(y) for y in data["Y"].values]

        return ProteinDataset(
            root=str(self.root),
            pdb_dir=self.pdb_dir,
            pdb_codes=data["Antibody_ID"].values,
            chains=["all"] * len(data),
            graph_labels=graph_labels,
            format=self.format,
            transform=self.transform,
            in_memory=self.in_memory,
        )

    def train_dataset(self) -> ProteinDataset:
        return self._get_dataset("train")

    def val_dataset(self) -> ProteinDataset:
        return self._get_dataset("valid")

    def test_dataset(self) -> ProteinDataset:
        return self._get_dataset("test")

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
