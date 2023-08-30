import os
import pathlib
import random
import tarfile
from typing import Callable, Dict, Iterable, List, Literal, Optional

import omegaconf
import pandas as pd
import wget
from graphein.protein.tensor.dataloader import ProteinDataLoader
from loguru import logger as log
from sklearn.model_selection import train_test_split

from proteinworkshop.datasets.base import ProteinDataModule, ProteinDataset
from proteinworkshop.datasets.utils import flatten_dir


class AstralDataModule(ProteinDataModule):
    def __init__(
        self,
        path: str,
        batch_size: int,
        pin_memory: bool,
        num_workers: int,
        release: str = "1.75",
        identity: Literal["40", "95"] = "95",
        dataset_fraction: float = 1.0,
        transforms: Optional[Iterable[Callable]] = None,
        in_memory: bool = False,
        train_val_test: List[float] = [0.8, 0.1, 0.1],
        overwrite: bool = False,
    ) -> None:
        """Data module for ASTRAL dataset.

        :param path: Path to store data.
        :type path: str
        :param batch_size: Batch size for dataloaders.
        :type batch_size: int
        :param pin_memory: Whether to pin memory for dataloaders.
        :type pin_memory: bool
        :param num_workers: Number of workers for dataloaders.
        :type num_workers: int
        :param release: ASTRAL release to use. Defaults to ``"1.75"``.
        :type release: str
        :param identity: ASTRAL sequence identity cutoff to use.
        :type identity: Literal["40", "95"]
        :param dataset_fraction: Fraction of dataset to use. Defaults to ``1.0``.
        :type dataset_fraction: float
        :param transforms: List of transforms to apply to dataset. Defaults to ``None``.
        :type transforms: Optional[List[Callable]]
        :param in_memory: Whether to load the entire dataset into memory. Defaults to ``False``.
        :type in_memory: bool
        :param train_val_test: Train/val/test split sizes. Defaults to ``[0.8, 0.1, 0.1]``.
        :type train_val_test: List[float]
        :param overwrite: Whether to overwrite existing data. Defaults to ``False``.
        :type overwrite: bool
        """
        super().__init__()
        self.data_dir = pathlib.Path(path)
        if not os.path.exists(self.data_dir):
            log.info(f"Creating dataset directory: {self.data_dir}")
            os.makedirs(self.data_dir, exist_ok=True)

        self.release = str(release)
        self.identity = str(identity)
        if self.identity not in {"95", "40"}:
            raise ValueError(
                f"Identity must be one of {95, 40} not {self.identity}"
            )

        self.ASTRAL_GZ_FNAME = (
            f"pdbstyle-sel-gs-bib-{identity}-{self.release}.tgz"
        )
        self.scop_url = f"https://scop.berkeley.edu/downloads/pdbstyle/{self.ASTRAL_GZ_FNAME}"

        self.structure_dir = self.data_dir / f"pdbstyle-{self.release}"

        self.in_memory = in_memory

        self.train_val_test: List[float] = train_val_test

        self.dataset_fraction = dataset_fraction
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        self.prepare_data_per_node = True

        if transforms is not None:
            self.transform = self.compose_transforms(
                omegaconf.OmegaConf.to_container(transforms, resolve=True)
            )
        else:
            self.transform = None

        self.overwrite = overwrite

    def download(self):
        """Downloads ASTRAL structures from SCOPe."""
        self._download_structures()

    def _download_structures(self):  # sourcery skip: extract-method
        """Downloads SCOPe structures."""
        if not os.path.exists(self.data_dir / self.ASTRAL_GZ_FNAME):
            log.info(
                f"Downloading SCOPe structures from: {self.scop_url} to {self.data_dir}"
            )
            wget.download(self.scop_url, str(self.data_dir))
        else:
            log.info(
                f"Found SCOP structure tarfile in: {self.data_dir / self.ASTRAL_GZ_FNAME}"
            )
        if not os.path.exists(self.structure_dir):
            log.info(f"Extracting tarfile to {self.data_dir}")
            tar = tarfile.open(
                str(
                    self.data_dir
                    / f"pdbstyle-sel-gs-bib-{self.identity}-{self.release}.tgz"
                )
            )
            tar.extractall(str(self.data_dir))
            tar.close()
            log.info("Flattening directory")
            flatten_dir(self.structure_dir)
        else:
            log.info("Found SCOPe structures in: ")  # TODO

    def parse_class_map(self) -> Dict[str, str]:
        """Parses class map from ASTRAL dataset.

        :return: Class map.
        :rtype: Dict[str, str]
        """
        log.info(f"Reading labels from: {self.data_dir / 'class_map.txt'}")
        class_map = pd.read_csv(
            self.data_dir / "class_map.txt", sep="\t", header=None
        )
        return dict(class_map.values)

    def setup(self, stage: Optional[str] = None):
        self.download()

    def parse_dataset(
        self, split: Literal["train", "val", "test"]
    ) -> List[str]:
        """Parses ASTRAL dataset. Returns a list of IDs for each split.

        :param split: Split to parse.
        :type split: Literal["train", "val", "test"]
        :return: List of IDs for split.
        :rtype: List[str]
        """
        # If we've already split, return the split data
        if hasattr(self, f"{split}_ids"):
            return getattr(self, f"{split}_ids")

        structs = os.listdir(self.structure_dir)
        structs = [s for s in structs if s.endswith(".ent")]
        structs = [s.replace(".ent", "") for s in structs]

        structs = random.sample(
            structs, int(len(structs) * self.dataset_fraction)
        )

        train_size, val_size, test_size = self.train_val_test
        log.info(
            f"Splitting {len(structs)} structures into {train_size}, {val_size}, {test_size} split"
        )

        train, val = train_test_split(structs, test_size=val_size + test_size)
        val, test = train_test_split(
            val, test_size=test_size / (val_size + test_size)
        )
        log.info(
            f"Split sizes: {len(train)} train, {len(val)} val, {len(test)} test"
        )

        self.train_ids = train
        self.val_ids = val
        self.test_ids = test

        return getattr(self, f"{split}_ids")

    def _get_dataset(self, split: str) -> ProteinDataset:
        ids = self.parse_dataset(split)
        return ProteinDataset(
            root=str(self.data_dir),
            pdb_dir=str(self.structure_dir),
            pdb_codes=ids,
            format="ent",
            # graph_labels=[torch.tensor(a) for a in list(df.label)],
            overwrite=self.overwrite,
            transform=self.transform,
            in_memory=self.in_memory,
        )

    def train_dataset(self) -> ProteinDataset:
        """Returns the training dataset.

        .. seealso::
            :py:class:`proteinworkshop.datasets.base.ProteinDataset`

        :return: Training dataset.
        :rtype: ProteinDataset
        """
        return self._get_dataset("train")

    def val_dataset(self) -> ProteinDataset:
        """Returns the validation dataset.

        .. seealso::
            :py:class:`proteinworkshop.datasets.base.ProteinDataset`

        :return: Validation dataset.
        :rtype: ProteinDataset
        """
        return self._get_dataset("val")

    def test_dataset(self) -> ProteinDataset:
        return self._get_dataset("test")

    def train_dataloader(self) -> ProteinDataLoader:
        """Returns the training dataloader.

        :return: Training dataloader.
        :rtype: ProteinDataLoader
        """
        return ProteinDataLoader(
            self.train_dataset(),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> ProteinDataLoader:
        """Returns the validation dataloader.

        :return: Validation dataloader.
        :rtype: ProteinDataLoader
        """
        return ProteinDataLoader(
            self.val_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> ProteinDataLoader:
        """Returns the test dataloader.

        :return: Test dataloader.
        :rtype: ProteinDataLoader
        """
        return ProteinDataLoader(
            self.test_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def get_test_loader(self, split: str) -> ProteinDataLoader:
        log.info(f"Getting test loader: {split}")
        test_ds = self._get_dataset(f"test_{split}")
        return ProteinDataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def parse_labels(self):
        """Not implemented for ASTRAL dataset."""
        pass

    def exclude_pdbs(self):
        """Not implemented for ASTRAL dataset."""
        pass
