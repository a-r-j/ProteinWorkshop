"""DataModule for IgFold antibody prediction corpuses."""
import glob
import os
import pathlib
import random
import tarfile
from typing import Callable, List, Literal, Optional

import omegaconf
import wget
from graphein.protein.tensor.dataloader import ProteinDataLoader
from loguru import logger
from sklearn.model_selection import train_test_split

from proteinworkshop.datasets.base import ProteinDataModule, ProteinDataset

IgFoldDatasetNames = Literal["paired_oas", "jaffe"]


class IgFoldDataModule(ProteinDataModule):
    """
    Datamodule for IgFold predictions for pOAS and Jaffe Antibody Datasets.

    :param path: Path to store data
    :type path: os.PathLike
    :param dataset_name: Name of dataset to use (``"paired_oas"`` or
        ``"jaffe"``).
    :type dataset_name: IgFoldDatasetNames
    :param batch_size: Batch size for dataloader.
    :type batch_size: int
    :param train_val_test: Split sizes for dataset partitions.
    :type train_val_test: List[float]
    :param format: Format of the structure files.
    :type format: Literal["pdb", "mmtf"]
    :param in_memory: Whether to load the entire dataset into memory.
    :type in_memory: bool
    :param pin_memory: Whether to pin memory for the dataloader.
    :type pin_memory: bool
    :param num_workers: Number of processes to use for the dataloader.
    :type num_workers: int
    :param dataset_fraction: Fraction of the dataset to use. Defaults to
        ``1.0``.
    :type dataset_fraction: float
    :param transforms: List of transforms to apply to the data.
    :type transforms: Optional[List[Callable]]
    """

    def __init__(
        self,
        path: os.PathLike,
        dataset_name: IgFoldDatasetNames,
        batch_size: int,
        train_val_test: List[float] = [0.8, 0.1, 0.1],
        format: Literal["pdb"] = "pdb",
        in_memory: bool = False,
        pin_memory: bool = True,
        num_workers: int = 16,
        dataset_fraction: float = 1.0,
        transforms: Optional[List[Callable]] = None,
    ):
        if dataset_name == "paired_oas":
            self.DATASET_URL = "https://data.graylab.jhu.edu/OAS_paired.tar.gz"
            self.TAR_FILENAME = "OAS_paired.tar.gz"
        elif dataset_name == "jaffe":
            self.DATASET_URL = "https://data.graylab.jhu.edu/Jaffe2022.tar.gz"
            self.TAR_FILENAME = "Jaffe2022.tar.gz"
        else:
            raise ValueError(
                f"Dataset {dataset_name} not supported. Must be one of 'jaffe' / 'paired_oas"
            )
        self.UNCOMPRESSED_DIR_NAME = "predictions_flat"
        # Dataset args
        assert sum(train_val_test) == 1, "Split sizes are invalid"
        self.train_val_test = train_val_test
        self.in_memory = in_memory
        self.format = format
        self.dataset_fraction = dataset_fraction

        # Paths
        self.data_dir: pathlib.Path = pathlib.Path(path)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        self.structure_dir = self.data_dir / "structures"

        # Dataloader args
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        if transforms is not None:
            self.transform = self.compose_transforms(
                omegaconf.OmegaConf.to_container(
                    transforms, resolve=True
                )  # type: ignore
            )
        else:
            self.transform = None

    def setup(self, stage: Optional[str] = None):
        """Perform datamodule setup tasks (downloads structure data).

        :param stage: Model training/val/test stage. Defaults to ``None``
        :type stage: Optional[str]
        """
        self.download()

    def download(self):
        """Download structure data to ``self.data_dir / self.TAR_FILENAME``."""
        if not os.path.exists(self.data_dir / self.TAR_FILENAME):
            logger.info(f"Downloading IgFold structures to {self.data_dir}")
            wget.download(
                self.DATASET_URL, out=str(self.data_dir / self.TAR_FILENAME)
            )
        else:
            logger.info(f"IgFold structures found in {self.data_dir}")

        if not os.path.exists(self.structure_dir):
            logger.info(f"Extracting tarfile to {self.structure_dir}")
            tar = tarfile.open(str(self.data_dir / self.TAR_FILENAME))
            tar.extractall(str(self.data_dir))
            tar.close()
            os.rename(
                self.data_dir / self.UNCOMPRESSED_DIR_NAME, self.structure_dir
            )

    def parse_dataset(
        self, split: Literal["train", "val", "test"]
    ) -> List[str]:
        """
        Parse dataset for a given ``split``.

        :param split: Name of split
        :type split: Literal["train", "val", "test"]
        :returns: List of filenames for a given ``split``.
        :rtype: List[str]
        """
        # If we've already split, return the split data
        if hasattr(self, f"{split}_pdbs"):
            return getattr(self, f"{split}_pdbs")

        structure_files = glob.glob(
            str(self.structure_dir / f"*.{self.format}"), recursive=True
        )
        logger.info(
            f"Found {len(structure_files)} pdbs in: {self.structure_dir}"
        )

        # Selection subset
        structure_files = random.sample(
            structure_files, int(self.dataset_fraction * len(structure_files))
        )

        structure_files = [f.split("/")[-1] for f in structure_files]
        structure_files = [f.replace(".pdb", "") for f in structure_files]

        train_size, val_size, test_size = self.train_val_test

        # Split dataset
        logger.info(
            f"Splitting {len(structure_files)} structures into {train_size}, {val_size}, {test_size} split"
        )
        train, val = train_test_split(
            structure_files, test_size=val_size + test_size
        )
        val, test = train_test_split(
            val, test_size=test_size / (val_size + test_size)
        )
        logger.info(
            f"Split sizes: {len(train)} train, {len(val)} val, {len(test)} test"
        )

        self.train_pdbs = train
        self.val_pdbs = val
        self.test_pdbs = test

        return getattr(self, f"{split}_pdbs")

    def parse_labels(self):
        """Parse labels.

        This is a no-op for this dataset as it has no labels.
        """
        pass

    def exclude_pdbs(self):
        """
        Identify PDBs to exclude.

        This is a no-op for this dataset as there are no PDBs to exclude..
        """
        pass

    def _get_dataset(
        self, split: Literal["train", "val", "test"]
    ) -> ProteinDataset:
        """
        Retrieve dataset for a given split.

        :param split: Name of split to retrieve.
        :type split: Literal["train", "val", "test"]
        :returns: ProteinDataset for a given ``split``.
        :rtype: ProteinDataset
        """
        pdb_codes = self.parse_dataset(split)
        logger.info(f"Initialising Graphein dataset for {split}...")

        return ProteinDataset(
            pdb_codes=pdb_codes,
            root=str(self.data_dir),
            pdb_dir=str(self.structure_dir),
            overwrite=False,
            transform=self.transform,
            format=self.format,
            in_memory=self.in_memory,
        )

    def train_dataset(self) -> ProteinDataset:
        """Return the train dataset.

        :returns: Train dataset
        :rtype: ProteinDataset
        """
        return self._get_dataset("train")

    def val_dataset(self) -> ProteinDataset:
        """Return the val dataset.

        :returns: Val dataset
        :rtype: ProteinDataset
        """
        return self._get_dataset("val")

    def test_dataset(self) -> ProteinDataset:
        """Return the test dataset.

        :returns: Test dataset
        :rtype: ProteinDataset
        """
        return self._get_dataset("test")

    def train_dataloader(self) -> ProteinDataLoader:
        """Return the train dataloader.

        :returns: Train dataloader
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
        """Return the val dataloader.

        :returns: val dataloader
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
        """Return the test dataloader.

        :returns: Test dataloader
        :rtype: ProteinDataLoader
        """
        return ProteinDataLoader(
            self.test_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    import hydra

    from proteinworkshop import constants

    cfg = omegaconf.OmegaConf.load(
        constants.HYDRA_CONFIG_PATH / "dataset" / "igfold_paired_oas.yaml"
    )
    cfg.datamodule.path = str(
        pathlib.Path(constants.DATA_PATH) / "igfold_paired_oas"
    )
    cfg.datamodule.transforms = None

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.download()
    dl = datamodule.test_dataloader()
    for i in dl:
        print(i)
        break
