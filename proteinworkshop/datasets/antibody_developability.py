import os
import pathlib
from typing import Callable, List, Literal, Optional

import omegaconf
import torch
from graphein.protein.tensor.dataloader import ProteinDataLoader
from loguru import logger as log

from proteinworkshop.datasets.base import ProteinDataModule, ProteinDataset

try:
    from tdc.single_pred import Develop
except ImportError:
    log.warning("Dependency TDC not installed. Run: pip install PyTDC")


class AntibodyDevelopabilityDataModule(ProteinDataModule):
    def __init__(
        self,
        path: str,
        pdb_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        in_memory=False,
        format: Literal["mmtf", "pdb"] = "mmtf",
        obsolete_strategy: str = "drop",
        transforms: Optional[List[Callable]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Data module for antibody developability dataset from Chen et al.

        :param path: Path to store data.
        :type path: str
        :param pdb_dir: Path to directory containing PDB files.
        :type pdb_dir: str
        :param batch_size: Batch size for dataloaders.
        :type batch_size: int
        :param num_workers: Number of workers for dataloaders.
        :type num_workers: int
        :param pin_memory: Whether to pin memory for dataloaders.
        :type pin_memory: bool
        :param in_memory: Whether to load the entire dataset into memory.
        :type in_memory: bool
        :param format: Format to load PDB files in.
        :type format: str
        :param obsolete_strategy: How to handle obsolete PDB structures.
        :type obsolete_strategy: str
        :param transforms: List of transforms to apply to dataset.
        :type transforms: Optional[List[Callable]]
        :param overwrite: Whether or not to overwrite existing processed data.
            Defaults t o ``False``.
        :type overwrite: bool
        """
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
        self.overwrite = overwrite

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
        """
        Parses the dataset from Chen (retrieved from PyTDC).

        Accounts for obsolete PDB structures by either dropping them or
        replacing them with a similar structure.

        Sets the relevant pd.DataFrames on the following attributes:

        - ``train_data``: Training dataset
        - ``valid_data``: Validation dataset
        - ``test_data``: Test dataset

        :raises NotImplementedError: Replace obsolete PDBs not implemented.
        :raises ValueError: Obsolete strategy not recognised.
        """
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
        """Gets the dataset object for a given split."""
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
            overwrite=self.overwrite,
        )

    def train_dataset(self) -> ProteinDataset:
        """Returns the training dataset.

        .. seealso::
            :py:class:`proteinworkshop.datasets.base.ProteinDataset`

        :return: Training dataset
        :rtype: ProteinDataset
        """
        return self._get_dataset("train")

    def val_dataset(self) -> ProteinDataset:
        """Returns the validation dataset.

        .. seealso::
            :py:class:`proteinworkshop.datasets.base.ProteinDataset`

        :return: Validation dataset
        :rtype: ProteinDataset
        """
        return self._get_dataset("valid")

    def test_dataset(self) -> ProteinDataset:
        """Returns the test dataset.

        .. seealso::
            :py:class:`proteinworkshop.datasets.base.ProteinDataset`

        :return: Test dataset
        :rtype: ProteinDataset
        """
        return self._get_dataset("test")

    def train_dataloader(self) -> ProteinDataLoader:
        """Returns the training dataloader.

        .. seealso::
            :py:class:`graphein.protein.tensor.dataloader.ProteinDataLoader`

        :return: Training dataloader
        :rtype: ProteinDataLoader
        """
        return ProteinDataLoader(
            self.train_dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> ProteinDataLoader:
        """Returns the validation dataloader.

        .. seealso::
            :py:class:`graphein.protein.tensor.dataloader.ProteinDataLoader`

        :return: Validation dataloader
        :rtype: ProteinDataLoader
        """
        return ProteinDataLoader(
            self.val_dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> ProteinDataLoader:
        """Returns the test dataloader.

        .. seealso::
            :py:class:`graphein.protein.tensor.dataloader.ProteinDataLoader`

        :return: Test dataloader
        :rtype: ProteinDataLoader
        """
        return ProteinDataLoader(
            self.test_dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

if __name__ == "__main__":
    import pathlib

    import hydra

    from proteinworkshop import constants

    cfg = omegaconf.OmegaConf.load(
        constants.SRC_PATH / "config" / "dataset" / "antibody_developability.yaml"
    )
    cfg.datamodule.path = pathlib.Path(constants.DATA_PATH) / "AntibodyDevelopability"  # type: ignore
    cfg.datamodule.pdb_dir = pathlib.Path(constants.DATA_PATH) / "pdb"  # type: ignore
    cfg.datamodule.transforms = []
    ds = hydra.utils.instantiate(cfg)
    print(ds)
    ds["datamodule"].parse_dataset()
    dl = ds["datamodule"].train_dataloader()
    for batch in dl:
        print(batch)
    dl = ds["datamodule"].val_dataloader()
    for batch in dl:
        print(batch)
    dl = ds["datamodule"].test_dataloader()
    for batch in dl:
        print(batch)
