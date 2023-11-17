import os
from pathlib import Path
from typing import Callable, Dict, Iterable, Literal, Optional

import omegaconf
import pandas as pd
import torch
import wget
from graphein.protein.tensor.dataloader import ProteinDataLoader
from loguru import logger

from proteinworkshop.datasets.base import ProteinDataModule, ProteinDataset


class EnzymeCommissionReactionDataset(ProteinDataModule):
    def __init__(
        self,
        path: str,
        batch_size: int,
        pdb_dir: Optional[str] = None,
        format: Literal["mmtf", "pdb"] = "mmtf",
        obsolete: str = "drop",
        in_memory: bool = False,
        pin_memory: bool = True,
        num_workers: int = 16,
        dataset_fraction: float = 1.0,
        shuffle_labels: bool = False,
        transforms: Optional[Iterable[Callable]] = None,
        overwrite: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = Path(path)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

        if transforms is not None:
            self.transform = self.compose_transforms(
                omegaconf.OmegaConf.to_container(transforms, resolve=True)
            )
        else:
            self.transform = None

        self.pdb_dir = pdb_dir
        self.in_memory = in_memory
        self.dataset_fraction = dataset_fraction
        self.obsolete = obsolete
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.shuffle_labels = shuffle_labels
        self.format = format
        self.overwrite = overwrite

        self.prepare_data_per_node = True
        logger.info(
            f"Setting up EC Prediction dataset. Fraction {self.dataset_fraction}"
        )
        self.BASE_URL = "https://raw.githubusercontent.com/phermosilla/IEConv_proteins/master/Datasets/data/ProtFunct/"
        self.TRAIN_URL = f"{self.BASE_URL}training.txt"
        self.VALIDATION_URL = f"{self.BASE_URL}validation.txt"
        self.LABELS_URL = f"{self.BASE_URL}chain_functions.txt"
        self.TEST_URL = f"{self.BASE_URL}testing.txt"

        self.TRAIN_FNAME = self.data_dir / "training.txt"
        self.VAL_FNAME = self.data_dir / "validation.txt"
        self.TEST_FNAME = self.data_dir / "testing.txt"
        self.LABEL_FNAME = self.data_dir / "chain_functions.txt"

    def exclude_pdbs(self):
        pass

    def download(self):  # sourcery skip: move-assign
        if not os.path.exists(self.TEST_FNAME):
            logger.info(f"Downloading training data to {self.data_dir}...")
            wget.download(self.TRAIN_URL, out=str(self.data_dir))

        if not os.path.exists(self.VAL_FNAME):
            logger.info(f"Downloading validation data to {self.data_dir}...")
            wget.download(self.VALIDATION_URL, out=str(self.data_dir))

        if not os.path.exists(self.data_dir / "testing.txt"):
            logger.info(f"Downloading test data to {self.data_dir}...")
            wget.download(self.TEST_URL, out=str(self.data_dir))

        if not os.path.exists(self.LABEL_FNAME):
            logger.info(f"Downloading class map data to {self.data_dir}...")
            wget.download(self.LABELS_URL, out=str(self.data_dir))

    def parse_labels(self) -> Dict[str, str]:
        class_map = pd.read_csv(
            self.data_dir / "chain_functions.txt", sep=",", header=None
        )
        return dict(class_map.values)

    def _get_dataset(
        self, split: Literal["training", "validation", "testing"]
    ) -> ProteinDataset:
        df = self.parse_dataset(split)
        return ProteinDataset(
            root=str(self.data_dir),
            pdb_dir=self.pdb_dir,
            pdb_codes=list(df.pdb),
            chains=list(df.chain),
            graph_labels=[torch.tensor(a) for a in list(df.label)],
            overwrite=self.overwrite,
            transform=self.transform,
            format=self.format,
            in_memory=self.in_memory,
        )

    def train_dataset(self) -> ProteinDataset:
        return self._get_dataset("training")

    def val_dataset(self) -> ProteinDataset:
        return self._get_dataset("validation")

    def test_dataset(self) -> ProteinDataset:
        return self._get_dataset("testing")

    def train_dataloader(self) -> ProteinDataLoader:
        return ProteinDataLoader(
            self.train_dataset(),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> ProteinDataLoader:
        return ProteinDataLoader(
            self.val_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> ProteinDataLoader:
        return ProteinDataLoader(
            self.test_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def parse_dataset(self, split: str) -> pd.DataFrame:
        """
        Parses the raw dataset files to Pandas DataFrames.
        Maps classes to numerical values.
        """
        # Load ID: label mapping
        class_map = self.parse_labels()

        # Read in IDs of structures in split
        if split == "training":
            data = pd.read_csv(
                self.data_dir / "training.txt", sep=",", header=None
            )
            data = data.sample(frac=self.dataset_fraction)
        elif split == "validation":
            data = pd.read_csv(
                self.data_dir / "validation.txt", sep=",", header=None
            )
        elif split == "testing":
            data = pd.read_csv(
                self.data_dir / "testing.txt", sep=",", header=None
            )
        else:
            raise ValueError(f"Unknown split: {split}")

        logger.info(f"Found {len(data)} original examples in {split}")
        # Assign IDs to DF
        data[["pdb", "chain"]] = data[0].str.split(".", expand=True)

        # Remove obsolete PDBs
        if self.obsolete == "drop":
            logger.info("Dropping obsolete PDBs")
            data = data.loc[~data["pdb"].isin(self.obsolete_pdbs.keys())]
            logger.info(
                f"Found {len(data)} examples in {split} after dropping obsolete PDBs"
            )
        else:
            raise NotImplementedError(
                "Obsolete PDB replacement not implemented"
            )

        # Map labels to IDs in dataframe
        data["label"] = data[0].map(class_map)
        logger.info(f"Identified {len(data['label'].unique())} classes")
        data["id"] = data["pdb"] + "_" + data["chain"]

        if self.shuffle_labels:
            logger.info("Shuffling labels. Expecting random performance.")
            data["label"] = data["label"].sample(frac=1).values

        return data.sample(frac=1)  # Shuffle dataset for batches


if __name__ == "__main__":
    import pathlib

    import hydra

    from proteinworkshop import constants

    cfg = omegaconf.OmegaConf.load(
        constants.SRC_PATH / "config" / "dataset" / "ec_reaction.yaml"
    )
    cfg.datamodule.path = pathlib.Path(constants.DATA_PATH) / "ECReaction"  # type: ignore
    cfg.datamodule.pdb_dir = pathlib.Path(constants.DATA_PATH) / "pdb"  # type: ignore
    cfg.datamodule.transforms = []
    ds = hydra.utils.instantiate(cfg)
    print(ds)
    dl = ds["datamodule"].val_dataloader()
    for batch in dl:
        print(batch)
    dl = ds["datamodule"].test_dataloader()
    for batch in dl:
        print(batch)
