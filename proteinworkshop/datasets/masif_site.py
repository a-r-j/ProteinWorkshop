import os
import pathlib
from typing import Callable, Iterable, List, Literal, Optional

import omegaconf
import wget
from graphein.protein.tensor.dataloader import ProteinDataLoader
from loguru import logger
from sklearn.model_selection import train_test_split

from proteinworkshop.datasets.base import ProteinDataModule, ProteinDataset


class MaSIFPPISP(ProteinDataModule):
    def __init__(
        self,
        path: str,
        pdb_dir: str,
        batch_size: int,
        dataset_fraction: float = 1.0,
        format: Literal["mmtf", "pdb"] = "mmtf",
        obsolete: str = "drop",
        val_fraction: float = 0.1,
        in_memory: bool = False,
        pin_memory: bool = True,
        num_workers: int = 16,
        shuffle_labels: bool = False,
        transforms: Optional[Iterable[Callable]] = None,
        overwrite: bool = False,
    ):
        super().__init__()

        self.path = pathlib.Path(path)
        self.pdb_dir = pdb_dir
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)

        if transforms is not None:
            self.transform = self.compose_transforms(
                omegaconf.OmegaConf.to_container(transforms, resolve=True)
            )
        else:
            self.transform = None

        self.overwrite = overwrite
        self.in_memory = in_memory
        self.dataset_fraction = dataset_fraction
        self.obsolete = obsolete
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.shuffle_labels = shuffle_labels
        self.format = format

        self.prepare_data_per_node = True
        self.val_fraction = val_fraction
        self.TRAIN_DATA_URL: str = "https://raw.githubusercontent.com/LPDI-EPFL/masif/master/data/masif_site/lists/training.txt"
        self.TEST_DATA_URL: str = "https://raw.githubusercontent.com/LPDI-EPFL/masif/master/data/masif_site/lists/testing.txt"
        self.TEST_TRANSIENT_DATA_URL: str = "https://raw.githubusercontent.com/LPDI-EPFL/masif/master/data/masif_site/lists/testing_transient.txt"

    def setup(self, stage: Optional[str] = None):
        self.download()

    def exclude_pdbs(self) -> List[str]:
        return ["1EXB_ABDC", "3LVK_AC"]

    def download(self):
        if not os.path.exists(self.path / "training.txt"):
            logger.info(
                f"Downloading training data from {self.TRAIN_DATA_URL}"
            )
            wget.download(
                self.TRAIN_DATA_URL,
                out=str(self.path / "training.txt"),
            )
        else:
            logger.info(
                f"Training data already exists at {self.path / 'training.txt'}"
            )

        if not os.path.exists(self.path / "testing.txt"):
            wget.download(
                self.TEST_DATA_URL,
                out=str(self.path / "testing.txt"),
            )
        else:
            logger.info(
                f"Test data already exists at {self.path / 'training.txt'}"
            )

    def parse_labels(self):
        pass

    def _get_dataset(self, split: str) -> ProteinDataset:
        if not hasattr(self, f"{split}_ids"):
            self.parse_dataset()

        ids = getattr(self, f"{split}_ids")
        pdb_codes = []
        chains = []
        for id in ids:
            pdb_codes.append(id[0])
            chains.append(id[1])

        return ProteinDataset(
            root=str(self.path),
            pdb_dir=self.pdb_dir,
            pdb_codes=pdb_codes,
            chains=["all"] * len(pdb_codes),
            graph_labels=chains,  # Hack to store the chain IDs to extract per-residue labels later via the transform.,
            overwrite=self.overwrite,
            transform=self.transform,
            format=self.format,
            in_memory=self.in_memory,
            out_names=[f"{id[0]}_{id[1]}" for id in ids],
        )

    def train_dataloader(self) -> ProteinDataLoader:
        return ProteinDataLoader(
            self._get_dataset("train"),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> ProteinDataLoader:
        return ProteinDataLoader(
            self._get_dataset("val"),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> ProteinDataLoader:
        return ProteinDataLoader(
            self._get_dataset("test"),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def train_dataset(self) -> ProteinDataset:
        return self._get_dataset("train")

    def val_dataset(self) -> ProteinDataset:
        return self._get_dataset("val")

    def test_dataset(self) -> ProteinDataset:
        return self._get_dataset("test")

    def parse_dataset(self):
        # Read files
        with open(self.path / "training.txt", "r") as f:
            train = f.readlines()
        train = [x.strip() for x in train]

        with open(self.path / "testing.txt", "r") as f:
            test = f.readlines()
        test = [x.strip() for x in test]

        train = [x for x in train if x not in self.exclude_pdbs()]
        test = [x for x in test if x not in self.exclude_pdbs()]

        # Train / val split
        train, val = train_test_split(train, test_size=self.val_fraction)

        # Split into PDB Code and Chain(s)
        train = [x.split("_") for x in train]
        val = [x.split("_") for x in val]
        test = [x.split("_") for x in test]

        logger.info(f"Found {len(train)} training examples")
        logger.info(f"Found {len(val)} validation examples")
        logger.info(f"Found {len(test)} test examples")

        # Remove obsolete PDBs
        for split in ["train", "val", "test"]:
            if self.obsolete == "drop":
                logger.info("Dropping obsolete PDBs")
                data = [
                    x
                    for x in eval(split)
                    if x[0].lower() not in self.obsolete_pdbs.keys()
                ]
                setattr(self, f"{split}_ids", data)
                logger.info(
                    f"Found {len(data)} examples in {split} after dropping obsolete PDBs"
                )
            else:
                raise NotImplementedError(
                    "Obsolete PDB replacement not implemented"
                )


if __name__ == "__main__":
    import hydra

    from proteinworkshop import constants

    cfg = omegaconf.OmegaConf.load(
        constants.SRC_PATH / "config" / "dataset" / "masif_site.yaml"
    )
    cfg.datamodule.path = pathlib.Path(constants.DATA_PATH) / "masif_site"  # type: ignore
    cfg.datamodule.pdb_dir = pathlib.Path(constants.DATA_PATH) / "pdb"  # type: ignore
    cfg.datamodule.transforms = []
    ds = hydra.utils.instantiate(cfg)
    print(ds)
    ds["datamodule"].setup()
    ds["datamodule"].parse_dataset()
    import torch

    dl = ds["datamodule"].train_dataloader()
    for i, batch in enumerate(dl):
        print(batch)
        bad_seq = torch.argwhere(batch["amino_acid_one_hot"][:, -2] == 1)

        if bad_seq.sum() > 10:
            break
    # dl = ds["datamodule"].val_dataloader()
    # for batch in dl:
    #    print(batch)
    # dl = ds["datamodule"].test_dataloader()
    # for batch in dl:
    #    print(batch)
