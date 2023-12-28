import os
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, Literal, Optional

import omegaconf
import pandas as pd
import torch
import wget
from graphein.protein.tensor.data import Protein
from graphein.protein.tensor.dataloader import ProteinDataLoader
from loguru import logger as log
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from proteinworkshop.datasets.base import ProteinDataModule, ProteinDataset

LABEL_LINE: Dict[str, int] = {
    "MF": 1,
    "BP": 5,
    "CC": 9,
}


class GeneOntologyDataset(ProteinDataModule):
    """

    Statistics (test_cutoff=0.95):
        - #Train: 27,496
        - #Valid: 3,053
        - #Test: 2,991

    """

    def __init__(
        self,
        path: str,
        batch_size: int,
        split: str = "BP",
        obsolete="drop",
        pdb_dir: Optional[str] = None,
        format: Literal["mmtf", "pdb"] = "mmtf",
        in_memory: bool = False,
        dataset_fraction: float = 1.0,
        shuffle_labels: bool = False,
        pin_memory: bool = True,
        num_workers: int = 16,
        transforms: Optional[Iterable[Callable]] = None,
        overwrite: bool = False,
    ) -> None:
        super().__init__()
        self.pdb_dir = pdb_dir
        self.data_dir = Path(path)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.dataset_fraction = dataset_fraction
        self.split = split
        self.obsolete = obsolete
        self.format = format

        self.in_memory = in_memory
        self.overwrite = overwrite

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.prepare_data_per_node = True

        self.shuffle_labels = shuffle_labels

        if transforms is not None:
            self.transform = self.compose_transforms(
                omegaconf.OmegaConf.to_container(transforms, resolve=True)
            )
        else:
            self.transform = None

        self.train_fname = self.data_dir / "nrPDB-GO_train.txt"
        self.val_fname = self.data_dir / "nrPDB-GO_valid.txt"
        self.test_fname = self.data_dir / "nrPDB-GO_test.txt"
        self.label_fname = self.data_dir / "nrPDB-GO_annot.tsv"
        self.url = "https://zenodo.org/record/6622158/files/GeneOntology.zip"

        log.info(
            f"Setting up Gene Ontology dataset. Fraction {self.dataset_fraction}"
        )

    @lru_cache
    def parse_labels(self) -> Dict[str, torch.Tensor]:
        """
        Parse the GO labels from the nrPDB-GO_annot.tsv file.
        """

        log.info(
            f"Loading GO labels for task {self.split} from file {self.label_fname}."
        )

        try:
            label_line = LABEL_LINE[self.split]
        except KeyError as e:
            raise ValueError(f"Task {self.split} not recognised.") from e

        # Load list of all labels
        with open(self.label_fname, "r") as f:
            all_labels = f.readlines()[label_line].strip("\n").split("\t")
        log.info(f"Found {len(all_labels)} labels for task {self.split}.")

        # Load labels for each PDB example
        df = pd.read_csv(self.label_fname, sep="\t", skiprows=12)
        df.columns = ["PDB", "MF", "BP", "CC"]
        df.set_index("PDB", inplace=True)

        # Remove rows with no labels for this task
        labels = df[self.split].dropna().to_dict()
        log.info(f"Found {len(labels)} examples for task {self.split}.")

        # Split GO terms string into list of individual terms
        labels = {k: v.split(",") for k, v in labels.items()}

        # Encode labels into numeric values
        log.info("Encoding labels...")
        label_encoder = LabelEncoder().fit(all_labels)
        labels = {
            k: torch.tensor(label_encoder.transform(v))
            for k, v in tqdm(labels.items())
        }
        log.info(f"Encoded {len(labels)} labels for task {self.split}.")
        return labels

    def _get_dataset(
        self, split: Literal["training", "validation", "testing"]
    ) -> ProteinDataset:
        df = self.parse_dataset(split)
        log.info("Initialising Graphein dataset...")
        return ProteinDataset(
            root=str(self.data_dir),
            pdb_dir=str(self.pdb_dir),
            pdb_codes=list(df.pdb),
            chains=list(df.chain),
            graph_labels=list(list(df.label)),
            overwrite=self.overwrite,
            transform=self.labeller
            if self.transform is None
            else self.compose_transforms([self.labeller] + [self.transform]),
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

    def download(self):
        if not all(
            os.path.exists(f)
            for f in [
                self.train_fname,
                self.val_fname,
                self.test_fname,
                self.label_fname,
            ]
        ):
            log.info("Downloading dataset...")
            wget.download(self.url, out=str(self.data_dir))
            with zipfile.ZipFile(self.data_dir / "GeneOntology.zip") as f:
                f.extractall(self.data_dir.parent)
        else:
            log.info(f"Found dataset at {self.data_dir}")

    def exclude_pdbs(self):
        pass

    def parse_dataset(
        self, split: Literal["training", "validation", "testing"]
    ) -> pd.DataFrame:
        # sourcery skip: remove-unnecessary-else, swap-if-else-branches, switch
        """
        Parses the raw dataset files to Pandas DataFrames.
        Maps classes to numerical values.
        """
        # Load ID: label mapping
        class_map = self.parse_labels()

        # Read in IDs of structures in split
        if split == "training":
            data = pd.read_csv(self.train_fname, sep="\t", header=None)
            data = data.sample(frac=self.dataset_fraction)
        elif split == "validation":
            data = pd.read_csv(self.val_fname, sep="\t", header=None)
        elif split == "testing":
            data = pd.read_csv(self.test_fname, sep="\t", header=None)
        else:
            raise ValueError(f"Unknown split: {split}")

        log.info(f"Found {len(data)} original examples in {split}")
        log.info("Removing unlabelled proteins for this task...")
        data = data.loc[data[0].isin(class_map.keys())]
        log.info(f"Found {len(data)} labelled examples in {split}")

        # Map labels to IDs in dataframe
        log.info("Mapping labels to IDs...")
        data["label"] = data[0].map(class_map)
        data.columns = ["pdb", "label"]

        to_drop = ["5EXC-I"]
        data = data.loc[~data["pdb"].isin(to_drop)]

        data["chain"] = data["pdb"].str[5:]
        data["pdb"] = data["pdb"].str[:4].str.lower()

        if self.obsolete == "drop":
            log.info("Dropping obsolete PDBs")
            data = data.loc[
                ~data["pdb"].str.lower().isin(self.obsolete_pdbs.keys())
            ]
            log.info(
                f"Found {len(data)} examples in {split} after dropping obsolete PDBs"
            )
        else:
            raise NotImplementedError(
                "Obsolete PDB replacement not implemented"
            )
        # logger.info(f"Identified {len(data['label'].unique())} classes in this split: {split}")

        if self.shuffle_labels:
            log.info("Shuffling labels. Expecting random performance.")
            data["label"] = data["label"].sample(frac=1).values

        # logger.info(f"Found {len(data)} examples in {split} after removing nonstandard proteins")
        self.labeller = GOLabeller(data)
        return data.sample(frac=1)  # Shuffle dataset for batches


class GOLabeller:
    """
    This labeller applies the graph labels to each example as a transform.

    This is required as chains can be used across tasks (e.g. CC, BP or MF) with
    different labels.
    """

    def __init__(self, label_df: pd.DataFrame):
        self.labels = label_df

    def __call__(self, data: Protein) -> Protein:
        pdb, chain = data.id.split("_")
        label = self.labels.loc[
            (self.labels.pdb == pdb) & (self.labels.chain == chain)
        ].label.item()
        data.graph_y = label
        return data


if __name__ == "__main__":
    import pathlib

    import hydra

    from proteinworkshop import constants

    log.info("Imported libs")
    cfg = omegaconf.OmegaConf.load(
        constants.SRC_PATH / "config" / "dataset" / "go-bp.yaml"
    )
    # cfg = omegaconf.OmegaConf.load(constants.SRC_PATH / "config" / "dataset" / "go-mf.yaml")
    # cfg = omegaconf.OmegaConf.load(constants.SRC_PATH / "config" / "dataset" / "go-bp.yaml")
    cfg.datamodule.path = pathlib.Path(constants.DATA_PATH) / "GeneOntology"
    cfg.datamodule.pdb_dir = pathlib.Path(constants.DATA_PATH) / "pdb"
    cfg.datamodule.num_workers = 1
    cfg.datamodule.transforms = []
    log.info("Loaded config")

    ds = hydra.utils.instantiate(cfg)
    print(ds)
    # labels = ds["datamodule"].parse_labels()
    ds.datamodule.setup()
    dl = ds["datamodule"].train_dataloader()
    for batch in dl:
        print(batch)
    dl = ds["datamodule"].val_dataloader()
    for batch in dl:
        print(batch)
    dl = ds["datamodule"].test_dataloader()
    for batch in dl:
        print(batch)
