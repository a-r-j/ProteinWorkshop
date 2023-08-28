import os
import shutil
import urllib
from pathlib import Path

import pandas as pd
import torch_geometric
from graphein.protein.utils import read_fasta
from loguru import logger as log
from tqdm import tqdm

from proteinworkshop.datasets.base import ProteinDataModule


def str2bool(v: str) -> bool:
    return v.lower() in {"yes", "true", "t", "1"}


class FLIPDatamodule(ProteinDataModule):
    def __init__(self, root: str, dataset_name: str, split: str) -> None:
        super().__init__()
        self.root = Path(root)
        os.makedirs(self.root / dataset_name, exist_ok=True)
        self.dataset_name = dataset_name
        self.split = split
        self.BASE_URL = "http://data.bioembeddings.com/public/FLIP/fasta/"
        self.DATA_URL = (
            self.BASE_URL + self.dataset_name + "/" + self.split + ".fasta"
        )
        self.data_fname = self.root / dataset_name / f"{split}.fasta"

    def download(self, overwrite: bool = False):
        req = urllib.request.Request(
            self.DATA_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64)"},
        )
        if not os.path.exists(self.data_fname) or overwrite:
            with urllib.request.urlopen(req) as response, open(
                self.data_fname, "wb"
            ) as outfile:
                log.info(
                    f"Downloading {self.split} split for {self.dataset_name} dataset from {self.DATA_URL} into: {self.data_fname}"
                )
                shutil.copyfileobj(response, outfile)
        else:
            log.info(
                f"Split {self.split} for {self.dataset_name} dataset already exists at {self.data_fname}, skipping download"
            )

    def parse_dataset(self, split: str) -> pd.DataFrame:
        log.info("Parsing dataset...")
        fasta_dict = read_fasta(self.data_fname)
        records = []
        for k, v in tqdm(fasta_dict.items()):
            keys = k.split(" ")
            record = {
                "name": keys[0],
                "label": float(keys[1].replace("TARGET=", "")),
            }
            record["set"] = keys[2].replace("SET=", "")
            record["validation"] = str2bool(keys[3].replace("VALIDATION=", ""))
            record["sequence"] = v
            records.append(record)
        df = pd.DataFrame.from_records(records)
        if split == "train":
            df = df[df["set"] == "train"]
        elif split == "test":
            df = df[df["set"] == "test"]
        elif split == "val":
            df = df[df["validation"] == True]
        else:
            raise ValueError(f"Invalid split: {split}")

        log.info(f"Loaded {len(df)} examples")
        return df

    def parse_labels(self, split: str):
        pass

    def train_dataset(self):
        data = self.parse_dataset("train")
        print(data)
        # return FASTADataset()
        raise NotImplementedError

    def val_dataset(self):
        self.parse_dataset("val")
        # return FASTADataset()
        raise NotImplementedError

    def test_dataset(self):
        self.parse_dataset("test")
        # return FASTADataset()
        raise NotImplementedError

    def exclude_pdbs(self):
        pass

    def train_dataloader(self) -> torch_geometric.loader.DataLoader:
        raise NotImplementedError

    def val_dataloader(self) -> torch_geometric.loader.DataLoader:
        raise NotImplementedError

    def test_dataloader(self) -> torch_geometric.loader.DataLoader:
        raise NotImplementedError
