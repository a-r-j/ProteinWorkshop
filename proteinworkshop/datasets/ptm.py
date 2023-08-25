import json
import os
import pathlib
import subprocess
import zipfile
from functools import partial
from typing import Callable, Iterable, Optional

import numpy as np
import omegaconf
import pandas as pd
import torch
from graphein.protein.tensor.dataloader import ProteinDataLoader
from graphein.protein.utils import download_alphafold_structure
from loguru import logger as log
from proteinworkshop.datasets.base import ProteinDataModule, ProteinDataset
from torch_geometric.data import Dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


class PTMDataModule(ProteinDataModule):
    def __init__(
        self,
        path: str,
        batch_size: int,
        dataset_name: str = "ptm_13",
        in_memory: bool = False,
        pin_memory: bool = True,
        num_workers: int = 16,
        transforms: Optional[Iterable[Callable]] = None,
    ) -> None:
        super().__init__()
        self.dataset_name: str = dataset_name

        self.root = pathlib.Path(path) / self.dataset_name
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)

        self.pdb_dir = pathlib.Path(path) / "structures"
        if not os.path.exists(self.pdb_dir):
            os.makedirs(self.pdb_dir, exist_ok=True)

        self.in_memory = in_memory

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

        self.PTM_13_URL: str = (
            "https://zenodo.org/record/7655709/files/13PTM%20.zip?download=1"
        )

        if os.path.exists(self.root / "unavailable_structures"):
            log.info(
                f"Loading cached unavailable structure list from {self.root / 'unavailable_structures.txt'}"
            )
            with open(self.root / "unavailable_structures.txt", "r") as f:
                self.unavailable_structures = f.read().splitlines()
                self.unavailable_structures_cached = True
        else:
            self.unavailable_structures = []
            self.unavailable_structures_cached = False

        # In the dataset these have incorrect sequences
        self.unavailable_structures += ["q8wyj6", "q95jn5", "q80xa6", "p30545"]

        if self.dataset_name == "ptm_13":
            self.SITE_TYPES = {
                "Hydro_K",
                "Hydro_P",
                "Methy_K",
                "Methy_R",
                "N6-ace_K",
                "Palm_C",
                "Phos_ST",
                "Phos_Y",
                "Pyro_Q",
                "SUMO_K",
                "Ubi_K",
                "glyco_N",
                "glyco_ST",
            }
            self.SITE_TO_NUM = {site: i for i, site in enumerate(self.SITE_TYPES)}
            self.NUM_TO_SITE = {v: k for k, v in self.SITE_TO_NUM.items()}
        elif self.dataset_name == "optm":
            raise NotImplementedError

    def exclude_pdbs(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.download_dataset()
        for split in {"train", "val", "test"}:
            data = self.parse_dataset(split)

            # If using cached list of unavailable structures,
            # remove them from the dataset to avoid trying to download them
            if self.unavailable_structures_cached:
                data = data[~data["uniprot_id"].isin(self.unavailable_structures)]

            self.download_structures(data)

            # Get unavailable structures if not cached
            ids = list(data["uniprot_id"].unique())
            if not self.unavailable_structures_cached:
                for id in ids:
                    if not os.path.exists(self.pdb_dir / f"{id}.pdb"):
                        self.unavailable_structures.append(id)
            log.info(f"Unavailable structures: {len(self.unavailable_structures)}")

        # Cache unavailable structures if not already cached
        if not self.unavailable_structures_cached:
            log.info(
                f"Caching unavailable structure list to {self.root / 'unavailable_structures.txt'}"
            )
            with open(self.root / "unavailable_structures.txt", "w") as f:
                f.write("\n".join(self.unavailable_structures))

    def download(self):
        self.download_dataset()

    def download_structures(self, data: pd.DataFrame):
        uniprot_ids = list(data["uniprot_id"].unique())
        to_download = [
            id for id in uniprot_ids if not os.path.exists(self.pdb_dir / f"{id}.pdb")
        ]
        log.info(f"Downloading {len(to_download)} PDBs...")
        dl_func = partial(
            download_alphafold_structure,
            version=4,
            out_dir=str(self.pdb_dir),
            aligned_score=False,
        )
        process_map(dl_func, to_download, max_workers=self.num_workers, chunksize=32)

    def download_dataset(self):
        file_paths = [
            self.root / f"PTM_{split}.json" for split in ["train", "val", "test"]
        ]

        if self.dataset_name == "ptm_13":
            if not all(os.path.exists(fpath) for fpath in file_paths):
                log.info("Downloading PTM_13 dataset...")
                cmd = f"wget {self.PTM_13_URL} -O {str(self.root / '13PTM.zip')}"
                subprocess.call(cmd, shell=True)
                with zipfile.ZipFile(self.root / "13PTM.zip", "r") as zip_ref:
                    zip_ref.extractall(self.root)
        else:
            raise NotImplementedError

    def parse_dataset(self, split: str) -> pd.DataFrame:
        data = json.load(open(self.root / f"PTM_{split}.json", "r"))
        data = pd.DataFrame.from_records(data).T
        data["uniprot_id"] = data.index
        data.columns = [
            "seq",
            "label",
            "uniprot_id",
        ]
        data["uniprot_id"] = data["uniprot_id"].str.lower()
        data["length"] = data["seq"].apply(len)
        data = data[~data["uniprot_id"].isin(self.unavailable_structures)]
        log.info(f"Found {len(data)} examples in {split}")
        return data

    def parse_labels(self, df: pd.DataFrame):
        labels = df["label"].values
        labels = [
            torch.zeros((length, len(self.SITE_TYPES)))
            for length in df["length"].values
        ]
        ids = list(df["uniprot_id"].values)

        label_map = {}
        for id, label_data, label_tensor in tqdm(zip(ids, df["label"], labels)):
            indices = np.array([i["site"] for i in label_data])
            site_type = [i["ptm_type"] for i in label_data]
            site_indices = [self.SITE_TO_NUM[i] for i in site_type]
            label_tensor[indices, site_indices] = 1
            label_map[id] = label_tensor

        print(len(label_map))
        return label_map

    def _get_dataset(self, split: str) -> ProteinDataset:
        data = self.parse_dataset(split)
        labels = self.parse_labels(data)

        return ProteinDataset(
            root=str(self.root),
            pdb_dir=str(self.pdb_dir),
            pdb_codes=list(data["uniprot_id"].values),
            node_labels=[labels[id] for id in data["uniprot_id"].values],
            transform=self.transform,
            format="pdb",
            in_memory=self.in_memory,
        )

    def train_dataset(self):
        return self._get_dataset("train")

    def val_dataset(self) -> Dataset:
        return self._get_dataset("val")

    def test_dataset(self) -> Dataset:
        return self._get_dataset("test")

    def train_dataloader(self):
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


if __name__ == "__main__":
    dm = PTMDataModule(".", "./pdb")
    # dm.download()
    df = dm.parse_dataset("val")
    a = dm.parse_labels(df)
