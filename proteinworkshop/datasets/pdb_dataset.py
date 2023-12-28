from typing import Callable, Iterable, List, Optional

import hydra
import omegaconf
import pandas as pd
from graphein.ml.datasets import PDBManager
from loguru import logger as log
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from proteinworkshop.datasets.base import ProteinDataModule, ProteinDataset
from proteinworkshop.datasets.utils import download_pdb_mmtf


class PDBData:
    def __init__(
        self,
        fraction: float,
        min_length: int,
        max_length: int,
        molecule_type: str,
        experiment_types: List[str],
        oligomeric_min: int,
        oligomeric_max: int,
        best_resolution: float,
        worst_resolution: float,
        has_ligands: List[str],
        remove_ligands: List[str],
        remove_non_standard_residues: bool,
        remove_pdb_unavailable: bool,
        split_sizes: List[float],
    ):
        self.fraction = fraction
        self.molecule_type = molecule_type
        self.experiment_types = experiment_types
        self.oligomeric_min = oligomeric_min
        self.oligomeric_max = oligomeric_max
        self.best_resolution = best_resolution
        self.worst_resolution = worst_resolution
        self.has_ligands = has_ligands
        self.remove_ligands = remove_ligands
        self.remove_non_standard_residues = remove_non_standard_residues
        self.remove_pdb_unavailable = remove_pdb_unavailable
        self.min_length = min_length
        self.max_length = max_length
        self.split_sizes = split_sizes

    def create_dataset(self):
        log.info(f"Initializing PDBManager in {self.path}...")
        pdb_manager = PDBManager(root_dir=self.path)
        num_chains = len(pdb_manager.df)
        log.info(f"Starting with: {num_chains} chains")

        log.info(f"Removing chains longer than {self.max_length}...")
        pdb_manager.length_longer_than(self.min_length, update=True)
        log.info(f"{len(pdb_manager.df)} chains remaining")

        log.info(f"Removing chains shorter than {self.min_length}...")
        pdb_manager.length_shorter_than(self.max_length, update=True)
        log.info(f"{len(pdb_manager.df)} chains remaining")

        log.info(
            f"Removing chains molecule types not in selection: {self.molecule_type}..."
        )
        pdb_manager.molecule_type(self.molecule_type, update=True)
        log.info(f"{len(pdb_manager.df)} chains remaining")

        # log.info(f"Removing chains experiment types not in selection: {self.experiment_types}...")
        # pdb_manager.experiment_type(self.experiment_types, update=True)
        # log.info(f"{len(pdb_manager.df)} chains remaining")

        log.info(
            f"Removing chains oligomeric state not in selection: {self.oligomeric_min} - {self.oligomeric_max}..."
        )
        pdb_manager.oligomeric(self.oligomeric_min, "greater", update=True)
        pdb_manager.oligomeric(self.oligomeric_max, "less", update=True)
        log.info(f"{len(pdb_manager.df)} chains remaining")

        log.info(
            f"Removing chains with resolution not in selection: {self.best_resolution} - {self.worst_resolution}..."
        )
        pdb_manager.resolution_better_than_or_equal_to(
            self.worst_resolution, update=True
        )
        pdb_manager.resolution_worse_than_or_equal_to(
            self.best_resolution, update=True
        )
        log.info(f"{len(pdb_manager.df)} chains remaining")

        if self.remove_ligands:
            log.info(
                f"Removing chains with ligands in selection: {self.remove_ligands}..."
            )
            pdb_manager.has_ligands(
                self.remove_ligands, inverse=True, update=True
            )
            log.info(f"{len(pdb_manager.df)} chains remaining")

        if self.has_ligands:
            log.info(
                f"Removing chains without ligands in selection: {self.has_ligands}..."
            )
            pdb_manager.has_ligands(self.has_ligands, update=True)
            log.info(f"{len(pdb_manager.df)} chains remaining")

        if self.remove_non_standard_residues:
            log.info("Removing chains with non-standard residues...")
            pdb_manager.remove_non_standard_alphabet_sequences(update=True)
            log.info(f"{len(pdb_manager.df)} chains remaining")
        if self.remove_pdb_unavailable:
            log.info("Removing chains with PDB unavailable...")
            pdb_manager.remove_unavailable_pdbs(update=True)
            log.info(f"{len(pdb_manager.df)} chains remaining")

        log.info(f"Splitting dataset into {self.split_sizes}...")
        split_names = ["train", "val", "test"]
        splits = pdb_manager.split_df_proportionally(
            df=pdb_manager.df,
            splits=split_names,
            split_ratios=self.split_sizes,
        )
        # log.info(splits["train"])
        # log.info(pdb_manager.df)
        # return pdb_manager.df
        log.info(splits["train"])
        return splits


class PDBDataModule(ProteinDataModule):
    def __init__(
        self,
        path: Optional[str] = None,
        pdb_dataset: Optional[PDBData] = None,
        transforms: Optional[Iterable[Callable]] = None,
        in_memory: bool = False,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        overwrite: bool = False,
    ):
        super().__init__()
        self.root = path
        self.dataset = pdb_dataset
        self.dataset.path = path
        self.format = "mmtf.gz"
        self.overwrite = overwrite

        self.in_memory = in_memory

        if transforms is not None:
            self.transform = self.compose_transforms(
                omegaconf.OmegaConf.to_container(transforms, resolve=True)
            )
        else:
            self.transform = None

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size

    def parse_dataset(self) -> pd.DataFrame:
        return self.dataset.create_dataset()

    def exclude_pdbs(self):
        pass

    def download(self):
        pdbs = self.parse_dataset()

        for k, v in pdbs:
            log.info(f"Downloading {k} PDBs")
            download_pdb_mmtf(pathlib.Path(self.root) / "raw", v.pdb.tolist())

    def parse_labels(self):
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def _get_dataset(self, split: str) -> Dataset:
        data = self.parse_dataset()[split]
        return ProteinDataset(
            pdb_codes=data["pdb"].tolist(),
            root=self.root,
            chains=data["chain"].tolist(),
            pdb_dir=self.root,
            format=self.format,
            transform=self.transform,
            in_memory=self.in_memory,
            overwrite=self.overwrite,
        )

    def train_dataset(self) -> Dataset:
        return self._get_dataset("train")

    def val_dataset(self) -> Dataset:
        return self._get_dataset("val")

    def test_dataset(self) -> Dataset:
        return self._get_dataset("test")


if __name__ == "__main__":
    import pathlib

    from proteinworkshop import constants

    cfg = omegaconf.OmegaConf.load(
        constants.SRC_PATH / "config" / "dataset" / "pdb.yaml"
    )
    cfg.datamodule.path = pathlib.Path(constants.DATA_PATH) / "pdb"
    print(cfg)
    ds = hydra.utils.instantiate(cfg)["datamodule"]
    print(ds)
    ds.val_dataset()
