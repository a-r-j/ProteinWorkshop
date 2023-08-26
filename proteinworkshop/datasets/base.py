import os
import pathlib
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from beartype import beartype
from graphein import verbose
from graphein.protein.tensor.dataloader import ProteinDataLoader
from graphein.protein.tensor.io import protein_to_pyg
from graphein.protein.utils import (download_pdb_multiprocessing,
                                    get_obsolete_mapping, read_fasta)
from loguru import logger
from sklearn.utils.class_weight import compute_class_weight
from proteinworkshop.features.sequence_features import amino_acid_one_hot
from torch_geometric import transforms as T
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

verbose(False)
from torch_geometric.data import Data


def pair_data(a: Data, b: Data) -> Data:
    """Pairs two graphs together in a single ``Data`` instance.

    The first graph is accessed via ``data.a`` (e.g. ``data.a.coords``)
    and the second via ``data.b``.

    :param a: The first graph.
    :type a: torch_geometric.data.Data
    :param b: The second graph.
    :type b: torch_geometric.data.Data
    :return: The paired graph.
    """
    out = Data()
    out.a = a
    out.b = b
    return out


class ProteinDataModule(L.LightningDataModule, ABC):
    @abstractmethod
    def download(self):
        """
        Implements downloading of raw data.

        Typically this will be an index file of structure
        identifiers (for datasets derived from the PDB) but
        may contain structures too.
        """
        ...

    def setup(self, stage: Optional[str] = None):
        self.download()
        logger.info("Preprocessing training data")
        self.train_ds = self.train_dataset()
        logger.info("Preprocessing validation data")
        self.val_ds = self.val_dataset()
        logger.info("Preprocessing test data")
        self.test_ds = self.test_dataset()
        # self.class_weights = self.get_class_weights()

    @property
    @lru_cache
    def obsolete_pdbs(self) -> Dict[str, str]:
        """This method returns a mapping of obsolete PDB codes
        to their updated replacements.

        :return: Mapping of obsolete PDB codes to their updated replacements.
        :rtype: Dict[str, str]
        """
        return get_obsolete_mapping()

    @beartype
    def compose_transforms(self, transforms: Iterable[Callable]) -> T.Compose:
        """Composes an iterable of Transforms into a single transform.

        :param transforms: An iterable of transforms.
        :type transforms: Iterable[Callable]
        :raises ValueError: If ``transforms`` is not a list or dict.
        :return: A single transform.
        :rtype: T.Compose
        """
        if isinstance(transforms, list):
            return T.Compose(transforms)
        elif isinstance(transforms, dict):
            return T.Compose(list(transforms.values()))
        else:
            raise ValueError("Transforms must be a list or dict")

    @abstractmethod
    def parse_dataset(self, split: str) -> pd.DataFrame:
        """
        This methods implements the parsing of the raw dataset to a dataframe.

        Override this method to implement custom parsing of raw data.

        :param split: The split to parse (e.g. train/val/test)
        :type split: str
        :return: The parsed dataset as a dataframe.
        :rtype: pd.DataFrame
        """
        ...

    @abstractmethod
    def parse_labels(self) -> Any:
        """Optional method to parse labels from the dataset.

        Labels may or may not be present in the dataframe returned by
        ``parse_dataset``.

        :return: The parsed labels in any format. We'd recommend:
            ``Dict[id, Tensor]``.
        :rtype: Any
        """
        ...

    @abstractmethod
    def exclude_pdbs(self):
        ...

    @abstractmethod
    def train_dataset(self) -> Dataset:
        """
        Implements the construction of the training dataset.

        :return: The training dataset.
        :rtype: Dataset
        """
        ...

    @abstractmethod
    def val_dataset(self) -> Dataset:
        """
        Implements the construction of the validation dataset.

        :return: The validation dataset.
        :rtype: Dataset
        """
        ...

    @abstractmethod
    def test_dataset(self) -> Dataset:
        """
        Implements the construction of the test dataset.

        :return: The test dataset.
        :rtype: Dataset
        """
        ...

    @abstractmethod
    def train_dataloader(self) -> ProteinDataLoader:
        """
        Implements the construction of the training dataloader.

        :return: The training dataloader.
        :rtype: ProteinDataLoader
        """
        ...

    @abstractmethod
    def val_dataloader(self) -> ProteinDataLoader:
        """Implements the construction of the validation dataloader.

        :return: The validation dataloader.
        :rtype: ProteinDataLoader
        """
        ...

    @abstractmethod
    def test_dataloader(self) -> ProteinDataLoader:
        """Implements the construction of the test dataloader.

        :return: The test dataloader.
        :rtype: ProteinDataLoader
        """
        ...

    def get_class_weights(self) -> torch.Tensor:
        labels: Dict[str, torch.Tensor] = self.parse_labels()
        labels = list(labels.values())  # type: ignore
        labels = np.array(labels)  # type: ignore
        weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(labels), y=labels
        )
        return torch.tensor(weights)


class ProteinDataset(Dataset):
    def __init__(
        self,
        pdb_codes: List[str],
        root: Optional[str] = None,
        pdb_dir: Optional[str] = None,
        processed_dir: Optional[str] = None,
        pdb_paths: Optional[List[str]] = None,
        chains: Optional[List[str]] = None,
        graph_labels: Optional[List[torch.Tensor]] = None,
        node_labels: Optional[List[torch.Tensor]] = None,
        transform: Optional[List[Callable]] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        overwrite: bool = False,
        format: str = "pdb",
        in_memory: bool = False,
        store_het: bool = False,
    ):
        self.pdb_codes = [pdb.lower() for pdb in pdb_codes]
        self.pdb_dir = pdb_dir
        self.pdb_paths = pdb_paths
        self.overwrite = overwrite
        self.chains = chains
        self.node_labels = node_labels
        self.graph_labels = graph_labels
        self.format = format
        self.root = root
        self.in_memory = in_memory
        self.store_het = store_het

        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.structures = pdb_codes if pdb_codes is not None else pdb_paths
        if self.in_memory:
            logger.info("Reading data into memory")
            self.data = [
                torch.load(pathlib.Path(self.root) / "processed" / f)
                for f in tqdm(self.processed_file_names)
            ]

    def download(self):
        """Downloads PDB files not already present in the raw directory."""
        if self.pdb_codes is not None:
            to_download = (
                self.pdb_codes
                if self.overwrite
                else [
                    pdb
                    for pdb in self.pdb_codes
                    if not (
                        os.path.exists(Path(self.raw_dir) / f"{pdb}.{self.format}")
                        or os.path.exists(
                            Path(self.raw_dir) / f"{pdb}.{self.format}.gz"
                        )
                    )
                ]
            )
            to_download = list(set(to_download))
            logger.info(f"Downloading {len(to_download)} structures")
            file_format = (
                self.format[:-3] if self.format.endswith(".gz") else self.format
            )
            download_pdb_multiprocessing(to_download, self.raw_dir, format=file_format)

    def len(self) -> int:
        return len(self.pdb_codes)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "raw") if self.pdb_dir is None else self.pdb_dir  # type: ignore

    @property
    def raw_file_names(self) -> List[str]:
        if self.pdb_paths is None:
            return [f"{pdb}.{format}" for pdb in self.pdb_codes]
        else:
            return list(self.pdb_paths)

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        if self.chains is not None:
            return [
                f"{pdb}_{chain}.pt" for pdb, chain in zip(self.pdb_codes, self.chains)
            ]
        else:
            return [f"{pdb}.pt" for pdb in self.pdb_codes]

    def process(self):
        if not self.overwrite:
            if self.chains is not None:
                pdb_codes = [
                    (i, pdb)
                    for i, pdb in enumerate(self.pdb_codes)
                    if not os.path.exists(
                        Path(self.processed_dir) / f"{pdb}_{self.chains[i]}.pt"
                    )
                ]
            else:
                pdb_codes = [
                    (i, pdb)
                    for i, pdb in enumerate(self.pdb_codes)
                    if not os.path.exists(Path(self.processed_dir) / f"{pdb}.pt")
                ]
            logger.info(f"Processing {len(pdb_codes)} unprocessed structures")
        else:
            pdb_codes = self.pdb_codes

        for i, pdb in tqdm(pdb_codes):
            try:
                graph = protein_to_pyg(
                    path=str(Path(self.raw_dir) / f"{pdb}.{self.format}"),
                    chain_selection=self.chains[i]
                    if self.chains is not None
                    else "all",
                    keep_insertions=True,
                    store_het=self.store_het,
                )
            except Exception as e:
                logger.error(
                    f"Error processing {pdb} {self.chains[i]}: {e}"
                )  # type: ignore
                raise e

            fname = f"{pdb}.pt" if self.chains is None else f"{pdb}_{self.chains[i]}.pt"

            graph.id = fname.split(".")[0]

            if self.graph_labels is not None:
                graph.graph_y = self.graph_labels[i]  # type: ignore

            if self.node_labels is not None:
                graph.node_y = self.node_labels[i]  # type: ignore

            torch.save(graph, Path(self.processed_dir) / fname)

    def get(self, idx: int) -> Data:
        """
        Returns PyTorch Geometric Data object for a given index.

        :param idx: Index to retrieve.
        :type idx: int
        :return: PyTorch Geometric Data object.
        """
        if self.in_memory:
            return self.data[idx]

        if self.chains is not None:
            fname = f"{self.pdb_codes[idx]}_{self.chains[idx]}.pt"
        else:
            fname = f"{self.pdb_codes[idx]}.pt"

        return self.batch_format(torch.load(Path(self.processed_dir) / fname))

    def batch_format(self, x: Data) -> Data:
        # x.coords = x.x
        # Set this to ensure proper batching behaviour
        x.x = torch.zeros(x.coords.shape[0])  # type: ignore
        x.amino_acid_one_hot = amino_acid_one_hot(x)
        return x
