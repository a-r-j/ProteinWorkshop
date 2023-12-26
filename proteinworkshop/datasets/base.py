"""Base classes for protein structure datamodules and datasets."""
import os
import pathlib
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import lightning as L
import numpy as np
import pandas as pd
import torch
from beartype import beartype as typechecker
from graphein import verbose
from graphein.protein.tensor.dataloader import ProteinDataLoader
from graphein.protein.tensor.io import protein_to_pyg
from graphein.protein.utils import (
    download_pdb_multiprocessing,
    get_obsolete_mapping,
)
from loguru import logger
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric import transforms as T
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from proteinworkshop.features.sequence_features import amino_acid_one_hot

verbose(False)


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
    """Base class for Protein datamodules.

    .. seealso::
        L.LightningDataModule
    """

    prepare_data_per_node = (
        True  # class default for lighting 2.0 compatability
    )

    @abstractmethod
    def download(self):
        """
        Implement downloading of raw data.

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
        """Returns a mapping of obsolete PDB codes to their updated replacement.

        :return: Mapping of obsolete PDB codes to their updated replacements.
        :rtype: Dict[str, str]
        """
        return get_obsolete_mapping()

    @typechecker
    def compose_transforms(self, transforms: Iterable[Callable]) -> T.Compose:
        """Compose an iterable of Transforms into a single transform.

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
        Implement the parsing of the raw dataset to a dataframe.

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
        """Return a list of PDBs/IDs to exclude from the dataset."""
        ...

    @abstractmethod
    def train_dataset(self) -> Dataset:
        """
        Implement the construction of the training dataset.

        :return: The training dataset.
        :rtype: Dataset
        """
        ...

    @abstractmethod
    def val_dataset(self) -> Dataset:
        """
        Implement the construction of the validation dataset.

        :return: The validation dataset.
        :rtype: Dataset
        """
        ...

    @abstractmethod
    def test_dataset(self) -> Dataset:
        """
        Implement the construction of the test dataset.

        :return: The test dataset.
        :rtype: Dataset
        """
        ...

    @abstractmethod
    def train_dataloader(self) -> ProteinDataLoader:
        """
        Implement the construction of the training dataloader.

        :return: The training dataloader.
        :rtype: ProteinDataLoader
        """
        ...

    @abstractmethod
    def val_dataloader(self) -> ProteinDataLoader:
        """Implement the construction of the validation dataloader.

        :return: The validation dataloader.
        :rtype: ProteinDataLoader
        """
        ...

    @abstractmethod
    def test_dataloader(self) -> ProteinDataLoader:
        """Implement the construction of the test dataloader.

        :return: The test dataloader.
        :rtype: ProteinDataLoader
        """
        ...

    def get_class_weights(self) -> torch.Tensor:
        """Return tensor of class weights."""
        labels: Dict[str, torch.Tensor] = self.parse_labels()
        labels = list(labels.values())  # type: ignore
        labels = np.array(labels)  # type: ignore
        weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(labels), y=labels
        )
        return torch.tensor(weights)


class ProteinDataset(Dataset):
    """Dataset for loading protein structures.

    :param pdb_codes: List of PDB codes to load. This can also be a list
        of identifiers to specific to your filenames if you have
        pre-downloaded structures.
    :type pdb_codes: List[str]
    :param root: Path to root directory, defaults to ``None``.
    :type root: Optional[str], optional
    :param pdb_dir: Path to directory containing raw PDB files,
        defaults to ``None``.
    :type pdb_dir: Optional[str], optional
    :param processed_dir: Directory to store processed data, defaults to
        ``None``.
    :type processed_dir: Optional[str], optional
    :param pdb_paths: If specified, the dataset will load structures from
        these paths instead of downloading them from the RCSB PDB or using
        the identifies in ``pdb_codes``. This is useful if you have already
        downloaded structures and want to use them. defaults to ``None``
    :type pdb_paths: Optional[List[str]], optional
    :param chains: List of chains to load for each PDB code,
        defaults to ``None``.
    :type chains: Optional[List[str]], optional
    :param graph_labels: List of tensors to set as graph labels for each
        examples. If not specified, no graph labels will be set.
        defaults to ``None``.
    :type graph_labels: Optional[List[torch.Tensor]], optional
    :param node_labels: List of tensors to set as node labels for each
        examples. If not specified, no node labels will be set.
        defaults to ``None``.
    :type node_labels: Optional[List[torch.Tensor]], optional
    :param transform: List of transforms to apply to each example,
        defaults to ``None``.
    :type transform: Optional[List[Callable]], optional
    :param pre_transform: Transform to apply to each example before
        processing, defaults to ``None``.
    :type pre_transform: Optional[Callable], optional
    :param pre_filter: Filter to apply to each example before processing,
        defaults to ``None``.
    :type pre_filter: Optional[Callable], optional
    :param log: Whether to log. If ``True``, logs will be printed to
        stdout, defaults to ``True``.
    :type log: bool, optional
    :param overwrite: Whether to overwrite existing files, defaults to
        ``False``.
    :type overwrite: bool, optional
    :param format: Format to save structures in, defaults to "pdb".
    :type format: Literal[mmtf, pdb, ent], optional
    :param in_memory: Whether to load data into memory, defaults to False.
    :type in_memory: bool, optional
    :param store_het: Whether to store heteroatoms in the graph,
        defaults to ``False``.
    :type store_het: bool, optional
    """

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
        format: Literal["mmtf", "pdb", "ent"] = "pdb",
        in_memory: bool = False,
        store_het: bool = False,
        out_names: Optional[List[str]] = None,
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
        self.out_names = out_names

        self._processed_files = []

        # Determine whether to download raw structures
        if not self.overwrite and all(
            os.path.exists(Path(self.root) / "processed" / p)
            for p in self.processed_file_names
        ):
            logger.info(
                "All structures already processed and overwrite=False. Skipping download."
            )
            self._skip_download = True
        else:
            self._skip_download = False

        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.structures = pdb_codes if pdb_codes is not None else pdb_paths
        if self.in_memory:
            logger.info("Reading data into memory")
            self.data = [
                torch.load(pathlib.Path(self.root) / "processed" / f)
                for f in tqdm(self.processed_file_names)
            ]

    def download(self):
        """
        Download structure files not present in the raw directory (``raw_dir``).

        Structures are downloaded from the RCSB PDB using the Graphein
        multiprocessed downloader.

        Structure files are downloaded in ``self.format`` format (``mmtf`` or
        ``pdb``). Downloading files in ``mmtf`` format is strongly recommended
        as it will be both faster and smaller than ``pdb`` format.

        Downloaded files are stored in ``self.raw_dir``.
        """
        if self.format == "ent":  # Skip downloads from ASTRAL
            logger.warning(
                "Downloads in .ent format are assumed to be from ASTRAL. These data should have already been downloaded"
            )
            return
        if self._skip_download:
            logger.info(
                "All structures already processed and overwrite=False. Skipping download."
            )
            return
        if self.pdb_codes is not None:
            to_download = (
                self.pdb_codes
                if self.overwrite
                else [
                    pdb
                    for pdb in self.pdb_codes
                    if not (
                        os.path.exists(
                            Path(self.raw_dir) / f"{pdb}.{self.format}"
                        )
                        or os.path.exists(
                            Path(self.raw_dir) / f"{pdb}.{self.format}.gz"
                        )
                    )
                ]
            )
            to_download = list(set(to_download))
            logger.info(f"Downloading {len(to_download)} structures")
            file_format = (
                self.format[:-3]
                if self.format.endswith(".gz")
                else self.format
            )
            download_pdb_multiprocessing(
                to_download, self.raw_dir, format=file_format
            )

    def len(self) -> int:
        """Return length of the dataset."""
        return len(self.pdb_codes)

    @property
    def raw_dir(self) -> str:
        """Returns the path to the raw data directory.

        :return: Raw data directory.
        :rtype: str
        """
        return os.path.join(self.root, "raw") if self.pdb_dir is None else self.pdb_dir  # type: ignore

    @property
    def raw_file_names(self) -> List[str]:
        """Returns the raw file names.

        :return: List of raw file names.
        :rtype: List[str]
        """
        if self._skip_download:
            return []
        if self.pdb_paths is None:
            return [f"{pdb}.{format}" for pdb in self.pdb_codes]
        else:
            return list(self.pdb_paths)

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """Returns the processed file names.

        This will either be a list in format [``{pdb_code}.pt``] or
        a list of [{pdb_code}_{chain(s)}.pt].

        :return: List of processed file names.
        :rtype: Union[str, List[str], Tuple]
        """
        if self._processed_files:
            return self._processed_files
        if self.overwrite:
            return ["this_forces_a_processing_cycle"]
        if self.out_names is not None:
            return [f"{name}.pt" for name in self.out_names]
        if self.chains is not None:
            return [
                f"{pdb}_{chain}.pt"
                for pdb, chain in zip(self.pdb_codes, self.chains)
            ]
        else:
            return [f"{pdb}.pt" for pdb in self.pdb_codes]

    def process(self):
        """Process raw data into PyTorch Geometric Data objects with Graphein.

        Processed data are stored in ``self.processed_dir`` as ``.pt`` files.
        """
        if not self.overwrite:
            if self.chains is not None:
                index_pdb_tuples = [
                    (i, pdb)
                    for i, pdb in enumerate(self.pdb_codes)
                    if not os.path.exists(
                        Path(self.processed_dir) / f"{pdb}_{self.chains[i]}.pt"
                    )
                ]
            else:
                index_pdb_tuples = [
                    (i, pdb)
                    for i, pdb in enumerate(self.pdb_codes)
                    if not os.path.exists(
                        Path(self.processed_dir) / f"{pdb}.pt"
                    )
                ]
            logger.info(
                f"Processing {len(index_pdb_tuples)} unprocessed structures"
            )
        else:
            index_pdb_tuples = [
                (i, pdb) for i, pdb in enumerate(self.pdb_codes)
            ]

        raw_dir = Path(self.raw_dir)
        for index_pdb_tuple in tqdm(index_pdb_tuples):
            try:
                (
                    i,
                    pdb,
                ) = index_pdb_tuple  # NOTE: here, we unpack the tuple to get each PDB's original index in `self.pdb_codes`
                path = raw_dir / f"{pdb}.{self.format}"
                if path.exists():
                    path = str(path)
                elif path.with_suffix("." + self.format + ".gz").exists():
                    path = str(path.with_suffix("." + self.format + ".gz"))
                else:
                    raise FileNotFoundError(
                        f"{pdb} not found in raw directory. Are you sure it's downloaded and has the format {self.format}?"
                    )
                graph = protein_to_pyg(
                    path=path,
                    chain_selection=self.chains[i]
                    if self.chains is not None
                    else "all",
                    keep_insertions=True,
                    store_het=self.store_het,
                )
            except Exception as e:
                logger.error(f"Error processing {pdb} {self.chains[i]}: {e}")  # type: ignore
                raise e

            if self.out_names is not None:
                fname = self.out_names[i] + ".pt"
            else:
                fname = (
                    f"{pdb}.pt"
                    if self.chains is None
                    else f"{pdb}_{self.chains[i]}.pt"
                )

            graph.id = fname.split(".")[0]

            if self.graph_labels is not None:
                graph.graph_y = self.graph_labels[i]  # type: ignore

            if self.node_labels is not None:
                graph.node_y = self.node_labels[i]  # type: ignore

            torch.save(graph, Path(self.processed_dir) / fname)
            self._processed_files.append(fname)
        logger.info("Completed processing.")

    def get(self, idx: int) -> Data:
        """
        Return PyTorch Geometric Data object for a given index.

        :param idx: Index to retrieve.
        :type idx: int
        :return: PyTorch Geometric Data object.
        """
        if self.in_memory:
            return self._batch_format(self.data[idx])

        if self.out_names is not None:
            fname = f"{self.out_names[idx]}.pt"
        elif self.chains is not None:
            fname = f"{self.pdb_codes[idx]}_{self.chains[idx]}.pt"
        else:
            fname = f"{self.pdb_codes[idx]}.pt"

        return self._batch_format(torch.load(Path(self.processed_dir) / fname))

    def _batch_format(self, x: Data) -> Data:
        # Set this to ensure proper batching behaviour
        x.x = torch.zeros(x.coords.shape[0])  # type: ignore
        x.amino_acid_one_hot = amino_acid_one_hot(x)
        x.seq_pos = torch.arange(x.coords.shape[0]).unsqueeze(
            -1
        )  # Add sequence position
        return x
