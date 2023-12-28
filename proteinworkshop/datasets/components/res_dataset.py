import math
import random

import numpy as np
import torch
from atom3d.datasets import LMDBDataset
from torch.utils.data import IterableDataset

from proteinworkshop.datasets.components.atom3d_dataset import BaseTransform

_amino_acids = lambda x: {  # noqa: E731
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLU": 5,
    "GLN": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
}.get(x, 20)


class RESDataset(IterableDataset):
    """
    A `torch.utils.data.IterableDataset` wrapper around a
    ATOM3D RES dataset.

    On each iteration, returns a `torch_geometric.data.Data`
    graph with the attribute `label` encoding the masked residue
    identity, `ca_idx` for the node index of the alpha carbon,
    and all structural attributes as described in BaseTransform.

    Adapted from:
    https://github.com/drorlab/gvp-pytorch/blob/82af6b22eaf8311c15733117b0071408d24ed877/gvp/atom3d.py

    Excludes hydrogen atoms.

    :param lmdb_dataset: path to ATOM3D dataset
    :param split_path: path to the ATOM3D split file
    """

    def __init__(self, lmdb_dataset, split_path):
        self.dataset = LMDBDataset(lmdb_dataset)
        self.idx = list(map(int, open(split_path).read().split()))
        self.transform = BaseTransform()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            gen = self._dataset_generator(
                list(range(len(self.idx))), shuffle=True
            )
        else:
            per_worker = int(
                math.ceil(len(self.idx) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.idx))
            gen = self._dataset_generator(
                list(range(len(self.idx)))[iter_start:iter_end], shuffle=True
            )
        return gen

    def _dataset_generator(self, indices, shuffle=True):
        if shuffle:
            random.shuffle(indices)
        with torch.no_grad():
            for idx in indices:
                data = self.dataset[self.idx[idx]]
                atoms = data["atoms"]
                for sub in data["labels"].itertuples():
                    _, num, aa = sub.subunit.split("_")
                    num, aa = int(num), _amino_acids(aa)
                    if aa == 20:
                        continue
                    my_atoms = atoms.iloc[
                        data["subunit_indices"][sub.Index]
                    ].reset_index(drop=True)
                    ca_idx = np.where(
                        (my_atoms.residue == num) & (my_atoms.name == "CA")
                    )[0]
                    if len(ca_idx) != 1:
                        continue

                    with torch.no_grad():
                        graph = self.transform(my_atoms)
                        graph.label = aa
                        graph.ca_idx = int(ca_idx)
                        yield graph
