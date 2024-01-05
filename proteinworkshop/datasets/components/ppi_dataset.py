import math
import random
from typing import Generator, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
import torch
from atom3d.datasets import LMDBDataset
from beartype import beartype as typechecker
from torch.utils.data import IterableDataset
from torch_geometric.data import Data

from proteinworkshop.datasets.components.atom3d_dataset import BaseTransform

PPI_DF_INDEX_COLUMNS = [
    "ensemble",
    "subunit",
    "structure",
    "model",
    "chain",
    "residue",
]


@typechecker
def get_res(df: pd.DataFrame) -> pd.DataFrame:
    """Get all residues."""
    # Adapted from: https://github.com/drorlab/atom3d/blob/master/examples/ppi/dataset/neighbors.py
    return df[PPI_DF_INDEX_COLUMNS].drop_duplicates()


@typechecker
def _get_idx_to_res_mapping(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Define mapping from residue index to single id number."""
    # Adapted from: https://github.com/drorlab/atom3d/blob/master/examples/ppi/dataset/neighbors.py
    idx_to_res = get_res(df).reset_index(drop=True)
    res_to_idx = idx_to_res.reset_index().set_index(PPI_DF_INDEX_COLUMNS)[
        "index"
    ]
    return idx_to_res, res_to_idx


@typechecker
def get_subunits(
    ensemble: pd.DataFrame,
) -> Tuple[
    Tuple[str, str, Optional[str], Optional[str]],
    Tuple[
        pd.DataFrame,
        pd.DataFrame,
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
    ],
]:
    """Get protein subunits from an ensemble."""
    # Adapted from: https://github.com/drorlab/atom3d/blob/master/examples/ppi/dataset/neighbors.py
    subunits = ensemble["subunit"].unique()

    if len(subunits) == 4:
        lb = [x for x in subunits if x.endswith("ligand_bound")][0]
        lu = [x for x in subunits if x.endswith("ligand_unbound")][0]
        rb = [x for x in subunits if x.endswith("receptor_bound")][0]
        ru = [x for x in subunits if x.endswith("receptor_unbound")][0]
        bdf0 = ensemble[ensemble["subunit"] == lb]
        bdf1 = ensemble[ensemble["subunit"] == rb]
        udf0 = ensemble[ensemble["subunit"] == lu]
        udf1 = ensemble[ensemble["subunit"] == ru]
        names = (lb, rb, lu, ru)
    elif len(subunits) == 2:
        udf0, udf1 = None, None
        bdf0 = ensemble[ensemble["subunit"] == subunits[0]]
        bdf1 = ensemble[ensemble["subunit"] == subunits[1]]
        names = (subunits[0], subunits[1], None, None)
    else:
        raise RuntimeError("Incorrect number of subunits for pair")
    return names, (bdf0, bdf1, udf0, udf1)


@typechecker
def get_negatives(
    neighbors, df0: pd.DataFrame, df1: pd.DataFrame
) -> pd.DataFrame:
    """Get negative pairs, given positives."""
    # Adapated from: https://github.com/drorlab/atom3d/blob/master/examples/ppi/dataset/neighbors.py
    idx_to_res0, res_to_idx0 = _get_idx_to_res_mapping(df0)
    idx_to_res1, res_to_idx1 = _get_idx_to_res_mapping(df1)
    all_pairs = np.zeros((len(idx_to_res0.index), len(idx_to_res1.index)))
    for i, neighbor in neighbors.iterrows():
        res0 = tuple(
            neighbor[
                [
                    "ensemble0",
                    "subunit0",
                    "structure0",
                    "model0",
                    "chain0",
                    "residue0",
                ]
            ]
        )
        res1 = tuple(
            neighbor[
                [
                    "ensemble1",
                    "subunit1",
                    "structure1",
                    "model1",
                    "chain1",
                    "residue1",
                ]
            ]
        )
        idx0 = res_to_idx0[res0]
        idx1 = res_to_idx1[res1]
        all_pairs[idx0, idx1] = 1
    pairs = np.array(np.where(all_pairs == 0)).T
    res0 = idx_to_res0.iloc[pairs[:, 0]][PPI_DF_INDEX_COLUMNS]
    res1 = idx_to_res1.iloc[pairs[:, 1]][PPI_DF_INDEX_COLUMNS]
    res0 = res0.reset_index(drop=True).add_suffix("0")
    res1 = res1.reset_index(drop=True).add_suffix("1")
    return pd.concat((res0, res1), axis=1)


class PPIDataset(IterableDataset):
    """
    A `torch.utils.data.IterableDataset` wrapper around an
    ATOM3D PPI dataset. Extracts (many) individual amino acid pairs
    from each structure of two interacting proteins. The returned graphs
    are separate and each represents a 30 angstrom radius from the
    selected residue's alpha carbon.

    On each iteration, returns a pair of `torch_geometric.data.Data`
    graphs with the (same) attribute `label` which is 1 if the two
    amino acids interact and 0 otherwise, `ca_idx` for the node index
    of the alpha carbon, and all structural attributes as
    described in BaseTransform.

    Adapted from:
    https://github.com/drorlab/gvp-pytorch/blob/82af6b22eaf8311c15733117b0071408d24ed877/gvp/atom3d.py

    Excludes hydrogen atoms.

    :param lmdb_dataset_path: path to ATOM3D dataset
    """

    def __init__(self, lmdb_dataset_path: str, dataset_type: str):
        self.dataset = LMDBDataset(lmdb_dataset_path)
        self.dataset_type = dataset_type
        self.transform = BaseTransform()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            gen = self._dataset_generator(
                list(range(len(self.dataset))), shuffle=True
            )
        else:
            per_worker = int(
                math.ceil(len(self.dataset) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.dataset))
            gen = self._dataset_generator(
                list(range(len(self.dataset)))[iter_start:iter_end],
                shuffle=True,
            )
        return gen

    @typechecker
    def _df_to_graph(
        self, struct_df: pd.DataFrame, chain_res: Iterable, label: float
    ) -> Optional[Data]:
        struct_df = struct_df[struct_df.element != "H"].reset_index(drop=True)

        chain, resnum = chain_res
        res_df = struct_df[
            (struct_df.chain == chain) & (struct_df.residue == resnum)
        ]
        if "CA" not in res_df.name.tolist():
            return None
        ca_pos = (
            res_df[res_df["name"] == "CA"][["x", "y", "z"]]
            .astype(np.float32)
            .to_numpy()[0]
        )

        kd_tree = scipy.spatial.KDTree(struct_df[["x", "y", "z"]].to_numpy())
        graph_pt_idx = kd_tree.query_ball_point(ca_pos, r=30.0, p=2.0)
        graph_df = struct_df.iloc[graph_pt_idx].reset_index(drop=True)

        ca_idx = np.where(
            (graph_df.chain == chain)
            & (graph_df.residue == resnum)
            & (graph_df.name == "CA")
        )[0]
        if len(ca_idx) != 1:
            return None

        data = self.transform(graph_df)
        data.label = label

        data.ca_idx = int(ca_idx)
        data.n_nodes = data.num_nodes

        return data

    @typechecker
    def _dataset_generator(
        self, indices: List[int], shuffle: bool = True
    ) -> Generator[Tuple[Data, Data], None, None]:
        if shuffle:
            random.shuffle(indices)
        with torch.no_grad():
            for idx in indices:
                data = self.dataset[idx]

                neighbors = data["atoms_neighbors"]
                pairs = data["atoms_pairs"]

                for ensemble_name, target_df in pairs.groupby(["ensemble"]):
                    sub_names, (
                        bound1,
                        bound2,
                        unbound1,
                        unbound2,
                    ) = get_subunits(target_df)
                    chain1 = unbound1 if "DB5" in self.dataset_type else bound1
                    chain2 = unbound2 if "DB5" in self.dataset_type else bound2
                    positives = neighbors[neighbors.ensemble0 == ensemble_name]
                    negatives = get_negatives(positives, chain1, chain2)
                    negatives["label"] = 0
                    labels = self._create_labels(
                        positives, negatives, num_pos=10, neg_pos_ratio=1
                    )

                    for _, row in labels.iterrows():
                        label = float(row["label"])
                        chain_res1 = row[["chain0", "residue0"]].values
                        chain_res2 = row[["chain1", "residue1"]].values
                        graph1 = self._df_to_graph(chain1, chain_res1, label)
                        graph2 = self._df_to_graph(chain2, chain_res2, label)
                        if (graph1 is None) or (graph2 is None):
                            continue
                        yield graph1, graph2

    @typechecker
    def _create_labels(
        self,
        positives: pd.DataFrame,
        negatives: pd.DataFrame,
        num_pos: int,
        neg_pos_ratio: int,
    ) -> pd.DataFrame:
        frac = min(1, num_pos / positives.shape[0])
        positives = positives.sample(frac=frac)
        n = positives.shape[0] * neg_pos_ratio
        n = min(negatives.shape[0], n)
        negatives = negatives.sample(n, random_state=0, axis=0)
        return pd.concat([positives, negatives])[
            ["chain0", "residue0", "chain1", "residue1", "label"]
        ]
