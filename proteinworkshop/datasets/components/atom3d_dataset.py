# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import torch
from beartype import beartype as typechecker
from graphein.protein.tensor.data import Protein
from graphein.protein.tensor.io import protein_to_pyg
from loguru import logger as log
from torch_geometric.data import Data

_atom_types_dict: Dict[str, int] = {
    "H": 0,
    "C": 1,
    "N": 2,
    "O": 3,
    "F": 4,
    "S": 5,
    "Cl": 6,
    "CL": 6,
    "P": 7,
}

NUM_ATOM_TYPES = len(_atom_types_dict)


@typechecker
def _element_mapping(x: str) -> int:
    return _atom_types_dict.get(x, 8)


"""
@typechecked
def _edge_features(
    coords: TensorType["num_nodes", 3],
    edge_index: TensorType[2, "num_edges"],
    D_max: float = 4.5,
    num_rbf: int = 16,
    device: Union[torch.device, str] = "cpu",
) -> Tuple[
    TensorType["num_edges", "num_edge_scalar_features"],
    TensorType["num_edges", "num_edge_vector_features", 3],
]:
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_max=D_max, D_count=num_rbf, device=device)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v
"""
"""
@typechecked
def _node_features(
    df: pd.DataFrame,
    coords: TensorType["num_nodes", 3],
    device: Union[torch.device, str] = "cpu",
) -> Tuple[
    TensorType["num_nodes"], TensorType["num_nodes", "num_node_vector_features", 3]
]:
    atoms = torch.as_tensor(
        list(map(_element_mapping, df.element)), dtype=torch.long, device=device
    )
    orientations = _orientations(coords)

    node_s = atoms
    node_v = orientations

    return node_s, node_v
"""

biopandas_mapping: Dict[str, str] = {
    "model": "model_id",
    "residue": "residue_number",
    "resname": "residue_name",
    "insertion_code": "insertion",
    "bfactor": "b_factor",
    "x": "x_coord",
    "y": "y_coord",
    "z": "z_coord",
    "element": "element_symbol",
    "name": "atom_name",
    "chain": "chain_id",
}


class BaseTransform:
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self):
        pass

    def __call__(self, df: pd.DataFrame) -> Union[Data, Protein]:
        with torch.no_grad():
            df = df.rename(columns=biopandas_mapping)
            # Rename water molecules
            df.residue_name = df.residue_name.str.replace("WAT", "HOH")
            # Assign HETATMs
            df["record_name"] = "ATOM"
            df.loc[df.residue_name == "HOH"].record_name = "HETATM"

            # Assign atom numbers
            df["atom_number"] = np.arange(1, len(df) + 1)
            # protein = Protein().from_dataframe(df)
            protein = protein_to_pyg(df=df)

            # Set this for correct batching. We overwrite it
            # with true features in featurisation.
            protein.x = torch.zeros(protein.coords.shape[0])
            return protein


########################################################################


class LBATransform(BaseTransform):
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __call__(self, elem: Any, index: int = -1):
        pocket, ligand = elem["atoms_pocket"], elem["atoms_ligand"]
        df = pd.concat([pocket, ligand], ignore_index=True)

        data = super().__call__(df)
        with torch.no_grad():
            data.graph_y = elem["scores"]["neglog_aff"]
            lig_flag = torch.zeros(
                df.shape[0], device=self.device, dtype=torch.bool
            )
            lig_flag[-len(ligand) :] = 1
            data.lig_flag = lig_flag
        return data


class PSRTransform(BaseTransform):
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __call__(self, elem: Any, index: int = -1):
        try:
            df = elem["atoms"]
            df = df[df.element != "H"].reset_index(drop=True)
            data = super().__call__(df)
            data.graph_y = torch.tensor(
                elem["scores"]["gdt_ts"]
            )  # .unsqueeze(0)
            data.id = eval(elem["id"])[0]
        except Exception as e:
            log.error(f"Failed to process {eval(elem['id'])[0]}")
            raise e
        return data


class MSPTransform(BaseTransform):
    """
    Transforms dict-style entries from the ATOM3D MSP dataset
    to featurized graphs. Returns a tuple (original, mutated) of
    `torch_geometric.data.Data` graphs with the (same) attribute
    `label` which is equal to 1. if the mutation stabilizes the
    complex and 0. otherwise, and all structural attributes as
    described in BaseTransform.

    The transform combines the atomic coordinates of the two proteins
    in each complex and treats them as a single structure/graph.

    From https://github.com/drorlab/gvp-pytorch.

    Excludes hydrogen atoms.
    """

    def __call__(self, elem):
        mutation = elem["id"].split("_")[-1]
        orig_df = elem["original_atoms"].reset_index(drop=True)
        mut_df = elem["mutated_atoms"].reset_index(drop=True)
        with torch.no_grad():
            original, mutated = self._transform(
                orig_df, mutation
            ), self._transform(mut_df, mutation)
        original.label = mutated.label = 1.0 if elem["label"] == "1" else 0.0
        return original, mutated

    def _transform(self, df, mutation):
        df = df[df.element != "H"].reset_index(drop=True)
        data = super().__call__(df)
        data.node_mask = self._extract_node_mask(df, mutation)
        return data

    def _extract_node_mask(self, df, mutation):
        chain, res = mutation[1], int(mutation[2:-1])
        idx = df.index[
            (df.chain.values == chain) & (df.residue.values == res)
        ].values
        mask = torch.zeros(len(df), dtype=torch.long)
        mask[idx] = 1
        return mask
