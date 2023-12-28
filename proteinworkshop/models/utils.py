from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import Bool, Float, Int64, jaxtyped
from omegaconf import DictConfig
from torch_geometric.data import Batch
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from proteinworkshop.types import ActivationType, LossType


def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, d: Any) -> Any:
    return val if exists(val) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def get_output_dim(output: str, cfg: DictConfig) -> int:
    """Gets dimensionality of output layer for a given output type."""
    if output in {"graph_label", "node_label"}:
        return cfg.get("dataset.num_classes")
    elif output == "dihedrals":
        return 6
    elif output == "residue_type":
        return 23
    elif output == "sidechain_torsions":
        return 8
    elif output == "rotation_frame_quaternion":
        return 4
    elif output == "plddt":
        return 1
    else:
        raise ValueError(f"Unknown output type: {output}")


def get_aggregation(aggregation: str) -> Callable:
    """Maps aggregation name (str) to aggregation function."""
    if aggregation == "max":
        return global_max_pool
    elif aggregation == "mean":
        return global_mean_pool
    elif aggregation in {"sum", "add"}:
        return global_add_pool
    else:
        raise ValueError(f"Unknown aggregation function: {aggregation}")


def get_input_dim(
    features_config: DictConfig,
    feature_config_name: str,
    task_config: DictConfig,
    recurse_for_node_features: bool = False,
) -> int:
    feat_sizes: Dict[str, int] = {
        ### scalar node features ###
        "amino_acid_one_hot": 23,  # 20 + 3
        "kappa": 2,
        "alpha": 2,
        "dihedrals": 6,
        "positional_encoding": 16,
        "sequence_positional_encoding": 16,
        "sidechain_torsions": 8,
        "rotation_frame_quaternion": 4,
        "chi1": 2,
        # TODO rotation_frame_matrix
        # TODO fill-in remaining scalar node features
        ### vector node features ###
        "orientation": 2,
        # TODO fill-in remaining vector node features
        ### scalar edge features ###
        "edge_distance": 1,
        "edge_type": 1,
        "node_features": (
            # avoid infinite recursion when parsing for features besides scalar node features
            0
            if not recurse_for_node_features
            # note: `2 *` to account for both source and destination nodes' scalar features
            else 2
            * get_input_dim(
                features_config,
                "scalar_node_features",
                task_config,
                recurse_for_node_features=(not recurse_for_node_features),
            )
        ),
        "sequence_distance": 1,
        # TODO fill-in remaining scalar edge features
        ### vector edge features ###
        "edge_vectors": 1
        # TODO fill-in remaining vector edge features
    }

    sizes = [
        feat_sizes[feat]
        for feat in getattr(features_config, feature_config_name)
        if feat
        not in {
            "edge_euler_angle_orientations",
            "rotation_frame_matrix",
            "invariant_bb_frames",
            "gnm",
        }
    ]
    return sum(sizes)


def get_activations(
    act_name: ActivationType, return_functional: bool = False
) -> Union[nn.Module, Callable]:
    """Maps activation name (str) to activation function module."""
    if act_name == "relu":
        return F.relu if return_functional else nn.ReLU()
    elif act_name == "elu":
        return F.elu if return_functional else nn.ELU()
    elif act_name == "leaky_relu":
        return F.leaky_relu if return_functional else nn.LeakyReLU()
    elif act_name == "tanh":
        return F.tanh if return_functional else nn.Tanh()
    elif act_name == "sigmoid":
        return F.sigmoid if return_functional else nn.Sigmoid()
    elif act_name == "none":
        return nn.Identity()
    elif act_name in {"silu", "swish"}:
        return F.silu if return_functional else nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {act_name}")


def has_nan(x: torch.Tensor) -> bool:
    return x.isnan().any()  # type: ignore


def get_loss(
    name: LossType,
    smoothing: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,
) -> Callable:
    """Return the loss function based on the name."""
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(
            label_smoothing=smoothing, weight=class_weights
        )
    if name == "bce":
        return nn.BCEWithLogitsLoss(weight=class_weights)
    elif name == "nll_loss":
        return F.nll_loss
    elif name == "mse_loss":
        return F.mse_loss
    elif name == "l1_loss":
        return F.l1_loss
    elif name == "dihedral_loss":
        raise NotImplementedError("Dihedral loss not implemented yet")
    else:
        raise ValueError(f"Incorrect Loss provided: {name}")


def flatten_list(l: List[List]) -> List:  # noqa: E741
    return [item for sublist in l for item in sublist]


@jaxtyped(typechecker=typechecker)
def centralize(
    batch: Union[Batch, ProteinBatch],
    key: str,
    batch_index: torch.Tensor,
    node_mask: Optional[Bool[torch.Tensor, " n_nodes"]] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor
]:  # note: cannot make assumptions on output shape
    if node_mask is not None:
        # derive centroid of each batch element
        entities_centroid = torch_scatter.scatter(
            batch[key][node_mask], batch_index[node_mask], dim=0, reduce="mean"
        )  # e.g., [batch_size, 3]

        # center entities using corresponding centroids
        entities_centered = batch[key] - (
            entities_centroid[batch_index] * node_mask.float().unsqueeze(-1)
        )
        masked_values = torch.ones_like(batch[key]) * torch.inf
        values = batch[key][node_mask]
        masked_values[node_mask] = (
            values - entities_centroid[batch_index][node_mask]
        )
        entities_centered = masked_values

    else:
        # derive centroid of each batch element, and center entities using corresponding centroids
        entities_centroid = torch_scatter.scatter(
            batch[key], batch_index, dim=0, reduce="mean"
        )  # e.g., [batch_size, 3]
        entities_centered = batch[key] - entities_centroid[batch_index]

    return entities_centroid, entities_centered


@jaxtyped(typechecker=typechecker)
def decentralize(
    batch: Union[Batch, ProteinBatch],
    key: str,
    batch_index: torch.Tensor,
    entities_centroid: torch.Tensor,
    node_mask: Optional[Bool[torch.Tensor, " n_nodes"]] = None,
) -> torch.Tensor:  # note: cannot make assumptions on output shape
    if node_mask is not None:
        masked_values = torch.ones_like(batch[key]) * torch.inf
        masked_values[node_mask] = (
            batch[key][node_mask] + entities_centroid[batch_index]
        )
        entities_centered = masked_values
    else:
        entities_centered = batch[key] + entities_centroid[batch_index]
    return entities_centered


@jaxtyped(typechecker=typechecker)
def localize(
    pos: Float[torch.Tensor, "batch_num_nodes 3"],
    edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
    norm_pos_diff: bool = True,
    node_mask: Optional[Bool[torch.Tensor, " n_nodes"]] = None,
) -> Float[torch.Tensor, "batch_num_edges 3 3"]:
    row, col = edge_index[0], edge_index[1]

    if node_mask is not None:
        edge_mask = node_mask[row] & node_mask[col]

        pos_diff = (
            torch.ones((edge_index.shape[1], 3), device=edge_index.device)
            * torch.inf
        )
        pos_diff[edge_mask] = pos[row][edge_mask] - pos[col][edge_mask]

        pos_cross = (
            torch.ones((edge_index.shape[1], 3), device=edge_index.device)
            * torch.inf
        )
        pos_cross[edge_mask] = torch.cross(
            pos[row][edge_mask], pos[col][edge_mask]
        )
    else:
        pos_diff = pos[row] - pos[col]
        pos_cross = torch.cross(pos[row], pos[col])

    if norm_pos_diff:
        # derive and apply normalization factor for `pos_diff`
        if node_mask is not None:
            norm = torch.ones((edge_index.shape[1], 1), device=pos_diff.device)
            norm[edge_mask] = (
                torch.sqrt(
                    torch.sum((pos_diff[edge_mask] ** 2), dim=1).unsqueeze(1)
                )
            ) + 1
        else:
            norm = torch.sqrt(torch.sum(pos_diff**2, dim=1).unsqueeze(1)) + 1
        pos_diff = pos_diff / norm

        # derive and apply normalization factor for `pos_cross`
        if node_mask is not None:
            cross_norm = torch.ones(
                (edge_index.shape[1], 1), device=pos_cross.device
            )
            cross_norm[edge_mask] = (
                torch.sqrt(
                    torch.sum((pos_cross[edge_mask]) ** 2, dim=1).unsqueeze(1)
                )
            ) + 1
        else:
            cross_norm = (
                torch.sqrt(torch.sum(pos_cross**2, dim=1).unsqueeze(1))
            ) + 1
        pos_cross = pos_cross / cross_norm

    if node_mask is not None:
        pos_vertical = (
            torch.ones((edge_index.shape[1], 3), device=edge_index.device)
            * torch.inf
        )
        pos_vertical[edge_mask] = torch.cross(
            pos_diff[edge_mask], pos_cross[edge_mask]
        )
    else:
        pos_vertical = torch.cross(pos_diff, pos_cross)

    f_ij = torch.cat(
        (
            pos_diff.unsqueeze(1),
            pos_cross.unsqueeze(1),
            pos_vertical.unsqueeze(1),
        ),
        dim=1,
    )
    return f_ij


@jaxtyped(typechecker=typechecker)
def safe_norm(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
    keepdim: bool = False,
    sqrt: bool = True,
) -> torch.Tensor:
    norm = torch.sum(x**2, dim=dim, keepdim=keepdim)
    if sqrt:
        norm = torch.sqrt(norm + eps)
    return norm + eps


@jaxtyped(typechecker=typechecker)
def is_identity(
    nonlinearity: Optional[Union[Callable, nn.Module]] = None
) -> bool:
    return nonlinearity is None or isinstance(nonlinearity, nn.Identity)
