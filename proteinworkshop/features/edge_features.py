"""Utilities for computing edge features."""
from typing import List, Union

import numpy as np
import torch
from beartype import beartype as typechecker
from graphein.protein.tensor.types import CoordTensor, EdgeTensor
from jaxtyping import jaxtyped
from omegaconf import ListConfig
from torch_geometric.data import Batch, Data

from proteinworkshop.features.utils import _normalize

EDGE_FEATURES: List[str] = [
    "edge_distance",
    "node_features",
    "edge_type",
    "sequence_distance",
]
"""List of edge features that can be computed."""


@jaxtyped(typechecker=typechecker)
def compute_scalar_edge_features(
    x: Union[Data, Batch], features: Union[List[str], ListConfig]
) -> torch.Tensor:
    """
    Computes scalar edge features from a :class:`~torch_geometric.data.Data` or :class:`~torch_geometric.data.Batch` object.

    :param x: :class:`~torch_geometric.data.Data` or :class:`~torch_geometric.data.Batch` protein object.
    :type x: Union[Data, Batch]
    :param features: List of edge features to compute.
    :type features: Union[List[str], ListConfig]

    """
    feats = []
    for feature in features:
        if feature == "edge_distance":
            feats.append(compute_edge_distance(x.pos, x.edge_index))
        elif feature == "node_features":
            n1, n2 = x.x[x.edge_index[0]], x.x[x.edge_index[1]]
            feats.append(torch.cat([n1, n2], dim=1))
        elif feature == "edge_type":
            feats.append(x.edge_type.T)
        elif feature == "orientation":
            raise NotImplementedError
        elif feature == "sequence_distance":
            feats.append(x.edge_index[1] - x.edge_index[0])
        elif feature == "pos_emb":
            feats.append(pos_emb(x.edge_index))
        else:
            raise ValueError(f"Unknown edge feature {feature}")
    feats = [feat.unsqueeze(1) if feat.ndim == 1 else feat for feat in feats]
    return torch.cat(feats, dim=1)


@jaxtyped(typechecker=typechecker)
def compute_vector_edge_features(
    x: Union[Data, Batch], features: Union[List[str], ListConfig]
) -> Union[Data, Batch]:
    vector_edge_features = []
    for feature in features:
        if feature == "edge_vectors":
            E_vectors = x.pos[x.edge_index[0]] - x.pos[x.edge_index[1]]
            vector_edge_features.append(_normalize(E_vectors).unsqueeze(-2))
        else:
            raise ValueError(f"Vector feature {feature} not recognised.")
    x.edge_vector_attr = torch.cat(vector_edge_features, dim=0)
    return x


@jaxtyped(typechecker=typechecker)
def compute_edge_distance(
    pos: CoordTensor, edge_index: EdgeTensor
) -> torch.Tensor:
    """
    Compute the euclidean distance between each pair of nodes connected by an edge.

    :param pos: Tensor of shape :math:`(|V|, 3)` containing the node coordinates.
    :type pos: CoordTensor
    :param edge_index: Tensor of shape :math:`(2, |E|)` containing the indices of the nodes forming the edges.
    :type edge_index: EdgeTensor
    :return: Tensor of shape :math:`(|E|, 1)` containing the euclidean distance between each pair of nodes connected by an edge.
    :rtype: torch.Tensor
    """
    return torch.pairwise_distance(
        pos[edge_index[0, :]], pos[edge_index[1, :]]
    )


@jaxtyped(typechecker=typechecker)
def pos_emb(edge_index: EdgeTensor, num_pos_emb: int = 16):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(
            0, num_pos_emb, 2, dtype=torch.float32, device=edge_index.device
        )
        * -(np.log(10000.0) / num_pos_emb)
    )
    angles = d.unsqueeze(-1) * frequency
    return torch.cat((torch.cos(angles), torch.sin(angles)), -1)
