"""Node feature computation functions."""
from typing import List, Union

import torch
import torch.nn.functional as F
from beartype import beartype
from graphein.protein.tensor.angles import (
    alpha,
    dihedrals,
    kappa,
    sidechain_torsion,
)
from graphein.protein.tensor.data import Protein, ProteinBatch
from graphein.protein.tensor.types import AtomTensor, CoordTensor
from jaxtyping import jaxtyped
from omegaconf import ListConfig
from torch_geometric.data import Batch, Data
from torch_geometric.nn.pool import knn_graph
from torch_geometric.utils import softmax

from proteinworkshop.models.utils import flatten_list
from proteinworkshop.types import OrientationTensor, ScalarNodeFeature

from .sequence_features import amino_acid_one_hot
from .utils import _normalize


@jaxtyped
@beartype
def compute_scalar_node_features(
    x: Union[Batch, Data, Protein, ProteinBatch],
    node_features: Union[ListConfig, List[ScalarNodeFeature]],
) -> torch.Tensor:
    """
    Factory function for node features.

    .. seealso::
        :py:class:`proteinworkshop.types.ScalarNodeFeature` for a list of node
        features that can be computed.

    This function operates on a :py:class:`torch_geometric.data.Data` or
    :py:class:`torch_geometric.data.Batch` object and computes the requested
    node features.

    :param x: :py:class:`~torch_geometric.data.Data` or
        :py:class:`~torch_geometric.data.Batch` protein object.
    :type x: Union[Data, Batch]
    :param node_features: List of node features to compute.
    :type node_features: Union[List[str], ListConfig]
    :return: Tensor of node features of shape (``N x F``), where ``N`` is the
        number of nodes and ``F`` is the number of features.
    :rtype: torch.Tensor
    """
    feats = []
    for feature in node_features:
        if feature == "amino_acid_one_hot":
            feats.append(amino_acid_one_hot(x, num_classes=23))
        elif feature == "alpha":
            feats.append(alpha(x.coords, x.batch, rad=True, embed=True))
        elif feature == "kappa":
            feats.append(kappa(x.coords, x.batch, rad=True, embed=True))
        elif feature == "dihedrals":
            feats.append(dihedrals(x.coords, x.batch, rad=True, embed=True))
        elif feature == "sidechain_torsions":
            feats.append(
                sidechain_torsion(
                    x.coords,
                    res_types=flatten_list(x.residues),
                    rad=True,
                    embed=True,
                )
            )
        elif feature.startswith("surface"):
            # k = int(feature.split("_")[1])
            # sigmas = [float(s) for s in feature.split("_")[2:]]
            # feats.append(compute_surface_feat(x.coords, k=k, sigma=sigmas))
            raise NotImplementedError(
                "Surface features not implemented yet. Does not handle batched data."
            )
        elif feature == "sequence_positional_encoding":
            continue
        else:
            raise ValueError(f"Node feature {feature} not recognised.")
    feats = [feat.unsqueeze(1) if feat.ndim == 1 else feat for feat in feats]
    # Return concatenated features or original features if no features were computed
    return torch.cat(feats, dim=1) if feats else x.x


@jaxtyped
@beartype
def compute_vector_node_features(
    x: Union[Batch, Data, Protein, ProteinBatch],
    vector_features: Union[ListConfig, List[str]],
) -> Union[Batch, Data, Protein, ProteinBatch]:
    """Factory function for vector features.

    Currently implemented vector features are:

        - ``orientation``: Orientation of each node in the protein backbone
        - ``virtual_cb_vector``: Virtual CB vector for each node in the protein
        backbone


    """
    vector_node_features = []
    for feature in vector_features:
        if feature == "orientation":
            vector_node_features.append(orientations(x.coords))
        elif feature == "virtual_cb_vector":
            raise NotImplementedError("Virtual CB vector not implemented yet.")
        else:
            raise ValueError(f"Vector feature {feature} not recognised.")
    x.x_vector_attr = torch.cat(vector_node_features, dim=0)
    return x


@jaxtyped
@beartype
def compute_surface_feat(
    coords: Union[CoordTensor, AtomTensor], k: int, sigma: List[float]
):
    """Coords: (N, 3)
    k: number of neighbors to consider in KNN graph
    """
    if coords.ndim == 3:
        coords = coords[:, 1, :]
    feat = []
    for s in sigma:
        SIGMA = torch.tensor([s], device=coords.device).unsqueeze(1)
        N_DIM = coords.shape[-1]
        N_SIGMA = SIGMA.shape[0]

        edge_index = knn_graph(coords, k=k)
        node_in = coords[edge_index[0]]
        node_out = coords[edge_index[1]]

        dists = torch.pairwise_distance(node_in, node_out).unsqueeze(0)
        dists = -(dists**2) / SIGMA
        weights = softmax(dists, index=edge_index[1], dim=-1)
        weights = weights.reshape(-1, N_SIGMA, k)

        diff_vecs = node_in - node_out
        diff_vecs = diff_vecs.reshape(-1, k, N_DIM)

        mean_vec = torch.bmm(weights, diff_vecs)
        denom = torch.bmm(
            weights, torch.norm(diff_vecs, dim=-1).unsqueeze(-1)
        ).squeeze(-1)
        feat.append(torch.norm(mean_vec, dim=-1) / denom)
    return torch.cat(feat, dim=1)


@jaxtyped
@beartype
def orientations(
    X: Union[CoordTensor, AtomTensor], ca_idx: int = 1
) -> OrientationTensor:
    if X.ndim == 3:
        X = X[:, ca_idx, :]
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat((forward.unsqueeze(-2), backward.unsqueeze(-2)), dim=-2)
