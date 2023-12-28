"""Edge construction and featurisation utils."""
import functools
from typing import List, Literal, Optional, Tuple, Union

import graphein.protein.tensor.edges as gp
import torch
from beartype import beartype as typechecker
from graphein.protein.tensor.data import Protein, ProteinBatch
from omegaconf import ListConfig
from torch_geometric.data import Batch, Data


@typechecker
def compute_edges(
    x: Union[Data, Batch, Protein, ProteinBatch],
    edge_types: Union[ListConfig, List[str]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Orchestrates the computation of edges for a given data object.

    This function returns a tuple of tensors, where the first tensor is a
    tensor indicating the edge type of shape (``|E|``) and the second are the
    edge indices of shape (``2 x |E|``).

    The edge type tensor can be used to mask out edges of a particular type
    downstream.

    .. warning::

        For spatial edges, (e.g. ``knn_``, ``eps_``), the input data/batch
        object must have a ``pos`` attribute of shape (``N x 3``).

    :param x: The input data object to compute edges for
    :type x: Union[Data, Batch, Protein, ProteinBatch]
    :param edge_types: List of edge types to compute. Must be a sequence of
        ``knn_{x}``, ``eps_{x}``, (where ``{x}`` should be replaced by a
        numerical value) ``seq_forward``, ``seq_backward``.
    :type edge_types: Union[ListConfig, List[str]]
    :raises ValueError: Raised if ``x`` is not a ``torch_geometric`` Data or
        Batch object
    :raises NotImplementedError: Raised if an edge type is not implemented
    :return: Tuple of tensors, where the first tensor is a tensor indicating
        the edge type of shape (``|E|``) and the second are the edge indices of
        shape (``2 x |E|``).
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    # Handle batch
    if isinstance(x, Batch):
        edge_fn = functools.partial(gp.compute_edges, batch=x.batch)
    elif isinstance(x, Data):
        edge_fn = gp.compute_edges
    else:
        raise ValueError("x must be a torch_geometric Data or Batch object")

    # Iterate over edge types
    edges = []
    for edge_type in edge_types:
        if edge_type.startswith("knn") or edge_type.startswith("eps"):
            edges.append(edge_fn(x.pos, edge_type))
        elif edge_type == "seq_forward":
            edges.append(
                sequence_edges(x, chains=x.chains, direction="forward")
            )
        elif edge_type == "seq_backward":
            edges.append(
                sequence_edges(x, chains=x.chains, direction="backward")
            )
        else:
            raise NotImplementedError(f"Edge type {edge_type} not implemented")

    # Compute edge types
    indxs = torch.cat(
        [
            torch.ones_like(e_idx[0, :]) * idx
            for idx, e_idx in enumerate(edges)
        ],
        dim=0,
    ).unsqueeze(0)
    edges = torch.cat(edges, dim=1)

    return edges, indxs


@typechecker
def sequence_edges(
    b: Union[Data, Batch, Protein, ProteinBatch],
    chains: Optional[torch.Tensor] = None,
    direction: Literal["forward", "backward"] = "forward",
):
    """Computes edges between adjacent residues in a sequence.

    :param b: Input data object to compute edges for
    :type b: Union[Data, Batch, Protein, ProteinBatch]
    :param chains: Tensor of shape (``N``) indicating the chain ID of each node.
        This is required for correct boundary handling. Defaults to ``None``
    :type chains: Optional[torch.Tensor], optional
    :param direction: Direction of edges to compute. Must be ``forward`` or ``backward``. Defaults to ``forward``
    :type direction: Literal["forward", "backward"], optional
    :raises ValueError: Raised if ``direction`` is not ``forward`` or ``backward``
    :return: Tensor of shape (``2 x |E|``) indicating the edge indices
    """
    if isinstance(b, Batch):
        idx_a = torch.arange(0, b.ptr[-1] - 1, device=b.ptr.device)
        idx_b = torch.arange(1, b.ptr[-1], device=b.ptr.device)
    elif isinstance(b, Data):
        idx_a = torch.arange(0, b.coords.shape[0] - 1, device=b.coords.device)
        idx_a = torch.arange(1, b.coords.shape[0] - 1, device=b.coords.device)
    # Concatenate indices to create edge list
    if direction == "forward":
        e_index = torch.stack([idx_a, idx_b], dim=0)
    elif direction == "backward":
        e_index = torch.stack([idx_b, idx_a], dim=0)
    else:
        raise ValueError(
            f"Unknown direction: {direction}. Must be 'forward' or 'backward'"
        )
    # Remove edges that cross batch boundaries
    if isinstance(b, Batch):
        mask = torch.ones_like(idx_a, device=b.coords.device).bool()
        mask[b.ptr[1:-2]] = 0
        e_index = e_index[:, mask]
    if chains is None and isinstance(b, Batch):
        chains = b.chains
    if chains is not None:
        # Remove edges between chains
        e_mask = chains[e_index]
        e_mask = (e_mask[0, :] - e_mask[1, :]) == 0
        e_index = e_index[:, e_mask]
    return e_index
