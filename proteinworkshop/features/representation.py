from itertools import chain
from typing import Literal, Tuple

import torch
from beartype import beartype as typechecker
from graphein.protein.tensor.types import AtomTensor, CoordTensor
from jaxtyping import jaxtyped
from torch_geometric.data import Batch, Data
from torch_geometric.utils import unbatch

from proteinworkshop.configs.config import ExperimentConfigurationError


@jaxtyped(typechecker=typechecker)
def get_full_atom_coords(
    atom_tensor: AtomTensor, fill_value: float = 1e-5
) -> Tuple[CoordTensor, torch.Tensor, torch.Tensor]:
    """Converts an AtomTensor to a full atom representation
    (e.g. dense to sparse).

    :param atom_tensor: AtomTensor of shape (``N_residues x 37 x 3``)
    :type atom_tensor: AtomTensor
    :param fill_value: Value indicating missing atoms, defaults to ``1e-5``
    :type fill_value: float, optional
    :return: Tuple of coords (``N_atoms x 3``), residue_index (``N_atoms``),
        atom_type (``N_atoms`` (``[0-36]``))
    :rtype: Tuple[CoordTensor, torch.Tensor, torch.Tensor]
    """
    # Get number of atoms per residue
    filled = atom_tensor[:, :, 0] != fill_value
    nz = filled.nonzero()

    residue_index = nz[:, 0]
    atom_type = nz[:, 1]

    coords = atom_tensor.reshape(-1, 3)
    coords = coords[coords != fill_value].reshape(-1, 3)

    return coords, residue_index, atom_type


@jaxtyped(typechecker=typechecker)
def transform_representation(
    x: Batch, representation_type: Literal["CA", "BB", "FA", "BB_SC", "CA_SC"]
) -> Batch:
    """
    Factory method to transform a batch into a specified representation.

    The ``AtomTensor`` (i.e. ``batch.coords`` with shape
    (:math:`|V| \times 37 \times 3`) is manipulated to produce the corresponding
    number of nodes according to the desired node representation.

    - ``CA`` simply selects the :math:`C_\alpha` atoms as nodes
        (i.e. ``batch.coords[:, 1, :]``)
    - ``BB`` selects and unravels the four backbone atoms
        (:math:`N, C_\alpha, C, O`) as nodes. Existing node features are tiled
        over the backbone atom nodes on a per-residue basis.
    - ``FA`` unravels all the a``AtomTensor`` to result in a full-atom graph,
        i.e. each atom in the structure becomes a node in the graph. Existing
        node features are tiled over the atom nodes on a per-residue basis.

    :param x: A minibatch of data
    :type x: Batch
    :param representation_type: _description_
    :type representation_type: Literal["CA", "BB", "FA", "BB_SC", "CA_SC"]
    :raises ExperimentConfigurationError: _description_
    :return: _description_
    :rtype: Batch
    """
    if representation_type == "CA":
        x.pos = x.coords[:, 1, :]
        return x
    elif representation_type == "BB":
        return ca_to_bb_repr(x)
    elif representation_type == "FA":
        return ca_to_fa_repr(x)
    elif representation_type == "BB_SC":
        return ca_to_bb_sc_repr(x)
    elif representation_type == "CA_SC":
        return ca_to_ca_sc_repr(x)
    else:
        raise ExperimentConfigurationError(
            f"Unsupported granularity type: \
            {representation_type}. Must be one of [CA, BB, BB_SC, FA]"
        )


@typechecker
def _ca_to_fa_repr(x: Data) -> Data:
    """Converts CA representation to full atom representation."""
    coords, residue_index, atom_type = get_full_atom_coords(x.coords)

    x.amino_acid_one_hot = x.amino_acid_one_hot[residue_index]
    x.dihedrals = x.dihedrals[residue_index]
    x.pos = coords
    x.residue_index = residue_index
    x.atom_type = atom_type
    x.num_nodes = x.pos.shape[0]
    return x


@typechecker
def _ca_to_bb_repr(x: Data) -> Data:
    """Converts CA representation to backbone representation."""
    x.pos = x.coords[:, :4, :].reshape(-1, 3)
    x.dihedrals = x.dihedrals.repeat_interleave(4, 0)
    x.amino_acid_one_hot = x.amino_acid_one_hot.repeat_interleave(4, 0)
    x.num_nodes = x.num_nodes * 4
    x.atom_type = torch.tensor([0.0, 1.0, 2.0]).repeat(x.num_nodes)

    n_id = [f"{n}:N" for n in x.node_id]
    ca_id = [f"{n}:Ca" for n in x.node_id]
    c_id = [f"{n}:C" for n in x.node_id]
    x.node_id = list(chain.from_iterable(zip(n_id, ca_id, c_id)))
    return x


@typechecker
def ca_to_bb_repr(batch: Batch) -> Batch:  # sourcery skip: assign-if-exp
    """
    Converts a batch of CA representations to backbone representations. I.e.
    1 node per residue -> 4 nodes per residue (N, CA, C, O)

    This function tiles any existing node features on the CA atoms over the
    additional nodes in the backbone representation.
    """
    if "sidechain_torsions" in batch.keys:
        sidechain_torsions = batch.sidechain_torsions.repeat_interleave(4, 0)
    else:
        sidechain_torsions = None

    if "chi1" in batch.keys:
        chi1 = batch.chi1.repeat_interleave(4, 0)
    else:
        chi1 = None

    if "positional_encoding" in batch.keys:
        positional_encoding = batch.positional_encoding.repeat_interleave(4, 0)
    else:
        positional_encoding = None

    if "true_dihedrals" in batch.keys:
        true_dihedrals = batch.true_dihedrals.repeat_interleave(4, 0)
    else:
        true_dihedrals = None

    if "mask" in batch.keys:
        mask = batch.mask.repeat_interleave(4, 0)
    else:
        mask = None

    batch_idx = batch.batch.repeat_interleave(4, 0)
    x = batch.x.repeat_interleave(4, 0) if "x" in batch.keys else None
    batch = Batch.from_data_list(
        [_ca_to_bb_repr(x) for x in batch.to_data_list()]
    )

    batch.batch = batch_idx
    if sidechain_torsions is not None:
        batch.sidechain_torsions = sidechain_torsions
        del sidechain_torsions
    if chi1 is not None:
        batch.chi1 = chi1
        del chi1
    if positional_encoding is not None:
        batch.positional_encoding = positional_encoding
        del positional_encoding
    if true_dihedrals is not None:
        batch.true_dihedrals = true_dihedrals
        del true_dihedrals
    if mask is not None:
        batch.mask = mask
        del mask
    if x is not None:
        batch.x = x
        del x

    return batch


@typechecker
def ca_to_bb_sc_repr(batch: Batch) -> Batch:
    """Converts a batch of CA representations to backbone + sidechain representations."""
    # Get centroids
    batch.coords[:, 3:, :] = 1e-5
    batch.coords[:, 4, :] = coarsen_sidechain(batch, aggr="mean")
    batch.coords = batch.coords[:, :4, :]
    return ca_to_fa_repr(batch)


@typechecker
def ca_to_ca_sc_repr(batch: Batch) -> Batch:
    """Converts a batch of CA representations to C + sidechain representations."""
    # Get centroids
    batch.coords[:, 2:, :] = 1e-5
    batch.coords[:, 0, :] = coarsen_sidechain(batch, aggr="mean")
    batch.coords = batch.coords[:, :2, :]
    return batch


@typechecker
def coarsen_sidechain(x: Data, aggr: str = "mean") -> CoordTensor:
    """Returns tensor of sidechain centroids: L x 3"""
    # sourcery skip: remove-unnecessary-else, swap-if-else-branches
    # Compute mean sidechain position
    sc_points = x.coords[:, 4:]
    if aggr == "mean":
        sc_points = torch.mean(sc_points, dim=1)
    else:
        raise NotImplementedError(
            f"Aggregation method {aggr} not implemented."
        )

    return sc_points


@typechecker
def ca_to_fa_repr(batch: Batch) -> Batch:  # sourcery skip: assign-if-exp
    """Converts a batch of CA representations to full atom representations."""
    if "sidechain_torsion" in batch.keys:
        sidechain_torsions = unbatch(batch.sidechain_torsion, batch.batch)
    else:
        sidechain_torsions = None

    if "chi1" in batch.keys:
        chi1 = unbatch(batch.chi1, batch.batch)
    else:
        chi1 = None

    if "mask" in batch.keys:
        mask = unbatch(batch.mask, batch.batch)
    else:
        mask = None

    if "true_dihedrals" in batch.keys:
        true_dihedrals = unbatch(batch.true_dihedrals, batch.batch)
    else:
        true_dihedrals = None

    if "true_amino_acid_one_hot" in batch.keys:
        true_amino_acid_one_hot = unbatch(
            batch.true_amino_acid_one_hot, batch.batch
        )
    else:
        true_amino_acid_one_hot = None

    if "positional_encoding" in batch.keys:
        positional_encoding = unbatch(batch.positional_encoding, batch.batch)
    else:
        positional_encoding = None

    batch = Batch.from_data_list(
        [_ca_to_fa_repr(x) for x in batch.to_data_list()]
    )

    residue_idxs = unbatch(batch.residue_index, batch.batch)

    if sidechain_torsions is not None:
        batch.sidechain_torsion = torch.cat(
            [
                sidechain_torsion[res_idx - torch.min(res_idx)]
                for sidechain_torsion, res_idx in zip(
                    sidechain_torsions, residue_idxs
                )
            ]
        )
        del sidechain_torsions

    if chi1 is not None:
        batch.chi1 = torch.cat(
            [
                chi1[res_idx - torch.min(res_idx)]
                for chi1, res_idx in zip(chi1, residue_idxs)
            ]
        )
        del chi1

    if true_dihedrals is not None:
        batch.true_dihedrals = torch.cat(
            [
                true_dihedrals[res_idx - torch.min(res_idx)]
                for true_dihedrals, res_idx in zip(
                    true_dihedrals, residue_idxs
                )
            ]
        )
        del true_dihedrals

    if true_amino_acid_one_hot is not None:
        batch.true_amino_acid_one_hot = torch.cat(
            [
                true_amino_acid_one_hot[res_idx - torch.min(res_idx)]
                for true_amino_acid_one_hot, res_idx in zip(
                    true_amino_acid_one_hot, residue_idxs
                )
            ]
        )
        del true_amino_acid_one_hot

    if mask is not None:
        batch.mask = torch.cat(
            [
                mask[res_idx - torch.min(res_idx)]
                for mask, res_idx in zip(mask, residue_idxs)
            ]
        )
        del mask

    if positional_encoding is not None:
        batch.positional_encoding = torch.cat(
            [
                pos_encoding[res_idx - torch.min(res_idx)]
                for pos_encoding, res_idx in zip(
                    positional_encoding, residue_idxs
                )
            ]
        )
        del positional_encoding

    return batch
