from typing import Set, Union

import torch
import torch_geometric.transforms as T
from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch


class EdgeDistancePredictionTransform(T.BaseTransform):
    """
    Self-supervision task to predict the pairwise distance between two nodes.

    We first sample ``num_samples`` edges randomly from the input batch. We then
    construct a mask to remove the sampled edges from the batch. We store the
    masked node indices and their pairwise distance as ``batch.node_mask`` and
    ``batch.edge_distance_labels``, respectively. Finally, it masks the edges
    (and their attributes) using the constructed mask and returns the modified
    batch.
    """

    def __init__(self, num_samples: int):
        """Initialise the transform.

        :param num_samples: Number of edges to mask
        :type num_samples: int
        """
        self.num_samples = num_samples

    @property
    def required_batch_attributes(self) -> Set[str]:
        """
        Returns the set of attributes that this transform requires to be
        present on the batch object for correct operation.

        :return: Set of required attributes
        :rtype: Set[str]
        """
        return {"num_edges", "edge_index", "pos"}

    def __call__(
        self, batch: Union[ProteinBatch, Batch]
    ) -> Union[Batch, ProteinBatch]:
        # Sample edges
        indices = torch.randint(
            0,
            batch.num_edges,
            (self.num_samples,),
            device=batch.edge_index.device,
        ).long()
        # Construct mask
        mask = torch.ones_like(
            batch.edge_index[0], device=batch.edge_index.device
        ).bool()
        mask[indices] = 0

        # Store masked node indices & labels
        nodes = batch.edge_index[:, indices]
        batch.node_mask = nodes
        batch.edge_distance_labels = torch.pairwise_distance(
            batch.pos[nodes[0]], batch.pos[nodes[1]]
        )

        # Mask edges and attributes
        batch.edge_index = batch.edge_index[:, mask]
        if hasattr(batch, "edge_type"):
            batch.edge_type = batch.edge_type[:, mask]
        if hasattr(batch, "edge_attr"):
            batch.edge_attr = batch.edge_attr[mask]
        # TODO - non scalar edge attributes

        return batch
