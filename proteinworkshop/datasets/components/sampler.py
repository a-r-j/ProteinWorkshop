# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import random
from operator import itemgetter
from typing import Iterator, List, Optional

import numpy as np
from torch.utils import data as data
from torch.utils.data import Dataset, DistributedSampler, Sampler


class BatchSampler(data.Sampler):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design.
    """

    def __init__(
        self,
        unit_counts: List[int],
        max_units: int = 3000,
        shuffle: bool = True,
        hard_shuffle: bool = False,
        **kwargs,
    ):
        self.hard_shuffle = hard_shuffle
        self.unit_counts = unit_counts
        self.idx = [
            i for i in range(len(unit_counts)) if unit_counts[i] <= max_units
        ]
        self.shuffle = shuffle
        self.max_units = max_units
        self._form_batches()

    def _form_batches(self):
        self.batches = []
        if self.shuffle:
            random.shuffle(self.idx)
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.unit_counts[idx[0]] <= self.max_units:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.unit_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)

    def __len__(self) -> int:
        if not self.batches:
            self._form_batches()
        return len(self.batches)

    def __iter__(self):
        if not self.batches or (self.shuffle and self.hard_shuffle):
            self._form_batches()
        elif self.shuffle:
            np.random.shuffle(self.batches)
        for batch in self.batches:
            yield batch


class DatasetFromSampler(Dataset):
    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    From https://github.com/catalyst-team/catalyst

    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler: Sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        **kwargs,
    ):
        """

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))
