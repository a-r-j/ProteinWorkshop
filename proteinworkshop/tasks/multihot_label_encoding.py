from typing import Union

import torch
from graphein.protein.tensor.data import Protein
from torch_geometric import transforms as T
from torch_geometric.data import Data


class MultiHotLabelEncoding(T.BaseTransform):
    """
    Transform to multihot encode labels for multilabel classification.

    :param num_classes: Number of classes to encode.
    :type num_classes: int
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def __call__(self, data: Union[Protein, Data]) -> Union[Protein, Data]:
        labels = torch.zeros((1, self.num_classes))
        labels[:, data.graph_y] = 1
        data.graph_y = labels
        return data
