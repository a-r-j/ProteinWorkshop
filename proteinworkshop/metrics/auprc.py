"""Implementation of the AUPRC metric in ``torchmetrics``."""

import torch
from torchmetrics import Metric


class AUPRC(Metric):
    """Class for AUPRC metric.

    :param compute_on_cpu: Whether to compute the metric on CPU,
            defaults to ``True``.
    :type compute_on_cpu: bool, optional
    """

    def __init__(self, compute_on_cpu: bool = True) -> None:
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.compute_on_cpu = compute_on_cpu

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Updates the current state of the metric.

        :param preds: Tensor of predictions
        :type preds: torch.Tensor
        :param target: Tensor of targets
        :type target: torch.Tensor
        """
        self.preds.append(preds.detach())
        self.targets.append(target.detach())

    def compute(self) -> torch.Tensor:
        """Computes the AUPRC metric.

        :return: AUPRC metric value
        :rtype: torch.Tensor
        """
        return self.auprc(
            torch.cat(self.preds).flatten(), torch.cat(self.targets).flatten()
        )

    @staticmethod
    def auprc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Area under precision-recall curve.

        Parameters:
        :param pred: predictions. Shape ``(N,)``
        :param target: binary targets of shape ``(N,)``
        """
        eps = 1e-10
        order = pred.argsort(descending=True)
        target = target[order]
        precision = target.cumsum(0) / torch.arange(
            1, len(target) + 1, device=target.device
        )
        return precision[target == 1].sum() / ((target == 1).sum() + eps)
