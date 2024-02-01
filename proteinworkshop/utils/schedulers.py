"""Implement custom learning rate schedulers."""

import warnings

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class InverseSquareRootLR(_LRScheduler):
    """Implement the InverseSquareRootLR learning rate scheduler.

    :param optimizer: The optimizer.
    :type optimizer: Optimizer
    :param warmup_steps: The number of warmup steps.
    :type warmup_steps: int
    :param last_epoch: The index of the last epoch. If -1, the scheduler will
        start at the initial learning rate.
    :type last_epoch: int
    """

    def __init__(
        self, optimizer: Optimizer, warmup_steps: int, last_epoch: int = -1
    ):
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be > 0")
        self._warmup_steps = warmup_steps
        self._lr_steps = [
            param_group["lr"] / warmup_steps
            for param_group in optimizer.param_groups
        ]
        self._decay_factors = [
            param_group["lr"] * warmup_steps**0.5
            for param_group in optimizer.param_groups
        ]

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch < self._warmup_steps:
            return [self.last_epoch * lr_step for lr_step in self._lr_steps]
        else:
            return [
                decay_factor * self.last_epoch**-0.5
                for decay_factor in self._decay_factors
            ]
