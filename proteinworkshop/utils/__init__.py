import torch
import os
import os.path
import warnings

import pytorch_lightning as pl

from loguru import logger as log
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT

from beartype.typing import Any, Dict, List, Optional

try:
    import amp_C

    apex_available = True
except Exception:
    apex_available = False


from .callbacks import instantiate_callbacks
from .extras import extras, task_wrapper
from .loggers import instantiate_loggers
from .logging_utils import log_hyperparameters


class EMA(Callback):
    """
    Implements Exponential Moving Averaging (EMA).
    When training a model, this callback will maintain moving averages of the trained parameters.
    When evaluating, we use the moving averages copy of the trained parameters.
    When saving, we save an additional set of parameters with the prefix `ema`.

    :param decay: Decay rate for EMA.
    :type decay: float
    :param apply_ema_every_n_steps: How often to apply EMA.
    :type apply_ema_every_n_steps: int
    :param start_step: Step to start applying EMA.
    :type start_step: int
    :save_ema_weights_in_callback_state: Whether to save EMA weights in callback state.
    :type save_ema_weights_in_callback_state: bool
    :param evaluate_ema_weights_instead: Whether to evaluate EMA weights instead of the original weights.
    :type evaluate_ema_weights_instead: bool

    Adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/callbacks/ema.py
    """

    def __init__(
        self,
        decay: float,
        apply_ema_every_n_steps: int = 1,
        start_step: int = 0,
        save_ema_weights_in_callback_state: bool = False,
        evaluate_ema_weights_instead: bool = False,
    ):
        if not apex_available:
            rank_zero_warn(
                "EMA has better performance when Apex is installed: https://github.com/NVIDIA/apex#installation."
            )
        if not (0 <= decay <= 1):
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        self._ema_model_weights: Optional[List[torch.Tensor]] = None
        self._overflow_buf: Optional[torch.Tensor] = None
        self._cur_step: Optional[int] = None
        self._weights_buffer: Optional[List[torch.Tensor]] = None
        self.apply_ema_every_n_steps = apply_ema_every_n_steps
        self.start_step = start_step
        self.save_ema_weights_in_callback_state = save_ema_weights_in_callback_state
        self.evaluate_ema_weights_instead = evaluate_ema_weights_instead
        self.decay = decay

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train begins. Creates a copy of the model's weights.""" 
        log.info("Creating EMA weights copy.")
        if self._ema_model_weights is None:
            self._ema_model_weights = [p.detach().clone() for p in pl_module.state_dict().values()]
        # ensure that all the weights are on the correct device
        self._ema_model_weights = [p.to(pl_module.device) for p in self._ema_model_weights]
        self._overflow_buf = torch.IntTensor([0]).to(pl_module.device)

    def ema(self, pl_module: "pl.LightningModule") -> None:
        """Updates EMA weights."""
        if apex_available and pl_module.device.type == "cuda":
            return self.apply_multi_tensor_ema(pl_module)
        return self.apply_ema(pl_module)

    def apply_multi_tensor_ema(self, pl_module: "pl.LightningModule") -> None:
        """Applies EMA using NVIDIA's apex.amp_C.multi_tensor_axpby()."""
        model_weights = list(pl_module.state_dict().values())
        amp_C.multi_tensor_axpby(
            65536,
            self._overflow_buf,
            [self._ema_model_weights, model_weights, self._ema_model_weights],
            self.decay,
            1 - self.decay,
            -1,
        )

    def apply_ema(self, pl_module: "pl.LightningModule") -> None:
        """Applies EMA using PyTorch operations."""
        for orig_weight, ema_weight in zip(list(pl_module.state_dict().values()), self._ema_model_weights):
            if ema_weight.data.dtype != torch.long and orig_weight.data.dtype != torch.long:
                # ensure that non-trainable parameters (e.g., feature distributions) are not included in EMA weight averaging
                diff = ema_weight.data - orig_weight.data
                diff.mul_(1.0 - self.decay)
                ema_weight.sub_(diff)

    def should_apply_ema(self, step: int) -> bool:
        """Checks whether EMA should be applied."""
        return step != self._cur_step and step >= self.start_step and step % self.apply_ema_every_n_steps == 0

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Applies EMA at the end of each training batch.
        
        :param trainer: The trainer object.
        :type trainer: pl.Trainer
        :param pl_module: The LightningModule.
        :type pl_module: pl.LightningModule
        :param outputs: The outputs of the training step.
        :type outputs: STEP_OUTPUT
        :param batch: The current batch.
        :type batch: Any
        :param batch_idx: The current batch index.
        :type batch_idx: int
        """
        if self.should_apply_ema(trainer.global_step):
            self._cur_step = trainer.global_step
            self.ema(pl_module)

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the callback as a dictionary."""
        if self.save_ema_weights_in_callback_state:
            return dict(cur_step=self._cur_step, ema_weights=self._ema_model_weights)
        return dict(cur_step=self._cur_step)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the state of the callback from the given `state_dict`."""
        self._cur_step = state_dict["cur_step"]
        # when loading within apps such as NeMo, EMA weights will be loaded by the experiment manager separately
        if self._ema_model_weights is None:
            self._ema_model_weights = state_dict.get("ema_weights")
            log.info("EMA weights have been loaded successfully through `state_dict`. Continuing training with saved EMA weights.")

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        """Called when restoring a model checkpoint. Loads the EMA weights if they are available.
        
        :param trainer: The trainer object.
        :type trainer: pl.Trainer
        :param pl_module: The LightningModule.
        :type pl_module: pl.LightningModule
        :param checkpoint: The checkpoint to load from.
        :type checkpoint: Dict[str, Any]
        """
        checkpoint_callback = trainer.checkpoint_callback

        if trainer.ckpt_path and checkpoint_callback is not None:
            ext = checkpoint_callback.FILE_EXTENSION
            if trainer.ckpt_path.endswith(f"-EMA{ext}"):
                log.info(
                    "loading EMA based weights. "
                    "The callback will treat the loaded EMA weights as the main weights"
                    " and create a new EMA copy when training."
                )
                return
            ema_path = trainer.ckpt_path.replace(ext, f"-EMA{ext}")
            ckpt_path = trainer.ckpt_path
            if os.path.exists(ema_path):
                ema_state_dict = torch.load(ema_path, map_location=torch.device("cpu"))
                self._ema_model_weights = ema_state_dict["state_dict"].values()
                del ema_state_dict
                log.info("EMA weights have been loaded successfully. Continuing training with saved EMA weights.")
            elif os.path.exists(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
                if "callbacks" in state_dict and "EMA" in state_dict["callbacks"] and "ema_weights" in state_dict["callbacks"]["EMA"]:
                    # note: this means we have found `ema_weights` which will subsequently be loaded via `load_state_dict()`
                    pass
                else:
                    warnings.warn(
                        "we were unable to find the associated EMA weights when re-loading, "
                        "training will start with new EMA weights.",
                        UserWarning,
                    )
                del state_dict
            else:
                warnings.warn(
                    "we were unable to find the associated EMA weights when re-loading, "
                    "training will start with new EMA weights.",
                    UserWarning,
                )

    def replace_model_weights(self, pl_module: "pl.LightningModule") -> None:
        """Replaces the model weights with the EMA weights.
        
        :param pl_module: The LightningModule.
        :type pl_module: pl.LightningModule"""
        self._weights_buffer = [p.detach().clone().to("cpu") for p in pl_module.state_dict().values()]
        new_state_dict = {k: v for k, v in zip(pl_module.state_dict().keys(), self._ema_model_weights)}
        pl_module.load_state_dict(new_state_dict)

    def restore_original_weights(self, pl_module: "pl.LightningModule") -> None:
        """Restores the original model weights.
        
        :param pl_module: The LightningModule.
        :type pl_module: pl.LightningModule"""
        state_dict = pl_module.state_dict()
        new_state_dict = {k: v for k, v in zip(state_dict.keys(), self._weights_buffer)}
        pl_module.load_state_dict(new_state_dict)
        del self._weights_buffer

    @property
    def ema_initialized(self) -> bool:
        """Returns whether the EMA weights have been initialized."""
        return self._ema_model_weights is not None

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)


class EMAModelCheckpoint(ModelCheckpoint):
    """
    Light wrapper around Lightning's `ModelCheckpoint` to, upon request, save an EMA copy of the model as well.

    Adapted from: https://github.com/NVIDIA/NeMo/blob/be0804f61e82dd0f63da7f9fe8a4d8388e330b18/nemo/utils/exp_manager.py#L744
    """

    def __init__(self, **kwargs):
        # call the parent class constructor with the provided kwargs
        super().__init__(**kwargs)

    @staticmethod
    def _get_ema_callback(trainer: "pl.Trainer") -> Optional[EMA]:
        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
                break
        return ema_callback

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """Saves the checkpoint to the given filepath."""
        super()._save_checkpoint(trainer, filepath)
        ema_callback = self._get_ema_callback(trainer)
        if ema_callback is not None:
            # save EMA copy of the model as well
            ema_callback.replace_model_weights(trainer.lightning_module)
            filepath = self._ema_format_filepath(filepath)
            if self.verbose:
                rank_zero_info(f"Saving EMA weights to separate checkpoint {filepath}")
            super()._save_checkpoint(trainer, filepath)
            ema_callback.restore_original_weights(trainer.lightning_module)

    def _ema_format_filepath(self, filepath: str) -> str:
        """Formats the filepath for the EMA checkpoint."""
        return filepath.replace(self.FILE_EXTENSION, f"-EMA{self.FILE_EXTENSION}")
