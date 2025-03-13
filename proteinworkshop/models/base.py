import abc
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Union

import hydra
import lightning as L
import torch
import torch.distributed as torch_dist
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype as typechecker
from graphein.protein.tensor.angles import dihedrals
from graphein.protein.tensor.data import ProteinBatch, get_random_protein
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from proteinworkshop.models.utils import get_loss
from proteinworkshop.types import EncoderOutput, Label, ModelOutput
from proteinworkshop.utils.memory_utils import clean_up_torch_gpu_memory


class BaseModel(L.LightningModule, abc.ABC):
    config: DictConfig
    featuriser: nn.Module
    losses: Dict[str, Callable]
    task_transform: Optional[Callable]
    metric_names: List[str]

    @abc.abstractmethod
    def forward(self, batch: Batch) -> torch.Tensor:
        """Implement forward pass of model.

        :param batch: Mini-batch of data.
        :type batch: Batch
        :return: Model output.
        :rtype: torch.Tensor
        """
        ...

    @abc.abstractmethod
    def training_step(
        self, batch: Batch, batch_idx: torch.Tensor
    ) -> torch.Tensor:
        """Implement training step.

        :param batch: Mini-batch of data.
        :type batch: Batch
        :param batch_idx: Index of batch.
        :type batch_idx: torch.Tensor
        :return: Return loss.
        :rtype: torch.Tensor
        """
        ...

    @abc.abstractmethod
    def validation_step(
        self, batch: Batch, batch_idx: torch.Tensor
    ) -> torch.Tensor:
        """Implement validation step.

        :param batch: Mini-batch of data.
        :type batch: Batch
        :param batch_idx: Index of batch.
        :type batch_idx: torch.Tensor
        :return: Return loss.
        :rtype: torch.Tensor
        """
        ...

    @abc.abstractmethod
    def test_step(self, batch: Batch, batch_idx: torch.Tensor) -> torch.Tensor:
        """Implement test step.

        :param batch: Mini-batch of data.
        :type batch: Batch
        :param batch_idx: Index of batch.
        :type batch_idx: torch.Tensor
        :return: Return loss.
        :rtype: torch.Tensor
        """
        ...

    def featurise(
        self, batch: Union[Batch, ProteinBatch]
    ) -> Union[Batch, ProteinBatch]:
        """Applies the featuriser (``self.featuriser``) to a batch of data.

        .. seealso::
            :py:class:proteinworkshop.features.factory.ProteinFeaturiser

        :param batch: Batch of data
        :type batch: Union[Batch, ProteinBatch]
        :return: Featurised batch
        :rtype: Union[Batch, ProteinBatch]
        """
        out = self.featuriser(batch)
        if self.task_transform is not None:
            out = self.task_transform(out)
        return out

    def get_labels(self, batch: Union[Batch, ProteinBatch]) -> Label:
        """
        Computes or retrieves labels from a batch of data.

        Labels are returned as a dictionary of tensors indexed by output name.

        :param batch: Batch of data to compute labels for
        :type batch: Union[Batch, ProteinBatch]
        :return: Dictionary of labels indexed by output name
        :rtype: Label
        """
        labels: Dict[str, torch.Tensor] = {}
        for output in self.config.task.supervise_on:
            if output == "node_label":
                labels["node_label"] = batch.node_y
                if isinstance(
                    self.losses["node_label"], torch.nn.BCEWithLogitsLoss
                ):
                    labels["node_label"] = F.one_hot(
                        labels["node_label"],
                        num_classes=self.config.dataset.num_classes,
                    ).float()
            elif output == "graph_label":
                labels["graph_label"] = batch.graph_y
                if (
                    isinstance(
                        self.losses["graph_label"], torch.nn.BCEWithLogitsLoss
                    )
                    and batch.graph_y.ndim == 1
                ):
                    labels["graph_label"] = F.one_hot(
                        labels["graph_label"],
                        num_classes=self.config.dataset.num_classes,
                    ).float()
            elif output == "dihedrals":
                # If we have dihedral labels in the batch, use those
                # These will have been stored by the torsional denoising
                # transform
                if hasattr(batch, "true_dihedrals"):
                    labels["dihedrals"] = batch.true_dihedrals
                # If we have stored uncorrupted coords, use those to compute
                elif hasattr(batch, "coords_uncorrupted"):
                    labels["dihedrals"] = dihedrals(
                        batch.coords_uncorrupted,
                        batch.batch,
                        rad=True,
                        embed=True,
                    )
                # Otherwise, compute dihedrals from the batch coordinates
                else:
                    labels["dihedrals"] = dihedrals(
                        batch.coords, batch.batch, rad=True, embed=True
                    )
            elif output == "torsional_noise":
                labels["torsional_noise"] = batch.torsional_noise
            elif output == "residue_type":
                # If we have stored uncorrupted labels, use those
                if hasattr(batch, "residue_type_uncorrupted"):
                    labels["residue_type"] = batch.residue_type_uncorrupted
                # Otherwise, use residue types
                else:
                    labels["residue_type"] = batch.residue_type
                # If we have stored a mask, apply it
                if hasattr(batch, "sequence_corruption_mask"):
                    labels["residue_type"] = labels["residue_type"][
                        batch.sequence_corruption_mask
                    ]
            elif output == "pos":
                labels["pos"] = batch.noise[
                    :, 1, :
                ]  # TODO this is hardcoded to only handle CA
            elif output == "edge_distance":
                labels["edge_distance"] = batch.edge_distance_labels
            elif output in {"b_factor", "plddt"}:
                labels["b_factor"] = batch.b_factor

        return Label(labels)

    @typechecker
    def compute_loss(
        self, y_hat: ModelOutput, y: Label
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss by iterating over all outputs.

        In the case of multiple losses, the total loss is also included in the
        output dictionary of losses.

        :param y_hat: Output of model. This should be a dictionary of outputs
            (torch.Tensor) indexed by the output name (str)
        :type y_hat: ModelOutput
        :param y: Labels. This should be a dictionary of labels (torch.Tensor)
            indexed by the output name (str)
        :type y: Label
        :return: Dictionary of losses indexed by output name (str)
        :rtype: Dict[str, torch.Tensor]
        """
        loss = {k: v(y_hat[k], y[k]) for k, v in self.losses.items()}

        # Scale loss terms by coefficient
        if self.config.get("task.aux_loss_coefficient"):
            for (
                output,
                coefficient,
            ) in self.config.task.aux_loss_coefficient.items():
                loss[output] = coefficient * loss[output]

        loss["total"] = sum(loss.values())
        return loss

    def configure_optimizers(self):  # sourcery skip: extract-method
        logger.info("Instantiating optimiser...")
        optimiser = hydra.utils.instantiate(self.config.optimiser)["optimizer"]
        logger.info(optimiser)
        optimiser = optimiser(self.parameters())

        if self.config.get("scheduler"):
            logger.info("Instantiating scheduler...")
            scheduler = hydra.utils.instantiate(
                self.config.scheduler, optimiser
            )
            scheduler = OmegaConf.to_container(scheduler)
            scheduler["scheduler"] = scheduler["scheduler"](
                optimizer=optimiser
            )
            optimiser_config = {
                "optimizer": optimiser,
                "lr_scheduler": scheduler,
            }
            logger.info(f"Optimiser configuration: {optimiser_config}")
            return optimiser_config
        return optimiser

    def _build_output_decoders(self) -> nn.ModuleDict:
        """
        Instantiate output decoders.

        Decoders are instantiated from their respective config files.

        Decoders are stored in :py:class:`nn.ModuleDict`, indexed by output
        name.

        :return: ModuleDict of decoders indexed by output name
        :rtype: nn.ModuleDict
        """
        decoders = nn.ModuleDict()
        for output_head in self.config.decoder.keys():
            cfg = self.config.decoder.get(output_head)
            logger.info(
                f"Building {output_head} decoder. Output dim {cfg.get('out_dim')}"
            )
            logger.info(cfg)
            decoders[output_head] = hydra.utils.instantiate(cfg)
        return decoders

    def configure_losses(
        self, loss_dict: Dict[str, str]
    ) -> Dict[str, Callable]:
        """
        Configures losses from config. Returns a dictionary of losses mapping
        each output name to its respective loss function.

        :param loss_dict: Config dictionary of losses indexed by output name
        :type loss_dict: Dict[str, str]
        :return: Dictionary of losses indexed by output name
        :rtype: Dict[str, Callable]
        """
        # weight_loss = False
        # if weight_loss:
        #    logger.info("Using class weights to weight loss")
        #    weights = torch.tensor(self.dataset.class_weights).float().cuda()
        # else:
        #    weights = None
        return {
            k: get_loss(v, self.config.task.label_smoothing)
            for k, v in loss_dict.items()
        }

    def on_after_batch_transfer(
        self, batch: Union[Batch, ProteinBatch], dataloader_idx: int
    ) -> Union[Batch, ProteinBatch]:
        """
        Featurise batch **after** it has been transferred to the correct device.

        :param batch: Batch of data
        :type batch: Batch
        :param dataloader_idx: Index of dataloader
        :type dataloader_idx: int
        :return: Featurised batch
        :rtype: Union[Batch, ProteinBatch]
        """
        return self.featurise(batch)

    def configure_metrics(self):
        """
        Instantiates metrics from config.

        Metrics are Torchmetrics Objects :py:class:`torchmetrics.Metric`
        (see `torchmetrics <https://torchmetrics.readthedocs.io/en/latest/>`_)

        Metrics are set as model attributes as:

        ``{stage}_{output}_{metric_name}`` (e.g. ``train_residue_type_f1_score``)
        """

        CLASSIFICATION_METRICS: Set[str] = {
            "f1_score",
            "auprc",
            "accuracy",
            "f1_max",
            "rocauc",
        }
        REGRESSION_METRICS: Set[str] = {"mse", "mae", "r2", "rmse"}
        CONTINUOUS_OUTPUTS: Set[str] = {
            "b_factor",
            "plddt",
            "pos",
            "dihedrals",
            "torsional_noise",
        }
        CATEGORICAL_OUTPUTS: Set[str] = {"residue_type"}

        metric_names = []
        for metric_name, metric_conf in self.config.metrics.items():
            for output in self.config.task.output:
                for stage in {"train", "val", "test"}:
                    metric = hydra.utils.instantiate(metric_conf)
                    if output == "residue_type":
                        if metric_name not in {"accuracy", "perplexity"}:
                            continue
                        metric.num_classes = 23
                        metric.task = "multiclass"

                    # Skip incompatible metrics
                    if (
                        output in CONTINUOUS_OUTPUTS
                        and metric_name in CLASSIFICATION_METRICS
                    ):
                        logger.info(
                            f"Skipping classification metric {metric_name} for output {output} as output is continuous"
                        )
                        continue
                    if (
                        output in CATEGORICAL_OUTPUTS
                        and metric_name in REGRESSION_METRICS
                    ):
                        logger.info(
                            f"Skipping regression metric {metric_name} for output {output} as output is categorical"
                        )
                        continue
                    setattr(self, f"{stage}_{output}_{metric_name}", metric)
            metric_names.append(f"{metric_name}")
        setattr(self, "metric_names", metric_names)

    @typechecker
    def log_metrics(
        self, loss, y_hat: ModelOutput, y: Label, stage: str, batch: Batch
    ):
        """
        Logs metrics to logger.

        :param loss: Dictionary of losses indexed by output name (str)
        :type loss: Dict[str, torch.Tensor]
        :param y_hat: Output of model. This should be a dictionary of outputs
            indexed by the output name (str)
        :type y_hat: ModelOutput
        :param y: Labels. This should be a dictionary of labels (torch.Tensor)
            indexed by the output name (str)
        :type y: Label
        :param stage: Stage of training (``"train"``, ``"val"``, ``"test"``)
        :type stage: str
        :param batch: Batch of data
        :type batch: Batch
        """
        # Log losses
        log_dict = {f"{stage}/loss/{k}": v for k, v in loss.items()}

        # Log metrics
        for m in self.metric_names:
            for output in self.config.task.output:
                if hasattr(self, f"{stage}_{output}_{m}"):
                    try:
                        metric = getattr(self, f"{stage}_{output}_{m}")
                        pred = y_hat[output]
                        target = y[output]

                        if m == "perplexity":
                            pred = to_dense_batch(pred, batch.batch)[0]
                            target = to_dense_batch(
                                target, batch.batch, fill_value=-100
                            )[0]
                        # This is a hack for MSE-type metrics which fail on e.g. [4,1] & [4]
                        try:
                            val = metric(pred, target)
                        except RuntimeError:
                            val = metric(pred, target.unsqueeze(-1))
                        log_dict[f"{stage}/{output}/{m}"] = val

                    except (ValueError, RuntimeError):
                        continue
        self.log_dict(log_dict, prog_bar=True)


class BenchMarkModel(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.config = cfg

        # self.encoder = get_protein_encoder(cfg)
        logger.info("Instantiating encoder...")
        self.encoder: nn.Module = hydra.utils.instantiate(cfg.encoder)
        logger.info(self.encoder)

        if hasattr(cfg.decoder, "disable") and cfg.decoder.disable:
            logger.info("Disabling decoder as requested")
            self.decoder = None
        else:
            logger.info("Instantiating decoders...")
            self.decoder: nn.ModuleDict = self._build_output_decoders()
            logger.info(self.decoder)

        logger.info("Instantiating losses...")
        self.losses = self.configure_losses(cfg.task.losses)
        logger.info(f"Using losses: {self.losses}")

        if self.config.get("task.aux_loss_coefficient"):
            logger.info(
                f"Using aux loss coefficient: {self.config.task.aux_loss_coefficient}"
            )
        else:
            logger.info("Not using aux loss scaling")

        logger.info("Configuring metrics...")
        self.metrics = self.configure_metrics()
        logger.info(self.metric_names)

        logger.info("Instantiating featuriser...")
        self.featuriser: nn.Module = hydra.utils.instantiate(cfg.features)
        logger.info(self.featuriser)

        logger.info("Instantiating task transform...")
        self.task_transform = hydra.utils.instantiate(
            cfg.get("task.transform")
        )
        logger.info(self.task_transform)

        self.save_hyperparameters()

        self.example_input_array = self._create_example_batch()

    def _create_example_batch(self) -> ProteinBatch:
        """Creates an example batch for model inspection (including
        featurisation and transformation as specified by the config).

        :return: Example batch of data, featurised and transformed as specified
            by the config.
        :rtype: ProteinBatch
        """
        with torch.no_grad():
            proteins = [
                get_random_protein()
                for _ in range(self.config.dataset.datamodule.batch_size)
            ]
            for p in proteins:
                setattr(p, "x", torch.zeros(p.coords.shape[0]))
                setattr(
                    p, "seq_pos", torch.arange(p.coords.shape[0]).unsqueeze(-1)
                )
            batch = ProteinBatch.from_data_list(proteins)
            return self.featurise(batch)

    @typechecker
    def forward(self, batch: Union[Batch, ProteinBatch]) -> ModelOutput:
        """
        Implements the forward pass of the model.


        1. Apply the model encoder (``self.encoder``) to the batch of data.
        2. (Optionally) apply any transformations to the encoder output
        (:py:meth:`BaseModel.transform_encoder_output`)
        3. Iterate over the decoder heads (``self.decoder``) and apply each
        decoder to the relevant part of the encoder output.
        4. (Optionally) apply any post-processing to the model output.
        (:py:meth:`BaseModel.compute_output`)

        :param batch: Mini-batch of data.
        :type batch: Union[Batch, ProteinBatch]
        :return: Model output.
        :rtype: ModelOutput
        """
        output: EncoderOutput = self.encoder(batch)

        output = self.transform_encoder_output(output, batch)

        if self.decoder is not None:
            for output_head in self.config.decoder.keys():
                if hasattr(self.decoder[output_head], "requires_pos"):
                    output[output_head] = self.decoder[output_head](
                        edge_index=batch.edge_index,
                        scalar_features=output["node_embedding"],
                        pos=batch.pos,
                    )
                else:
                    emb_type = self.decoder[
                        output_head
                    ].input  # node_embedding or graph_embedding
                    output[output_head] = self.decoder[output_head](
                        output[emb_type]
                    )

        return self.compute_output(output, batch)

    @typechecker
    def transform_encoder_output(
        self, output: EncoderOutput, batch
    ) -> EncoderOutput:
        """
        Modifies graph encoder output.

        - If we are computing edge distances, we concatenate the node embeddings
        of the two nodes connected by the masked edge.

        :param output: Encoder output (dictionary mapping output name to the
            output tensor)
        :type output: EncoderOutput
        :param batch: Batch of data
        :type batch: Batch
        :return: Encoder output (dictionary mapping output name to the
            transformed output)
        """
        if "edge_distance" in self.config.decoder.keys():
            output["edge_distance"] = torch.cat(
                [
                    output["node_embedding"][batch.node_mask[0]],
                    output["node_embedding"][batch.node_mask[1]],
                ],
                dim=-1,
            )

        return output

    @typechecker
    def compute_output(self, output: ModelOutput, batch: Batch) -> ModelOutput:
        """
        Computes output from model output.

        - For dihedral angle prediction, this involves normalising the
        'sin'/'cos' pairs for each angle such that the have norm 1.
        - For sequence denoising, this masks the output such that we only
        supervise on the corrupted residues.

        :param output: Model output (dictionary mapping output name to the
            output tensor)
        :type: ModelOutput
        :param batch: Batch of data
        :type batch: Batch
        :return: Model output (dictionary mapping output name to the
            transformed output)
        :rtype: ModelOutput
        """
        if "dihedrals" in output.keys():
            # Normalize output so each pair of sin(ang) and cos(ang) sum to 1.
            output["dihedrals"] = F.normalize(
                output["dihedrals"].view(-1, 3, 2), dim=-1
            ).view(-1, 6)
        # If we have a mask, apply it
        if hasattr(batch, "sequence_corruption_mask"):
            output["residue_type"] = output["residue_type"][
                batch.sequence_corruption_mask
            ]

        return output

    @typechecker
    def _do_step(
        self,
        batch: Batch,
        batch_idx: int,
        stage: Literal["train", "val", "test"],
    ) -> torch.Tensor:
        """Performs a training/validation/test step.

        1. Obtains labels from :py:meth:`get_labels`
        2. Computes model output :py:meth:`forward`
        3. Computes loss :py:meth:`compute_loss`
        4. Logs metrics :py:meth:`log_metrics`

        Returns the total loss.

        :param batch: Mini-batch of data.
        :type batch: Batch
        :param batch_idx: Index of batch.
        :type batch_idx: int
        :param stage: Stage of training (``"train"``, ``"val"``, ``"test"``)
        :type stage: Literal["train", "val", "test"]
        :return: Loss
        :rtype: torch.Tensor
        """
        y = self.get_labels(batch)
        y_hat = self(batch)

        loss = self.compute_loss(y_hat, y)
        self.log_metrics(loss, y_hat, y, stage, batch=batch)
        return loss["total"]

    @typechecker
    def _do_step_catch_oom(
        self,
        batch: Batch,
        batch_idx: int,
        stage: Literal["train", "val"],
    ) -> Optional[torch.Tensor]:
        """Performs a training/validation step
        while catching out of memory errors.
        Note that this should not be used for
        test steps for proper benchmarking.

        1. Obtains labels from :py:meth:`get_labels`
        2. Computes model output :py:meth:`forward`
        3. Computes loss :py:meth:`compute_loss`
        4. Logs metrics :py:meth:`log_metrics`

        Returns the total loss.

        :param batch: Mini-batch of data.
        :type batch: Batch
        :param batch_idx: Index of batch.
        :type batch_idx: int
        :param stage: Stage of training (``"train"``, ``"val"``)
        :type stage: Literal["train", "val"]
        :return: Loss
        :rtype: torch.Tensor
        """
        # by default, do not skip the current batch
        skip_flag = torch.zeros(
            (), device=self.device, dtype=torch.bool
        )  # NOTE: for skipping batches in a multi-device setting
        
        try:
            y = self.get_labels(batch)
            y_hat = self(batch)

            loss = self.compute_loss(y_hat, y)
            self.log_metrics(loss, y_hat, y, stage, batch=batch)

        except Exception as e:
            skip_flag = torch.ones((), device=self.device, dtype=torch.bool)

            if "out of memory" in str(e):
                logger.warning(
                    f"Ran out of memory in the forward pass. Skipping current {stage} batch with index {batch_idx}."
                )
                if not torch_dist.is_initialized():
                    # NOTE: for skipping batches in a single-device setting
                    if self.training:
                        for p in self.trainer.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                    return None
            else:
                if not torch_dist.is_initialized():
                    raise e
                
        # NOTE: for skipping batches in a multi-device setting
        # credit: https://github.com/Lightning-AI/lightning/issues/5243#issuecomment-1553404417
        if torch_dist.is_initialized():
            # if any rank skips a batch, then all other ranks need to skip
            # their batches as well so DDP can properly keep all ranks synced
            world_size = torch_dist.get_world_size()
            torch_dist.barrier()
            result = [torch.zeros_like(skip_flag) for _ in range(world_size)]
            torch_dist.all_gather(result, skip_flag)
            any_skipped = torch.sum(torch.stack(result)).bool().item()
            if any_skipped:
                if self.training:
                    for p in self.trainer.model.parameters():
                        if p.grad is not None:
                            del p.grad
                logger.warning(
                    f"Failed to perform the forward pass for at least one rank. Skipping {stage} batches for all ranks."
                )
                return None

        return loss["total"]

    def training_step(
        self, batch: Union[Batch, ProteinBatch], batch_idx: int
    ) -> Optional[torch.Tensor]:
        """
        Perform training step.

        1. Obtains labels from :py:meth:`get_labels`
        2. Computes model output :py:meth:`forward`
        3. Computes loss :py:meth:`compute_loss`
        4. Logs metrics :py:meth:`log_metrics`

        Returns the total loss.

        :param batch: Mini-batch of data.
        :type batch: Batch
        :param batch_idx: Index of batch.
        :type batch_idx: int
        :return: Loss
        :rtype: Optional[torch.Tensor]
        """
        return self._do_step_catch_oom(batch, batch_idx, "train")

    def validation_step(
        self, batch: Union[Batch, ProteinBatch], batch_idx: int
    ) -> Optional[torch.Tensor]:
        """
        Perform validation step.

        1. Obtains labels from :py:meth:`get_labels`
        2. Computes model output :py:meth:`forward`
        3. Computes loss :py:meth:`compute_loss`
        4. Logs metrics :py:meth:`log_metrics`

        Returns the total loss.

        :param batch: Mini-batch of data.
        :type batch: Batch
        :param batch_idx: Index of batch.
        :type batch_idx: int
        :return: Loss
        :rtype: Optional[torch.Tensor]
        """
        return self._do_step_catch_oom(batch, batch_idx, "val")

    def test_step(
        self, batch: Union[Batch, ProteinBatch], batch_idx: int
    ) -> torch.Tensor:
        """Perform test step.

        1. Obtains labels from :py:meth:`get_labels`
        2. Computes model output :py:meth:`forward`
        3. Computes loss :py:meth:`compute_loss`
        4. Logs metrics :py:meth:`log_metrics`

        Returns the total loss.

        :param batch: Mini-batch of data.
        :type batch: Batch
        :param batch_idx: Index of batch.
        :type batch_idx: int
        :return: Loss
        :rtype: torch.Tensor
        """
        return self._do_step(batch, batch_idx, "test")

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Dict[str, Any]):
        """Overrides Lightning's `backward` hook to add an out-of-memory (OOM) check.

        :param loss: The loss value to backpropagate.
        :param args: Additional positional arguments to pass to `torch.Tensor.backward`.
        :param kwargs: Additional keyword arguments to pass to `torch.Tensor.backward`.
        """
        # by default, do not skip the current batch
        skip_flag = torch.zeros(
            (), device=self.device, dtype=torch.bool
        )  # NOTE: for skipping batches in a multi-device setting

        try:
            loss.backward(*args, **kwargs)
        except Exception as e:
            skip_flag = torch.ones((), device=self.device, dtype=torch.bool)
            logger.warning(f"Failed the backward pass. Skipping it for the current rank due to: {e}")
            for p in self.trainer.model.parameters():
                if p.grad is not None:
                    del p.grad
            logger.warning("Finished cleaning up all gradients following the failed backward pass.")
            if "out of memory" not in str(e) and not torch_dist.is_initialized():
                raise e

        # NOTE: for skipping batches in a multi-device setting
        # credit: https://github.com/Lightning-AI/lightning/issues/5243#issuecomment-1553404417
        if torch_dist.is_initialized():
            # if any rank skips a batch, then all other ranks need to skip
            # their batches as well so DDP can properly keep all ranks synced
            world_size = torch_dist.get_world_size()
            torch_dist.barrier()
            result = [torch.zeros_like(skip_flag) for _ in range(world_size)]
            torch_dist.all_gather(result, skip_flag)
            any_skipped = torch.sum(torch.stack(result)).bool().item()
            if any_skipped:
                logger.warning(
                    "Skipping backward for all ranks after detecting a failed backward pass."
                )
                del loss  # delete the computation graph
                logger.warning(
                    "Finished cleaning up the computation graph following one of the rank's failed backward pass."
                )
                for p in self.trainer.model.parameters():
                    if p.grad is not None:
                        del p.grad
                logger.warning(
                    "Finished cleaning up all gradients following one of the rank's failed backward pass."
                )
                clean_up_torch_gpu_memory()
                logger.warning(
                    "Finished manually freeing up memory following one of the rank's failed backward pass."
                )
