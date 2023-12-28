"""Module to validate the hydra config."""
import json
import os

import hydra
import psutil
import torch
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf

from proteinworkshop.constants import PROJECT_PATH


class ExperimentConfigurationError(Exception):
    """Exception to raise when the experiment configuration is invalid."""


TASKS = [
    "graph_property_prediction",
    "node_property_prediction",
    "inverse_folding",
    "backbone_dihedral_angle_prediction",
    "torsional_denoising",
]
"""List of task names."""

SELF_SUPERVISION_OUTPUTS = [
    "residue_type",
    "dihedrals",
    "pos",
    "edge_distance",
    "b_factor",
]

CLASSIFICATION_OUTPUTS = ["graph_label", "node_label"]

REGRESSION_OUTPUTS = [
    "graph_label",
    "node_label",
    "torsional_noise",
    "dihedrals",
    "pos",
    "edge_distance",
    "b_factor",
]

OUTPUTS = list(
    set(SELF_SUPERVISION_OUTPUTS + CLASSIFICATION_OUTPUTS + REGRESSION_OUTPUTS)
)


def load_config(run_id: str) -> DictConfig:
    """
    Loads a model config from a WANDB run id.

    :param run_id: WANDB run id
    :type run_id: str
    :returns: Config associated with the run
    :rtype: DictConfig
    """
    WANDB_DIR = PROJECT_PATH / "wandb"

    # Search for run
    run_dirs = os.listdir(WANDB_DIR)
    run_dir = [f for f in run_dirs if f.endswith(run_id)][0]

    # Load config and extract overrides
    metadata = WANDB_DIR / run_dir / "files" / "wandb-metadata.json"
    logger.info(f"Extracting config overrides from: {metadata}")
    metadata = json.load(open(metadata, "r"))
    overrides = metadata["args"]

    # Compose config
    return hydra.compose(
        "template", return_hydra_config=False, overrides=overrides
    )


def get_start_time(run_id: str) -> str:
    """
    Get the start time of a WANDB run from metadata.
    Used to identify checkpoints.

    :param run_id: WANDB run id
    :type run_id: str
    :returns: Path to the run
    :rtype: str
    """
    WANDB_DIR = PROJECT_PATH / "wandb"
    # Search for run
    run_dirs = os.listdir(WANDB_DIR)
    run_dir = [f for f in run_dirs if f.endswith(run_id)][0]

    cfg = OmegaConf.load(WANDB_DIR / run_dir / "files" / "config.yaml")
    run_dir = cfg["env/paths/run_dir"]["value"]
    return run_dir.split("/")[-1]


def validate_inverse_folding(cfg: DictConfig):
    """Validate the inverse folding task config.

    If the user is using ``amino_acid_one_hot`` or ``sidechain_torsions``
    as a feature, this will raise an error as these features leak information
    about the target.

    :param cfg: Config
    :type cfg: DictConfig
    :raises ExperimentConfigurationError: If the user is using
        ``amino_acid_one_hot`` or sidechain_torsions as a feature.
    :raises ExperimentConfigurationError: If the user is using no scalar node
        features.
    """
    if "amino_acid_one_hot" in cfg.features.scalar_node_features:
        logger.warning(
            "You are launching an inverse folding experiment with amino_acid_one_hot as a feature. This will be removed."
        )
        cfg.features.scalar_node_features.remove("amino_acid_one_hot")
    if "sidechain_torsions" in cfg.features.scalar_node_features:
        raise ExperimentConfigurationError(
            "You are launching an inverse folding experiment with sidechain_torsions as a feature. This will be removed."
        )
    if cfg.features.scalar_node_features == []:
        raise ExperimentConfigurationError(
            "You are launching an inverse folding experiment with no scalar node features."
        )


def validate_early_stopping_config(cfg: DictConfig):
    """Validate the early stopping config.

    - If early stopping is conditioned on MSE, the mode must be ``min``.
    - If early stopping is conditioned on accuracy, the mode must be ``max``.
    - The early stopping metric must be in the list of metrics.

    :param cfg: Config
    :type cfg: DictConfig
    :raises ExperimentConfigurationError: If the user is using MSE as an early
        stopping metric and the mode is not ``min``.
    :raises ExperimentConfigurationError: If the user is using accuracy as an
        early stopping metric and the mode is not ``max``.
    :raises ExperimentConfigurationError: If the early stopping metric is not
        in the list of metrics.
    """
    if cfg.get("callbacks.early_stopping") is not None:
        monitor = cfg.callbacks.early_stopping.monitor
        mode = cfg.callbacks.early_stopping.mode
        monitor = monitor.split("/")[2]
        # Check mode is valid
        if monitor == "mse" and mode != "min":
            raise ExperimentConfigurationError(
                f"Using MSE as early stopping metric requires mode to be 'min' not {mode}."
            )
        elif monitor.startswith("accuracy") and mode != "max":
            raise ExperimentConfigurationError(
                f"Using accuracy as an early stopping metric requires mode to 'max' not {mode}."
            )
        # Check we are computing this metric

        if monitor not in cfg.metrics:
            raise ExperimentConfigurationError(
                f"Early stopping metric {monitor} not in metrics {cfg.metrics}"
            )
    else:
        logger.warning("You are not using early stopping.")


def validate_loss_config(cfg: DictConfig):
    """Validate the loss config.

    - The losses must be specified as a dictionary mapping output to loss
        function.
    - The supervision target must be in the list of outputs.

    :param cfg: Config
    :type cfg: DictConfig
    :raises ExperimentConfigurationError: If the losses are not specified as a
        dictionary.
    """
    # Assert losses are valid
    if not isinstance(cfg.task.losses, DictConfig):
        raise ExperimentConfigurationError(
            f"The `task.losses` argument must be a dictionary. ({cfg.task.losses})"
        )

    for k in cfg.task.losses.keys():
        assert k in OUTPUTS, f"Loss {k} not in {OUTPUTS}"

    # Assert each supervision target has a loss:
    for k in cfg.task.supervise_on:
        assert (
            k in cfg.task.losses.keys()
        ), f"Supervision target {k} has no loss"


def validate_supervision_config(cfg: DictConfig) -> DictConfig:
    if not isinstance(cfg.task.supervise_on, ListConfig):
        print(type(cfg.task.supervise_on))
        raise ExperimentConfigurationError(
            f"The `task.supervise_on` argument must be a list. ({cfg.task.supervise_on})"
        )

    for supervision_target in cfg.task.supervise_on:
        if supervision_target not in cfg.task.output:
            logger.warning(
                f"Supervision target: {supervision_target} not in outputs ({cfg.task.output}). Appending {supervision_target} to outputs."
            )
            cfg.task.output.append(supervision_target)
    return cfg


def validate_output_config(cfg: DictConfig) -> DictConfig:
    # TODO
    return cfg


def validate_classification_config(cfg: DictConfig):
    if (cfg.task.task == "classification") & (
        "graph_label" not in cfg.task.output
        and "node_label" not in cfg.task.output
    ):
        raise ExperimentConfigurationError(
            "The `task.output` argument must contain `graph_label` or node_label when \
                training a classification model."
        )
    if cfg.task.task not in [
        "inverse_folding",
        "backbone_dihedral_angle_prediction",
        "sequence_denoising",
        "structure_denoising",
        "sequence_structure_denoising",
        "edge_distance_prediction",
        "plddt_prediction",
        "torsional_denoising",
    ] and set(cfg.task.output).isdisjoint(
        {"graph_label", "node_label", "dataset_features"}
    ):
        raise ExperimentConfigurationError(
            f"Incorrect configuration for task: {cfg.task.task}, \
                Output: {cfg.task.output}"
        )


def validate_cuda(cfg: DictConfig) -> DictConfig:
    # Make sure cuda config is correct
    # if cfg.trainer.devices <= 1:
    # cfg.trainer.strategy = None
    logger.debug(f"CUDA available: {torch.cuda.is_available()}")
    logger.debug(f"Requested GPUs: {cfg.trainer.get('devices')}.")
    if isinstance(cfg.trainer.get("devices"), int):
        cfg.trainer.devices = min(
            torch.cuda.device_count(), cfg.trainer.devices
        )
        logger.debug(f"GPU count set to: {cfg.trainer.devices}")

    requesting_multiple_device_indices = (
        isinstance(cfg.trainer.get("devices"), list)
        and len(cfg.trainer.get("devices")) > 1
    )
    requesting_multiple_devices = (
        isinstance(cfg.trainer.get("devices"), int)
        and cfg.trainer.get("devices") > 1
    )
    if (
        requesting_multiple_device_indices or requesting_multiple_devices
    ) and cfg.get("test"):
        logger.warning(
            "You are running a test with multiple GPUs. This is not recommended and testing will be disabled on this run."
        )
        del cfg.test
    return cfg


def validate_egnn_config(cfg: DictConfig):
    pass


def validate_gnn_config(cfg: DictConfig):
    # sourcery skip: merge-nested-ifs, remove-empty-nested-block, remove-redundant-if
    # Set number of edge types
    if cfg.features.edge_types:
        cfg.encoder.edge_types = len(cfg.features.edge_types)
    if cfg.features.scalar_edge_features == ["edge_distance"]:
        cfg.encoder.edge_weight = True
    if cfg.features.scalar_edge_features:
        cfg.encoder.edge_features = True

    if cfg.features.vector_node_features:
        logger.warning("You are using vector-valued node features with a GNN.")
    if cfg.features.vector_edge_features:
        logger.warning("You are using vector-valued edge features with a GNN.")


def validate_gcpnet_config(cfg: DictConfig):
    """Validate the GCPNet config.

    - Injects config values for requested features into corresponding `features`
        object

    :param cfg: Config
    :type cfg: DictConfig
    """
    # inject config values for requested features into corresponding `features`
    # object
    for feature in cfg.encoder.features:
        cfg.features[feature] = cfg.encoder.features[feature]


def validate_multiprocessing(cfg: DictConfig) -> DictConfig:
    # Make sure num_workers isn't too high.
    core_count = psutil.cpu_count(logical=False)
    if cfg.num_workers > core_count:
        logger.debug(
            (
                f"Requested CPUs: {cfg.num_workers}. Avialable CPUs (physical): {core_count}. "
                "Requested CPU count will therefore be set to maximum number of"
                "available physical cores. NOTE: It is recommended to use N-1"
                "cores or less to avoid memory flush overheads."
            )
        )
        cfg.num_workers = core_count
    return cfg


def validate_backbone_dihedral_angle_prediction(cfg: DictConfig):
    if "dihedrals" in cfg.features.scalar_node_features:
        raise ExperimentConfigurationError(
            "You are launching a masked attribute dihedral prediction experiment with dihedrals as a feature. This will be removed."
        )
    if cfg.features.scalar_node_features == []:
        raise ExperimentConfigurationError(
            "You are launching a  masked attribute dihedral prediction experiment with no scalar node features."
        )


def validate_config(cfg: DictConfig) -> DictConfig:
    """
    Validate the config and make any necessary alterations to the parameters.
    """
    if cfg.name is None:
        raise TypeError("The `run_name` argument is mandatory.")
    validate_classification_config(cfg)
    cfg = validate_output_config(cfg)

    cfg = validate_supervision_config(cfg)
    validate_loss_config(cfg)

    if (
        cfg.encoder._target_
        == "proteinworkshop.model.graph_encoders.egnn.EGNNModel"
    ):
        validate_egnn_config(cfg)
    elif (
        cfg.encoder._target_
        == "proteinworkshop.models.graph_encoders.gnn.GNNModel"
    ):
        validate_gnn_config(cfg)
    elif (
        cfg.encoder._target_
        == "proteinworkshop.models.graph_encoders.gcpnet.GCPNetModel"
    ):
        validate_gcpnet_config(cfg)

    # Validate task
    if cfg.task.task == "inverse_folding":
        validate_inverse_folding(cfg)
    elif cfg.task.task == "backbone_dihedral_angle_prediction":
        validate_backbone_dihedral_angle_prediction(cfg)

    cfg = validate_multiprocessing(cfg)
    cfg = validate_cuda(cfg)

    validate_early_stopping_config(cfg)
    return cfg
