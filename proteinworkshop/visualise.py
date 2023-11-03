import collections
import sys

import hydra
import lightning as L
import omegaconf
import torch
import numpy as np
import umap
import umap.plot
import matplotlib.pyplot as plt
from beartype.typing import Any, Dict, List, Optional
from loguru import logger as log
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from torchmetrics.functional.clustering import dunn_index
from tqdm import tqdm

from proteinworkshop import (
    constants,
    register_custom_omegaconf_resolvers,
    utils,
)
from proteinworkshop.configs import config
from proteinworkshop.models.base import BenchMarkModel


def draw_simple_ellipse(
    position: np.ndarray,
    width: float,
    height: float,
    angle: float,
    ax: Optional[plt.Axes] = None,
    from_size: float = 0.1,
    to_size: float = 0.5,
    n_ellipses: int = 3,
    alpha: float = 0.1,
    color: Optional[str] = None,
    **kwargs: Dict[str, Any],
):
    ax = ax or plt.gca()
    angle = (angle / np.pi) * 180
    width, height = np.sqrt(width), np.sqrt(height)
    for nsig in np.linspace(from_size, to_size, n_ellipses):
        ax.add_patch(
            Ellipse(
                position,
                nsig * width,
                nsig * height,
                angle,
                alpha=alpha,
                lw=0,
                color=color,
                **kwargs
            )
        )


def visualise(cfg: omegaconf.DictConfig):
    assert cfg.ckpt_path, "A checkpoint path must be provided."
    assert cfg.plot_filepath, "A plot name must be provided."
    if cfg.use_cuda_device and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    cfg = config.validate_config(cfg)

    L.seed_everything(cfg.seed)

    log.info("Instantiating datamodule:... ")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(
        cfg.dataset.datamodule
    )

    log.info("Instantiating model:... ")
    model: L.LightningModule = BenchMarkModel(cfg)

    # Initialize lazy layers for parameter counts
    # This is also required for the model to be able to load weights
    # Otherwise lazy layers will have their parameters reset
    # https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin
    log.info("Initializing lazy layers...")
    with torch.no_grad():
        datamodule.setup()  # type: ignore
        batch = next(iter(datamodule.val_dataloader()))
        log.info(f"Unfeaturized batch: {batch}")
        batch = model.featurise(batch)
        log.info(f"Featurized batch: {batch}")
        out = model.forward(batch)
        log.info(f"Model output: {out}")
        del batch, out

    # Load weights
    # We only want to load weights
    log.info(f"Loading weights from checkpoint {cfg.ckpt_path}...")
    state_dict = torch.load(cfg.ckpt_path)["state_dict"]

    if cfg.finetune.encoder.load_weights:
        encoder_weights = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("encoder"):
                encoder_weights[k.replace("encoder.", "")] = v
        log.info(f"Loading encoder weights: {encoder_weights}")
        err = model.encoder.load_state_dict(encoder_weights, strict=False)
        log.warning(f"Error loading encoder weights: {err}")

    if cfg.finetune.decoder.load_weights:
        decoder_weights = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("decoder"):
                decoder_weights[k.replace("decoder.", "")] = v
        log.info(f"Loading decoder weights: {decoder_weights}")
        err = model.decoder.load_state_dict(decoder_weights, strict=False)
        log.warning(f"Error loading decoder weights: {err}")
    else:
        model.decoder = None

    log.info("Freezing encoder!")
    for param in model.encoder.parameters():
        param.requires_grad = False

    log.info("Freezing decoder!")
    model.decoder = None

    # Select CUDA computation device, otherwise default to CPU
    if cfg.use_cuda_device:
        device = torch.device(f"cuda:{cfg.cuda_device_index}")
        model = model.to(device)
    else:
        device = torch.device("cpu")

    # Setup datamodule
    datamodule.setup()

    # Get class map
    class_map_available = hasattr(datamodule, "parse_class_map")
    if class_map_available:
        # NOTE: e.g., for fold classification, by default the `class_map`
        # is a mapping from fold name to fold index, so we need to invert it
        class_map = {v: k for k, v in datamodule.parse_class_map().items()}

    # Iterate over batches and perform visualisation
    dataloaders = {}
    if "train" in cfg.visualise.split:
        dataloaders["train"] = datamodule.train_dataloader()
    if "val" in cfg.visualise.split:
        dataloaders["val"] = datamodule.val_dataloader()
    if "test" in cfg.visualise.split:
        dataloaders["test"] = datamodule.test_dataloader()

    collection = []
    with torch.inference_mode():
        for split, dataloader in dataloaders.items():
            log.info(f"Performing visualisation for split: {split}")

            for batch in tqdm(dataloader):
                if cfg.visualise.label in batch:
                    labels = batch[cfg.visualise.label].tolist()
                elif "graph_y" in batch:
                    labels = batch.graph_y.tolist()
                    if class_map_available:
                        labels = [class_map[label] for label in labels]
                else:
                    labels = [id.split("_")[0] for id in batch.id]
                batch = batch.to(device)
                batch = model.featuriser(batch)
                out = model.forward(batch)
                graph_embeddings = out["graph_embedding"]
                node_embeddings = graph_embeddings.tolist()
                collection.append({"embedding": node_embeddings, "labels": labels})

    # Derive clustering of embeddings using UMAP
    assert len(collection) > 0 and len(collection[0]["embedding"]) > 0, "At least one batch of embeddings must be present to plot with UMAP."
    clustering_data = np.array([batch for x in collection for batch in x["embedding"]])
    clustering_labels = np.array([label for x in collection for label in x["labels"]])
    umap_embeddings = umap.UMAP(random_state=cfg.seed).fit_transform(clustering_data)

    graph_label_available = cfg.visualise.label in batch
    if graph_label_available:
        clustering_label_indices = np.array(list(range(len(clustering_labels))))
    elif class_map_available:
        orig_class_map = datamodule.parse_class_map()
        clustering_label_indices = np.array([orig_class_map[label] for label in clustering_labels])
    else:
        clustering_label_indices = clustering_labels

    # Report Dunn index of clustering
    dunn_index_data = torch.from_numpy(umap_embeddings)
    dunn_index_labels = torch.from_numpy(clustering_label_indices)
    clustering_dunn_index = dunn_index(dunn_index_data, dunn_index_labels)
    log.info(f"Dunn index of clustering: {clustering_dunn_index.item()}")

    # Plot UMAP clustering
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    num_unique_labels = len(class_map) if class_map_available else len(np.unique(clustering_label_indices))
    colors = plt.get_cmap("Spectral")(np.linspace(0, 1, num_unique_labels))

    ax.scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        c=clustering_label_indices,
        cmap="Spectral",
        s=3
    )

    # Create the legend with only the 20 most common labels
    if class_map_available:
        label_counts = {}
        for label in clustering_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        top_20_labels = sorted(label_counts, key=label_counts.get, reverse=True)[:20]
        legend_handles = [Line2D([0], [0], color=colors[orig_class_map[label]], lw=3, label=label) for label in top_20_labels]
        plt.legend(handles=legend_handles)

    plt.xlabel("")  # Remove x-axis label
    plt.ylabel("")  # Remove y-axis label
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    plt.savefig(cfg.plot_filepath)


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    version_base="1.3",
    config_path=str(constants.HYDRA_CONFIG_PATH),
    config_name="visualise",
)
def _main(cfg: omegaconf.DictConfig) -> None:
    """Load and validate the hydra config."""
    utils.extras(cfg)
    cfg = config.validate_config(cfg)
    visualise(cfg)


def _script_main(args: List[str]) -> None:
    """
    Provides an entry point for the script dispatcher.
    Sets the sys.argv to the provided args and calls the main train function.
    """
    sys.argv = args
    register_custom_omegaconf_resolvers()
    _main()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    register_custom_omegaconf_resolvers()
    _main()  # type: ignore
