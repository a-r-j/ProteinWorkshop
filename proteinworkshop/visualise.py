import collections

import hydra
import lightning as L
import omegaconf
import torch
import numpy as np
import umap
import umap.plot
from loguru import logger as log
from tqdm import tqdm

from proteinworkshop import (
    constants,
    register_custom_omegaconf_resolvers,
    utils,
)
from proteinworkshop.configs import config
from proteinworkshop.models.base import BenchMarkModel


def visualise(cfg: omegaconf.DictConfig):
    assert cfg.ckpt_path, "No checkpoint path provided."
    assert cfg.plot_filepath, "No plot name provided."

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
    model.decoder = None  # TODO make this controllable by config
    # for param in model.decoder.parameters():
    #    param.requires_grad = False

    # Setup datamodule
    datamodule.setup()

    collection = []
    with torch.inference_mode():
        for batch in tqdm(datamodule.train_dataloader()):
            ids = [id.split("_")[0] for id in batch.id] # e.g., acquire PDB codes as IDs/labels
            batch = model.featuriser(batch)
            out = model.forward(batch)
            graph_embeddings = out["graph_embedding"]
            node_embeddings = graph_embeddings.tolist()
            collection.append({"embedding": node_embeddings, "ids": ids})
            break

    # Plot embeddings using UMAP
    assert len(collection) > 0 and len(collection[0]["embedding"]) > 0, "At least one batch of embeddings must be present to plot with UMAP."
    emb_dim = len(collection[0]["embedding"][0])
    umap_data = np.array([x["embedding"] for x in collection]).reshape(-1, emb_dim)
    umap_labels = np.array([x["ids"] for x in collection]).reshape(-1)
    mapper = umap.UMAP().fit(umap_data)
    # umap_axes = umap.plot.points(mapper, labels=umap_labels)
    umap_axes = umap.plot.points(mapper)
    umap_figure = umap_axes.figure
    umap_figure.savefig(cfg.plot_filepath)


@hydra.main(
    version_base="1.3",
    config_path=str(constants.HYDRA_CONFIG_PATH),
    config_name="visualise.yaml",
)
def _main(cfg: omegaconf.DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    visualise(cfg)


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    _main()
