import collections

import chromadb
import hydra
import lightning as L
import omegaconf
import torch
from chromadb.config import Settings
from loguru import logger as log
from tqdm import tqdm

from proteinworkshop import (
    constants,
    register_custom_omegaconf_resolvers,
    utils,
)
from proteinworkshop.models.base import BenchMarkModel


def embed(cfg: omegaconf.DictConfig):
    assert cfg.ckpt_path, "No checkpoint path provided."
    assert cfg.collection_name, "No collection name provided."

    L.seed_everything(cfg.seed)

    log.info("Instantiating datamodule:... ")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(
        cfg.dataset.datamodule
    )

    log.info("Instantiating model:... ")
    model: L.LightningModule = BenchMarkModel(cfg)

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

    # Initialise chromadb
    chroma_client = chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=".chromadb",  # Optional, defaults to .chromadb/ in the current directory
            anonymized_telemetry=False,
        )
    )
    chroma_client.persist()

    collection = chroma_client.create_collection(name=cfg.collection_name)

    for i, batch in enumerate(tqdm(datamodule.train_dataloader())):
        ids = batch.id
        batch = model.featuriser(batch)
        out = model.forward(batch)
        # node_embeddings = out["node_embedding"] # TODO: add node embeddings
        graph_embeddings = out["graph_embedding"]
        node_embeddings = graph_embeddings.tolist()
        collection.add(embeddings=node_embeddings, ids=ids)
    chroma_client.persist()


@hydra.main(
    version_base="1.3",
    config_path=str(constants.HYDRA_CONFIG_PATH),
    config_name="embed.yaml",
)
def _main(cfg: omegaconf.DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras.extras(cfg)

    embed(cfg)


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    _main()
