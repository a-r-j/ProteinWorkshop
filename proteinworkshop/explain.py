import collections
import functools
import os
import pathlib
import sys
from typing import List, Union

import hydra
import lightning as L
import omegaconf
import torch
from captum.attr import IntegratedGradients
from graphein.protein.tensor.data import ProteinBatch
from graphein.protein.tensor.io import to_dataframe
from loguru import logger as log
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
from tqdm import tqdm

from proteinworkshop import (
    constants,
    register_custom_omegaconf_resolvers,
    utils,
)
from proteinworkshop.configs import config
from proteinworkshop.models.base import BenchMarkModel


def explain(cfg: omegaconf.DictConfig):
    assert cfg.ckpt_path, "No checkpoint path provided."
    assert (
        cfg.output_dir
    ), "No output directory for attributed PDB files provided."

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
        # log.info(f"Loading encoder weights: {encoder_weights}")
        err = model.encoder.load_state_dict(encoder_weights, strict=False)
        log.warning(f"Error loading encoder weights: {err}")

    if cfg.finetune.decoder.load_weights:
        decoder_weights = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("decoder"):
                decoder_weights[k.replace("decoder.", "")] = v
        # log.info(f"Loading decoder weights: {decoder_weights}")
        err = model.decoder.load_state_dict(decoder_weights, strict=False)
        log.warning(f"Error loading decoder weights: {err}")
    else:
        model.decoder = None

    OUTPUT = cfg.explain.output
    output_dir = pathlib.Path(cfg.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Wrap forward function and pass to captum
    def _forward_wrapper(
        model: L.LightningModule,
        node_feats: torch.Tensor,
        batch: Union[Batch, ProteinBatch],
        output: str,
    ):
        """Wrapper function around forward pass.

        Sets node features to the provided node features and returns the
        model output for the specified output.

        The node feature update is necessary to set the interpolated features
        from IntegratedGradients.
        """
        batch.edge_index = batch.edge_index[:2, :]  # Handle GearNet edge case
        batch.x = node_feats  # Update node features
        return model.forward(batch)[output]

    fwd = functools.partial(_forward_wrapper, model.cuda(), output=OUTPUT)
    ig = IntegratedGradients(fwd)

    dataloaders = {}
    if "train" in cfg.explain.split:
        dataloaders["train"] = datamodule.train_dataloader()
    if "val" in cfg.explain.split:
        dataloaders["val"] = datamodule.val_dataloader()
    if "test" in cfg.explain.split:
        dataloaders["test"] = datamodule.test_dataloader()

    # Iterate over batches and perform attribution
    for split, dataloader in dataloaders.items():
        log.info(f"Performing attribution for split: {split}")

        for batch in tqdm(dataloader):
            batch = batch.cuda()
            batch = model.featurise(batch)
            node_features = batch.x
            attribution = ig.attribute(
                node_features,
                baselines=torch.ones_like(batch.x),
                additional_forward_args=batch,
                target=model.get_labels(batch)[OUTPUT].long(),
                internal_batch_size=cfg.dataset.datamodule.batch_size,
                n_steps=cfg.explain.n_steps,
            )
            attribution = attribution.sum(-1)

            # Unbatch and write each protein to disk
            batch = batch.cpu()
            batch_items = batch.to_data_list()
            attribution_scores = unbatch(attribution, batch.batch)
            for elem, score in tqdm(zip(batch_items, attribution_scores)):
                # Scale score between 0-100
                score = score - score.min()
                score = score / score.max()
                score = score * 100

                df = to_dataframe(
                    x=elem.coords,
                    residue_types=elem.residues,
                    chains=elem.chains,
                    insertions=None,
                    b_factors=score.cpu(),  # Write attribution score in B factor column
                    occupancy=None,
                    charge=None,
                    alt_loc=None,
                    segment_id=None,
                    biopandas=True,
                )
                output_path = output_dir / f"{elem.id}.pdb"
                df.to_pdb(str(output_path))


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    version_base="1.3",
    config_path=str(constants.HYDRA_CONFIG_PATH),
    config_name="explain",
)
def _main(cfg: omegaconf.DictConfig) -> None:
    """Load and validate the hydra config."""
    utils.extras(cfg)
    cfg = config.validate_config(cfg)
    explain(cfg)


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
