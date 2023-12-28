Tutorials
---------------------

.. mdinclude:: ../../README.md
    :start-line: 118
    :end-line: 127


Training a New Model
=====================

1. Using provided options via the CLI:

.. code-block:: bash

    workshop train dataset=DATASET encoder=MODEL task=TASK features=FEATURES trainer=cpu env.paths.data=where/you/want/data/
    # or 
    python proteinworkshop/train.py dataset=DATASET encoder=MODEL task=TASK features=FEATURES trainer=cpu # or trainer=gpu

To override hparams, you can either edit the relevant :doc:`/configs/` files directly or via the CLI using Hydra syntax:

.. code-block:: bash

    workshop train ... optimiser.optimizer.lr=0.001 dataset.datamodule.batch_size=32
    # or
    python proteinworkshop/train.py ... optimiser.optimizer.lr=0.001 dataset.datamodule.batch_size=32

2. In a jupyter notebook:

    See: :doc:`/notebooks/Training a New Model`


Evaluating a pre-trained Model
===============================

1. Define a new model class

.. code-block::

    from typing import Union, Set, Dict

    import torch
    import torch.nn as nn
    from torch_geometric.data import Batch
    from graphein.protein.tensor.data import ProteinBatch
    from proteinworkshop.models.utils import get_aggregation
    from jaxtyping import jaxtyped
    from beartype import beartype as typechecker


    class IdentityModel(nn.Module):
        def __init__(self, readout: str = "sum"):
            super().__init__()
            self.readout = get_aggregation(readout)

        @property
        def required_batch_attributes(self) -> Set[str]:
            """This property describes the required attributes of the input batch."""
            return {"x", "batch"}

        @jaxtyped(typechecker=typechecker)
        def forward(self, batch: Union[Batch, ProteinBatch]) -> Dict[str, torch.Tensor]:
            """
            This method does the forward pass of the model.

            It should take in a batch of data and return a dictionary of outputs.
            """
            output = {
                "node_embedding": batch.x,
                "graph_embedding": self.readout(batch.x, batch.batch)
            }
            return output

2. Load the model weights

.. code-block::

    encoder = IdentityModel()
    encoder.load_state_dict(torch.load("path/to/model.pt"))

3. Configure the task

.. code-block:: python

    # Misc. tools
    import os

    # Hydra tools
    import hydra

    from hydra.compose import GlobalHydra
    from hydra.core.hydra_config import HydraConfig

    from proteinworkshop.constants import HYDRA_CONFIG_PATH
    from proteinworkshop.utils.notebook import init_hydra_singleton

    version_base = "1.2"  # Note: Need to update whenever Hydra is upgraded
    init_hydra_singleton(reload=True, version_base=version_base)

    path = HYDRA_CONFIG_PATH
    rel_path = os.path.relpath(path, start=".")

    GlobalHydra.instance().clear()
    hydra.initialize(rel_path, version_base=version_base)

    cfg = hydra.compose(config_name="train", overrides=["encoder=schnet", "task=inverse_folding", "dataset=afdb_swissprot_v4", "features=ca_angles", "+aux_task=none"], return_hydra_config=True)

    # Note: Customize as needed e.g., when running a sweep
    cfg.hydra.job.num = 0
    cfg.hydra.job.id = 0
    cfg.hydra.hydra_help.hydra_help = False
    cfg.hydra.runtime.output_dir = "outputs"

    HydraConfig.instance().set_config(cfg)

4. Run the model

.. code-block::

    from proteinworkshop.train import train_model

    train_model(cfg, model)


            