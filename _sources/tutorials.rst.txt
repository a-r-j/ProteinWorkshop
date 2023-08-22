Tutorials
---------------------

.. mdinclude:: ../../README.md
    :start-line: 68
    :end-line: 76


Training a New Model
=====================

1. Using provided options via the CLI:

.. code-block:: bash

    python src/train.py dataset=DATASET model=MODEL task=TASK features=FEATURES

To override hparams, you can either edit the relevant :doc:`/configs` files directly or via the CLI using Hydra syntax:

.. code-block::

    python src/train.py ... optimiser.optimizer.lr=0.001 dataset.datamodule.batch_size=32

2. In a jupyter notebook:

    See: :doc:`/notebooks/Training a New Model`


Evaluating a pre-trained Model
===============================