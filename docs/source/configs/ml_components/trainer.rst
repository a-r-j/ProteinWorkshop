Trainer
================

These configs configure PyTorch Lightning ``Trainer`` objects.

.. seealso::
   :py:mod:`lightning`

.. note::
    To use an alternative trainer config, use a command like:

    .. code-block:: bash

        python proteinworkshop/train.py trainer=<TRAINER_NAME> encoder=gvp dataset=cath task=inverse_folding

    where ``<TRAINER_NAME>`` is the name of the trainer config.


Default (``default``)
----------------------

.. code-block:: bash

   # Example usage
   python proteinworkshop/train.py ... trainer.max_epochs=1000

.. literalinclude:: ../../../../configs/trainer/default.yaml
   :language: yaml


GPU (``gpu``)
----------------

.. literalinclude:: ../../../../configs/trainer/gpu.yaml
   :language: yaml


CPU (``cpu``)
----------------

.. literalinclude:: ../../../../configs/trainer/cpu.yaml
   :language: yaml


DDP (``ddp``)
----------------

.. literalinclude:: ../../../../configs/trainer/ddp.yaml
   :language: yaml


DDP Sim (``ddp_sim``)
----------------------

.. literalinclude:: ../../../../configs/trainer/ddp_sim.yaml
   :language: yaml


MPS (``mps``)
----------------

.. literalinclude:: ../../../../configs/trainer/mps.yaml
   :language: yaml
