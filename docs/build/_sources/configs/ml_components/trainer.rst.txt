Trainer
================

These configs configure PyTorch Lightning ``Trainer`` objects.

.. seealso::
   :py:mod:`lightning`

.. note::
    To use an alternative trainer config, use a command like:

    .. code-block:: bash

         workshop train trainer=<TRAINER_NAME> encoder=gvp dataset=cath task=inverse_folding
         # or
         python proteinworkshop/train.py trainer=<TRAINER_NAME> encoder=gvp dataset=cath task=inverse_folding

    where ``<TRAINER_NAME>`` is the name of the trainer config.


Default (``default``)
----------------------

.. code-block:: bash

   # Example usage
   python proteinworkshop/train.py ... trainer.max_epochs=1000

.. literalinclude:: ../../../../proteinworkshop/config/trainer/default.yaml
   :language: yaml


GPU (``gpu``)
----------------

.. literalinclude:: ../../../../proteinworkshop/config/trainer/gpu.yaml
   :language: yaml


CPU (``cpu``)
----------------

.. literalinclude:: ../../../../proteinworkshop/config/trainer/cpu.yaml
   :language: yaml


DDP (``ddp``)
----------------

.. literalinclude:: ../../../../proteinworkshop/config/trainer/ddp.yaml
   :language: yaml


DDP Sim (``ddp_sim``)
----------------------

.. literalinclude:: ../../../../proteinworkshop/config/trainer/ddp_sim.yaml
   :language: yaml


MPS (``mps``)
----------------

.. literalinclude:: ../../../../proteinworkshop/config/trainer/mps.yaml
   :language: yaml
