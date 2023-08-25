Schedulers
================

These configs configure various learning rate schedulers.

.. note::
    To use an alternative scheduler, use a command like:

    .. code-block:: bash

        python proteinworkshop/train.py scheduler=<SCHEDULER_NAME> encoder=gvp dataset=cath task=inverse_folding

    where ``<SCHEDULER_NAME>`` is the name of the scheduler config.


ReduceLROnPlateau (``plateau``)
--------------------------------------------

.. code-block:: bash

    # Example usage:
    python proteinworkshop/train.py ... scheduler=plateau scheduler.scheduler.patience=10


.. literalinclude:: ../../../../configs/scheduler/plateau.yaml
   :language: yaml


LinearWarmupCosineDecay (``linear_warmup_cosine_decay``)
------------------------------------------------------------------

.. code-block:: bash

    # Example usage:
    python proteinworkshop/train.py ... scheduler=linear_warmup_cosine_decay scheduler.scheduler.warmup_epochs=10

.. literalinclude:: ../../../../configs/scheduler/linear_warmup_cosine_decay.yaml
   :language: yaml
