Optimiser
================

These configs (generally) configure PyTorch ``Optimizer`` objects.

.. note::
    To use an alternative optimiser config, use a command like:

    .. code-block:: bash

        workshop train optimiser=<OPTIMISER_NAME> encoder=gvp dataset=cath task=inverse_folding trainer=cpu
        # or
        python proteinworkshop/train.py optimiser=<OPTIMISER_NAME> encoder=gvp dataset=cath task=inverse_folding trainer=cpu # or trainer=gpu

    where ``<OPTIMISER_NAME>`` is the name of the optimiser config.

.. note::
    To change the learning rate, use a command like:

    .. code-block:: bash

        workshop train optimizer.lr=0.0001 encoder=gvp dataset=cath task=inverse_folding trainer=cpu
        # or
        python proteinworkshop/train.py optimizer.lr=0.0001 encoder=gvp dataset=cath task=inverse_folding trainer=cpu # or trainer=gpu

    where ``0.0001`` is the new learning rate.


ADAM (``adam``)
----------------------

.. code-block:: bash

    # Example usage:
    python proteinworkshop/train.py ... optimiser=adam optimiser.optimizer.lr=0.0001 ...

.. literalinclude:: ../../../../proteinworkshop/config/optimiser/adam.yaml
   :language: yaml


ADAM-W (``adamw``)
----------------------

.. code-block:: bash

    # Example usage:
    python proteinworkshop/train.py ... optimiser=adamw optimiser.optimizer.lr=0.0001 ...


.. literalinclude:: ../../../../proteinworkshop/config/optimiser/adamw.yaml
   :language: yaml


Lion (``lion``)
----------------------

.. code-block:: bash

    # Example usage:
    python proteinworkshop/train.py ... optimiser=lion optimiser.optimizer.lr=0.0001 ...

.. literalinclude:: ../../../../proteinworkshop/config/optimiser/lion.yaml
   :language: yaml
