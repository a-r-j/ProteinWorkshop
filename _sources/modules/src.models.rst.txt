protein_workshop.models
-------------------------
To switch between different encoder architectures, simply change the ``encoder`` argument in the launch command. For example:

.. code-block:: bash

    python src/train.py encoder=<encoder_name> dataset=cath task=inverse_folding

Where ``<encoder_name>`` is given by bracketed name in the listing below. For example, the encoder name for SchNet is ``schnet``.


.. note::
    To change encoder hyperparameters, either

    1. Edit the config file directly, or
    2. Provide commands in the form:

    .. code-block:: bash

        python src/train.py encoder=<encoder_name> encoder.num_layer=3 encoder.readout=mean dataset=cath task=inverse_folding

The following structural encoders are currently supported:


Base Classes
=============================
.. automodule:: src.models.base
    :members:
    :undoc-members:
    :show-inheritance:


Invariant Encoders
=============================

SchNet (``schnet``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: src.models.graph_encoders.schnet
    :members:
    :undoc-members:
    :show-inheritance:


DimeNet++ (``dimenet_plus_plus``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: src.models.graph_encoders.dimenetpp
    :members:
    :undoc-members:
    :show-inheritance:


GearNet (``gear_net``, ``gear_net_edge``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: src.models.graph_encoders.gear_net
    :members:
    :undoc-members:
    :show-inheritance:


Vector-Equivariant Encoders
=============================

EGNN (``egnn``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: src.models.graph_encoders.egnn
    :members:
    :undoc-members:
    :show-inheritance:


GVP (``gvp``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: src.models.graph_encoders.gvp
    :members:
    :undoc-members:
    :show-inheritance:


GCPNet (``gcpnet``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: src.models.graph_encoders.gcpnet
    :members:
    :undoc-members:
    :show-inheritance:


Tensor-Equivariant Encoders
=============================

TFN (``tfn``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: src.models.graph_encoders.gcpnet
    :members:
    :undoc-members:
    :show-inheritance:


Multi-Atomic Cluster Expansion (``mace``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: src.models.graph_encoders.gcpnet
    :members:
    :undoc-members:
    :show-inheritance:


Decoders
=============================

.. automodule:: src.models.decoders.mlp_decoder
    :members:
    :undoc-members:
    :show-inheritance: