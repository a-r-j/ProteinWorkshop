protein_workshop.models
-------------------------
To switch between different encoder architectures, simply change the ``encoder`` argument in the launch command. For example:

.. code-block:: bash

    workshop train encoder=<encoder_name> dataset=cath task=inverse_folding trainer=cpu
    # or
    python proteinworkshop/train.py encoder=<encoder_name> dataset=cath task=inverse_folding trainer=cpu # or trainer=gpu

Where ``<encoder_name>`` is given by bracketed name in the listing below. For example, the encoder name for SchNet is ``schnet``.


.. note::
    To change encoder hyperparameters, either

    1. Edit the config file directly, or
    2. Provide commands in the form:

    .. code-block:: bash

        workshop train encoder=<encoder_name> encoder.num_layer=3 encoder.readout=mean dataset=cath task=inverse_folding trainer=cpu
        # or
        python proteinworkshop/train.py encoder=<encoder_name> encoder.num_layer=3 encoder.readout=mean dataset=cath task=inverse_folding trainer=cpu # or trainer=gpu

The following structural encoders are currently supported:


.. contents:: Contents
    :local:
    :class: this-will-duplicate-information-and-it-is-still-useful-here


Base Classes
=============================
.. automodule:: proteinworkshop.models.base
    :members:
    :undoc-members:
    :show-inheritance:


Invariant Encoders
=============================

SchNet (``schnet``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: proteinworkshop.models.graph_encoders.schnet
    :members:
    :undoc-members:
    :show-inheritance:


DimeNet++ (``dimenet_plus_plus``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: proteinworkshop.models.graph_encoders.dimenetpp
    :members:
    :undoc-members:
    :show-inheritance:


GearNet (``gear_net``, ``gear_net_edge``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: proteinworkshop.models.graph_encoders.gear_net
    :members:
    :undoc-members:
    :show-inheritance:

CDConv (``cdconv``)
^^^^^^^^^^^^^^^^^^^^^
.. automodule:: proteinworkshop.models.graph_encoders.cdconv
    :members:
    :undoc-members:
    :show-inheritance:

Vector-Equivariant Encoders
=============================

EGNN (``egnn``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: proteinworkshop.models.graph_encoders.egnn
    :members:
    :undoc-members:
    :show-inheritance:


GVP (``gvp``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: proteinworkshop.models.graph_encoders.gvp
    :members:
    :undoc-members:
    :show-inheritance:


GCPNet (``gcpnet``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: proteinworkshop.models.graph_encoders.gcpnet
    :members:
    :undoc-members:
    :show-inheritance:


Tensor-Equivariant Encoders
=============================

TFN (``tfn``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: proteinworkshop.models.graph_encoders.tfn
    :members:
    :undoc-members:
    :show-inheritance:


Multi-Atomic Cluster Expansion (``mace``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: proteinworkshop.models.graph_encoders.mace
    :members:
    :undoc-members:
    :show-inheritance:


Sequence-Based Encoders
=============================

Evolutionary Scale Modeling (``esm``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: proteinworkshop.models.graph_encoders.esm_embeddings
    :members:
    :undoc-members:
    :show-inheritance:


Decoders
=============================

.. automodule:: proteinworkshop.models.decoders.mlp_decoder
    :members:
    :undoc-members:
    :show-inheritance:
