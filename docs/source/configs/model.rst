Models
------

.. image:: ../_static/box_models.png
  :alt: Summary of model types
  :align: center
  :width: 400

To switch between different encoder architectures, simply change the ``encoder`` argument in the launch command. For example:

.. code-block:: bash

    workshop train encoder=<ENCODER_NAME> dataset=cath task=inverse_folding trainer=cpu
    # or
    python proteinworkshop/train.py encoder=<ENCODER_NAME> dataset=cath task=inverse_folding trainer=cpu # or trainer=gpu

Where ``<ENCODER_NAME>`` is given by bracketed name in the listing below. For example, the encoder name for SchNet is ``schnet``.


.. note::
    To change encoder hyperparameters, either

    1. Edit the config file directly, or
    2. Provide commands in the form:

    .. code-block:: bash

        workshop train encoder=<ENCODER_NAME> encoder.num_layer=3 encoder.readout=mean dataset=cath task=inverse_folding trainer=cpu
        # or
        python proteinworkshop/train.py encoder=<ENCODER_NAME> encoder.num_layer=3 encoder.readout=mean dataset=cath task=inverse_folding trainer=cpu # or trainer=gpu


Invariant Encoders
=============================

.. mdinclude:: ../../../README.md
    :start-line: 319
    :end-line: 326

:py:class:`SchNet <proteinworkshop.models.graph_encoders.schnet.SchNetModel>` (``schnet``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SchNet is one of the most popular and simplest instantiation of E(3) invariant message passing GNNs. SchNet constructs messages through element-wise multiplication of scalar features modulated by a radial filter conditioned on the pairwise distance :math:`\Vert \vec{\vx}_{ij} \Vert`` between two neighbours.
Scalar features are updated from iteration :math:`t`` to :math:`t+1` via:

.. math::
    \begin{align}
        \vs_i^{(t+1)} & \defeq \vs_i^{(t)} + \sum_{j \in \mathcal{N}_i} f_1 \left( \vs_j^{(t)} , \ \Vert \vec{\vx}_{ij} \Vert \right) \label{eq:schnet}
    \end{align}

.. literalinclude:: ../../../proteinworkshop/config/encoder/schnet.yaml
    :language: yaml
    :caption: config/encoder/schnet.yaml


:py:class:`DimeNet++ <proteinworkshop.models.graph_encoders.dimenetpp.DimeNetPPModel>` (``dimenet_plus_plus``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DimeNet is an E(3) invariant GNN which uses both distances :math:`\Vert \vec{\vx}_{ij} \Vert` and angles :math:`\vec{\vx}_{ij} \cdot \vec{\vx}_{ik}` to perform message passing among triplets, as follows:


.. math::
    \begin{align}
        \vs_i^{(t+1)} & \defeq \sum_{j \in \mathcal{N}_i} f_1 \Big( \vs_i^{(t)} , \ \vs_j^{(t)} , \sum_{k \in \mathcal{N}_i \backslash \{j\}} f_2 \left( \vs_j^{(t)} , \ \vs_k^{(t)} , \ \Vert \vec{\vx}_{ij} \Vert , \ \vec{\vx}_{ij} \cdot \vec{\vx}_{ik} \right) \Big) \label{eq:dimenet}
    \end{align}


.. literalinclude:: ../../../proteinworkshop/config/encoder/dimenet_plus_plus.yaml
    :language: yaml
    :caption: config/encoder/dimenet_plus_plus.yaml

:py:class:`GearNet <proteinworkshop.models.graph_encoders.gear_net.GearNet>` (``gear_net``, ``gear_net_edge``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GearNet-Edge is an SE(3) invariant architecture leveraging
relational graph convolutional layers and edge message passing. The original
GearNet-Edge formulation presented in Zhang et al. (2023) operates on
multirelational protein structure graphs making use of several edge construction
schemes (:math:`k`-NN, euclidean distance and sequence distance based).
Our benchmark contains full capabilities for working with multirelational graphs
but use a single edge type (i.e. :math:`|\mathcal{R}| = 1`) in our experiments to
enable more direct architectural comparisons.

The relational graph convolutional layer is defined for relation type :math:`r` as:

.. math::
    \begin{equation}
        \vs_{i}^{(t+1)} \defeq \vs_i^{(t)} + \sigma \left(\mathrm{BN}\left( \sum_{r \in \mathcal{R}} \mathbf{W_r} \sum_{j \in \mathcal{N}_{r}(i)} \vs_j^{(t)}) \right) \right) \\
    \end{equation}

The edge message passing layer is defined for relation type :math:`r` as:

.. math::
    \begin{equation}
        \vm_{(i,j,r_{1})}^{(t+1)} \defeq \sigma \left( \mathrm{BN} \left( \sum_{r \in {|R|\prime}} \mathbf{W}^{\prime}_r \sum_{(w, k, r_2) \in \mathcal{N}_{r}^{\prime}((i,j,r_{1}))}\vm_{(w,k,r_{2})}^{(t)}\right)\right)
    \end{equation}

.. math::
    \begin{equation}
        \vs_{i}^{(t+1)} \defeq \sigma \left( \mathrm{BN} \left( \sum_{r \in {|R|}} \mathbf{W}_r \sum_{j \in \mathcal{N}_{r}(i)}\left(s_{j}^{(t)} + \mathrm{FC}(\vm_{(i,j,r)}^{(t + 1)})\right)\right)\right),
    \end{equation}

where :math:`\mathrm{FC(\cdot)}` denotes a linear transformation upon the message function.


.. literalinclude:: ../../../proteinworkshop/config/encoder/gear_net.yaml
    :language: yaml
    :caption: config/encoder/gear_net.yaml


.. literalinclude:: ../../../proteinworkshop/config/encoder/gear_net_edge.yaml
    :language: yaml
    :caption: config/encoder/gear_net_edge.yaml



:py:class:`CDConv <proteinworkshop.models.graph_encoders.cdconv.CDConvModel>` (``cdconv``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CDConv is an SE(3) invariant architecture that uses independent learnable weights for sequential displacement, whilst directly encoding geometric displacements.

As a result of the downsampling procedures, this architecture is only suitable for graph-level prediction tasks.

.. literalinclude:: ../../../proteinworkshop/config/encoder/cdconv.yaml
    :language: yaml
    :caption: config/encoder/cdconv.yaml


Vector-Equivariant Encoders
=============================

.. mdinclude:: ../../../README.md
    :start-line: 330
    :end-line: 336

:py:class:`EGNN <proteinworkshop.models.graph_encoders.egnn.EGNNModel>` (``egnn``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We consider E(3) equivariant GNN layers proposed by Satorras et al. (2021) which updates both scalar features :math:`\vs_i` as well as node coordinates :math:`\vec{\vx}_{i}`, as follows:

.. math::
    \begin{align}
        \vs_i^{(t+1)} & \defeq f_2 \left( \vs_i^{(t)} \ , \ \sum_{j \in \mathcal{N}_i} f_1 \left( \vs_i^{(t)} , \vs_j^{(t)} , \ \Vert \vec{\vx}_{ij}^{(t)} \Vert \right) \right)      \\
        \vec{\vx}_i^{(t+1)} & \defeq \vec{\vx}_i^{(t)} + \sum_{j \in \mathcal{N}_i} \vec{\vx}_{ij}^{(t)} \odot f_3 \left( \vs_i^{(t)} , \vs_j^{(t)} , \ \Vert \vec{\vx}_{ij}^{(t)} \Vert \right)
    \end{align}

.. literalinclude:: ../../../proteinworkshop/config/encoder/egnn.yaml
    :language: yaml
    :caption: config/encoder/egnn.yaml


:py:class:`GVP <proteinworkshop.models.graph_encoders.gvp.GVPGNNModel>` (``gvp``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../proteinworkshop/config/encoder/gvp.yaml
    :language: yaml
    :caption: config/encoder/gvp.yaml


:py:class:`GCPNet <proteinworkshop.models.graph_encoders.gcpnet.GCPNetModel>` (``gcpnet``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GCPNet is an SE(3) equivariant architecture that jointly learns scalar and vector-valued features from geometric protein structure inputs and, through the use of geometry-complete frame embeddings, sensitises its predictions to account for potential changes induced by the effects of molecular chirality on protein structure. In contrast to the original GCPNet formulation presented in Morehead et al. (2022), the implementation we provide in the benchmark incorporates the architectural enhancements proposed in Morehead et al. (2023) which include the addition of a scalar message attention gate (i.e., :math:`f_{a}(\cdot)``) and a simplified structure for the model's geometric graph convolution layers (i.e., :math:`f_{n}(\cdot)`). With geometry-complete graph convolution in mind, for node :math:`i` and layer :math:`t`, scalar edge features :math:`\vs_{e^{ij}}^{(t)}` and vector edge features :math:`\vv_{e^{ij}}^{(t)}` are used along with scalar node features :math:`\vs_{n^{i}}^{(t)}` and vector node features :math:`\vv_{n^{i}}^{(t)}` to update each node feature type as:

.. math::
    \begin{equation}
        (\vs_{m^{ij}}^{(t+1)}, \vv_{m^{ij}}^{(t+1)}) \defeq f_{e}^{(t+1)}\left((\vs_{n^{i}}^{(t)}, \vv_{n^{i}}^{(t)}),(\vs_{n^{j}}^{(t)}, \vv_{n^{j}}^{(t)}),(f_{a}^{(t + 1)}(\vs_{e^{ij}}^{(t)}), \vv_{e^{ij}}^{(t)}),\mathbf{\mathcal{F}}_{ij}\right)
    \end{equation}

.. math::
    \begin{equation}
        (\vs_{n^{i}}^{(t+1)}, \vv_{n^{i}}^{(t+1)}) \defeq f_{n}^{(t+1)}\left((\vs_{n^{i}}^{(t)}, \vv_{n^{i}}^{(t)}), \sum_{j \in \mathcal{N}(i)} (\vs_{m^{ij}}^{(t+1)}, \vv_{m^{ij}}^{(t+1)}) \right),
    \end{equation}

where the geometry-complete and chirality-sensitive local frames for node :math:`i` (i.e., its edges) are defined as :math:`\mathbf{\mathcal{F}}_{ij} = (\va_{ij}, \vb_{ij}, \vc_{ij})`, with :math:`\va_{ij} = \frac{\vx_{i} - \vx_{j}}{ \lVert \vx_{i} - \vx_{j} \rVert }, \vb_{ij} = \frac{\vx_{i} \times \vx_{j}}{ \lVert \vx_{i} \times \vx_{j} \rVert },` and :math:`\vc_{ij} = \va_{ij} \times \vb_{ij}`, respectively.


.. literalinclude:: ../../../proteinworkshop/config/encoder/gcpnet.yaml
    :language: yaml
    :caption: config/encoder/gcpnet.yaml


Tensor-Equivariant Encoders
=============================

.. mdinclude:: ../../../README.md
    :start-line: 338
    :end-line: 343


:py:class:`Tensor Field Networks <proteinworkshop.models.graph_encoders.tfn.TensorProductModel>` (``tfn``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tensor Field Networks are E(3) or SE(3) equivariant GNNs that have been successfully used in protein structure prediction (Baek et al., 2021) and protein-ligand docking (Corso et al., 2022).
These models use higher order spherical tensors :math:`\tilde \vh_{i,l} \in \mathbb{R}^{2l+1 \times f}` as node features, starting from order :math:`l = 0` up to arbitrary :math:`l = L`.
The first two orders correspond to scalar features :math:`\vs_i` and vector features :math:`\vec{\vv}_i`, respectively.
The higher order tensors :math:`\tilde \vh_{i}` are updated via tensor products :math:`\otimes` of neighbourhood features :math:`\tilde \vh_{j}`` for all :math:`j \in \mathcal{N}_i` with the higher order spherical harmonic representations :math:`Y` of the relative displacement :math:`\frac{\vec{\vx}_{ij}}{\Vert \vec{\vx}_{ij} \Vert} = \hat{\vx}_{ij}`:

.. math::
    \begin{align}
        \tilde \vh_{i}^{(t+1)} & \defeq \tilde \vh_{i}^{(t)} + \sum_{j \in \mathcal{N}_i} Y \left( \hat{\vx}_{ij} \right) \otimes_{\vw} \tilde \vh_{j}^{(t)} ,
    \end{align}

where the weights :math:`\vw` of the tensor product are computed via a learnt radial basis function of the relative distance, i.e. :math:`\vw = f \left( \Vert \vec{\vx}_{ij} \Vert \right)`.


.. literalinclude:: ../../../proteinworkshop/config/encoder/tfn.yaml
    :language: yaml
    :caption: config/encoder/tfn.yaml


:py:class:`Multi-Atomic Cluster Expansion <proteinworkshop.models.graph_encoders.mace.MACEModel>` (``mace``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MACE (Batatia et al., 2022) is a higher order E(3) or SE(3) equivariant GNN originally developed for molecular dynamics simulations.
MACE provides an efficient approach to computing high body order equivariant features in the Tensor Field Network framework via Atomic Cluster Expansion:
They first aggregate neighbourhood features analogous to the node update equation for TFN above (the :math:`A` functions in Batatia et al. (2022) (eq.9)) and then take :math:`k-1` repeated self-tensor products of these neighbourhood features.
In our formalism, this corresponds to:

.. math::
    \begin{align}
        \label{eq:e3nn-3}
        \tilde \vh_{i}^{(t+1)} & \defeq \underbrace {\tilde \vh_{i}^{(t+1)} \otimes_{\vw} \dots \otimes_{\vw} \tilde \vh_{i}^{(t+1)} }_\text{$k-1$ times} \ ,
    \end{align}

.. literalinclude:: ../../../proteinworkshop/config/encoder/mace.yaml
    :language: yaml
    :caption: config/encoder/mace.yaml


Sequence-Based Encoders
=============================

.. mdinclude:: ../../../README.md
    :start-line: 345
    :end-line: 349


:py:class:`Evolutionary Scale Modeling <proteinworkshop.models.graph_encoders.esm_embeddings.EvolutionaryScaleModeling>` (``esm``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evolutionary Scale Modeling is a series of Transformer-based protein sequence encoders (Vaswani et al., 2017) that has been successfully used in protein structure prediction (Lin et al., 2023), protein design (Verkuil et al., 2022), and beyond.
This model class has commonly been used as a baseline for protein-related representation learning tasks, and we included it in our benchmark for this reason.

.. literalinclude:: ../../../proteinworkshop/config/encoder/esm.yaml
    :language: yaml
    :caption: config/encoder/esm.yaml


Decoder Models
=============================

Decoder models are used to predict the target property from the learned representation.

Decoder configs are dictionaries indexed by the name of the output to which they are applied.

These are configured in the task config. See :doc:`/configs/task` for more details.

For example, the ``residue_type`` decoder:

.. seealso::
    :doc:`/configs/task`

.. literalinclude:: ../../../proteinworkshop/config/decoder/residue_type.yaml
    :language: yaml
    :caption: config/decoder/residue_type.yaml
