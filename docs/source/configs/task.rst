Tasks
----------------

.. list-table::

   * - .. figure:: ../_static/box_aux_tasks.png
     - .. figure:: ../_static/box_downstream_tasks.png


Task configs are 'high-level' configs which configure the transforms outputs, losses and metrics of the model.

Specific training objects are achieved through the use of :doc:`/configs/transforms`.

.. note::
    To change the task, use a command with a format like:

    .. code-block:: bash

        workshop train task=<TASK_NAME> dataset=cath encoder=gvp ...
        # or
        python proteinworkshop/train.py task=<TASK_NAME> dataset=cath encoder=gvp ...

    Where ``<TASK_NAME>`` is one of the tasks listed below.

.. seealso::
    :doc:`/configs/transforms`

Denoising Tasks
================

Sequence Denoising (``sequence_denoising``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This config trains a model to predict the original identies of the corrupted residues in a protein sequence.

The corruption is configured by the :py:class:`sequence noise transform <proteinworkshop.tasks.sequence_denoising.SequenceNoiseTransform`>, which can be configured to apply masking or mutation
corruptions in various amounts to the input sequence.

.. seealso::
    :py:class:`proteinworkshop.tasks.sequence_denoising.SequenceNoiseTransform`


.. code-block:: bash

    config/task/sequence_denoising.yaml

.. literalinclude:: ../../../proteinworkshop/config/task/sequence_denoising.yaml
    :language: yaml
    :caption: config/task/sequence_denoising.yaml

Structure Denoising (``structure_denoising``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This config trains a model to predict the original Cartesian coordinates of the corrupted residues in a protein structure.

Noise is applied to the Cartesian coordinates and the model is tasked with predicting either the per-residue Cartesian noise or the original Cartesian coordinates.

The corruption is configured by the :py:class:`structure noise transform <proteinworkshop.tasks.structure_denoising.StructureNoiseTransform`>`, which can be configured to apply uniform or gaussian random noise in various amounts to the input structure.

.. seealso::
    :py:class:`proteinworkshop.tasks.structure_denoising.StructureNoiseTransform`

.. literalinclude:: ../../../proteinworkshop/config/task/structure_denoising.yaml
   :language: yaml
   :caption: config/task/structure_denoising.yaml

Sequence & Structure Denoising (``sequence_structure_denoising``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This config trains a model to predict the original identies of the corrupted residues in a protein sequence and the original Cartesian coordinates of the corrupted residues in a protein structure.

This config demonstrates how we can compose transforms in a modular fashion to create new training regimes.

.. seealso::
    :py:class:`proteinworkshop.tasks.sequence_denoising.SequenceNoiseTransform`
    :py:class:`proteinworkshop.tasks.structure_denoising.StructureNoiseTransform`

.. literalinclude:: ../../../proteinworkshop/config/task/sequence_structure_denoising.yaml
   :language: yaml
   :caption: config/task/sequence_structure_denoising.yaml

Torsional Denoising (``torsional_denoising``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This config trains a model to predict the original dihedral angles of the corrupted residues in a protein structure.

The torsional noise transform applies noise in dihedral angle space. The cartesian coordinates are the recomputed using pNeRF to enable structure-based featurisation.

.. seealso::
    :py:class:`proteinworkshop.tasks.torsional_denoising.TorsionalNoiseTransform`

.. literalinclude:: ../../../proteinworkshop/config/task/torsional_denoising.yaml
    :language: yaml
    :caption: config/task/torsional_denoising.yaml


Node-level Tasks
================


Protein-Protein Interaction Site Prediction (``ppi_site_prediction``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../proteinworkshop/config/task/ppi_site_prediction.yaml
    :language: yaml
    :caption: config/task/ppi_site_prediction.yaml


Ligand Binding Site Prediction (``binding_site_identification``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::
    :py:class:`proteinworkshop.tasks.binding_site_identification.BindingSiteIdentificationTransform`

.. literalinclude:: ../../../proteinworkshop/config/task/binding_site_identification.yaml
   :language: yaml
   :caption: config/task/binding_site_identification.yaml



Multiclass Node Classification (``multiclass_node_classification``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../proteinworkshop/config/task/muliclass_node_classification.yaml
   :language: yaml
   :caption: config/task/muliclass_node_classification.yaml

pLDDT Prediction (``plddt_prediction``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This config specifies a self-supervision task to predict the per-residue pLDDT score of each node.

.. warning::
    This task requires the input data to have a ``b_factor`` attribute.

    If the input structure are not predicted structures, this task will be a B factor prediction task.

.. literalinclude:: ../../../proteinworkshop/config/task/plddt_prediction.yaml
   :language: yaml
   :caption: config/task/plddt_prediction.yaml


Edge-level Tasks
================

Edge Distance Prediction (``edge_distance_prediction``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This config specifies a self-supervision task to predict the pairwise distance between two nodes.

We first sample ``num_samples`` edges randomly from the input batch. We then
construct a mask to remove the sampled edges from the batch. We store the
masked node indices and their pairwise distance as ``batch.node_mask`` and
``batch.edge_distance_labels``, respectively. Finally, it masks the edges
(and their attributes) using the constructed mask and returns the modified
batch. The distance is then predicted from the concantenated node embeddings of the two nodes.

.. literalinclude:: ../../../proteinworkshop/config/task/edge_distance_prediction.yaml
   :language: yaml
   :caption: config/task/edge_distance_prediction.yaml


.. mdinclude:: ../../../proteinworkshop/config/aux_task/README.md

.. note::
    Aux tasks are specified with the following syntax:

    .. code-block::bash

        python proteinworkshop/train.py ... +aux_task=nn_sequence

Sequence Denoising (``nn_sequence``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Example:
    python proteinworkshop/train.py dataset=cath encoder=gvp task=plddt_prediction +aux_task=nn_sequence

.. literalinclude:: ../../../proteinworkshop/config/aux_task/nn_sequence.yaml
    :language: yaml
    :caption: config/aux_task/nn_sequence.yaml

Structure Denoising (``nn_structure_r3``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Example:
    python proteinworkshop/train.py dataset=cath encoder=gvp task=plddt_prediction +aux_task=nn_structure_r3

.. literalinclude:: ../../../proteinworkshop/config/aux_task/nn_structure_r3.yaml
    :language: yaml
    :caption: config/aux_task/nn_structure_r3.yaml

Torsional Denoising (``nn_structure_torsion``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Noise is applied to the backbone torsion angles and Cartesian coordinates are recomputed using pNeRF and the uncorrupted bond
lengths and angles prior to feature computation. Similarly to the coordinate denoising task, the model
is then tasked with predicting either the per-residue angular noise or the original dihedral angles

.. warning::
    This will subset the data to only include the backbone atoms
    (N, Ca, C). The backbone oxygen can be placed with:
    :py:func:`graphein.protein.tensor.reconstruction.place_fourth_coord`.

    This will break, for example, sidechain torsion angle computation for
    the first few chi angles that are partially defined by backbone atoms.

.. code-block:: bash

    # Example:
    python proteinworkshop/train.py dataset=cath encoder=gvp task=plddt_prediction +aux_task=nn_structure_torsion

.. literalinclude:: ../../../proteinworkshop/config/aux_task/nn_structure_torsion.yaml
    :language: yaml
    :caption: config/aux_task/nn_structure_torsion.yaml

Inverse Folding (``inverse_folding``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This adds an additional inverse folding objective to the model. I.e. the model is trained to predict the sequence of the input structure.

.. warning::

    This will remove the residue-type node feature and sidechain torsion angles to avoid leakage of information from the target structure.

.. code-block:: bash

    # Example:
    python proteinworkshop/train.py dataset=cath encoder=gvp task=plddt_prediction +aux_task=inverse_folding

.. literalinclude:: ../../../proteinworkshop/config/aux_task/inverse_folding.yaml
    :language: yaml
    :caption: config/aux_task/inverse_folding.yaml

None (``none``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Example:
    python proteinworkshop/train.py dataset=cath encoder=gvp task=plddt_prediction
