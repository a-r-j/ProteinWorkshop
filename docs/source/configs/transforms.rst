Transforms
==============

This section describes the configurations for various :py:class:`torch_geometric.transforms.Transforms`
objects used throughout the frame.


.. note::
    These are typically not used in the CLI - they are rather primitives that we
    use in defining :doc:`/configs/task`.

.. seealso::
    :doc:`/configs/task`


None
--------------------

This config file is used to specify no transforms.

.. literalinclude:: ../../../configs/transforms/none.yaml
    :language: yaml
    :caption: :file:`transforms/none.yaml`



Data Transforms
--------------------

These transforms are used to modify the input data in some way, such as handling edge cases.


Remove Missing :math:`C_{\alpha}` Atoms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../configs/transforms/remove_missing_ca.yaml
    :language: yaml
    :caption: :file:`transforms/remove_missing_ca.yaml`



Generic Task Transforms
------------------------

Binding Site Prediction
^^^^^^^^^^^^^^^^^^^^^^^^
.. seealso::
    :py:class:`proteinworkshop.tasks.binding_site_prediction.BindingSiteTransform`


Protein Protein Site Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::
    :py:class:`proteinworkshop.tasks.ppi_site_prediction.BindingSiteTransform`

.. literalinclude:: ../../../configs/transforms/ppi_site_prediction.yaml
    :language: yaml
    :caption: :file:`transforms/generic.yaml`




Denoising Transforms
--------------------

Sequence Denoising
^^^^^^^^^^^^^^^^^^
.. seealso::
    :py:class:`proteinworkshop.tasks.sequence_denoising.SequenceNoiseTransform`

.. literalinclude:: ../../../configs/transforms/sequence_denoising.yaml
    :language: yaml
    :caption: :file:`transforms/sequence_denoising.yaml`

Structure Denoising
^^^^^^^^^^^^^^^^^^^

.. seealso::
    :py:class:`proteinworkshop.tasks.structural_denoising.StructuralNoiseTransform`

.. literalinclude:: ../../../configs/transforms/structure_denoising.yaml
    :language: yaml
    :caption: :file:`transforms/structure_denoising.yaml`

Torsional Denoising
^^^^^^^^^^^^^^^^^^^

.. seealso::
    :py:class:`proteinworkshop.tasks.torsional_denoising.TorsionalNoiseTransform`

.. literalinclude:: ../../../configs/transforms/torsional_denoising.yaml
    :language: yaml
    :caption: :file:`transforms/torsion_denoising.yaml`



