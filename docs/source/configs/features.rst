Features
---------

.. image:: ../_static/box_featurisation_schemes.png
  :alt: Summary of featurisation schemes
  :align: center
  :width: 400

.. mdinclude:: ../../../README.md
    :start-line: 459
    :end-line: 508


Default Features
================


:math:`C_{\alpha}` Only (``ca_base``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../proteinworkshop/config/features/ca_base.yaml
    :language: yaml
    :caption: config/features/ca_base.yaml


:math:`C_{\alpha}` + Sequence (``ca_seq``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../proteinworkshop/config/features/ca_seq.yaml
    :language: yaml
    :caption: config/features/ca_seq.yaml


:math:`C_{\alpha}` + Virtual Angles (``ca_angles``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../proteinworkshop/config/features/ca_angles.yaml
    :language: yaml
    :caption: config/features/ca_angles.yaml


:math:`C_{\alpha}` + Sequence + Backbone (``ca_bb``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../proteinworkshop/config/features/ca_bb.yaml
    :language: yaml
    :caption: config/features/ca_seq_bb.yaml


:math:`C_{\alpha}` + Sequence + Backbone + Sidechains (``ca_sc``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../proteinworkshop/config/features/ca_sc.yaml
    :language: yaml
    :caption: config/features/ca_sc.yaml

