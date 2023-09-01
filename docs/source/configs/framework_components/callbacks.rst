Callbacks
=============

Default
---------

.. literalinclude:: ../../../../proteinworkshop/config/callbacks/default.yaml
   :language: yaml


Training
------------------

Early Stopping (``early_stopping``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../../proteinworkshop/config/callbacks/early_stopping.yaml
   :language: yaml
   :caption: config/callbacks/early_stopping.yaml

Checkpointing (``model_checkpoint``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../../proteinworkshop/config/callbacks/model_checkpoint.yaml
   :language: yaml
   :caption: config/callbacks/model_checkpoint.yaml

Stop on NaN (``stop_on_nan``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../../proteinworkshop/config/callbacks/stop_on_nan.yaml
   :language: yaml
   :caption: config/callbacks/stop_on_nan.yaml


Exponential Moving Average (``ema``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../proteinworkshop/config/callbacks/ema.yaml
   :language: yaml
   :caption: config/callbacks/ema.yaml



Logging
---------

Model Summary (``model_summary``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::

   :py:class:`lightning.pytorch.callbacks.RichModelSummary`

.. literalinclude:: ../../../../proteinworkshop/config/callbacks/model_summary.yaml
   :language: yaml
   :caption: config/callbacks/model_summary.yaml


Rich Progress Bar (``rich_progress_bar``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::

   :py:class:`lightning.pytorch.callbacks.RichProgressBar`

.. literalinclude:: ../../../../proteinworkshop/config/callbacks/rich_progress_bar.yaml
   :language: yaml
   :caption: config/callbacks/rich_progress_bar.yaml

Learning Rate Monitor (``learning_rate_monitor``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is automatically configured when using a learning rate scheduler. See :doc:`/configs/ml_components/scheduler`

.. seealso::

   :py:class:`lightning.pytorch.callbacks.LearningRateMonitor`

.. literalinclude:: ../../../../proteinworkshop/config/callbacks/learning_rate_monitor.yaml
   :language: yaml
   :caption: config/callbacks/learning_rate_monitor.yaml