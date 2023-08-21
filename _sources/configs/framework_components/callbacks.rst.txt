Callbacks
=============

Default
---------

.. literalinclude:: ../../../../configs/callbacks/default.yaml
   :language: yaml


Training
------------------

Early Stopping (``early_stopping``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../../configs/callbacks/early_stopping.yaml
   :language: yaml
   :caption: configs/callbacks/early_stopping.yaml

Checkpointing (``model_checkpoint``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../../configs/callbacks/model_checkpoint.yaml
   :language: yaml
   :caption: configs/callbacks/model_checkpoint.yaml

Stop on NaN (``stop_on_nan``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../../configs/callbacks/stop_on_nan.yaml
   :language: yaml
   :caption: configs/callbacks/stop_on_nan.yaml


Exponential Moving Average (``ema``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../configs/callbacks/ema.yaml
   :language: yaml
   :caption: configs/callbacks/ema.yaml



Logging
---------

Model Summary (``model_summary``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::

   :py:class:`lightning.pytorch.callbacks.RichModelSummary`

.. literalinclude:: ../../../../configs/callbacks/model_summary.yaml
   :language: yaml
   :caption: configs/callbacks/model_summary.yaml


Rich Progress Bar (``rich_progress_bar``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::

   :py:class:`lightning.pytorch.callbacks.RichProgressBar`

.. literalinclude:: ../../../../configs/callbacks/rich_progress_bar.yaml
   :language: yaml
   :caption: configs/callbacks/rich_progress_bar.yaml

Learning Rate Monitor (``learning_rate_monitor``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is automatically configured when using a learning rate scheduler. See :doc:`/configs/ml_components/scheduler`

.. seealso::

   :py:class:`lightning.pytorch.callbacks.LearningRateMonitor`

.. literalinclude:: ../../../../configs/callbacks/learning_rate_monitor.yaml
   :language: yaml
   :caption: configs/callbacks/learning_rate_monitor.yaml