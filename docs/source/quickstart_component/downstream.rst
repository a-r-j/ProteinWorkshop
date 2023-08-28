Downstream
==========

To get up and running quickly, the following widgets can be used to configure commands to run various functions of the benchmark.

.. note::
   The widgets provide only a subset of options. For example, any dataset can be used for pre-training, but only a few are available in the widget. For more options, see the `Configuration` section.

.. note::
   If you have pip-installed ``proteinworkshop``, the ``python proteinworkshop/train.py ...`` syntax can be replaced with ``workshop train ...``
   or ``workshop finetune ...``.

.. note::
   To use specify CPU / GPU, add ``trainer=cpu`` or ``trainer=gpu`` to the command.

.. raw:: html
   :file: downstream_component.html
