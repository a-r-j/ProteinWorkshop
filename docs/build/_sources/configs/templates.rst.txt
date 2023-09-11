Config Templates
------------------

For configuration management in the benchmark, we use `hydra <https://hydra.cc/docs/intro/>`_. The configuration files are stored in ``config`` folder.

Configs are built through composing different individual configs for each component according to a template schema.


Below, we document the different config templates used in the benchmark.

Training
==================

This config can be used for both training a model from scratch and pre-training a model.

.. literalinclude:: ../../../proteinworkshop/config/train.yaml
    :language: yaml
    :caption: config/train.yaml
    :linenos:



Finetuning
==================

This config should be used to finetune a pre-trained model on a downstream task.

.. literalinclude:: ../../../proteinworkshop/config/finetune.yaml
    :language: yaml
    :caption: config/finetune.yaml
    :linenos: