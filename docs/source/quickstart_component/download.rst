Downloading Datasets
=====================

Raw datasets will be downloaded from their respective sources and built the first time it is used. This may take a while.

Processed datasets are available from Zenodo (https://zenodo.org/record/8282470) and we provide a CLI tool to download them.

.. note::

    If you wish to specify a custom location for the datasets, you can set the ``DATA_PATH`` environment variable.

    .. code-block::

            export DATA_PATH=/path/to/where/you/want/datasets


.. code-block:: bash

    workshop download <DATASET_NAME>
    # Download pre-training datasets
    workshop download pdb
    workshop download afdb_rep_v4
    workshop download cath

    # Download downstream datasets
    workshop download ec_reaction
    workshop download fold_classification
    workshop download antibody_developability
    ...

.. seealso::

    :doc:`/configs/datasets`