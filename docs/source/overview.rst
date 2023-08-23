Architectural Overview
-----------------------


The benchmark is constructed in a modular fashion, with each module configured via a ``.yaml`` config using `Hydra <https://hydra.cc/docs/intro/>`_.

The predominant ingredients are:

- **Datasets**: The benchmark supports a variety of datasets, described in :doc:`/configs/dataset` and documented in :doc:`/modules/src.datasets`
- **Models**: The benchmark supports a variety of models, described in :doc:`/configs/model` and documented in :doc:`/modules/src.models`
- **Tasks**: The benchmark supports a variety of tasks, described in :doc:`/configs/task` and documented in :doc:`/modules/src.tasks`
- **Features**: The benchmark supports a variety of features, described in :doc:`/configs/features` and documented in :doc:`/modules/src.features`


Datasets
==========

:py:class:`Protein <graphein.protein.tensor.data.Protein>` and :py:class:`ProteinBatch <<graphein.protein.tensor.data.ProteinBatch>` Data structures
================================================================================================================================================================================================

We make extensive use of Graphein for data processing and featurisation in the framework.
To familiarise yourself with the data structures used in the framework, please see the
`tutorials provided by Graphein <https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/protein_tensors.ipynb>`_. In essence, these objects inherit from :py:class:`torch_geometric.data.Data`
and :py:class:`torch_geometric.data.Batch` respectively, and are used to represent a single protein or a batch of proteins.


:py:class:`ProteinDataModule <src.datasets.base.ProteinDataModule>` Base classes
==============================================================================================================

The framework provides base classes for datasets and datamodules, which can be extended to create new datasets.

The datamodule is the only object that needs to be configured to add a new dataset. The :py:class:`src.datasets.base.ProteinDataModule` class is a subclass of :py:class:`pytorch_lightning.LightningDataModule` and is used to represent a datamodule for a dataset of proteins. This class is used to create dataloaders for training, validation and testing.

To do so, the datamodule for the new dataset should inherit from :py:class:`src.datasets.base.ProteinDataModule` and implement the following methods:

- :py:meth:`src.datasets.base.ProteinDataModule.download`
- :py:meth:`src.datasets.base.ProteinDataModule.parse_dataset`
- (optionally) :py:meth:`src.datasets.base.ProteinDataModule.parse_labels`
- (optionally) :py:meth:`src.datasets.base.ProteinDataModule.exclude_pdbs`
- :py:meth:`src.datasets.base.ProteinDataModule.train_dataset`
- :py:meth:`src.datasets.base.ProteinDataModule.val_dataset`
- :py:meth:`src.datasets.base.ProteinDataModule.test_dataset`
- :py:meth:`src.datasets.base.ProteinDataModule.train_dataloader`
- :py:meth:`src.datasets.base.ProteinDataModule.val_dataloader`
- :py:meth:`src.datasets.base.ProteinDataModule.test_dataloader`

The methods :py:meth:`src.datasets.base.ProteinDataModule.train_dataset`, :py:meth:`src.datasets.base.ProteinDataModule.val_dataset` and :py:meth:`src.datasets.base.ProteinDataModule.test_dataset` should return a :py:class:`src.datasets.base.ProteinDataset` object, which is a subclass of :py:class:`torch.utils.data.Dataset` and is used to represent a dataset of proteins.

The methods :py:meth:`src.datasets.base.ProteinDataModule.train_dataloader`, :py:meth:`src.datasets.base.ProteinDataModule.val_dataloader` and :py:meth:`src.datasets.base.ProteinDataModule.test_dataloader` should return a :py:class:`from graphein.protein.tensor.dataloader.ProteinDataLoader` object, which is used to represent a dataloader for a dataset of proteins.

The methods :py:meth:`src.datasets.base.ProteinDataModule.download` :py:meth:`src.datasets.base.ProteinDataModule.parse_dataset`, handles all of the dataset-specific logic for downloading, and parsing labels, ids/filenames and chains.



Models
==========

:py:class:`src.models.base.BaseModel` and :py:class:`src.models.base.BenchMarkModel` Base classes
==============================================================================================================

These objects orchestrate model training and validation logic. The :py:class:`src.models.base.BaseModel` class is a subclass of :py:class:`pytorch_lightning.LightningModule`.
The :py:class:`src.models.base.BenchMarkModel` class is a subclass of :py:class:`src.models.base.BaseModel` and is used as the primary orchestrator in the framework.

To use a different structural encoder, the user should overwrite :py:attr:`src.models.base.BenchMarkModel.encoder` with a new encoder class. The encoder class should be a subclass of :py:class:`torch.nn.Module` and should implement the following methods:

- :py:meth:`torch.nn.Module.forward`

The forward method should be of the form:

.. code-block:: python

    from src.types import EncoderOutput

    def forward(self, x: [Batch, ProteinBatch]) -> EncoderOutput:
        node_emb = x.x
        graph_emb = self.readout(node_emb, x.batch)
        return EncoderOutput({"node_embedding": node_emb, "graph_embedding": graph_embedding})

Consuming a Batch object and returning a dictionary with keys ``node_embedding`` and ``graph_embedding``.

.. note::

    Both keys in the output dictionary are not required to be present, depending on whether the task is node-level or graph-level.
