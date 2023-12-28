### 0.2.5 (28/12/2023)

#### Datasets

* Adds to antibody-specific datasets using the IGFold corpuses for paired OAS and Jaffe 2022 [#53](https://github.com/a-r-j/ProteinWorkshop/pull/53/)
* Set `in_memory=True` as default for most (small) datasets for improved performance [#53](https://github.com/a-r-j/ProteinWorkshop/pull/53/)
* Fix `num_classes` for GO datamodules * Set `in_memory=True` as default for most (downstream) datasets for improved performance [#53](https://github.com/a-r-j/ProteinWorkshop/pull/53/)
* Fixes GO labelling [#53](https://github.com/a-r-j/ProteinWorkshop/pull/53/)

### Features

* Improves positional encoding performance by adding a `seq_pos` attribute on `Data/Protein` objects in the base dataset getter. [#53](https://github.com/a-r-j/ProteinWorkshop/pull/53/)
* Ensure correct batched computation of orientation features. [#58](https://github.com/a-r-j/ProteinWorkshop/pull/58/)

### Models

* Implement ESM embedding encoder ([#33](https://github.com/a-r-j/ProteinWorkshop/pull/33), [#41](https://github.com/a-r-j/ProteinWorkshop/pull/33))
* Adds CDConv implementation [#53](https://github.com/a-r-j/ProteinWorkshop/pull/53/)
* Adds tuned hparams for models [#53](https://github.com/a-r-j/ProteinWorkshop/pull/53/)

### Framework

* Refactors beartype/jaxtyping to use latest recommended syntax [#53](https://github.com/a-r-j/ProteinWorkshop/pull/53/)
* Adds explainability module for performing attribution on a trained model [#53](https://github.com/a-r-j/ProteinWorkshop/pull/53/)
* Change default finetuning features in config: `ca_base` -> `ca_seq` [#53](https://github.com/a-r-j/ProteinWorkshop/pull/53/)
* Add optional hparam entry point to finetuning config [#53](https://github.com/a-r-j/ProteinWorkshop/pull/53/)
* Fixes GPU memory accumulation for some metrics [#53](https://github.com/a-r-j/ProteinWorkshop/pull/53/)
* Updates zenodo URL for processed datasets to reflect upstream API change [#53](https://github.com/a-r-j/ProteinWorkshop/pull/53/)
* Adds multi-hot label encoding transform [#53](https://github.com/a-r-j/ProteinWorkshop/pull/53/)
* Fixes auto PyG install for `torch>2.1.0` [#53](https://github.com/a-r-j/ProteinWorkshop/pull/53/)
* Adds `proteinworkshop.model_io` containing utils for loading trained models [#53](https://github.com/a-r-j/ProteinWorkshop/pull/53/)
* Add script for plotting UMAP embeddings of any dataset given a pre-trained encoder model

### 0.2.4 (10/09/2024)

* Fixes error in Metal3D processed download link ([#28](https://github.com/a-r-j/ProteinWorkshop/pull/28))
* Fixes typo in wandb run name setting ([#30](https://github.com/a-r-j/ProteinWorkshop/pull/30))
* Fixes paths for models and datasets when testing instantiation of each module ([#32](https://github.com/a-r-j/ProteinWorkshop/pull/32))
* Improvements to TFN, MACE and EGNN models and layers, including DiffDock-style intermediate edge feature creation (TFN), dropout, gaussian RBF, mean global pooling ([#38](https://github.com/a-r-j/ProteinWorkshop/pull/38))

### 0.2.3 (31/08/2023)

* Minor patch; adds missing `overwrite` attribute to `CATHDataModule`, `FoldClassificationDataModule` and `GeneOntologyDataModule`. ([#25](https://github.com/a-r-j/ProteinWorkshop/pull/25))


### 0.2.2 (30/08/2023)

* Fixes raw data download triggered by absence of PDB when using pre-processed datasets ([#24](https://github.com/a-r-j/ProteinWorkshop/pull/24))
* Fixes bug where batches created from `in_memory=True` data were not correctly formatted ([#24](https://github.com/a-r-j/ProteinWorkshop/pull/24))
* Consistently exposes the `overwrite` argument for datamodules to users ([#24](https://github.com/a-r-j/ProteinWorkshop/pull/24))
* Fixes bug where downloading FoldComp datasets into directories with the same name as the dataset throws an error ([#24](https://github.com/a-r-j/ProteinWorkshop/pull/24))
* Increments `graphein` dependency to `1.7.3` ([#24](https://github.com/a-r-j/ProteinWorkshop/pull/24))

### 0.2.1 (29/08/2023)

* Fixes incorrect lookup of `DATA_PATH` env var ([#19](https://github.com/a-r-j/ProteinWorkshop/pull/19))

### 0.2.0 - 28/08/2023

* First public release
