### 0.2.4

* Fixes error in Metal3D processed download link ([#28](https://github.com/a-r-j/ProteinWorkshop/pull/28))
* Fixes typo in wandb run name setting ([#30](https://github.com/a-r-j/ProteinWorkshop/pull/30))
* Fixes paths for models and datasets when testing instantiation of each module ([#32](https://github.com/a-r-j/ProteinWorkshop/pull/32))

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
