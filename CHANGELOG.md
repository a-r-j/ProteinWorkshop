### 2.0.3

* Minor patch; adds missing `overwrite` attribute to `CATHDataModule`.

### 2.0.2 (Unreleased)

* Fixes raw data download triggered by absence of PDB when using pre-processed datasets ([#24](https://github.com/a-r-j/ProteinWorkshop/pull/24))
* Fixes bug where batches created from `in_memory=True` data were not correctly formatted ([#24](https://github.com/a-r-j/ProteinWorkshop/pull/24))
* Consistently exposes the `overwrite` argument for datamodules to users ([#24](https://github.com/a-r-j/ProteinWorkshop/pull/24))
* Fixes bug where downloading FoldComp datasets into directories with the same name as the dataset throws an error ([#24](https://github.com/a-r-j/ProteinWorkshop/pull/24))
* Increments `graphein` dependency to `1.7.3` ([#24](https://github.com/a-r-j/ProteinWorkshop/pull/24))

### 2.0.1 (29/08/2023)

* Fixes incorrect lookup of `DATA_PATH` env var ([#19](https://github.com/a-r-j/ProteinWorkshop/pull/19))

### 2.0.0 - 28/08/2023

* First public release
