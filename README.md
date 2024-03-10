# Protein Workshop

[![PyPI version](https://badge.fury.io/py/proteinworkshop.svg)](https://badge.fury.io/py/proteinworkshop)
[![Zenodo doi badge](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.8282470-blue.svg)](https://zenodo.org/record/8282470)
![Tests](https://github.com/a-r-j/ProteinWorkshop/actions/workflows/code-tests.yaml/badge.svg)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://assets.readthedocs.org/static/projects/badges/passing-flat.svg)](https://www.proteins.sh)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

![Overview of the Protein Workshop](https://github.com/a-r-j/ProteinWorkshop/blob/main/docs/source/_static/workshop_overview.png)

[Documentation](https://www.proteins.sh)


This [repository](https://github.com/a-r-j/ProteinWorkshop) provides the code for the protein structure representation learning benchmark detailed in the paper [*Evaluating Representation Learning on the Protein Structure Universe*](https://openreview.net/forum?id=sTYuRVrdK3) (ICLR 2024).

In the benchmark, we implement numerous [featurisation](https://www.proteins.sh/configs/features) schemes, [datasets](https://www.proteins.sh/configs/dataset) for [self-supervised pre-training](https://proteins.sh/quickstart_component/pretrain.html) and [downstream evaluation](https://proteins.sh/quickstart_component/downstream.html), [pre-training](https://proteins.sh/configs/task) tasks, and [auxiliary tasks](https://proteins.sh/configs/task.html#auxiliary-tasks).

The benchmark can be used as a working template for a protein representation learning research project, [a library](#using-proteinworkshop-modules-functionally) of drop-in components for use in your projects, or as a CLI tool for quickly running [protein representation learning evaluation](https://proteins.sh/quickstart_component/downstream.html) and [pre-training](https://proteins.sh/quickstart_component/pretrain.html) configurations.

[Processed datasets](https://zenodo.org/record/8282470) and [pre-trained weights](https://zenodo.org/record/8287754) are made available. Downloading datasets is not required; upon first run all datasets will be downloaded and processed from their respective source.

Configuration files to run the experiments described in the manuscript are provided in the `proteinworkshop/config/sweeps/` directory.

## Contents

- [Protein Workshop](#protein-workshop)
  - [Contents](#contents)
  - [Installation](#installation)
    - [From PyPI](#from-pypi)
    - [Building from source](#building-from-source)
  - [Tutorials](#tutorials)
  - [Quickstart](#quickstart)
    - [Downloading datasets](#downloading-datasets)
    - [Training a model](#training-a-model)
    - [Finetuning a model](#finetuning-a-model)
    - [Running a sweep/experiment](#running-a-sweepexperiment)
    - [Embedding a dataset](#embedding-a-dataset)
    - [Visualising a dataset's embeddings](#visualising-pre-trained-model-embeddings-for-a-given-dataset)
    - [Performing attribution of a pre-trained model](#performing-attribution-of-a-pre-trained-model)
    - [Verifying a config](#verifying-a-config)
    - [Using `proteinworkshop` modules functionally](#using-proteinworkshop-modules-functionally)
  - [Models](#models)
    - [Invariant Graph Encoders](#invariant-graph-encoders)
    - [Equivariant Graph Encoders](#equivariant-graph-encoders)
      - [(Vector-type)](#vector-type)
      - [(Tensor-type)](#tensor-type)
  - [Datasets](#datasets)
    - [Structure-based Pre-training Corpuses](#structure-based-pre-training-corpuses)
    - [Supervised Datasets](#supervised-datasets)
  - [Tasks](#tasks)
    - [Self-Supervised Tasks](#self-supervised-tasks)
    - [Generic Supervised Tasks](#generic-supervised-tasks)
  - [Featurisation Schemes](#featurisation-schemes)
    - [Invariant Node Features](#invariant-node-features)
    - [Equivariant Node Features](#equivariant-node-features)
    - [Edge Construction](#edge-construction)
    - [Invariant Edge Features](#invariant-edge-features)
    - [Equivariant Edge Features](#equivariant-edge-features)
  - [For Developers](#for-developers)
    - [Dependency Management](#dependency-management)
    - [Code Formatting](#code-formatting)
    - [Documentation](#documentation)

## Installation

Below, we outline how one may set up a virtual environment for `proteinworkshop`. Note that these installation instructions currently target Linux-like systems with NVIDIA CUDA support. Note that Windows and macOS are currently not officially supported.

### From PyPI

`proteinworkshop` is available for install [from PyPI](https://pypi.org/project/proteinworkshop/). This enables training of specific configurations via the CLI **or** using individual components from the benchmark, such as datasets, featurisers, or transforms, as drop-ins to other projects. Make sure to install [PyTorch](https://pytorch.org/) (specifically version `2.1.2` or newer) using its official `pip` installation instructions, with CUDA support as desired.

```bash
# install `proteinworkshop` from PyPI
pip install proteinworkshop

# install PyTorch Geometric using the (now-installed) CLI
workshop install pyg

# set a custom data directory for file downloads; otherwise, all data will be downloaded to `site-packages`
export DATA_PATH="where/you/want/data/" # e.g., `export DATA_PATH="proteinworkshop/data"`
```

However, for full exploration we recommend cloning the repository and building from source.

### Building from source
With a local virtual environment activated (e.g., one created with `conda create -n proteinworkshop python=3.10`):
1. Clone and install the project

    ```bash
    git clone https://github.com/a-r-j/ProteinWorkshop
    cd ProteinWorkshop
    pip install -e .
    ```

2. Install [PyTorch](https://pytorch.org/) (specifically version `2.1.2` or newer) using its official `pip` installation instructions, with CUDA support as desired

    ```bash
    # e.g., to install PyTorch with CUDA 11.8 support on Linux:
    pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
    ```

3. Then use the newly-installed `proteinworkshop` CLI to install [PyTorch Geometric](https://pyg.org/)

    ```bash
    workshop install pyg
    ```

4. Configure paths in `.env` (optional, will override default paths if set). See [`.env.example`](https://github.com/a-r-j/proteinworkshop/blob/main/.env.example) for an example.

5. Download PDB data:

    ```bash
    python proteinworkshop/scripts/download_pdb_mmtf.py
    ```

## Tutorials

We provide a five-part tutorial series of Jupyter notebooks to provide users with examples
of how to use and extend `proteinworkshop`, as outlined below.

1. [Training a new model](https://github.com/a-r-j/ProteinWorkshop/blob/main/notebooks/training_new_model_tutorial.ipynb)
2. [Customizing an existing dataset](https://github.com/a-r-j/ProteinWorkshop/blob/main/notebooks/customizing_existing_dataset_tutorial.ipynb)
3. [Adding a new dataset](https://github.com/a-r-j/ProteinWorkshop/blob/main/notebooks/adding_new_dataset_tutorial.ipynb)
4. [Adding a new model](https://github.com/a-r-j/ProteinWorkshop/blob/main/notebooks/adding_new_model_tutorial.ipynb)
5. [Adding a new task](https://github.com/a-r-j/ProteinWorkshop/blob/main/notebooks/adding_new_task_tutorial.ipynb)

## Quickstart

### Downloading datasets

Datasets can either be built from the source structures or downloaded from [Zenodo](https://zenodo.org/record/8282470). Datasets will be built from source the first time a dataset is used in a run (or by calling the appropriate `setup()` method in the corresponding datamodule). We provide a CLI tool for downloading datasets:

```bash
workshop download <DATASET_NAME>
workshop download pdb
workshop download cath
workshop download afdb_rep_v4
# etc..
```

If you wish to build datasets from source, we recommend first downloading the entire PDB first (in MMTF format, c. 24 Gb) to reuse shared PDB data as much as possible:

```bash
workshop download pdb
# or
python proteinworkshop/scripts/download_pdb_mmtf.py
```

### Training a model

Launching an experiment minimally requires specification of a dataset, structural encoder, and task (devices can be specified with `trainer=cpu/gpu`):

```bash
workshop train dataset=cath encoder=egnn task=inverse_folding trainer=cpu env.paths.data=where/you/want/data/
# or
python proteinworkshop/train.py dataset=cath encoder=egnn task=inverse_folding trainer=cpu # or trainer=gpu
```

This command uses the default configurations in `configs/train.yaml`, which can be overwritten by equivalently named options. For instance, you can use a different input featurisation using the `features` option, or set the display name of your experiment on wandb using the `name` option:

```bash
workshop train dataset=cath encoder=egnn task=inverse_folding features=ca_bb name=MY-EXPT-NAME trainer=cpu env.paths.data=where/you/want/data/
# or
python proteinworkshop/train.py dataset=cath encoder=egnn task=inverse_folding features=ca_bb name=MY-EXPT-NAME trainer=cpu # or trainer=gpu
```

### Finetuning a model

Finetuning a model additionally requires specification of a checkpoint.

```bash
workshop finetune dataset=cath encoder=egnn task=inverse_folding ckpt_path=PATH/TO/CHECKPOINT trainer=cpu env.paths.data=where/you/want/data/
# or
python proteinworkshop/finetune.py dataset=cath encoder=egnn task=inverse_folding ckpt_path=PATH/TO/CHECKPOINT trainer=cpu # or trainer=gpu
```

### Running a sweep/experiment

We can make use of the hydra wandb sweeper plugin to configure experiments as sweeps, allowing searches over hyperparameters, architectures, pre-training/auxiliary tasks and datasets.

See `proteinworkshop/config/sweeps/` for examples.

1. Create the sweep with weights and biases

  ```bash
  wandb sweep proteinworkshop/config/sweeps/my_new_sweep_config.yaml
  ```

2. Launch job workers

With wandb:

  ```bash
  wandb agent mywandbgroup/proteinworkshop/2wwtt7oy --count 8
  ```

Or an example SLURM submission script:

  ```bash
  #!/bin/bash
  #SBATCH --nodes 1
  #SBATCH --ntasks-per-node=1
  #SBATCH --gres=gpu:1
  #SBATCH --array=0-32

  source ~/.bashrc
  source $(conda info --base)/envs/proteinworkshop/bin/activate

  wandb agent mywandbgroup/proteinworkshop/2wwtt7oy --count 1
  ```

Reproduce the sweeps performed in the manuscript:

```bash
# reproduce the baseline tasks sweep (i.e., those performed without pre-training each model)
wandb sweep proteinworkshop/config/sweeps/baseline_fold.yaml
wandb agent mywandbgroup/proteinworkshop/2awtt7oy --count 8
wandb sweep proteinworkshop/config/sweeps/baseline_ppi.yaml
wandb agent mywandbgroup/proteinworkshop/2bwtt7oy --count 8
wandb sweep proteinworkshop/config/sweeps/baseline_inverse_folding.yaml
wandb agent mywandbgroup/proteinworkshop/2cwtt7oy --count 8

# reproduce the model pre-training sweep
wandb sweep proteinworkshop/config/sweeps/pre_train.yaml
wandb agent mywandbgroup/proteinworkshop/2dwtt7oy --count 8

# reproduce the pre-trained tasks sweep (i.e., those performed after pre-training each model)
wandb sweep proteinworkshop/config/sweeps/pt_fold.yaml
wandb agent mywandbgroup/proteinworkshop/2ewtt7oy --count 8
wandb sweep proteinworkshop/config/sweeps/pt_ppi.yaml
wandb agent mywandbgroup/proteinworkshop/2fwtt7oy --count 8
wandb sweep proteinworkshop/config/sweeps/pt_inverse_folding.yaml
wandb agent mywandbgroup/proteinworkshop/2gwtt7oy --count 8
```

### Embedding a dataset
We provide a utility in `proteinworkshop/embed.py` for embedding a dataset using a pre-trained model.
To run it:
```bash
python proteinworkshop/embed.py ckpt_path=PATH/TO/CHECKPOINT collection_name=COLLECTION_NAME
```
See the `embed` section of `proteinworkshop/config/embed.yaml` for additional parameters.

### Visualising pre-trained model embeddings for a given dataset
We provide a utility in `proteinworkshop/visualise.py` for visualising the UMAP embeddings of a pre-trained model for a given dataset.
To run it:
```bash
python proteinworkshop/visualise.py ckpt_path=PATH/TO/CHECKPOINT plot_filepath=VISUALISATION/FILEPATH.png
```
See the `visualise` section of `proteinworkshop/config/visualise.yaml` for additional parameters.

### Performing attribution of a pre-trained model

We provide a utility in `proteinworkshop/explain.py` for performing attribution of a pre-trained model using integrated gradients.

This will write PDB files for all the structures in a dataset for a supervised task with residue-level attributions in the `b_factor` column. To visualise the attributions, we recommend using the [Protein Viewer VSCode extension](https://marketplace.visualstudio.com/items?itemName=ArianJamasb.protein-viewer) and changing the 3D representation to colour by `Uncertainty/Disorder`.

To run the attribution:

```bash
python proteinworkshop/explain.py ckpt_path=PATH/TO/CHECKPOINT output_dir=ATTRIBUTION/DIRECTORY
```

See the `explain` section of `proteinworkshop/config/explain.yaml` for additional parameters.


### Verifying a config

```bash
python proteinworkshop/validate_config.py dataset=cath features=full_atom task=inverse_folding
```

### Using `proteinworkshop` modules functionally

One may use the modules (e.g., datasets, models, featurisers, and utilities) of `proteinworkshop`
functionally by importing them directly. When installing this package using PyPi, this makes building
on top of the assets of `proteinworkshop` straightforward and convenient.

For example, to use any datamodule available in `proteinworkshop`:

```python
from proteinworkshop.datasets.cath import CATHDataModule

datamodule = CATHDataModule(path="data/cath/", pdb_dir="data/pdb/", format="mmtf", batch_size=32)
datamodule.download()

train_dl = datamodule.train_dataloader()
```

To use any model or featuriser available in `proteinworkshop`:

```python
from proteinworkshop.models.graph_encoders.dimenetpp import DimeNetPPModel
from proteinworkshop.features.factory import ProteinFeaturiser
from proteinworkshop.datasets.utils import create_example_batch

model = DimeNetPPModel(hidden_channels=64, num_layers=3)
ca_featuriser = ProteinFeaturiser(
    representation="CA",
    scalar_node_features=["amino_acid_one_hot"],
    vector_node_features=[],
    edge_types=["knn_16"],
    scalar_edge_features=["edge_distance"],
    vector_edge_features=[],
)

example_batch = create_example_batch()
batch = ca_featuriser(example_batch)

model_outputs = model(example_batch)
```

Read [the docs](https://www.proteins.sh) for a full list of modules available in `proteinworkshop`.

## Models

### Invariant Graph Encoders

| Name      | Source   | Protein Specific |
| ----------- | ----------- | ----------- |
| `GearNet`| [Zhang et al.](https://arxiv.org/abs/2203.06125) | ✓
| `DimeNet++`   | [Gasteiger et al.](https://arxiv.org/abs/2011.14115) | ✗
| `SchNet`   | [Schütt et al.](https://arxiv.org/abs/1706.08566) | ✗
| `CDConv`   | [Fan et al.](https://openreview.net/forum?id=P5Z-Zl9XJ7) | ✓

### Equivariant Graph Encoders

#### (Vector-type)

| Name      |  Source | Protein Specific |
| ----------- |  ----------- | --------- |
| `GCPNet`   | [Morehead et al.](https://academic.oup.com/bioinformatics/article/40/2/btae087/7610880) | ✓
| `GVP-GNN` | [Jing et al.](https://arxiv.org/abs/2009.01411) | ✓
| `EGNN`  | [Satorras et al.](https://arxiv.org/abs/2102.09844) | ✗

#### (Tensor-type)

| Name      |  Source | Protein Specific |
| ----------- |  ----------- | --------- |
| `Tensor Field Network` | [Corso et al.](https://arxiv.org/abs/2210.01776) | ✓
| `Multi-ACE` | [Batatia et al.](https://arxiv.org/abs/2206.07697) | ✗

### Sequence-based Encoders

| Name      | Source   | Protein Specific |
| ----------- | ----------- | ----------- |
| `ESM2`| [Lin et al.](https://www.science.org/doi/10.1126/science.ade2574) | ✓

## Datasets

To download a (processed) dataset from Zenodo, you can run

```bash
workshop download <DATASET_NAME>
```

where `<DATASET_NAME>` is given the first column in the tables below.

Otherwise, simply starting a training run will download and process the data from source.

### Structure-based Pre-training Corpuses

Pre-training corpuses (with the exception of `pdb`, `cath`, and `astral`) are provided in FoldComp database format. This format is highly compressed, resulting in very small disk space requirements despite the large size. `pdb` is provided as a collection of
`MMTF` files, which are significantly smaller in size than conventional `.pdb` or `.cif` file.

| Name      | Description   | Source |   Size  | Disk Size | License |
| ----------- | ----------- | ----------- | --- |  -- | ---- |
| `astral`| [SCOPe](https://scop.berkeley.edu/) domain structures       |  [SCOPe/ASTRAL](https://scop.berkeley.edu/)      | | 1 - 2.2 Gb | [Publicly available](https://scop.berkeley.edu/about/ver=2.08)
| `afdb_rep_v4`| Representative structures identified from the [AlphaFold database](https://alphafold.com/) by [FoldSeek](https://github.com/steineggerlab/foldseek) structural clustering        | [Barrio-Hernandez et al.](https://www.biorxiv.org/content/10.1101/2023.03.09.531927v1)       | 2.27M Chains | 9.6 Gb|  [GPL-3.0](https://github.com/steineggerlab/foldcomp/blob/master/LICENSE.txt) |
| `afdb_rep_dark_v4`| Dark proteome structures identied by structural clustering of the [AlphaFold database](https://alphafold.com/).       |  [Barrio-Hernandez et al.](https://www.biorxiv.org/content/10.1101/2023.03.09.531927v1)      | ~800k | 2.2 Gb| [GPL-3.0](https://github.com/steineggerlab/foldcomp/blob/master/LICENSE.txt) |
| `afdb_swissprot_v4`| AlphaFold2 predictions for [SwissProt/UniProtKB](https://www.uniprot.org/help/uniprotkb)       | [Kim et al.](https://academic.oup.com/bioinformatics/article/39/4/btad153/7085592)        | 542k Chains | 2.9 Gb | [GPL-3.0](https://github.com/steineggerlab/foldcomp/blob/master/LICENSE.txt) |
| `afdb_uniprot_v4`| AlphaFold2 predictions for [UniProt](https://www.uniprot.org/)       | [Kim et al.](https://academic.oup.com/bioinformatics/article/39/4/btad153/7085592)       | 214M Chains| 1 Tb| [GPL-3.0](https://github.com/steineggerlab/foldcomp/blob/master/LICENSE.txt) / [CC-BY 4.0](https://alphafold.ebi.ac.uk/assets/License-Disclaimer.pdf)|
| `cath`| [CATH](https://www.cathdb.info/) 4.2 40% split by CATH topologies.      |  [Ingraham et al.](https://www.mit.edu/~vgarg/GenerativeModelsForProteinDesign.pdf)      | ~18k chains| 4.3 Gb| [CC-BY 4.0](http://cathdb.info/)
| `esmatlas` | [ESMAtlas](https://esmatlas.com/) predictions  (full)     | [Kim et al.](https://academic.oup.com/bioinformatics/article/39/4/btad153/7085592) | | 1 Tb | [GPL-3.0](https://github.com/steineggerlab/foldcomp/blob/master/LICENSE.txt) / [CC-BY 4.0](https://esmatlas.com/about)
| `esmatlas_v2023_02`| [ESMAtlas](https://esmatlas.com/) predictions (v2023_02 release)      | [Kim et al.](https://academic.oup.com/bioinformatics/article/39/4/btad153/7085592)       | | 137 Gb| [GPL-3.0](https://github.com/steineggerlab/foldcomp/blob/master/LICENSE.txt) / [CC-BY 4.0](https://esmatlas.com/about)
| `highquality_clust30`| [ESMAtlas](https://esmatlas.com/) High Quality predictions       |  [Kim et al.](https://academic.oup.com/bioinformatics/article/39/4/btad153/7085592)      | 37M Chains | 114 Gb |  [GPL-3.0](https://github.com/steineggerlab/foldcomp/blob/master/LICENSE.txt) / [CC-BY 4.0](https://esmatlas.com/about)
| `igfold_paired_oas` | IGFold Predictions for [Paired OAS](https://journals.aai.org/jimmunol/article/201/8/2502/107069/Observed-Antibody-Space-A-Resource-for-Data-Mining) | [Ruffolo et al.](https://www.nature.com/articles/s41467-023-38063-x) | 104,994 paired Ab chains | | [CC-BY 4.0](https://www.nature.com/articles/s41467-023-38063-x#rightslink)
| `igfold_jaffe` | IGFold predictions for [Jaffe2022](https://www.nature.com/articles/s41586-022-05371-z) data | [Ruffolo et al.](https://www.nature.com/articles/s41467-023-38063-x) | 1,340,180 paired Ab chains   | | [CC-BY 4.0](https://www.nature.com/articles/s41467-023-38063-x#rightslink)
| `pdb`| Experimental structures deposited in the [RCSB Protein Data Bank](https://www.rcsb.org/)       |  [wwPDB consortium](https://academic.oup.com/nar/article/47/D1/D520/5144142)      | ~800k Chains |23 Gb | [CC0 1.0](https://www.rcsb.org/news/feature/611e8d97ef055f03d1f222c6) |


<details>
  <summary>Additionally, we provide several species-specific compilations (mostly reference species)</summary>

  | Name            | Description   | Source | Size |
  | ----------------| ----------- | ------ | ------ |
  | `a_thaliana`    | _Arabidopsis thaliana_ (thale cress) proteome | AlphaFold2|
  | `c_albicans`    | _Candida albicans_ (a fungus) proteome | AlphaFold2|
  | `c_elegans`     | _Caenorhabditis elegans_ (roundworm) proteome        | AlphaFold2       | |
  | `d_discoideum`  | _Dictyostelium discoideum_ (slime mold) proteome | AlphaFold2| |
  | `d_melanogaster`  | [_Drosophila melanogaster_](https://www.uniprot.org/taxonomy/7227) (fruit fly) proteome        | AlphaFold2        | |
  | `d_rerio`  | [_Danio rerio_](https://www.uniprot.org/taxonomy/7955) (zebrafish) proteome        | AlphaFold2        | |
  | `e_coli`  | _Escherichia coli_ (a bacteria) proteome        |  AlphaFold2       | |
  | `g_max`  | _Glycine max_ (soy bean) proteome       | AlphaFold2        | |
  | `h_sapiens`  | _Homo sapiens_ (human) proteome       |  AlphaFold2       | |
  | `m_jannaschii`  | _Methanocaldococcus jannaschii_ (an archaea) proteome        |  AlphaFold2       | |
  | `m_musculus`  | _Mus musculus_ (mouse) proteome        |   AlphaFold2      | |
  | `o_sativa`  | _Oryza sative_ (rice) proteome     |   AlphaFold2      | |
  | `r_norvegicus`  | _Rattus norvegicus_ (brown rat) proteome       |   AlphaFold2      | |
  | `s_cerevisiae`  | _Saccharomyces cerevisiae_ (brewer's yeast) proteome       |   AlphaFold2      | |
  | `s_pombe`  | _Schizosaccharomyces pombe_ (a fungus) proteome      |  AlphaFold2       | |
  | `z_mays`  | _Zea mays_ (corn) proteome    |   AlphaFold2      | |

</details>

### Supervised Datasets

| Name      | Description   | Source | License |
| ----------- | ----------- | ----------- | ---- |
| `antibody_developability` | Antibody developability prediction | [Chen et al.](https://www.biorxiv.org/content/10.1101/2020.06.18.159798v1.abstract) | [CC-BY 3.0](https://tdcommons.ai/single_pred_tasks/develop/#sabdab-chen-et-al) |
| `atom3d_msp` | Mutation stability prediction      | [Townshend et al.](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/c45147dee729311ef5b5c3003946c48f-Paper-round1.pdf) | [MIT](https://github.com/drorlab/atom3d/blob/master/LICENSE) |
| `atom3d_ppi` | Protein-protein interaction prediction      | [Townshend et al.](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/c45147dee729311ef5b5c3003946c48f-Paper-round1.pdf) | [MIT](https://github.com/drorlab/atom3d/blob/master/LICENSE) |
| `atom3d_psr` | Protein structure ranking      | [Townshend et al.](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/c45147dee729311ef5b5c3003946c48f-Paper-round1.pdf) | [MIT](https://github.com/drorlab/atom3d/blob/master/LICENSE) |
| `atom3d_res` | Residue identity prediction      | [Townshend et al.](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/c45147dee729311ef5b5c3003946c48f-Paper-round1.pdf) | [MIT](https://github.com/drorlab/atom3d/blob/master/LICENSE) |
|`ccpdb_ligands`| Ligand binding residue prediction | [Agrawal et al.](https://academic.oup.com/database/article/doi/10.1093/database/bay142/5298333#130010908) | [Publicly Available](https://academic.oup.com/database/article/doi/10.1093/database/bay142/5298333#130010908)
|`ccpdb_metal`| Metal ion binding residue prediction | [Agrawal et al.](https://academic.oup.com/database/article/doi/10.1093/database/bay142/5298333#130010908)  | [Publicly Available](https://academic.oup.com/database/article/doi/10.1093/database/bay142/5298333#130010908)
|`ccpdb_nucleic`| Nucleic acid binding residue prediction | [Agrawal et al.](https://academic.oup.com/database/article/doi/10.1093/database/bay142/5298333#130010908)  | [Publicly Available](https://academic.oup.com/database/article/doi/10.1093/database/bay142/5298333#130010908)
|`ccpdb_nucleotides`| Nucleotide binding residue prediction | [Agrawal et al.](https://academic.oup.com/database/article/doi/10.1093/database/bay142/5298333#130010908)  | [Publicly Available](https://academic.oup.com/database/article/doi/10.1093/database/bay142/5298333#130010908)
| `deep_sea_proteins` | Gene Ontology prediction (Biological Process)      |    [Sieg et al.](https://onlinelibrary.wiley.com/doi/10.1002/prot.26337uuujj)    | [Public domain](https://onlinelibrary.wiley.com/doi/10.1002/prot.26337)
| `go-bp` | Gene Ontology prediction (Biological Process)      |    [Gligorijevic et al](https://www.nature.com/articles/s41467-021-23303-9)    | [CC-BY 4.0](https://www.nature.com/articles/s41467-021-23303-9)|
| `go-cc` | Gene Ontology (Cellular Component)       | [Gligorijevic et al](https://www.nature.com/articles/s41467-021-23303-9)       | [CC-BY 4.0](https://www.nature.com/articles/s41467-021-23303-9) |
| `go-mf` | Gene Ontology (Molecular Function)       | [Gligorijevic et al](https://www.nature.com/articles/s41467-021-23303-9)       | [CC-BY 4.0](https://www.nature.com/articles/s41467-021-23303-9) |
| `ec_reaction` | Enzyme Commission (EC) Number Prediction       |  [Hermosilla et al.](https://arxiv.org/abs/2007.06252)      | [MIT](https://github.com/phermosilla/IEConv_proteins/blob/master/LICENSE)
| `fold_fold` |   Fold prediction, split at the fold level      |  [Hou et al.](https://academic.oup.com/bioinformatics/article/34/8/1295/4708302)       | [CC-BY 4.0](https://academic.oup.com/bioinformatics/article/34/8/1295/4708302)
| `fold_family` |  Fold prediction, split at the family level       |  [Hou et al.](https://academic.oup.com/bioinformatics/article/34/8/1295/4708302)      | [CC-BY 4.0](https://academic.oup.com/bioinformatics/article/34/8/1295/4708302)
| `fold_superfamily` |   Fold prediction, split at the superfamily level      | [Hou et al.](https://academic.oup.com/bioinformatics/article/34/8/1295/4708302)       | [CC-BY 4.0](https://academic.oup.com/bioinformatics/article/34/8/1295/4708302)
| `masif_site` | Protein-protein interaction site prediction | [Gainza et al.](https://www.nature.com/articles/s41592-019-0666-6)       | [Apache 2.0](https://github.com/LPDI-EPFL/masif/blob/master/LICENSE)
| `metal_3d` | Zinc Binding Site Prediction | [Duerr et al.](https://www.nature.com/articles/s41467-023-37870-6) | [MIT](https://zenodo.org/record/7594085)
| `ptm` | Post Translational Modification Side Prediction | [Yan et al.](https://www.sciencedirect.com/science/article/pii/S2667237523000450?via%3Dihub) | [CC-BY 4.0](https://zenodo.org/record/7655709) |


## Tasks

### Self-Supervised Tasks

| Name      | Description   | Source |
| ----------- | ----------- | ----------- |
| `inverse_folding` | Predict amino acid sequence given structure      |        |
| `residue_prediction` | Masked residue type prediction      |      |
| `distance_prediction` | Masked edge distance prediction      | [Zhang et al.](https://arxiv.org/pdf/2203.06125.pdf)        |
| `angle_prediction` | Masked triplet angle prediction     | [Zhang et al.](https://arxiv.org/pdf/2203.06125.pdf)        |
| `dihedral_angle_prediction` | Masked quadruplet dihedral prediction       |  [Zhang et al.](https://arxiv.org/pdf/2203.06125.pdf)       |
| `multiview_contrast` | Contrastive learning with multiple crops and InfoNCE loss       | [Zhang et al.](https://arxiv.org/pdf/2203.06125.pdf)       |
| `structural_denoising` |  Denoising of atomic coordinates with SE(3) decoders      |        |

### Generic Supervised Tasks

Generic supervised tasks can be applied broadly across datasets. The labels are directly extracted from the PDB structures.

These are likely to be most frequently used with the [`pdb`](https://github.com/a-r-j/ProteinWorkshop/blob/main/configs/dataset/pdb.yaml) dataset class which wraps the [PDB Dataset curator](https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/creating_datasets_from_the_pdb.ipynb) from [Graphein](https://github.com/a-r-j/graphein).

| Name      | Description   | Requires |
| ----------- | ----------- | ----------- |
| `binding_site_prediction` | Predict ligand binding residues | HETATM ligands (for training) |
| `ppi_site_prediction` | Predict protein binding residues | `graph_y` attribute in data objects specifying the desired chain to select interactions for (for training) |

## Featurisation Schemes

Part of the goal of the `proteinworkshop` benchmark is to investigate the impact of the degree to which increasing granularity of structural detail affects performance. To achieve this, we provide several featurisation schemes for protein structures.

### Invariant Node Features

N.B. All angular features are provided in [sin, cos] transformed form. E.g.: $\textrm{dihedrals} = [sin(\phi), cos(\phi), sin(\psi), cos(\psi), sin(\omega), \cos(\omega)]$, hence their dimensionality will be double the number of angles.

| Name      | Description   | Dimensionality |
| ----------- | ----------- | ----------- |
| `residue_type` | One-hot encoding of amino acid type       |      21  |
| `positional_encoding` | Transformer-like positional encoding of sequence position       |      16  |
| `alpha` | Virtual torsion angle defined by four $C_\alpha$ atoms of residues $I_{-1}, I, I_{+1}, I_{+2}$       |      2  |
| `kappa` | Virtual bond angle (bend angle) defined by the three $C_\alpha$ atoms of residues $I_{-2}, I, I_{+2}$       |      2  |
| `dihedrals` | Backbone dihedral angles $(\phi, \psi, \omega)$      |      6  |
| `sidechain_torsions` | Sidechain torsion angles  $(\chi_{1-4})$     |    8    |

### Equivariant Node Features

| Name      | Description   | Dimensionality |
| ----------- | ----------- | ----------- |
| `orientation` | Forward and backward node orientation vectors (unit-normalized)        |      2  |

### Edge Construction

We predominanty support two types of edges: $k$-NN and $\epsilon$ edges.

Edge types can be specified as follows:

```bash
python proteinworkshop/train.py ... features.edge_types=[knn_16, knn_32, eps_16]
```

Where the suffix after `knn` or `eps` specifies $k$ (number of neighbours) or $\epsilon$ (distance threshold in angstroms).

### Invariant Edge Features

| Name      | Description   | Dimensionality |
| ----------- | ----------- | ----------- |
| `edge_distance` | Euclidean distance between source and target nodes        |    1  |
| `node_features` | Concatenated scalar node features of the source and target nodes        |     Number of scalar node features $\times 2$  |
| `edge_type` | Type annotation for each edge        |    1  |
| `sequence_distance` | Sequence-based distance between source and target nodes        |    1  |
| `pos_emb` | Structured Transformer-inspired positional embedding of $i - j$ for source node $i$ and target node $j$        |    16  |

### Equivariant Edge Features

| Name      | Description   | Dimensionality |
| ----------- | ----------- | ----------- |
| `edge_vectors` | Edge directional vectors (unit-normalized)        |      1  |

## For Developers

### Dependency Management
We use `poetry` to manage the project's underlying dependencies and to push updates to the project's PyPI package. To make changes to the project's dependencies, follow the instructions below to (**1**) install `poetry` on your local machine; (**2**) customize the dependencies; or (**3**) (de)activate the project's virtual environment using `poetry`:
1. Install `poetry` for platform-agnostic dependency management using its [installation instructions](https://python-poetry.org/docs/)

    After installing `poetry`, to avoid potential [keyring errors](https://github.com/python-poetry/poetry/issues/1917#issuecomment-1235998997), disable its keyring usage by adding `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` to your shell's startup configuration and restarting your shell environment (e.g., `echo 'export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring' >> ~/.bashrc && source ~/.bashrc` for a Bash shell environment and likewise for other shell environments).

2. Install, add, or upgrade project dependencies

    ```bash
      poetry install  # install the latest project dependencies
      # or
      poetry add XYZ  # add dependency `XYZ` to the project
      # or
      poetry show  # list all dependencies currently installed
      # or
      poetry lock  # standardize the (now-)installed dependencies
    ```

3. Activate the newly-created virtual environment following `poetry`'s [usage documentation](https://python-poetry.org/docs/basic-usage/)

    ```bash
      # activate the environment on a `posix`-like (e.g., macOS or Linux) system
      source $(poetry env info --path)/bin/activate
    ```
    ```powershell
      # activate the environment on a `Windows`-like system
      & ((poetry env info --path) + "\Scripts\activate.ps1")
    ```
    ```bash
      # if desired, deactivate the environment
      deactivate
    ```

### Code Formatting
To keep with the code style for the `proteinworkshop` repository, using the following lines, please format your commits before opening a pull request:
```bash
# assuming you are located in the `ProteinWorkshop` top-level directory
isort .
autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports --ignore-init-module-imports .
black --config=pyproject.toml .
```

### Documentation
To build a local version of the project's Sphinx documentation web pages:
```bash
# assuming you are located in the `ProteinWorkshop` top-level directory
pip install -r docs/.docs.requirements # one-time only
rm -rf docs/build/ && sphinx-build docs/source/ docs/build/ # NOTE: errors can safely be ignored
```

## Citing `ProteinWorkshop`

Please consider citing `proteinworkshop` if it proves useful in your work.

```bibtex
@inproceedings{
  jamasb2024evaluating,
  title={Evaluating Representation Learning on the Protein Structure Universe},
  author={Arian R. Jamasb, Alex Morehead, Chaitanya K. Joshi, Zuobai Zhang, Kieran Didi, Simon V. Mathis, Charles Harris, Jian Tang, Jianlin Cheng, Pietro Lio, Tom L. Blundell},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
}

```
