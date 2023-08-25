# Protein Workshop

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://assets.readthedocs.org/static/projects/badges/passing-flat.svg)](https://www.proteins.sh)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

![Overview of the Protein Workshop](https://github.com/a-r-j/ProteinWorkshop/blob/main/docs/source/_static/workshop_overview.png)

[Documentation](https://www.proteins.sh)

This repository provides the code for the protein structure representation learning benchmark detailed in the paper *Evaluating Representation Learning on the Protein Structure Universe*.

In the benchmark, we implement numerous [featurisation](https://www.proteins.sh/configs/features) schemes, [datasets](https://www.proteins.sh/configs/dataset) for [self-supervised pre-training](https://proteins.sh/quickstart_component/pretrain.html) and [downstream evaluation](https://proteins.sh/quickstart_component/downstream.html), [pre-training](https://proteins.sh/configs/task) tasks, and [auxiliary tasks](https://proteins.sh/configs/task.html#auxiliary-tasks).

[Processed datasets](https://drive.google.com/drive/folders/18i8rLST6ZICTBu6Q67ClT0KqN9AHeqoW?usp=sharing) and [pre-trained weights](https://drive.google.com/drive/folders/1zK1r8FpmGaqV_QwUJuvDacwSL0RW-Vw9?usp=sharing) are made available. Downloading datasets is not required; upon first run all datasets will be downloaded and processed from their respective source.

Configuration files to run the experiments described in the manuscript are provided in the `configs/sweeps/` directory.

## Contents

* [Installation](#installation)
* [Tutorials](#tutorials)
* [Quickstart](#quickstart)
* [Datasets](#datasets)
  * [Pre-training corpuses](#structure-based-pre-training-corpuses)
  * [Supervised graph-level tasks](#supervised-datasets)
  * [Supervised node-level tasks](#supervised-datasets)
* [Models](#models)
  * [Invariant structure encoders](#invariant-graph-encoders)
  * [Equivariant structure encoders](#equivariant-graph-encoders)
* [Featurisation Schemes](#featurisation-schemes)
  * [Invariant Node Features](#invariant-node-features)
  * [Equivariant Node Features](#equivariant-node-features)
  * [Edge Construction](#edge-construction)
  * [Invariant Edge Features](#invariant-edge-features)
  * [Equivariant Edge Features](#equivariant-edge-features)

## Installation

Below, we outline how one may set up a virtual environment for the `ProteinWorkshop`. Note that these installation instructions currently target Linux-like systems with NVIDIA CUDA support. Note that Windows and macOS are currently not officially supported.

1. Install `poetry` for dependency management using its [installation instructions](https://python-poetry.org/docs/)

2. Install project dependencies

    ```bash
    poetry install
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

4. With the environment activated, install [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pyg.org/) using their official `pip` installation instructions (with CUDA support as desired)

    ```bash
    # hint: to see the list of dependencies that are currently installed in the environment, run:
    poetry show
    ```

5. Configure paths in `.env`. See [`.env.example`](https://github.com/a-r-j/ProteinWorkshop/blob/main/.env.example) for an example.

6. Download PDB data:

    ```bash
    python scripts/download_pdb_mmtf.py
    ```

## Tutorials

We provide a five-part tutorial series of Jupyter notebooks to provide users with examples
of how to use and extend the `Protein Workshop`, as outlined below.

1. [Training a new model](https://github.com/a-r-j/ProteinWorkshop/blob/main/notebooks/training_new_model_tutorial.ipynb)
2. [Customizing an existing dataset](https://github.com/a-r-j/ProteinWorkshop/blob/main/notebooks/customizing_existing_dataset_tutorial.ipynb)
3. [Adding a new dataset](https://github.com/a-r-j/ProteinWorkshop/blob/main/notebooks/adding_new_dataset_tutorial.ipynb)
4. [Adding a new model](https://github.com/a-r-j/ProteinWorkshop/blob/main/notebooks/adding_new_model_tutorial.ipynb)
5. [Adding a new task](https://github.com/a-r-j/ProteinWorkshop/blob/main/notebooks/adding_new_task_tutorial.ipynb)

### Quickstart

#### Training a model

Minimally, launching an experiment minimally requires specification of a dataset, structural encoder, and task:

```bash
python proteinworkshop/train.py dataset=cath encoder=gcn task=inverse_folding
```

#### Finetuning a model

Finetuning a model additionally requires specification of a checkpoint.

```bash
python proteinworkshop/finetune.py dataset=cath encoder=gcn task=inverse_folding ckpt_path=PATH/TO/CHECKPOINT
```

### Running a sweep/experiment

We can make use of the hydra wandb sweeper plugin to configure experiments as sweeps, allowing searches over hyperparameters, architectures, pre-training/auxiliary tasks and datasets.

See `configs/sweeps/` for examples.

1. Create the sweep with weights and biases

  ```bash
  wandb sweep configs/sweeps/my_new_sweep_config.yaml
  ```

2. Launch job workers

With wandb:

  ```bash
  wandb agent mywandbgroup/ProteinWorkshop/2wwtt7oy --count 8
  ```

Or an example SLURM submission script:

  ```bash
  #!/bin/bash
  #SBATCH --nodes 1
  #SBATCH --ntasks-per-node=1
  #SBATCH --gres=gpu:1
  #SBATCH --array=0-32

  source ~/.bashrc
  source $(poetry env info --path)/bin/activate

  wandb agent mywandbgroup/ProteinWorkshop/2wwtt7oy --count 1
  ```

Reproduce the sweeps performed in the manuscript:

```bash
# reproduce the baseline tasks sweep (i.e., those performed without pre-training each model)
wandb sweep configs/sweeps/baseline_fold.yaml
wandb agent mywandbgroup/ProteinWorkshop/2awtt7oy --count 8
wandb sweep configs/sweeps/baseline_ppi.yaml
wandb agent mywandbgroup/ProteinWorkshop/2bwtt7oy --count 8
wandb sweep configs/sweeps/baseline_inverse_folding.yaml
wandb agent mywandbgroup/ProteinWorkshop/2cwtt7oy --count 8

# reproduce the model pre-training sweep
wandb sweep configs/sweeps/pre_train.yaml
wandb agent mywandbgroup/ProteinWorkshop/2dwtt7oy --count 8

# reproduce the pre-trained tasks sweep (i.e., those performed after pre-training each model)
wandb sweep configs/sweeps/pt_fold.yaml
wandb agent mywandbgroup/ProteinWorkshop/2ewtt7oy --count 8
wandb sweep configs/sweeps/pt_ppi.yaml
wandb agent mywandbgroup/ProteinWorkshop/2fwtt7oy --count 8
wandb sweep configs/sweeps/pt_inverse_folding.yaml
wandb agent mywandbgroup/ProteinWorkshop/2gwtt7oy --count 8
```

#### Embedding a dataset

```bash
python proteinworkshop/embed.py dataset=cath encoder=gnn ckpt_path=PATH/TO/CHECKPOINT
```

#### Verify a config

```bash
python proteinworkshop/validate_config.py dataset=cath features=full_atom task=inverse_folding
```

## Models

### Invariant Graph Encoders

| Name      | Source   | Protein Specific |
| ----------- | ----------- | ----------- |
| `GearNet`| [Zhang et al.](https://arxiv.org/pdf/2203.06125) | ✓ |
| `ProNet`   | [Wang et al.](https://arxiv.org/abs/2207.12600) | ✓ |
| `DimeNet++`   | [Gasteiger et al.](https://arxiv.org/abs/2011.14115) | ✗ |
| `SchNet`   | [Schütt et al.](https://arxiv.org/abs/1706.08566) | ✗ |

### Equivariant Graph Encoders

#### (Vector-type)

| Name      |  Source | Protein Specific |
| ----------- |  ----------- | --------- |
| `GCPNet`   | [Morehead et al.](https://arxiv.org/abs/2211.02504) | ✓
| `GVP-GNN` | [Jing et al.](https://arxiv.org/abs/2009.01411) | ✓
| `EGNN`  | [Satorras et al.](https://arxiv.org/abs/2102.09844) | ✗

#### (Tensor-type)

| Name      |  Source | Protein Specific |
| ----------- |  ----------- | --------- |
| `Tensor Field Network` | [Corso et al.](https://arxiv.org/abs/2210.01776) | ❓
| `Multi-ACE` | [Batatia et al.](https://arxiv.org/abs/2206.07697) | ✗

## Datasets

### Structure-based Pre-training Corpuses

Pre-training corpuses (with the exception of `pdb`, `cath`, and `astral`) are provided in FoldComp database format. This format is highly compressed, resulting in very small disk space requirements despite the large size. `pdb` is provided as a collection of
`MMTF` files, which are significantly smaller in size than conventional `.pdb` or `.cif` file.

| Name      | Description   | Source |   Size  | Disk Size | License |
| ----------- | ----------- | ----------- | --- |  -- | ---- |
| `astral`| [SCOPe](https://scop.berkeley.edu/) domain structures       |  [SCOPe/ASTRAL](https://scop.berkeley.edu/)      | | 1 - 2.2 Gb | [Publicly available](https://scop.berkeley.edu/about/ver=2.08)
| `afdb_rep_v4`| Representative structures identified from the [AlphaFold database](https://alphafold.com/) by [FoldSeek](https://github.com/steineggerlab/foldseek) structural clustering        | [Barrio-Hernandez et al.](https://www.biorxiv.org/content/10.1101/2023.03.09.531927v1)       | 2.27M Chains | 9.6 Gb|  [GPL-3.0](https://github.com/steineggerlab/foldcomp/blob/master/LICENSE.txt) |
| `afdb_rep_dark_v4`| Dark proteome structures identied by structural clustering of the [AlphaFold database](https://alphafold.com/).       |  [Barrio-Hernandez et al.](https://www.biorxiv.org/content/10.1101/2023.03.09.531927v1)      | ~800k | 2.2 Gb| [GPL-3.0](https://github.com/steineggerlab/foldcomp/blob/master/LICENSE.txt) |
| `afdb_swissprot_v4`| AlphaFold2 predictions for [SwissProt/UniProtKB](https://www.uniprot.org/help/uniprotkb)       | [Kim et al.](https://academic.oup.com/bioinformatics/article/39/4/btad153/7085592)        | 542k Chains | 2.9 Gb | [GPL-3.0](https://github.com/steineggerlab/foldcomp/blob/master/LICENSE.txt) |
| `afdb_uniprot_v4`| AlphaFold2 predictions for [UniProt](https://www.uniprot.org/)       | [Kim et al.](https://academic.oup.com/bioinformatics/article/39/4/btad153/7085592)       | | 1 Tb| [GPL-3.0](https://github.com/steineggerlab/foldcomp/blob/master/LICENSE.txt) / [CC-BY 4.0](https://alphafold.ebi.ac.uk/assets/License-Disclaimer.pdf)|
| `cath`| [CATH](https://www.cathdb.info/) 4.2 40% split by CATH topologies.      |  [Ingraham et al.](https://www.mit.edu/~vgarg/GenerativeModelsForProteinDesign.pdf)      | ~18k chains| | [CC-BY 4.0](http://cathdb.info/)
| `esmatlas_v2023_02`| [ESMAtlas](https://esmatlas.com/) predictions       | [Kim et al.](https://academic.oup.com/bioinformatics/article/39/4/btad153/7085592)       | | 137 Gb| [GPL-3.0](https://github.com/steineggerlab/foldcomp/blob/master/LICENSE.txt) / [CC-BY 4.0](https://esmatlas.com/about)
| `highquality_clust30`| [ESMAtlas](https://esmatlas.com/) High Quality predictions       |  [Kim et al.](https://academic.oup.com/bioinformatics/article/39/4/btad153/7085592)      | 37M Chains | 114 Gb |  [GPL-3.0](https://github.com/steineggerlab/foldcomp/blob/master/LICENSE.txt) / [CC-BY 4.0](https://esmatlas.com/about)
| `pdb`| Experimental structures deposited in the [RCSB Protein Data Bank](https://www.rcsb.org/)       |  [wwPDB consortium](https://academic.oup.com/nar/article/47/D1/D520/5144142)      | ~800k Chains |23 Gb | [CC0 1.0](https://www.rcsb.org/news/feature/611e8d97ef055f03d1f222c6) |

<details>
  <summary>Additionally, we provide several species-specific compilations</summary>

  | Name            | Description   | Source | Size |
  | ----------------| ----------- | ------ | ------ |
  | `a_thaliana`    | _Arabidopsis thaliana_ proteome | AlphaFold2|
  | `c_albicans`    | _Candida albicans_ proteome | AlphaFold2|
  | `c_elegans`     | _Caenorhabditis elegans_ proteome        | AlphaFold2       | |
  | `d_discoideum`  | _Dictyostelium discoideum_ proteome | AlphaFold2| |
  | `d_melanogaster`  | [_Drosophila melanogaster_](https://www.uniprot.org/taxonomy/7227) proteome        | AlphaFold2        | |
  | `d_rerio`  | [_Danio rerio_](https://www.uniprot.org/taxonomy/7955) proteome        | AlphaFold2        | |
  | `e_coli`  | Text        |  AlphaFold2       | |
  | `g_max`  | Text        | AlphaFold2        | |
  | `h_sapiens`  | Text        |  AlphaFold2       | |
  | `m_jannaschii`  | Text        |  AlphaFold2       | |
  | `m_musculus`  | Text        |   AlphaFold2      | |
  | `o_sativa`  | Text        |   AlphaFold2      | |
  | `r_norvegicus`  | Text        |   AlphaFold2      | |
  | `s_cerevisiae`  | Text        |   AlphaFold2      | |
  | `s_pombe`  | Text        |  AlphaFold2       | |
  | `z_mays`  | Text        |   AlphaFold2      | |

</details>

### Supervised Datasets

| Name      | Description   | Source | License |
| ----------- | ----------- | ----------- | ---- |
| `atom3d_msp` | Mutation stability prediction      | [Townshend et al.](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/c45147dee729311ef5b5c3003946c48f-Paper-round1.pdf) | [MIT](https://github.com/drorlab/atom3d/blob/master/LICENSE) |
| `atom3d_ppi` | Protein-protein interaction prediction      | [Townshend et al.](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/c45147dee729311ef5b5c3003946c48f-Paper-round1.pdf) | [MIT](https://github.com/drorlab/atom3d/blob/master/LICENSE) |
| `atom3d_psr` | Protein structure ranking      | [Townshend et al.](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/c45147dee729311ef5b5c3003946c48f-Paper-round1.pdf) | [MIT](https://github.com/drorlab/atom3d/blob/master/LICENSE) |
| `atom3d_res` | Residue identity prediction      | [Townshend et al.](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/c45147dee729311ef5b5c3003946c48f-Paper-round1.pdf) | [MIT](https://github.com/drorlab/atom3d/blob/master/LICENSE) |
| `deep_sea_proteins` | Gene Ontology prediction (Biological Process)      |    [Sieg et al.](https://onlinelibrary.wiley.com/doi/10.1002/prot.26337uuujj)    | [Public domain](https://onlinelibrary.wiley.com/doi/10.1002/prot.26337)
| `go-bp` | Gene Ontology prediction (Biological Process)      |    [Gligorijevic et al](https://www.nature.com/articles/s41467-021-23303-9)    | [CC-BY 4.0](https://www.nature.com/articles/s41467-021-23303-9)|
| `go-cc` | Gene Ontology (Cellular Component)       | [Gligorijevic et al](https://www.nature.com/articles/s41467-021-23303-9)       | [CC-BY 4.0](https://www.nature.com/articles/s41467-021-23303-9) |
| `go-mf` | Gene Ontology (Molecular Function)       | [Gligorijevic et al](https://www.nature.com/articles/s41467-021-23303-9)       | [CC-BY 4.0](https://www.nature.com/articles/s41467-021-23303-9) |
| `ec-reaction` | Enzyme Commission (EC) Number Prediction       |  [Hermosilla et al.](https://arxiv.org/abs/2007.06252)      | [MIT](https://github.com/phermosilla/IEConv_proteins/blob/master/LICENSE)
| `fold-fold` |   Fold prediction, split at the fold level      |  [Hou et al.](https://academic.oup.com/bioinformatics/article/34/8/1295/4708302)       | [CC-BY 4.0](https://academic.oup.com/bioinformatics/article/34/8/1295/4708302)
| `fold-family` |  Fold prediction, split at the family level       |  [Hou et al.](https://academic.oup.com/bioinformatics/article/34/8/1295/4708302)      | [CC-BY 4.0](https://academic.oup.com/bioinformatics/article/34/8/1295/4708302)
| `fold-superfamily` |   Fold prediction, split at the superfamily level      | [Hou et al.](https://academic.oup.com/bioinformatics/article/34/8/1295/4708302)       | [CC-BY 4.0](https://academic.oup.com/bioinformatics/article/34/8/1295/4708302)
| `masif-site` | Protein-protein interaction site prediction | [Gainza et al.](https://www.nature.com/articles/s41592-019-0666-6)       | [Apache 2.0](https://github.com/LPDI-EPFL/masif/blob/master/LICENSE)

## Tasks

### Self-supervision Tasks

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

Part of the goal of the benchmark is to investigate the impact of the degree to which increasing granularity of structural detail affects performance. To achieve this, we provide several featurisation schemes for protein structures.

### Invariant Node Features

N.B. All angular features are provided in [sin, cos] transformed form. E.g.: $\textrm{dihedrals} = [sin(\phi), cos(\phi), sin(\psi), cos(\psi), sin(\omega), \cos(\omega)]$, hence their dimensionality will be double the number of angles.

| Name      | Description   | Dimensionality |
| ----------- | ----------- | ----------- |
| `residue_type` | One-hot encoding of amino acid type       |      21  |
| `positional_encoding` | Transformer-like positional encoding of sequence position       |      16  |
| `alpha` | Virtual torsion angle defined by four $C_\alpha$ atoms of residues $I_{-1},I,I_{+1},I_{+2}$       |      2  |
| `kappa` | Virtual bond angle (bend angle) defined by the three $C_\alpha$ atoms of residues $I_{-2},I,_{+2}$       |      2  |
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
