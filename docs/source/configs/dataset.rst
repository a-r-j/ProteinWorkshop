Dataset
---------

.. image:: ../_static/box_datasets.png
  :alt: Summary of dataset sources
  :align: center
  :width: 400

To switch between different datasets, simply change the `dataset` argument in the launch command. For example:

.. code-block:: bash

    workshop train encoder=gear_net dataset=<DATASET_NAME> task=inverse_folding trainer=cpu
    # or
    python proteinworkshop/train.py encoder=gear_net dataset=<DATASET_NAME> task=inverse_folding trainer=cpu # or trainer=gpu

Where ``<DATASET_NAME>`` is given by bracketed name in the listing below. For example, the dataset name for CATH is ``cath``.


.. note::

    If you have pip-installed `proteinworkshop`, you can download pre-training or processed downstream datasets from Zenodo with:

    .. code-block:: bash

        workshop download <DATASET_NAME>


Unlabelled Datasets
===================


.. mdinclude:: ../../../README.md
    :start-line: 361
    :end-line: 406


:py:class:`ASTRAL <proteinworkshop.datasets.astral.AstralDataModule>` (``astral``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ASTRAL provides compendia of protein domain structures, regions of proteins that
can maintain their structure and function independently of the rest of the protein. Domains typically
exhibit highly-specific functions and can be considered structural building blocks of proteins.

.. literalinclude:: ../../../proteinworkshop/config/dataset/astral.yaml
    :language: yaml
    :caption: config/dataset/astral.yaml


:py:class:`CATH <proteinworkshop.datasets.cath.CATHDataModule>` (``cath``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../proteinworkshop/config/dataset/cath.yaml
    :language: yaml
    :caption: config/dataset/cath.yaml

:py:class:`PDB <proteinworkshop.datasets.pdb_dataset.PDBDataModule>` (``pdb``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::
    :py:class:`proteinworkshop.datasets.pdb_dataset.PDBData`

.. literalinclude:: ../../../proteinworkshop/config/dataset/pdb.yaml
    :language: yaml
    :caption: config/dataset/pdb.yaml



:py:class:`AFdb Rep. v4 <graphein.ml.datasets.foldcomp_dataset.FoldCompLightningDataModule>` (``afdb_rep_v4``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is a dataset of approximately 3 million protein structures from the AlphaFold database, structurally clustered using FoldSeek.

.. literalinclude:: ../../../proteinworkshop/config/dataset/afdb_rep_v4.yaml
    :language: yaml
    :caption: config/dataset/afdb_rep_v4.yaml


:py:class:`AFdb Dark Proteome <graphein.ml.datasets.foldcomp_dataset.FoldCompLightningDataModule>` (``afdb_rep_dark_v4``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../proteinworkshop/config/dataset/afdb_rep_dark_v4.yaml
    :language: yaml
    :caption: config/dataset/afdb_rep_dark_v4.yaml


:py:class:`ESM Atlas <graphein.ml.datasets.foldcomp_dataset.FoldCompLightningDataModule>` (``esmatlas_v2023_02``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../proteinworkshop/config/dataset/esmatlas_v2023_02.yaml
    :language: yaml
    :caption: config/dataset/esmatlas_v2023_02.yaml

:py:class:`ESM Atlas (High Quality) <graphein.ml.datasets.foldcomp_dataset.FoldCompLightningDataModule>` (``highquality_clust30``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../proteinworkshop/config/dataset/highquality_clust30.yaml
    :language: yaml
    :caption: config/dataset/highquality_clust30.yaml


:py:class:`UniProt (Alphafold) <graphein.ml.datasets.foldcomp_dataset.FoldCompLightningDataModule>` (``afdb_uniprot_v4``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../proteinworkshop/config/dataset/afdb_uniprot_v4.yaml
    :language: yaml
    :caption: config/dataset/afdb_uniprot_v4.yaml



:py:class:`SwissProt (Alphafold) <graphein.ml.datasets.foldcomp_dataset.FoldCompLightningDataModule>` (``afdb_swissprot_v4``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../proteinworkshop/config/dataset/afdb_swissprot_v4.yaml
    :language: yaml
    :caption: config/dataset/afdb_swissprot_v4.yaml



Species-Specific Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Stay tuned!


Graph-level Datasets
=============================

:py:class:`Antibody Developability <proteinworkshop.datasets.antibody_developability.AntibodyDevelopabilityDataModule>` (``antibody_developability``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Therapeutic antibodies must be optimised for favourable
physicochemical properties in addition to target binding affinity and specificity
to be viable development candidates. Consequently, this task frames prediction of antibody
developability as a binary graph classification task indicating whether a given antibody is developable

Dataset: We adopt the antibody developability dataset originally curated from SabDab by TDC.

Impact: From a benchmarking perspective, this task is important as it enables targeted performance
assessment of models on a specific (immunoglobulin) fold, providing insight into whether general-
purpose structure-based encoders can be applicable to fold-specific tasks.

.. literalinclude:: ../../../proteinworkshop/config/dataset/antibody_developability.yaml
    :language: yaml
    :caption: config/dataset/antibody_developability.yaml

:py:class:`Atom3D Mutation Stability Prediction <proteinworkshop.datasets.atom3d_datamodule.ATOM3DDataModule>` (``atom3d_msp``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This task is defined in the `Atom3D benchmark <https://www.atom3d.ai/msp.html>`_.

As per their documentation:

**Impact**: Identifying mutations that stabilize a protein's interactions is a key task in designing new proteins. Experimental techniques for probing these are labor intensive, motivating the development of efficient computational methods.

**Dataset description**: We derive a novel dataset by collecting single-point mutations from the SKEMPI database (Jankauskaitė et al., 2019) and model each mutation into the structure to produce mutated structures.

**Task**: We formulate this as a binary classification task where we predict whether the stability of the complex increases as a result of the mutation.

**Splitting criteria**: We split protein complexes by sequence identity at 30%.

**Downloads**: The full dataset, split data, and split indices are available for download via Zenodo (doi:10.5281/zenodo.4962515)


.. literalinclude:: ../../../proteinworkshop/config/dataset/atom3d_msp.yaml
    :language: yaml
    :caption: config/dataset/atom3d_msp.yaml

:py:class:`Atom3D Protein Structure Ranking <proteinworkshop.datasets.atom3d_datamodule.ATOM3DDataModule>` (``atom3d_psr``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This task is defined in the `Atom3D benchmark <https://www.atom3d.ai/psr.html>`_.

As per their documentation:

**Impact**: Proteins are one of the primary workhorses of the cell, and knowing their structure is often critical to understanding (and engineering) their function.

**Dataset description**: The Critical Assessment of Structure Prediction (CASP) (Kryshtafovych et al., 2019) is a blind international competition for predicting protein structure.

**Task**: We formulate this as a regression task, where we predict the global distance test (GDT_TS) from the true structure for each of the predicted structures submitted in the last 18 years of CASP.

**Splitting criteria**: We split structures temporally by competition year.

**Downloads**: The full dataset, split data, and split indices are available for download via Zenodo (doi:10.5281/zenodo.4915648)


.. literalinclude:: ../../../proteinworkshop/config/dataset/atom3d_psr.yaml
    :language: yaml
    :caption: config/dataset/atom3d_psr.yaml


:py:class:`Deep Sea Protein Classification <proteinworkshop.datasets.deep_sea_proteins.DeepSeaProteinsDataModule>` (``deep_sea_proteins``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../proteinworkshop/config/dataset/deep_sea_proteins.yaml
    :language: yaml
    :caption: config/dataset/deepsea.yaml


:py:class:`Enzyme Commission Number Prediction <proteinworkshop.datasets.ec_reaction.EnzymeCommissionReactionDataset>` (``ec_reaction``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../proteinworkshop/config/dataset/ec_reaction.yaml
    :language: yaml
    :caption: config/dataset/ec_reaction.yaml


:py:class:`Fold Classification <proteinworkshop.datasets.fold_classification.FoldClassificationDataModule>` (``fold-family``, ``fold-superfamily``, ``fold-fold``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is a multiclass graph classification task where each protein, G, is mapped to a
label y ∈ {1, . . . , 1195} denoting the fold class.

**Dataset**: We adopt the fold classification dataset originally curated from SCOP 1.75 by Hermosilla et al. In
particular, this dataset contains three distinct test splits across which we average a method’s results.

**Impact**: The utility of this task is that it serves as a litmus test for the ability of a model to distinguish
different structural folds. It stands to reason that models that perform poorly on distinguishing fold
classes likely learn limited or low-quality structural representations.214

**Splitting Criteria**:

.. literalinclude:: ../../../proteinworkshop/config/dataset/fold_family.yaml
    :language: yaml
    :caption: config/dataset/fold_family.yaml

Gene Ontology (``go-bp``, ``go-cc``, ``go-mf``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../proteinworkshop/config/dataset/go-bp.yaml
    :language: yaml
    :caption: config/dataset/go-bp.yaml / config/dataset/go-cc.yaml / config/dataset/go-mf.yaml


Node-level Datasets
=============================

Atom3D Residue Identity Prediction (``atom3d_res``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This task is defined in the `Atom3D benchmark <https://www.atom3d.ai/res.html>`_.

As per their documentation:

**Impact**: Understanding the structural role of individual amino acids is important for engineering new proteins. We can understand this role by predicting the substitutabilities of different amino acids at a given protein site based on the surrounding structural environment.

**Dataset description**: We generate a novel dataset consisting of atomic environments extracted from nonredundant structures in the PDB.

**Task**: We formulate this as a classification task where we predict the identity of the amino acid in the center of the environment based on all other atoms.

**Splitting criteria**: We split residue environments by domain-level CATH protein topology class.

.. literalinclude:: ../../../proteinworkshop/config/dataset/atom3d_res.yaml
    :language: yaml
    :caption: config/dataset/atom3d_res.yaml

CCPDB Ligand Binding (``ccpdb_ligand``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../proteinworkshop/config/dataset/ccpdb_ligands.yaml
    :language: yaml
    :caption: config/dataset/ccpdb_ligand.yaml

CCPDB Metal Binding (``ccpdb_metal``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../proteinworkshop/config/dataset/ccpdb_metal.yaml
    :language: yaml
    :caption: config/dataset/ccpdb_metal.yaml

CCPDB Nucleic Acid Binding (``ccpdb_nucleic``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../proteinworkshop/config/dataset/ccpdb_nucleic.yaml
    :language: yaml
    :caption: config/dataset/ccpdb_nucleic.yaml


CCPDB Nucleotide Binding (``ccpdb_nucleotides``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../proteinworkshop/config/dataset/ccpdb_nucleotides.yaml
    :language: yaml
    :caption: config/dataset/ccpdb_nucleotides.yaml


Post Translational Modifications (``ptm``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../proteinworkshop/config/dataset/ptm.yaml
    :language: yaml
    :caption: config/dataset/ptm.yaml


PPI Site Prediction (``masif_site``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use the dataset of experimental structures curated from the PDB by Gainza et al.
and retain the original splits, though we modify the labelling scheme to be based on inter-atomic
proximity (3.5 Å), which can be user-defined, rather than solvent exclusion.

The dataset is composed by selecting PPI pairs from the PRISM list of nonredundant proteins,
the ZDock benchmark, PDBBind and SabDab. Splits are performed using CD-HIT and structural splits
are performed using TM-algin.


.. literalinclude:: ../../../proteinworkshop/config/dataset/masif_site.yaml
    :language: yaml
    :caption: config/dataset/masif_site.yaml