"""Types used in the library."""
from typing import Dict, List, Literal, NewType

import torch
from jaxtyping import Float

TASK_TYPES = Literal[
    "NodePrediction",
    "GraphPrediction",
    "EdgePrediction",
    "StructuralDenoising",
    "InverseFolding",
]

ActivationType = Literal[
    "relu", "elu", "leaky_relu", "tanh", "sigmoid", "none", "silu", "swish"
]
LossType = Literal[
    "cross_entropy", "nll_loss", "mse_loss", "l1_loss", "dihedral_loss"
]

EncoderOutput = NewType("EncoderOutput", Dict[str, torch.Tensor])

ModelOutput = NewType("ModelOutput", Dict[str, torch.Tensor])

Label = NewType("Label", Dict[str, torch.Tensor])

GNNLayerType = Literal["GCN", "GATv2", "GAT", "GRAPH_TRANSFORMER"]

NodeFeatureTensor = NewType(
    "NodeFeatureTensor", Float[torch.Tensor, "n_nodes n_features"]
)

OrientationTensor = NewType(
    "OrientationTensor", Float[torch.Tensor, "n_nodes 2 3"]
)

GRAPH_CLASSIFICATION_DATASETS = [
    "EnzymeCommission",
    "GeneOntology",
    "ProteinFamily",
]

GRAPH_REGRESSION_DATASETS = ["LBA", "PSR"]

NODE_CLASSIFICATION_DATASETS = []  # TODO

NODE_REGRESSION_DATASETS = []  # TODO

SELF_SUPERVISION_DATASETS = [
    "afdb_swissprot_v4",
    "afdb_uniprot_v4",
    "highquality_clust30",
    "afdb_rep_v4",
    "cath",
    "a_thaliana",
    "c_albicans",
    "c_elegans",
    "d_melanogaster",
    "d_discoideum",
    "d_rerio",
    "e_coli",
    "g_max",
    "h_sapiens",
    "m_jannaschii",
    "m_musculus",
    "o_sativa",
    "r_norvegicus",
    "s_cerevisiae",
    "s_pombe",
    "z_mays",
]

NODE_PROPERTY_PREDICTION_DATASETS = (
    NODE_CLASSIFICATION_DATASETS + NODE_REGRESSION_DATASETS
)

GRAPH_PROPERTY_PREDICTION_DATASETS = (
    GRAPH_CLASSIFICATION_DATASETS + GRAPH_REGRESSION_DATASETS
)

SEQUENCE_ENCODERS: List[str] = [
    "ProteinLSTM",
    "ProteinTransformer",
    "ProteinBert",
    "ProteinCNN",
]

GRAPH_ENCODERS: List[str] = ["EGNN", "GVP", "GNN"]


ScalarNodeFeature = Literal[
    "amino_acid_one_hot",
    "alpha",
    "kappa",
    "dihedrals",
    "sidechain_torsions",
    "sequence_positional_encoding",
]
VectorNodeFeature = Literal["orientation", "virtual_cb_vector"]
ScalarEdgeFeature = Literal["edge_distance", "sequence_distance"]
VectorEdgeFeature = Literal["edge_vectors", "pos_emb"]
