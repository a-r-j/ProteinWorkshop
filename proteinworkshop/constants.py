"""
Contains project-level constants used to configure paths and wandb logging.

Paths are configured using the `.env` file in the project root.
"""
import logging
import os
import pathlib

import dotenv

# ---------------- PATH CONSTANTS ----------------
SRC_PATH = pathlib.Path(__file__).parent
"""Path to the project source code. """

PROJECT_PATH = SRC_PATH.parent
"""Path to the project root."""

# Data paths are configured using the `.env` file in the project root.

if not os.path.exists(PROJECT_PATH / ".env"):
    DATA_PATH = str(PROJECT_PATH / "data")
    os.environ["DATA_PATH"] = str(DATA_PATH)

else:
    dotenv.load_dotenv(PROJECT_PATH / ".env")

    DATA_PATH = os.environ.get("DATA_PATH")
    """Root path to the data directory. """

# ---------------- HYDRA CONSTANTS ----------------
HYDRA_CONFIG_PATH = PROJECT_PATH / "configs"

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
"""API key for wandb."""

WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
"""Entity for wandb logging."""

WANDB_PROJECT = os.environ.get("WANDB_PROJECT")
"""Project name for wandb logging."""

# ---------------- LOGGING CONSTANTS ----------------
DEFAULT_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s [in %(funcName)s at %(pathname)s:%(lineno)d]"
)
DEFAULT_LOG_FILE = PROJECT_PATH / "logs" / "default_log.log"
DEFAULT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_LEVEL = logging.DEBUG  # verbose logging per default

# ---------------- VALID DATASETS CONSTANTS ---------
FOLDCOMP_DATASET_NAMES = [
    "afdb_rep_v4",
    "afdb_rep_dark_v4",
    "afdb_swissprot",
    "afdb_swissprot_v4",
    "afdb_uniprot_v4",
    "esmatlas",
    "highquality_clust30",
    "a_thaliana",
    "c_albicans",
    "c_elegans",
    "d_discoideum",
    "d_melanogaster",
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

ZENODO_DATASET_NAMES = [
    "antibody_developability",
    "cath",
    "ccpdb",
    "ccpdb_ligands",
    "ccpdb_metal",
    "ccpdb_nucleic",
    "ccpdb_nucleotides",
    "deep_sea_proteins",
    "ec_reaction",
    "fold_classification",
    "fold_fold",
    "fold_family",
    "fold_superfamily",
    "go-bp",
    "go-cc",
    "go-mf",
    "masif_site",
    "metal_3d",
    "ptm",
]
