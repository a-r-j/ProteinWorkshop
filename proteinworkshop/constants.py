"""
Contains project-level constants used to configure paths and wandb logging.

Paths are configured using the `.env` file in the project root.
"""
import logging
import os
import pathlib
from loguru import logger

# ---------------- PATH CONSTANTS ----------------
SRC_PATH = pathlib.Path(__file__).parent
"""Path to the project source code. """

PROJECT_PATH = SRC_PATH.parent
"""Path to the project root."""

# ---------------- ENVIRONMENT VARIABLES ----------------
# Data paths are configured using the `.env` file in the project root.

if not os.path.exists(PROJECT_PATH / ".env"):
    logger.debug("No `.env` file found in project root. Checking for env vars...")
    # If no `.env` file found, check for an env var
    if os.environ.get("DATA_PATH") is not None:
        logger.debug("Found env var `DATA_PATH`:.")
        DATA_PATH = os.environ.get("DATA_PATH")
    else:
        logger.debug("No env var `DATA_PATH` found. Setting default...")
        DATA_PATH = str(SRC_PATH / "data")
        os.environ["DATA_PATH"] = str(DATA_PATH)
else:
    import dotenv  # lazy import to avoid dependency on dotenv

    dotenv.load_dotenv(PROJECT_PATH / ".env")

    DATA_PATH = os.environ.get("DATA_PATH")
    """Root path to the data directory. """

logger.info(f"DATA_PATH: {DATA_PATH}")
# Set default environment paths as fallback if not specified in .env file
#  NOTE: These will be overridden by paths in the hydra config or by
#   the corresponding `.env` environment variables if they are set.
#   We provide them simply as a fallback for users who do not want to
#   use hydra or environment variables. Plese see the README and
#   `proteinworkshop/config/env/default.yaml` for more information on how to configure paths.
if os.environ.get("ROOT_DIR") is None:
    ROOT_DIR = str(PROJECT_PATH)
    os.environ["ROOT_DIR"] = str(ROOT_DIR)
if os.environ.get("DATA_PATH") is None:
    DATA_PATH = str(PROJECT_PATH / "data")
    os.environ["DATA_PATH"] = str(DATA_PATH)
if os.environ.get("RUNS_PATH") is None:
    RUNS_PATH = str(PROJECT_PATH / "runs")
    os.environ["RUNS_PATH"] = str(RUNS_PATH)

# ---------------- HYDRA CONSTANTS ----------------
# HYDRA_CONFIG_PATH = PROJECT_PATH / "configs"
HYDRA_CONFIG_PATH = SRC_PATH / "config"

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
    # available foldcomp datasets
    #   documentation:           https://github.com/steineggerlab/foldcomp
    #   up-to-date dataset list: https://foldcomp.steineggerlab.workers.dev/
    "afdb_rep_v4",  # representative structures identified from the AF2 database by FoldSeek structural clustering
    "afdb_rep_dark_v4",  # dark proteome structures identied by structural clustering of the AlphaFold database.
    "afdb_swissprot",  # AlphaFold2 predictions for SwissProt/UniProtKB -- OLD VERSION
    "afdb_swissprot_v4",  # AlphaFold2 predictions for SwissProt/UniProtKB
    "afdb_uniprot_v4",  # AlphaFold2 predictions for all of UniProt
    "esmatlas",  # ESMFold predictions for all sequences in the ESMAtlas
    "highquality_clust30",  # ESMFold high-quality predictions for the ESMAtlas (30% seq. id clustering)
    # ... species-specific proteome-wide datasets (all predicted by AlphaFold 2)
    "a_thaliana",  # Arabidopsis thaliana (thale cress)
    "c_albicans",  # Candida albicans (a fungus)
    "c_elegans",  # Caenorhabditis elegans (roundworm)
    "d_discoideum",  # Dictyostelium discoideum (slime mold)
    "d_melanogaster",  # Drosophila melanogaster (fruit fly)
    "d_rerio",  # Danio rerio (zebrafish)
    "e_coli",  # Escherichia coli (a bacterium)
    "g_max",  # Glycine max (soybean)
    "h_sapiens",  # Homo sapiens (human)
    "m_jannaschii",  # Methanocaldococcus jannaschii (an archaea)
    "m_musculus",  # Mus musculus (mouse) proteome predicted by AlphaFold 2
    "o_sativa",  # Oryza sative (rice)
    "r_norvegicus",  # Rattus norvegicus (brown rat)
    "s_cerevisiae",  # Saccharomyces cerevisiae (brewer's yeast)
    "s_pombe",  #  Schizosaccharomyces pombe (a fungus)
    "z_mays",  # Zea mays (corn)
]
"""List of available foldcomp datasets for download. For documentation see https://github.com/steineggerlab/foldcomp.
For an up-to-date list of all datsets see https://foldcomp.steineggerlab.workers.dev."""

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
"""List of pre-processed datasets for `proteinworkshop` available on Zenodo. For documentation see
https://github.com/a-r-j/ProteinWorkshop/#datasets"""
