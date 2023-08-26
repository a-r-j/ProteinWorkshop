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
    DATA_PATH = PROJECT_PATH / "data"
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
