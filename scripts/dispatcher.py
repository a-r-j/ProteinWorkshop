import sys


from proteinworkshop.finetune import _script_main as finetune_main
from proteinworkshop.train import _script_main as train_main

from .download_foldcomp import _DATASET_NAMES, download_foldcomp
from .download_pdb_mmtf import download_pdb_mmtf
from .download_processed_data import (
    _ZENODO_DATASET_NAMES,
    download_processed_data,
)
from .install_pyg import _install_pyg


def main():
    """Dispatches the command to the appropriate function."""
    if sys.argv[1] == "download":
        if sys.argv[2] == "pdb":
            download_pdb_mmtf()
        elif sys.argv[2] in _DATASET_NAMES:
            download_foldcomp(*sys.argv[2:])
        elif sys.argv[2] in _ZENODO_DATASET_NAMES:
            download_processed_data(*sys.argv[2:])
        else:
            raise ValueError("Invalid dataset name")
    elif sys.argv[1] == "install":
        if sys.argv[2] == "pyg":
            _install_pyg()

    elif sys.argv[1] == "train":
        train_main(sys.argv[2:])

    elif sys.argv[1] == "finetune":
        finetune_main(sys.argv[2:])

    else:
        raise ValueError("Invalid command")


if __name__ == "__main__":
    main()
