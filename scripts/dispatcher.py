import sys
from proteinworkshop import constants


def main():
    """Dispatches the command to the appropriate function."""
    if len(sys.argv) < 2:
        _valid = "\n\t".join(["\n\tdownload", "install", "train", "finetune"])
        raise ValueError(f"Did not provide a command. Valid commands are: {_valid}")

    # Options to download data
    if sys.argv[1] == "download":
        if sys.argv[2] == "pdb":
            # lazy import
            from .download_pdb_mmtf import download_pdb_mmtf  
            download_pdb_mmtf()
        elif sys.argv[2] in constants.FOLDCOMP_DATASET_NAMES:
            # lazy import
            from .download_foldcomp import download_foldcomp
            download_foldcomp(*sys.argv[2:])
        elif sys.argv[2] in constants.ZENODO_DATASET_NAMES:
            # lazy import
            from .download_processed_data import download_processed_data
            download_processed_data(*sys.argv[2:])
        else:
            _valid = "\n\t".join(["\n\tpdb", *constants.FOLDCOMP_DATASET_NAMES, *constants.ZENODO_DATASET_NAMES])
            raise ValueError(f"Invalid dataset name. Valid datasets are: {_valid}")
    
    # Options to install dependencies
    elif sys.argv[1] == "install":
        if sys.argv[2] == "pyg":
            # lazy import
            from .install_pyg import _install_pyg
            _install_pyg()

    # Options to run scripts
    elif sys.argv[1] == "train":
        from proteinworkshop.train import _script_main as train_main
        train_main(sys.argv[2:])

    elif sys.argv[1] == "finetune":
        from proteinworkshop.finetune import _script_main as finetune_main
        finetune_main(sys.argv[2:])

    else:
        _valid = "\n\t".join(["\n\tdownload", "install", "train", "finetune"])
        raise ValueError(f"Invalid command. Valid commands are: {_valid}")


if __name__ == "__main__":
    main()
