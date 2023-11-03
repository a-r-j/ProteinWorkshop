import argparse

from proteinworkshop import constants


def main():
    """Dispatches the command to the appropriate function."""
    parser = argparse.ArgumentParser(description="Protein Workshop CLI")

    # Add various sub-commands
    subparsers = parser.add_subparsers(
        dest="command", help="command explanation:"
    )

    # ... install sub-command
    install_parser = subparsers.add_parser(
        "install",
        help="install specific dependencies, such as pyg (pytorch geometric) which require careful checking against pytorch and cuda versions",
    )
    install_parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help="force reinstall",
        default=False,
        required=False,
    )
    install_parser.add_argument(
        "dependency", choices=["pyg"], help="dependency help"
    )

    # ... download sub-command
    _dataset_names = [
        "pdb",
        *constants.FOLDCOMP_DATASET_NAMES,
        *constants.ZENODO_DATASET_NAMES,
    ]
    download_parser = subparsers.add_parser(
        "download",
        help="download raw or precomputed datasets. The README contains details of the individual datasets. In short, the options are:  "
        + ", ".join(_dataset_names),
    )
    download_parser.add_argument(
        "dataset",
        choices=[
            "pdb",
            *constants.FOLDCOMP_DATASET_NAMES,
            *constants.ZENODO_DATASET_NAMES,
        ],
        help="dataset help",
    )

    # ... train sub-command
    train_parser = subparsers.add_parser(
        "train",
        help="train a basic model. See proteinworkshop/train.py for more details.",
    )
    train_parser.add_argument(
        "train_args",
        nargs="*",
        help="Additional arguments passed to the train script.",
    )

    # ... finetune sub-command
    finetune_parser = subparsers.add_parser(
        "finetune",
        help="finetune a basic model. See proteinworkshop/finetune.py for more details.",
    )
    finetune_parser.add_argument(
        "finetune_args",
        nargs="*",
        help="Additional arguments passed to the finetune script.",
    )

    # ... explain sub-command
    explain_parser = subparsers.add_parser(
        "explain",
        help="explain a model. See proteinworkshop/explain.py for more details.",
    )
    explain_parser.add_argument(
        "explain_args",
        nargs="*",
        help="Additional arguments passed to the explain script.",
    )

    # ... embed sub-command
    embed_parser = subparsers.add_parser(
        "embed",
        help="embed a dataset. See proteinworkshop/embed.py for more details.",
    )
    embed_parser.add_argument(
        "embed_args",
        nargs="*",
        help="Additional arguments passed to the embed script.",
    )

    # ... visualise sub-command
    visualise_parser = subparsers.add_parser(
        "visualise",
        help="visualise a dataset's embeddings. See proteinworkshop/visualise.py for more details.",
    )
    visualise_parser.add_argument(
        "visualise_args",
        nargs="*",
        help="Additional arguments passed to the visualise script.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Route to appropriate function
    if args.command == "install":
        # lazy import
        from .install_pyg import _install_pyg

        _install_pyg(args.force_reinstall)

    elif args.command == "download":
        if args.dataset == "pdb":
            # lazy import
            from .download_pdb_mmtf import download_pdb_mmtf

            download_pdb_mmtf()

        elif args.dataset in constants.FOLDCOMP_DATASET_NAMES:
            # lazy import
            from .download_foldcomp import download_foldcomp

            download_foldcomp(args.dataset)
        elif args.dataset in constants.ZENODO_DATASET_NAMES:
            # lazy import
            from .download_processed_data import download_processed_data

            download_processed_data(args.dataset)
        else:
            _valid = "\n\t".join(_dataset_names)
            raise ValueError(
                f"Invalid dataset name. Valid datasets are: \n\t{_valid}"
            )

    elif args.command == "train":
        from proteinworkshop.train import _script_main as train_main

        train_main(args.train_args)

    elif args.command == "finetune":
        from proteinworkshop.finetune import _script_main as finetune_main

        finetune_main(args.finetune_args)

    elif args.command == "explain":
        from proteinworkshop.explain import _script_main as explain_main

        explain_main(args.explain_args)

    elif args.command == "embed":
        from proteinworkshop.embed import _script_main as embed_main

        embed_main(args.embed_args)

    elif args.command == "visualise":
        from proteinworkshop.visualise import _script_main as visualise_main

        visualise_main(args.visualise_args)

    else:
        _valid = "\n\t".join(
            ["\n\tinstall", "\tdownload", "\ttrain", "\tfinetune", "\tembed", "\tvisualise", "\texplain"]

        )
        if args.command is None:
            raise ValueError(f"Missing command. Valid commands are: {_valid}")
        else:
            raise ValueError(f"Invalid command. Valid commands are: {_valid}")

    print("Done.")


if __name__ == "__main__":
    main()
