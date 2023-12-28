import argparse
import os
import pathlib
from typing import Optional

import foldcomp
from loguru import logger

from proteinworkshop import constants

_DATASET_NAMES = constants.FOLDCOMP_DATASET_NAMES


def download_foldcomp(dataset_name: str, data_dir: Optional[str] = None):
    """
    Downloads a FoldComp dataset.

    :param dataset_name: Name of the FoldComp dataset to download
    :type dataset_name: str
    :param data_dir: Parent directory to download the dataset to.
        If ``None``, defaults to :py:const:`constants.DATA_PATH` / dataset_name,
       defaults to ``None``
    :type data_dir: Optional[str], optional
    :raises ValueError: If dataset_name is not one of the FoldComp datasets
    """

    if dataset_name not in _DATASET_NAMES:
        raise ValueError(
            f"Dataset name must be one of {_DATASET_NAMES}. Got {dataset_name}."
        )

    if data_dir is None:
        data_dir = pathlib.Path(constants.DATA_PATH) / dataset_name
    else:
        data_dir = pathlib.Path(data_dir) / dataset_name

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    logger.info(f"Downloading {dataset_name} to {data_dir}.")
    run_dir = os.getcwd()
    os.chdir(data_dir)
    foldcomp.setup(dataset_name)
    os.chdir(run_dir)
    # Move the downloaded files to the specified directory
    # for i in ["", ".index", ".dbtype", ".lookup", ".source"]:
    #    fname = dataset_name + i
    #    shutil.move(fname, data_dir / fname)

    logger.info(f"Download of {dataset_name} complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_name",
        type=str,
        default="afdb_rep_v4",
        help="Name of the FoldComp dataset to download",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        required=False,
        help="Parent directory to download the dataset to. If None, defaults to constants.DATA_PATH / dataset_name",
    )

    args = parser.parse_args()

    download_foldcomp(dataset_name=args.dataset_name, data_dir=args.data_dir)
