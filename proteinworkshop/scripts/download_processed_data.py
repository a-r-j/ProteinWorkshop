"""Downloads processed datasets for the benchmark from Zenodo."""
import os
import pathlib
import tarfile
from typing import Dict, Optional

import wget
from loguru import logger

from proteinworkshop import constants

_ZENODO_RECORD = "8282470"

# NOTE: This is a list of all datasets available on Zenodo, these are
#  the same as the keys in `dataset_fname_map` and `_MD5_CHECKSUMS`
#  below.
_ZENODO_DATASET_NAMES = constants.ZENODO_DATASET_NAMES

dataset_fname_map = {
    "antibody_developability": "AntibodyDevelopability",
    "cath": "cath",
    "ccpdb": "ccpdb",
    "ccpdb_ligands": "ccpdb",
    "ccpdb_metal": "ccpdb",
    "ccpdb_nucleic": "ccpdb",
    "ccpdb_nucleotides": "ccpdb",
    "deep_sea_proteins": "deep-sea-proteins",
    "ec_reaction": "ECReaction",
    "fold_classification": "FoldClassification",
    "fold_fold": "FoldClassification",
    "fold_family": "FoldClassification",
    "fold_superfamily": "FoldClassification",
    "gene_ontology": "GeneOntology",
    "go-bp": "GeneOntology",
    "go-cc": "GeneOntology",
    "go-mf": "GeneOntology",
    "masif_site": "masif_site",
    "metal_3d": "Metal3D",
    "ptm": "PostTranslationalModification",
}

_MD5_CHECKSUMS: Dict[str, str] = {
    "antibody_developability": "2d7ad11284bbd1c561f3c828a38d29cc",
    "cath": "d1c77941ad390660ddcfee948a4f7b3f",
    "ccpdb": "b64e73ede0550212ab220dce446242b4",
    "ccpdb_ligands": "b64e73ede0550212ab220dce446242b4",
    "ccpdb_metal": "b64e73ede0550212ab220dce446242b4",
    "ccpdb_nucleic": "b64e73ede0550212ab220dce446242b4",
    "ccpdb_nucleotides": "b64e73ede0550212ab220dce446242b4",
    "deep_sea_proteins": "cd17fd7230f710f70cea5162ec73a784",
    "ec_reaction": "8a201370939453ed86847c923c7cd48d",
    "fold_classification": "810fc8b24c6fb6b887f6bd4fc7389838",
    "fold_fold": "810fc8b24c6fb6b887f6bd4fc7389838",
    "fold_family": "810fc8b24c6fb6b887f6bd4fc7389838",
    "fold_superfamily": "810fc8b24c6fb6b887f6bd4fc7389838",
    "go-bp": "a59a559aceb265d8b8b9e15211a864f1",
    "go-cc": "a59a559aceb265d8b8b9e15211a864f1",
    "go-mf": "a59a559aceb265d8b8b9e15211a864f1",
    "masif_site": "65187f457449f977a84eb226d98fc79",
    "metal_3d": "72bc625f68f874dc4702229fae991372",
    "ptm": "314839e6073a8d2f289bd89bbd42c9e1",
}


def _compute_md5(file_name, chunk_size=65536):
    """
    Compute MD5 of the file.

    Parameters:
        file_name (str): file name
        chunk_size (int, optional): chunk size for reading large files
    """
    import hashlib

    md5 = hashlib.md5()
    with open(file_name, "rb") as fin:
        chunk = fin.read(chunk_size)
        while chunk:
            md5.update(chunk)
            chunk = fin.read(chunk_size)
    return md5.hexdigest()


def download_processed_data(dataset_name: str, data_dir: Optional[str] = None):
    if dataset_name not in _ZENODO_DATASET_NAMES:
        raise ValueError(f"Dataset {dataset_name} not found in Zenodo")
    if data_dir is None:
        data_dir = pathlib.Path(constants.DATA_PATH)
    else:
        data_dir = pathlib.Path(data_dir)

    if not os.path.exists(data_dir):
        logger.info(f"Creating data directory at {data_dir}")
        os.makedirs(parents=True, exist_ok=True)

    fname = dataset_fname_map[dataset_name]
    save_file = data_dir / f"{fname}.tar.gz"

    # Skip download if file exists or checksum matches
    checksum = _MD5_CHECKSUMS[dataset_name]
    if not os.path.exists(save_file) or _compute_md5(save_file) != checksum:
        logger.info(
            f"Downloading {dataset_name} dataset from Zenodo Record {_ZENODO_RECORD}"
        )
        zenodo_url = f"https://zenodo.org/records/{_ZENODO_RECORD}/files/{fname}.tar.gz?download=1"
        wget.download(zenodo_url, out=str(data_dir))
    else:
        logger.info(
            f"Dataset {dataset_name} already downloaded. Skipping download"
        )

    assert os.path.exists(data_dir / f"{fname}.tar.gz"), "Download failed"

    # Check the checksum of the downloaded file
    md5 = _compute_md5(data_dir / f"{fname}.tar.gz")
    if not md5 == checksum:
        logger.warning(f"Checksum mismatch: DL ({md5}), expected ({checksum})")

    # Untar
    logger.info("Untarring dataset...")
    tar = tarfile.open(data_dir / f"{fname}.tar.gz")
    tar.extractall(path=data_dir)
    tar.close()

    if not os.path.exists(data_dir / fname / "processed"):
        logger.warning(f"Processed data not found in {data_dir / fname}")

    logger.info("Done!")


if __name__ == "__main__":
    download_processed_data("ec_reaction")
