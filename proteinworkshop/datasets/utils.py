# Code source: Patrick Kunzmann
# License: BSD 3 clause

import concurrent.futures
import functools
import os
import os.path
import pathlib
import tarfile
from typing import List, Optional

import biotite.database.rcsb as rcsb
import torch
import torch.nn.functional as F
from graphein.protein.tensor.data import ProteinBatch, get_random_protein
from tqdm import tqdm

from proteinworkshop.features.edge_features import pos_emb
from proteinworkshop.features.node_features import orientations
from proteinworkshop.features.utils import _normalize


def flatten_dir(dir: os.PathLike):
    """
    Flattens the nested directory structure of a directory into a single level.

    :param dir: Path to directory
    :type dir: os.PathLike
    """
    for dirpath, _, filenames in os.walk(dir):
        for filename in filenames:
            try:
                os.rename(
                    os.path.join(dirpath, filename),
                    os.path.join(dir, filename),
                )
            except OSError:
                print(f"Could not move {os.path.join(dirpath, filename)}")


def download_pdb_mmtf(
    mmtf_dir: pathlib.Path,
    ids: Optional[List[str]] = None,
    create_tar: bool = False,
):
    """Download PDB files in MMTF format from RCSB PDB and create archive.
    MMTF files are downloaded into a new directory in this path
    and the .tar archive is created here.
    Obtain all PDB IDs using a query that includes all entries.
    Each PDB entry has a title.

    :param mmtf_dir: Path to directory to store MMTF files.
    :type mmtf_dir: pathlib.Path
    :param ids: List of PDB IDs to download.
    :type ids: Optional[List[str]]
    :param create_tar: Whether to create a .tar archive from the downloaded files.
    :type create_tar: bool
    """
    ### Download of PDB and archive creation ###

    # MMTF files are downloaded into a new directory in this path
    # and the .tar archive is created here

    # Obtain all PDB IDs using a query that includes all entries
    # Each PDB entry has a title
    if ids is None:
        all_id_query = rcsb.FieldQuery("struct.title")
        pdb_ids = rcsb.search(all_id_query)
        pdb_ids = [pdb_id.lower() for pdb_id in pdb_ids]

    # Name for download directory
    if not os.path.isdir(mmtf_dir):
        os.mkdir(mmtf_dir)

    # Download all PDB IDs with parallelized HTTP requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        num_requests = len(pdb_ids)
        pbar = tqdm(pdb_ids)
        for pdb_id in pbar:
            pbar.set_description(
                f"Submitting PDB download request for {pdb_id}"
            )
            futures.append(
                executor.submit(rcsb.fetch, pdb_id, "mmtf", mmtf_dir)
            )
        pbar = tqdm(concurrent.futures.as_completed(futures))
        for request_index, future in enumerate(pbar):
            pbar.set_description(
                f"Waiting for PDB download request #{request_index + 1}/{num_requests} to complete"
            )
            # Wait for the future to complete
            future.result()

    if create_tar:
        # Create .tar archive file from MMTF files in directory
        with tarfile.open(f"{mmtf_dir}.tar", mode="w") as file:
            pbar = tqdm(pdb_ids)
            for pdb_id in pbar:
                pbar.set_description(
                    f"Adding downloaded PDB {pdb_id} to {f'{mmtf_dir}.tar'}"
                )
                file.add(
                    os.path.join(mmtf_dir, f"{pdb_id}.mmtf"), f"{pdb_id}.mmtf"
                )

    ### File access for analysis ###

    # Iterate over all files in archive;
    # Instead of extracting the files from the archive,
    # the `.tar` file is directly accessed
    # with tarfile.open(f"{mmtf_dir}.tar", mode="r") as file:
    # for member in file.getnames():
    # mmtf_file = mmtf.MMTFFile.read(file.extractfile(member))
    ## Do some fancy stuff with the data...


@functools.lru_cache()
def create_example_batch(n: int = 4) -> ProteinBatch:
    """Returns a batch of random proteins.

    :param n: Number of proteins to include in batch.
    :type n: int, optional
    :return: Batch of random proteins.
    :rtype: ProteinBatch
    """
    proteins = []
    for _ in range(n):
        p = get_random_protein()
        p.x = p.residue_type
        proteins.append(p)

    batch = ProteinBatch.from_protein_list(proteins)

    batch.edges("knn_8", cache="edge_index")
    batch.edge_index = batch.edge_index.long()
    batch.pos = batch.coords[:, 1, :]
    batch.x = F.one_hot(batch.residue_type, num_classes=23).float()

    batch.x_vector_attr = orientations(batch.pos, batch._slice_dict["coords"])
    batch.graph_y = torch.randint(0, 2, (n, 1))

    batch.edge_attr = pos_emb(batch.edge_index, 9)
    batch.edge_vector_attr = _normalize(
        batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]
    )
    return batch
