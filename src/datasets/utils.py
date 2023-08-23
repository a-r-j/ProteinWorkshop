# Code source: Patrick Kunzmann
# License: BSD 3 clause

import concurrent.futures
import datetime
import os
import os.path
import pathlib
import tarfile
from typing import List, Optional

import biotite.database.rcsb as rcsb
from tqdm import tqdm


def flatten_dir(dir: os.PathLike):
    """
    Flattens the nested directory structure of a directory into a single level.
    """
    for dirpath, _, filenames in os.walk(dir):
        for filename in filenames:
            try:
                os.rename(os.path.join(dirpath, filename), os.path.join(dir, filename))
            except OSError:
                print(f"Could not move {os.path.join(dirpath, filename)}")


def download_pdb_mmtf(
    mmtf_dir: pathlib.Path, ids: Optional[List[str]] = None, create_tar: bool = False
):
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
    now = datetime.datetime.now()
    if not os.path.isdir(mmtf_dir):
        os.mkdir(mmtf_dir)

    # Download all PDB IDs with parallelized HTTP requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for pdb_id in tqdm(pdb_ids):
            executor.submit(rcsb.fetch, pdb_id, "mmtf", mmtf_dir)

    if create_tar:
        # Create .tar archive file from MMTF files in directory
        with tarfile.open(f"{mmtf_dir}.tar", mode="w") as file:
            for pdb_id in pdb_ids:
                file.add(os.path.join(mmtf_dir, f"{pdb_id}.mmtf"), f"{pdb_id}.mmtf")

    ### File access for analysis ###

    # Iterate over all files in archive
    # Instead of extracting the files from the archive,
    # the .tar file is directly accessed
    # with tarfile.open(mmtf_dir+".tar", mode="r") as file:
    #    for member in file.getnames():
    #        mmtf_file = mmtf.MMTFFile.read(file.extractfile(member))
    ###
    # Do some fancy stuff with the data...
    ###
