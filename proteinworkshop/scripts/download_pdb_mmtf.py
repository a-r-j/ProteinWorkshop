# Code source: Patrick Kunzmann
# License: BSD 3 clause

import concurrent.futures
import os
import os.path
import pathlib
import tarfile

import biotite.database.rcsb as rcsb
from tqdm import tqdm

from proteinworkshop.constants import DATA_PATH


def download_pdb_mmtf(create_tar: bool = True):
    ### Download of PDB and archive creation ###

    # MMTF files are downloaded into a new directory in this path
    # and the .tar archive is created here
    mmtf_dir = pathlib.Path(DATA_PATH) / "pdb"

    # Obtain all PDB IDs using a query that includes all entries
    # Each PDB entry has a title
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


if __name__ == "__main__":
    download_pdb_mmtf()
