import shutil

import requests


def is_tool(name: str) -> bool:
    """Checks whether ``name`` is on ``PATH`` and is marked as an executable.

    Source:
    https://stackoverflow.com/questions/11210104/check-if-a-program-exists-from-a-python-script

    :param name: Name of program to check for execution ability.
    :type name: str
    :return: Whether ``name`` is on PATH and is marked as an executable.
    :rtype: bool
    """
    found = shutil.which(name) is not None
    return found


def test_commands_found():
    """Test that the CLI commands are found."""
    assert is_tool("workshop"), "Workshop CLI tool not found"
    assert is_tool("train_workshop"), "Workshop training CLI tool not found"
    assert is_tool(
        "finetune_workshop"
    ), "Workshop finetuning CLI tool not found"
    assert is_tool(
        "install_pyg_workshop"
    ), "Workshop PyG installation CLI tool not found"


def test_download_urls():
    """Assert downloads are found."""
    from proteinworkshop.scripts.download_processed_data import (
        _ZENODO_RECORD,
        dataset_fname_map,
    )

    fnames = list(set(dataset_fname_map.values()))

    for f in fnames:
        url = f"https://zenodo.org/records/{_ZENODO_RECORD}/files/{f}.tar.gz?download=1"
        response = requests.head(url)
        assert response.status_code == 200, f"URL {url} not found."
