import shutil

import pytest


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
