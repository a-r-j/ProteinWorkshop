import importlib
import os
import pathlib

import pytest


def test_env_var():
    os.environ["DATA_PATH"] = "test"
    from proteinworkshop import constants

    importlib.reload(constants)

    try:
        assert (
            constants.DATA_PATH == "test"
        ), f"ENV VAR {constants.DATA_PATH} NOT SET!"
    except AssertionError as e:
        del os.environ["DATA_PATH"]
        raise e
    del os.environ["DATA_PATH"]


def test_dot_env():
    data_path_str = "DATA_PATH=TEST_DOT_ENV"
    dot_env_path = pathlib.Path(__file__).parent.parent / ".env"

    with open(dot_env_path, "w") as f:
        f.write(data_path_str)

    from proteinworkshop import constants

    importlib.reload(constants)

    try:
        assert (
            constants.DATA_PATH == "TEST_DOT_ENV"
        ), f"DOT ENV {constants.DATA_PATH} NOT SET!"
    except Exception as e:
        os.remove(dot_env_path)
        raise e
    os.remove(dot_env_path)


def test_datapath_fallback():
    dot_env_path = pathlib.Path(__file__).parent.parent / ".env"
    if os.path.exists(dot_env_path):
        os.remove(dot_env_path)
    del os.environ["DATA_PATH"]

    from proteinworkshop import constants

    importlib.reload(constants)

    assert pathlib.Path(constants.DATA_PATH) == (
        pathlib.Path(constants.SRC_PATH) / "data"
    ), f"FALLBACK ({constants.DATA_PATH}) NOT SET!"
