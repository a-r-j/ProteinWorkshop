"""
Convenient setup for jupyter notebook explorations
Use via inserting:
   `from proteinworkshop.utils.notebook import *`
as the last import in your jupyter notebook imports cell.
"""

import os
import pathlib

import hydra
from loguru import logger as log

from proteinworkshop import constants

try:
    # Convenient for debugging
    import lovely_tensors as lt

    lt.monkey_patch()
except:
    pass


def init_hydra_singleton(
    path: os.PathLike = constants.HYDRA_CONFIG_PATH,
    reload: bool = False,
    version_base: str = "1.2",
) -> None:
    """Initialises the hydra singleton.

    .. seealso::
        https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main

    :param path: Path to hydra config, defaults to ``constants.HYDRA_CONFIG_PATH``
    :type path: os.PathLike, optional
    :param reload: Whether to reload the hydra config if it has already been
        initialised, defaults to ``False``
    :type reload: bool, optional
    :raises ValueError: If hydra has already been initialised and ``reload`` is
        ``False``
    """
    # See: https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main
    if reload:
        clear_hydra_singleton()
    try:
        path = pathlib.Path(path)
        # Note: hydra needs to be initialised with a relative path. Since the hydra
        #  singleton is first created here, it needs to be created relative to this
        #  file. The `rel_path` below takes care of that.
        rel_path = os.path.relpath(path, start=pathlib.Path(__file__).parent)
        hydra.initialize(rel_path, version_base=version_base)
        log.info(f"Hydra initialised at {path.absolute()}.")
    except ValueError:
        log.info("Hydra already initialised.")


def clear_hydra_singleton() -> None:
    """
    Clears the initialised hydra singleton.
    """
    if hydra.core.global_hydra.GlobalHydra not in hydra.core.singleton.Singleton._instances:  # type: ignore
        return
    hydra_singleton = hydra.core.singleton.Singleton._instances[hydra.core.global_hydra.GlobalHydra]  # type: ignore
    hydra_singleton.clear()
    log.info("Hydra singleton cleared and ready to re-initialise.")
