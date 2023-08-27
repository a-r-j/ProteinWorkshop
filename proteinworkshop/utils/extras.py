import warnings
from importlib.util import find_spec
from typing import Callable

from loguru import logger as log
from omegaconf import DictConfig

from proteinworkshop.utils import rich_utils


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.
    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info(
            "Disabling python warnings! <cfg.extras.ignore_warnings=True>"
        )
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info(
            "Printing config tree with Rich! <cfg.extras.print_config=True>"
        )
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.
    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)
    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[dict, dict]:
        ...
        return metric_dict, object_dict
    ```
    """

    def wrap(cfg: DictConfig):
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.env.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap
