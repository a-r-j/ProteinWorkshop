import importlib.metadata

from graphein import verbose
from omegaconf import OmegaConf

# Disable graphein import warnings
verbose(False)


__version__ = importlib.metadata.version("proteinworkshop")


def register_custom_omegaconf_resolvers():
    """
    Register custom OmegaConf resolvers for use in Hydra config files.
    """
    # lazy import
    from proteinworkshop.models.utils import get_input_dim  # noqa: F401

    OmegaConf.register_new_resolver("plus", lambda x, y: x + y)
    OmegaConf.register_new_resolver(
        "resolve_feature_config_dim",
        lambda features_config, feature_config_name, task_config, recurse_for_node_features: get_input_dim(
            features_config,
            feature_config_name,
            task_config,
            recurse_for_node_features=recurse_for_node_features,
        ),
    )
    OmegaConf.register_new_resolver(
        "resolve_num_edge_types",
        lambda features_config: len(features_config.edge_types),
    )
