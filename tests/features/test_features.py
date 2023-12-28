import os
import torch
import omegaconf
from hydra.utils import instantiate

from proteinworkshop import constants
from proteinworkshop.features.factory import ProteinFeaturiser
from proteinworkshop.models.utils import get_input_dim

FEATURE_CONFIG_DIR = constants.HYDRA_CONFIG_PATH / "features"
TRANSFORMS = os.listdir(FEATURE_CONFIG_DIR)


def test_instantiate_featuriser():
    """Tests we can instantiate all featurisers."""
    for t in TRANSFORMS:
        config_path = FEATURE_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)
        featuriser = instantiate(cfg)

        assert featuriser, f"Featuriser {t} not instantiated!"
        assert isinstance(featuriser, ProteinFeaturiser)


def test_feature_shapes(example_batch):
    """Test all featurisers return the correct shapes."""
    for t in TRANSFORMS:
        config_path = FEATURE_CONFIG_DIR / t
        cfg = omegaconf.OmegaConf.load(config_path)
        featuriser = instantiate(cfg)

        example_batch.seq_pos = torch.arange(
            example_batch.coords.shape[0], dtype=torch.long
        )
        out = featuriser(example_batch)
        out_features = out.x

        # Test we have edges
        assert out.edge_index.shape[0] == 2

        # Test we have node features of the correct shape
        assert out_features.shape[0] == example_batch.num_nodes
        assert out_features.shape[1] == get_input_dim(
            cfg, "scalar_node_features", None
        )
