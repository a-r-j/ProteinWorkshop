import pytest
import omegaconf
from hydra.utils import instantiate
from proteinworkshop import constants


ENCODERS = ["gvp", "gcn", "schnet", "dimenet_plus_plus", "egnn", "gear_net", "gear_net_edge", "identity", "]

def test_instantiate_encoders():

    for encoder in ENCODERS:
        config_path = constants.PROJECT_PATH / "configs" / "encoder" / f"{encoder}.yaml"
        cfg = omegaconf.OmegaConf.load(config_path)
        enc = instantiate(cfg)

        assert enc, f"Encoder {encoder} not instantiated!"


@pytest.mark.slow
def test_encoder_forward_pass(example_batch):

    for encoder in ENCODERS:
        config_path = constants.PROJECT_PATH / "configs" / "encoder" / f"{encoder}.yaml"
        cfg = omegaconf.OmegaConf.load(config_path)
        enc = instantiate(cfg)

        out = enc(example_batch)
        assert out
        assert isinstance(out, dict)
        assert "node_embedding" in out
        assert "graph_embedding" in out