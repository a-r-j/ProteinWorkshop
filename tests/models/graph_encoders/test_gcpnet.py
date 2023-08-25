import omegaconf
from hydra.utils import instantiate

from src import constants


def test_instantiate_encoder():
    config_path = constants.PROJECT_PATH / "configs" / "encoder" / "gcpnet.yaml"
    cfg = omegaconf.OmegaConf.load(config_path)
    enc = instantiate(cfg)

    print(enc)

if __name__ == "__main__":
    test_instantiate_encoder()