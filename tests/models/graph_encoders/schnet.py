
import pytest
import hydra
import omegaconf
import pyrootutils
from graphein.protein.tensor.data import get_random_protein, ProteinBatch
from loguru import logger


def test_instantiate_schnet():
    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "encoder" / "schnet.yaml")

    logger.info(cfg)
    encoder = hydra.utils.instantiate(cfg)
    logger.info(encoder)
    batch = ProteinBatch().from_protein_list([get_random_protein() for _ in range(4)], follow_batch=["coords"])

    batch.batch = batch.coords_batch
    batch.edges("knn_8", cache="edge_index")
    batch.pos = batch.coords[:, 1, :]
    batch.x = batch.residue_type

    logger.info(batch)
    out = encoder.forward(batch)
    logger.info(out)

    assert "node_embedding" in out.keys()
    assert "graph_embedding" in out.keys()

    assert out["node_embedding"].shape[0] == batch.num_nodes
    assert out["graph_embedding"].shape[0] == batch.num_graphs


if __name__ == "__main__":
    test_instantiate_schnet()