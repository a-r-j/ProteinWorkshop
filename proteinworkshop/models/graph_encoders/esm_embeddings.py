"""Modified from TorchDrug."""
import os
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from beartype import beartype as typechecker
from graphein.protein.resi_atoms import RESI_THREE_TO_1
from graphein.protein.tensor.data import ProteinBatch
from loguru import logger
from six.moves.urllib.request import urlretrieve
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

import esm
from proteinworkshop.models.utils import get_aggregation
from proteinworkshop.types import EncoderOutput


@typechecker
def _compute_md5(file_name: str, chunk_size: int = 65536) -> str:
    """
    Compute MD5 of the file.

    :param file_name (str): file name
    :param chunk_size (int, optional): chunk size for reading large files
    """
    import hashlib

    md5 = hashlib.md5()
    with open(file_name, "rb") as fin:
        chunk = fin.read(chunk_size)
        while chunk:
            md5.update(chunk)
            chunk = fin.read(chunk_size)
    return md5.hexdigest()


@typechecker
def _download(
    url: str,
    path: str,
    save_file: Optional[str] = None,
    md5: Optional[str] = None,
):
    """
    Download a file from the specified url.
    Skip the downloading step if there exists a file satisfying the given MD5.

    :param url (str): URL to download
    :param path (str): path to store the downloaded file
    :param save_file (str, optional): name of save file. If not specified, infer the file name from the URL.
    :param md5 (str, optional): MD5 of the file
    """
    if save_file is None:
        save_file = os.path.basename(url)
        if "?" in save_file:
            save_file = save_file[: save_file.find("?")]
    save_file = os.path.join(path, save_file)

    if not os.path.exists(save_file) or (
        md5 is not None and _compute_md5(save_file) != md5
    ):
        logger.info(f"Downloading {url} to {save_file}")
        urlretrieve(url, save_file)
    return save_file


class EvolutionaryScaleModeling(nn.Module):
    """
    The protein language model, Evolutionary Scale Modeling (ESM) proposed in
    `Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences`_.

    .. _Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences:
        https://www.biorxiv.org/content/10.1101/622803v1.full.pdf


    :param path (str): path to store ESM model weights
    :param model (str): model name. Available model names are ``ESM-1b``, ``ESM-1v`` and ``ESM-1b-regression``.
    :param readout (str): readout function. Available functions are ``sum`` and ``mean``.
    :param mlp_post_embed (bool): whether to use MLP to combine ESM embeddings with input features
    :param dropout (float): dropout rate for MLP
    :param finetune (bool): whether to finetune ESM model
    """

    url: Dict[str, str] = {
        "ESM-1b": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt",
        "ESM-1v": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt",
        "ESM-1b-regression": "https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt",
        "ESM-2-8M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt",
        "ESM-2-35M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt",
        "ESM-2-150M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt",
        "ESM-2-650M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
        "ESM-2-3B": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt",
        "ESM-2-15B": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt",
    }

    model_names: List[str] = list(url.keys())

    md5: Dict[str, str] = {
        "ESM-1b": "ba8914bc3358cae2254ebc8874ee67f6",
        "ESM-1v": "1f04c2d2636b02b544ecb5fbbef8fefd",
        "ESM-1b-regression": "e7fe626dfd516fb6824bd1d30192bdb1",
        "ESM-2-8M": "8039fc9cee7f71cd2633b13b5a38ff50",
        "ESM-2-35M": "a894ddb31522e511e1273abb23b5f974",
        "ESM-2-150M": "229fcf8f9f3d4d442215662ca001b906",
        "ESM-2-650M": "ba6d997e29db07a2ad9dca20e024b102",
        "ESM-2-3B": "d37a0d0dbe7431e48a72072b9180b16b",
        "ESM-2-15B": "af61a9c0b792ae50e244cde443b7f4ac",
    }

    output_dim: Dict[str, int] = {
        "ESM-1b": 1280,
        "ESM-1v": 1280,
        "ESM-2-8M": 320,
        "ESM-2-35M": 480,
        "ESM-2-150M": 640,
        "ESM-2-650M": 1280,
        "ESM-2-3B": 2560,
        "ESM-2-15B": 5120,
    }

    num_layer: Dict[str, int] = {
        "ESM-1b": 33,
        "ESM-1v": 33,
        "ESM-2-8M": 6,
        "ESM-2-35M": 12,
        "ESM-2-150M": 30,
        "ESM-2-650M": 33,
        "ESM-2-3B": 36,
        "ESM-2-15B": 48,
    }

    max_input_length = 1024 - 2

    def __init__(
        self,
        path: Union[str, os.PathLike],
        model: str = "ESM-2-650M",
        readout: str = "mean",
        mlp_post_embed: bool = True,
        dropout: float = 0.1,
        finetune: bool = False,
    ):
        super(EvolutionaryScaleModeling, self).__init__()
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        _model, alphabet = self.load_weight(path, model)
        self.alphabet = alphabet
        self.output_dim = self.output_dim[model]
        self.model = _model
        self.alphabet = alphabet
        self.batch_converter = self.alphabet.get_batch_converter()
        self.repr_layer = self.num_layer[model]
        self.mlp_post_embed = mlp_post_embed
        self.finetune = finetune

        if self.mlp_post_embed:
            self.mlp = nn.Sequential(
                nn.LazyLinear(self.output_dim),
                nn.LayerNorm(self.output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        if not self.finetune:
            self.model.eval()

        self.readout = get_aggregation(readout)

        self.residue_map = RESI_THREE_TO_1
        self.residue_map["UNK"] = "<unk>"

    @property
    @typechecker
    def required_batch_attributes(self) -> Set[str]:
        """
        Return the requied attributes for each batch.

        :return: set of required attributes
        """
        return {"residues", "id", "coords", "batch"}

    @typechecker
    def load_weight(
        self, path: str, model: str
    ) -> Tuple[nn.Module, esm.data.Alphabet]:
        """
        Load ESM model weights and their corresponding alphabet.

        :param path (str): path to store ESM model weights
        :param model (str): model name. Available model names are ``ESM-1b``, ``ESM-1v`` and ``ESM-1b-regression``.
        :return: ESM model and its alphabet as `nn.Module` and `esm.data.Alphabet` objects, respectively.
        """
        if model not in self.model_names:
            raise ValueError(f"Unknown model {model}")
        model_file = _download(self.url[model], path, md5=self.md5[model])
        model_data = torch.load(model_file, map_location="cpu")
        if model != "ESM-1v" and not model.startswith("ESM-2"):
            regression_model = f"{model}-regression"
            regression_file = _download(
                self.url[regression_model],
                path,
                md5=self.md5[regression_model],
            )
            regression_data = torch.load(regression_file, map_location="cpu")
        else:
            regression_data = None
        model_name = os.path.basename(self.url[model])
        return esm.pretrained.load_model_and_alphabet_core(
            model_name, model_data, regression_data
        )

    @typechecker
    def esm_embed(
        self,
        batch: Union[Batch, ProteinBatch],
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.Tensor:
        """
        Compute residue ESM embeddings for input proteins
        """
        device = device if device is not None else batch.coords.device

        seqs = [
            "".join([self.residue_map[s] for s in seq])
            for seq in batch.residues
        ]
        seqs = ["".join(seq) for seq in seqs]
        data = list(tuple(zip(batch.id, seqs)))

        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        output = self.model(batch_tokens, repr_layers=[self.repr_layer])
        node_embedding = output["representations"][self.repr_layer]
        # NOTE: tokens `0` and `N` are always beginning-of-sequence and end-of-sequence tokens,
        # so the first (real) residue is token `1` and the last is `N - 1`.
        node_embedding = node_embedding[:, 1 : node_embedding.shape[1] - 1, :]

        _, batch_mask = to_dense_batch(
            x=torch.rand(
                batch.coords.shape[0], self.output_dim, device=device
            ),
            batch=batch.batch,
        )
        node_embedding = node_embedding[batch_mask]
        return node_embedding

    @typechecker
    def forward(
        self,
        batch: Union[Batch, ProteinBatch],
        device: Optional[Union[torch.device, str]] = None,
    ) -> EncoderOutput:
        """
        Compute the residue representations and the graph representation(s).

        :param graph (Protein): :math:`n` protein(s)
        :param input (Tensor): input node representations
        :param device (torch.device or str, optional): device on which to compute and update representations

        :return: dict with ``residue_feature`` and ``graph_feature`` fields:
                residue representations of shape :math:`(|V_{res}|, d)`, graph representations of shape :math:`(n, d)`
        """
        if self.finetune:
            node_embedding = self.esm_embed(batch, device)
        else:
            with torch.no_grad():
                node_embedding = self.esm_embed(batch, device)

        if self.mlp_post_embed:
            # combine ESM embeddings with node features
            node_embedding = self.mlp(
                torch.concatenate([node_embedding, batch.x], dim=-1)
            )

        graph_embedding = self.readout(node_embedding, batch.batch)

        return EncoderOutput(
            {
                "graph_embedding": graph_embedding,
                "node_embedding": node_embedding,
            }
        )


if __name__ == "__main__":
    from proteinworkshop.datasets.utils import create_example_batch

    num_steps = 100
    pbar = tqdm(range(num_steps))
    for _ in pbar:
        pbar.set_description("Embedding random batch")
        b = create_example_batch()
        b = b.to("cuda")
        m = EvolutionaryScaleModeling(path=".")
        m.model = m.model.to("cuda")
        m(b, device="cuda")
