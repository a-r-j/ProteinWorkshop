from typing import Set, Union

import e3nn
import torch
import torch.nn.functional as F
from beartype import beartype
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import jaxtyped
from torch_geometric.data import Batch

import proteinworkshop.models.graph_encoders.layers.tfn as tfn
from proteinworkshop.models.graph_encoders.components import blocks
from proteinworkshop.models.utils import get_aggregation
from proteinworkshop.types import EncoderOutput


class TensorProductModel(torch.nn.Module):
    def __init__(
        self,
        r_max: float = 10.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        max_ell: int = 2,
        num_layers: int = 4,
        emb_dim: int = 64,
        mlp_dim: int = 256,
        aggr: str = "sum",
        pool: str = "sum",
        residual: bool = True,
        batch_norm: bool = True,
        gate: bool = False,
        hidden_irreps=None,
    ):
        """e3nn-based Tensor Product Convolution Network (Tensor Field Network)

        Initialise an instance of the TensorProductModel class with the provided
        parameters.

        :param r_max: Maximum distance for Bessel basis functions
            (default: ``10.0``)
        :type r_max: float, optional
        :param num_bessel: Number of Bessel basis functions (default: ``8``)
        :type num_bessel: int, optional
        :param num_polynomial_cutoff: Number of polynomial cutoff basis
            functions (default: ``5``)
        :type num_polynomial_cutoff: int, optional
        :param max_ell: Maximum degree/order of spherical harmonics basis
            functions and node feature tensors (default: ``2``)
        :type max_ell: int, optional
        :param num_layers: Number of layers in the model (default: ``5``)
        :type num_layers: int, optional
        :param emb_dim: Number of hidden channels/embedding dimension for each
            node feature tensor order (default: ``64``)
        :type emb_dim: int, optional
        :param mlp_dim: Dimension of MLP for computing tensor product
            weights (default: ``256``)
        :type: int, optional
        :param aggr: Aggregation function to use, defaults to ``"sum"``
        :type aggr: str, optional
        :param pool: Pooling operation to use, defaults to ``"sum"``
        :type pool: str, optional
        :param residual: Whether to use residual connections, defaults to
            ``True``
        :type residual: bool, optional
        :param batch_norm: Whether to use e3nn batch normalisation, defaults to
            ``True``
        :type batch_norm: bool, optional
        :param gate: Whether to use gated non-linearity, defaults to ``False``
        :type gate: bool, optional
        :param hidden_irreps: Irreps for intermediate layer node feature tensors
            (default: ``None``)
        :type hidden_irreps: e3nn.o3.Irreps, optional

        .. note::
            If ``hidden_irreps`` is None, irreps for node feature tensors are
            computed using ``max_ell`` order of spherical harmonics and ``emb_dim``.
        """
        super().__init__()
        self.r_max = r_max
        self.max_ell = max_ell
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.mlp_dim = mlp_dim
        self.residual = residual
        self.batch_norm = batch_norm
        self.gate = gate
        self.hidden_irreps = hidden_irreps

        # Edge embedding
        self.radial_embedding = blocks.RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        sh_irreps = e3nn.o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = e3nn.o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Embedding lookup for initial node features
        self.emb_in = torch.nn.LazyLinear(emb_dim)

        # Set hidden irreps if none are provided
        if hidden_irreps is None:
            hidden_irreps = (sh_irreps * emb_dim).sort()[0].simplify()
            # Note: This defaults to O(3) equivariant layers. It is
            #       possible to use SO(3) equivariance by passing
            #       the appropriate irreps to `hidden_irreps`.

        self.convs = torch.nn.ModuleList()
        # First conv layer: scalar only -> tensor
        self.convs.append(
            tfn.TensorProductConvLayer(
                in_irreps=e3nn.o3.Irreps(f"{emb_dim}x0e"),
                out_irreps=hidden_irreps,
                sh_irreps=sh_irreps,
                edge_feats_dim=self.radial_embedding.out_dim,
                mlp_dim=mlp_dim,
                aggr=aggr,
                batch_norm=batch_norm,
                gate=gate,
            )
        )
        # Intermediate conv layers: tensor -> tensor
        for _ in range(num_layers - 2):
            conv = tfn.TensorProductConvLayer(
                in_irreps=hidden_irreps,
                out_irreps=hidden_irreps,
                sh_irreps=sh_irreps,
                edge_feats_dim=self.radial_embedding.out_dim,
                mlp_dim=mlp_dim,
                aggr=aggr,
                batch_norm=batch_norm,
                gate=gate,
            )
            self.convs.append(conv)
        # Last conv layer: tensor -> scalar only
        self.convs.append(
            tfn.TensorProductConvLayer(
                in_irreps=hidden_irreps,
                out_irreps=e3nn.o3.Irreps(f"{emb_dim}x0e"),
                sh_irreps=sh_irreps,
                edge_feats_dim=self.radial_embedding.out_dim,
                mlp_dim=mlp_dim,
                aggr=aggr,
                batch_norm=batch_norm,
                gate=gate,
            )
        )

        # Global pooling/readout function
        self.readout = get_aggregation(pool)

    @property
    def required_batch_attributes(self) -> Set[str]:
        """
        Required batch attributes for this encoder.

        - ``x``: Node features (shape: :math:`(n, d)`)
        - ``pos``: Node positions (shape: :math:`(n, 3)`)
        - ``edge_index``: Edge indices (shape: :math:`(2, e)`)
        - ``batch``: Batch indices (shape: :math:`(n,)`)

        :return: Set of required batch attributes
        :rtype: Set[str]
        """
        return {"edge_index", "pos", "x", "batch"}

    @jaxtyped
    @beartype
    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Returns the node embedding and graph embedding in a dictionary.

        :param batch: Batch of data to encode.
        :type batch: Union[Batch, ProteinBatch]
        :return: Dictionary of node and graph embeddings. Contains
            ``node_embedding`` and ``graph_embedding`` fields. The node
            embedding is of shape :math:`(|V|, d)` and the graph embedding is
            of shape :math:`(n, d)`, where :math:`|V|` is the number of nodes
            and :math:`n` is the number of graphs in the batch and :math:`d` is
            the dimension of the embeddings.
        :rtype: EncoderOutput
        """
        # Node embedding
        h = self.emb_in(batch.x)  # (n,) -> (n, d)

        # Edge features
        vectors = (
            batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]
        )  # [n_edges, 3]
        lengths = torch.linalg.norm(
            vectors, dim=-1, keepdim=True
        )  # [n_edges, 1]
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        for conv in self.convs:
            # Message passing layer
            h_update = conv(h, batch.edge_index, edge_attrs, edge_feats)
            # TODO it may be useful to concatenate node scalar type l=0 features
            # from both src and dst nodes into edge_feats (RBF of displacement), as in
            # https://github.com/gcorso/DiffDock/blob/main/models/score_model.py#L263

            # Update node features
            h = (
                h_update + F.pad(h, (0, h_update.shape[-1] - h.shape[-1]))
                if self.residual
                else h_update
            )

        return EncoderOutput(
            {
                "node_embedding": h,
                "graph_embedding": self.readout(
                    h, batch.batch
                ),  # (n, d) -> (batch_size, d)
            }
        )


if __name__ == "__main__":
    import hydra
    import omegaconf

    from proteinworkshop import constants

    cfg = omegaconf.OmegaConf.load(
        constants.PROJECT_PATH / "configs" / "encoder" / "tfn.yaml"
    )
    enc = hydra.utils.instantiate(cfg)
    print(enc)
