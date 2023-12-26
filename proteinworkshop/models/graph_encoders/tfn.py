from functools import partial
from typing import Set, Union

import e3nn
import torch
import torch.nn.functional as F
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import jaxtyped
from torch_geometric.data import Batch
from torch_geometric.utils import to_undirected

import proteinworkshop.models.graph_encoders.layers.tfn as tfn
from proteinworkshop.models.graph_encoders.components import radial
from proteinworkshop.models.utils import get_aggregation
from proteinworkshop.types import EncoderOutput


class TensorProductModel(torch.nn.Module):
    def __init__(
        self,
        r_max: float = 10.0,
        num_basis: int = 16,
        max_ell: int = 2,
        num_layers: int = 4,
        hidden_irreps = "32x0e + 32x0o + 8x1e + 8x1o + 4x2e + 4x2o",
        mlp_dim: int = 256,
        aggr: str = "mean",
        pool: str = "sum",
        residual: bool = True,
        batch_norm: bool = True,
        gate: bool = False,
        dropout: float = 0.1,
    ):
        """e3nn-based Tensor Product Convolution Network (Tensor Field Network)

        Initialise an instance of the TensorProductModel class with the provided
        parameters.

        :param r_max: Maximum distance for radial basis functions
            (default: ``10.0``)
        :type r_max: float, optional
        :param num_basis: Number of radial basis functions (default: ``16``)
        :type num_basis: int, optional
        :param max_ell: Maximum degree/order of spherical harmonics basis
            functions and node feature tensors (default: ``2``)
        :type max_ell: int, optional
        :param num_layers: Number of layers in the model (default: ``4``)
        :type num_layers: int, optional
        :param hidden_irreps: Irreps string for intermediate layer node 
            feature tensors; converted to e3nn.o3.Irreps format 
            (default: SO(3) equivariance: ``32x0e + 32x0o + 8x1e + 8x1o + 4x2e + 4x2o``
            alternative: O(3) equivariance: ``64x0e + 16x1o + 8x2e``)
        :type hidden_irreps: str, optional
        :param mlp_dim: Dimension of MLP for computing tensor product
            weights (default: ``256``)
        :type: int, optional
        :param aggr: Aggregation function to use, defaults to ``"mean"``
        :type aggr: str, optional
        :param pool: Pooling operation to use, defaults to ``"mean"``
        :type pool: str, optional
        :param residual: Whether to use residual connections, defaults to
            ``True``
        :type residual: bool, optional
        :param batch_norm: Whether to use e3nn batch normalisation, defaults to
            ``True``
        :type batch_norm: bool, optional
        :param gate: Whether to use gated non-linearity, defaults to ``False``
        :type gate: bool, optional
        :param dropout: Dropout rate, defaults to ``0.1``
        :type dropout: float, optional
        """
        super().__init__()
        self.r_max = r_max
        self.max_ell = max_ell
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.residual = residual
        self.batch_norm = batch_norm
        self.gate = gate
        self.hidden_irreps = e3nn.o3.Irreps(hidden_irreps)
        self.emb_dim = self.hidden_irreps[0].dim  # scalar embedding dimension

        # Edge embedding
        self.radial_embedding = partial(
            radial.compute_rbf, max_distance=r_max, num_rbf=num_basis
        )

        self.sh_irreps = e3nn.o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = e3nn.o3.SphericalHarmonics(
            self.sh_irreps, normalize=True, normalization="component"
        )

        # Embedding lookup for initial node features
        self.emb_in = torch.nn.LazyLinear(self.emb_dim)

        self.convs = torch.nn.ModuleList()
        # First conv layer: scalar only -> tensor
        self.convs.append(
            tfn.TensorProductConvLayer(
                in_irreps=e3nn.o3.Irreps(f"{self.emb_dim}x0e"),
                out_irreps=self.hidden_irreps,
                sh_irreps=self.sh_irreps,
                edge_feats_dim=num_basis + 2*self.emb_dim,
                mlp_dim=self.mlp_dim,
                aggr=aggr,
                batch_norm=batch_norm,
                gate=gate,
                dropout=dropout,
            )
        )
        # Intermediate conv layers: tensor -> tensor
        for _ in range(num_layers - 2):
            conv = tfn.TensorProductConvLayer(
                in_irreps=self.hidden_irreps,
                out_irreps=self.hidden_irreps,
                sh_irreps=self.sh_irreps,
                edge_feats_dim=num_basis + 2*self.emb_dim,
                mlp_dim=self.mlp_dim,
                aggr=aggr,
                batch_norm=batch_norm,
                gate=gate,
                dropout=dropout,
            )
            self.convs.append(conv)
        # Last conv layer: tensor -> scalar only
        self.convs.append(
            tfn.TensorProductConvLayer(
                in_irreps=self.hidden_irreps,
                out_irreps=e3nn.o3.Irreps(f"{self.emb_dim}x0e"),
                sh_irreps=self.sh_irreps,
                edge_feats_dim=num_basis + 2*self.emb_dim,
                mlp_dim=self.mlp_dim,
                aggr=aggr,
                batch_norm=batch_norm,
                gate=gate,
                dropout=dropout,
            )
        )

        # Global pooling/readout function
        self.readout = get_aggregation(pool)

    @property
    def required_batch_attributes(self) -> Set[str]:
        """Required batch attributes for this encoder.

        - ``x``: Node features (shape: :math:`(n, d)`)
        - ``pos``: Node positions (shape: :math:`(n, 3)`)
        - ``edge_index``: Edge indices (shape: :math:`(2, e)`)
        - ``batch``: Batch indices (shape: :math:`(n,)`)

        :return: Set of required batch attributes
        :rtype: Set[str]
        """
        return {"edge_index", "pos", "x", "batch"}

    @jaxtyped(typechecker=typechecker)
    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Implements the forward pass of the TFN encoder.
        
        Returns the node embedding and graph embedding in a dictionary.

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
        # Convert to undirected edges
        edge_index = to_undirected(batch.edge_index)

        # Node embedding
        h = self.emb_in(batch.x)  # (n,) -> (n, d)

        # Edge features
        vectors = (
            batch.pos[edge_index[0]] - batch.pos[edge_index[1]]
        )  # [n_edges, 3]
        lengths = torch.linalg.norm(
            vectors, dim=-1,
        )  # [n_edges, 1]
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        for conv in self.convs:
            # concatenate RBF with scalar features from src and dst nodes
            edge_feats_expanded = torch.cat(
                [
                    edge_feats, 
                    h[edge_index[0], :self.emb_dim], 
                    h[edge_index[1], :self.emb_dim]
                ], 
                dim=1
            )

            # Message passing layer
            h_update = conv(h, edge_index, edge_attrs, edge_feats_expanded)
            
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
        constants.SRC_PATH / "config" / "encoder" / "tfn.yaml"
    )
    enc = hydra.utils.instantiate(cfg)
    print(enc)