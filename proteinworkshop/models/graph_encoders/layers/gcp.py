###########################################################################################
# Implementation of Geometric-Complete Perceptron layers
#
# Papers:
# (1) Geometry-Complete Perceptron Networks for 3D Molecular Graphs,
#     by A Morehead, J Cheng
# (2) Geometry-Complete Diffusion for 3D Molecule Generation,
#     by A Morehead, J Cheng
#
# Orginal repositories:
# (1) https://github.com/BioinfoMachineLearning/GCPNet
# (2) https://github.com/BioinfoMachineLearning/Bio-Diffusion
###########################################################################################

from copy import copy
from functools import partial
from typing import Any, Optional, Tuple, Union

import torch
import torch_scatter
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import Bool, Float, Int64, jaxtyped
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch_geometric.data import Batch

from proteinworkshop.models.graph_encoders.components import radial
from proteinworkshop.models.graph_encoders.components.wrappers import (
    ScalarVector,
)
from proteinworkshop.models.utils import (
    get_activations,
    is_identity,
    safe_norm,
)


class VectorDropout(nn.Module):
    """
    Adapted from https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, drop_rate: float):
        super().__init__()
        self.drop_rate = drop_rate

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        device = x[0].device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class GCPDropout(nn.Module):
    """
    Adapted from https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, drop_rate: float, use_gcp_dropout: bool = True):
        super().__init__()
        self.scalar_dropout = (
            nn.Dropout(drop_rate) if use_gcp_dropout else nn.Identity()
        )
        self.vector_dropout = (
            VectorDropout(drop_rate) if use_gcp_dropout else nn.Identity()
        )

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, x: Union[torch.Tensor, ScalarVector]
    ) -> Union[torch.Tensor, ScalarVector]:
        if isinstance(x, torch.Tensor) and x.shape[0] == 0:
            return x
        elif isinstance(x, ScalarVector) and (
            x.scalar.shape[0] == 0 or x.vector.shape[0] == 0
        ):
            return x
        elif isinstance(x, torch.Tensor):
            return self.scalar_dropout(x)
        return ScalarVector(self.scalar_dropout(x[0]), self.vector_dropout(x[1]))


class GCPLayerNorm(nn.Module):
    """
    Adapted from https://github.com/drorlab/gvp-pytorch
    """

    def __init__(
        self, dims: ScalarVector, eps: float = 1e-8, use_gcp_norm: bool = True
    ):
        super().__init__()
        self.scalar_dims, self.vector_dims = dims
        self.scalar_norm = (
            nn.LayerNorm(self.scalar_dims) if use_gcp_norm else nn.Identity()
        )
        self.use_gcp_norm = use_gcp_norm
        self.eps = eps

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    def norm_vector(
        v: torch.Tensor, use_gcp_norm: bool = True, eps: float = 1e-8
    ) -> torch.Tensor:
        v_norm = v
        if use_gcp_norm:
            vector_norm = torch.clamp(
                torch.sum(torch.square(v), dim=-1, keepdim=True), min=eps
            )
            vector_norm = torch.sqrt(torch.mean(vector_norm, dim=-2, keepdim=True))
            v_norm = v / vector_norm
        return v_norm

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, x: Union[torch.Tensor, ScalarVector]
    ) -> Union[torch.Tensor, ScalarVector]:
        if isinstance(x, torch.Tensor) and x.shape[0] == 0:
            return x
        elif isinstance(x, ScalarVector) and (
            x.scalar.shape[0] == 0 or x.vector.shape[0] == 0
        ):
            return x
        elif not self.vector_dims:
            return self.scalar_norm(x)
        s, v = x
        return ScalarVector(
            self.scalar_norm(s),
            self.norm_vector(v, use_gcp_norm=self.use_gcp_norm, eps=self.eps),
        )


class GCP(nn.Module):
    def __init__(
        self,
        input_dims: ScalarVector,
        output_dims: ScalarVector,
        nonlinearities: Tuple[Optional[str]] = ("silu", "silu"),
        scalar_out_nonlinearity: Optional[str] = "silu",
        scalar_gate: int = 0,
        vector_gate: bool = True,
        feedforward_out: bool = False,
        bottleneck: int = 1,
        scalarization_vectorization_output_dim: int = 3,
        enable_e3_equivariance: bool = False,
        **kwargs,
    ):
        super().__init__()

        if nonlinearities is None:
            nonlinearities = ("none", "none")

        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.scalar_nonlinearity, self.vector_nonlinearity = (
            get_activations(nonlinearities[0], return_functional=True),
            get_activations(nonlinearities[1], return_functional=True),
        )
        self.scalar_gate, self.vector_gate = scalar_gate, vector_gate
        self.enable_e3_equivariance = enable_e3_equivariance

        if self.scalar_gate > 0:
            self.norm = nn.LayerNorm(self.scalar_output_dim)

        if self.vector_input_dim:
            assert (
                self.vector_input_dim % bottleneck == 0
            ), f"Input channel of vector ({self.vector_input_dim}) must be divisible with bottleneck factor ({bottleneck})"

            self.hidden_dim = (
                self.vector_input_dim // bottleneck
                if bottleneck > 1
                else max(self.vector_input_dim, self.vector_output_dim)
            )

            scalar_vector_frame_dim = scalarization_vectorization_output_dim * 3
            self.vector_down = nn.Linear(
                self.vector_input_dim, self.hidden_dim, bias=False
            )
            self.scalar_out = (
                nn.Sequential(
                    nn.Linear(
                        self.hidden_dim
                        + self.scalar_input_dim
                        + scalar_vector_frame_dim,
                        self.scalar_output_dim,
                    ),
                    get_activations(scalar_out_nonlinearity),
                    nn.Linear(self.scalar_output_dim, self.scalar_output_dim),
                )
                if feedforward_out
                else nn.Linear(
                    self.hidden_dim + self.scalar_input_dim + scalar_vector_frame_dim,
                    self.scalar_output_dim,
                )
            )

            self.vector_down_frames = nn.Linear(
                self.vector_input_dim,
                scalarization_vectorization_output_dim,
                bias=False,
            )

            if self.vector_output_dim:
                self.vector_up = nn.Linear(
                    self.hidden_dim, self.vector_output_dim, bias=False
                )
                if self.vector_gate:
                    self.vector_out_scale = nn.Linear(
                        self.scalar_output_dim, self.vector_output_dim
                    )
        else:
            self.scalar_out = (
                nn.Sequential(
                    nn.Linear(self.scalar_input_dim, self.scalar_output_dim),
                    get_activations(scalar_out_nonlinearity),
                    nn.Linear(self.scalar_output_dim, self.scalar_output_dim),
                )
                if feedforward_out
                else nn.Linear(self.scalar_input_dim, self.scalar_output_dim)
            )

    @jaxtyped(typechecker=typechecker)
    def create_zero_vector(
        self,
        scalar_rep: Float[torch.Tensor, "batch_num_entities merged_scalar_dim"],
    ) -> Float[torch.Tensor, "batch_num_entities o 3"]:
        return torch.zeros(
            scalar_rep.shape[0],
            self.vector_output_dim,
            3,
            device=scalar_rep.device,
        )

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    def scalarize(
        vector_rep: Float[torch.Tensor, "batch_num_entities 3 3"],
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        node_inputs: bool,
        enable_e3_equivariance: bool,
        dim_size: int,
        node_mask: Optional[Bool[torch.Tensor, "n_nodes"]] = None,
    ) -> Float[torch.Tensor, "effective_batch_num_entities 9"]:
        row, col = edge_index[0], edge_index[1]

        # gather source node features for each `entity` (i.e., node or edge)
        # note: edge inputs are already ordered according to source nodes
        vector_rep_i = vector_rep[row] if node_inputs else vector_rep

        # project equivariant values onto corresponding local frames
        if vector_rep_i.ndim == 2:
            vector_rep_i = vector_rep_i.unsqueeze(-1)
        elif vector_rep_i.ndim == 3:
            vector_rep_i = vector_rep_i.transpose(-1, -2)

        if node_mask is not None:
            edge_mask = node_mask[row] & node_mask[col]
            # Initialize destination tensor
            local_scalar_rep_i = torch.zeros(
                (edge_index.shape[1], 3, 3), device=edge_index.device
            )
            # Calculate the source value (result of matmul, likely Half under AMP)
            matmul_result = torch.matmul(
                frames[edge_mask], vector_rep_i[edge_mask]
            )
            # Explicitly cast the source value to the destination's dtype before assignment
            local_scalar_rep_i[edge_mask] = matmul_result.to(local_scalar_rep_i.dtype)

            local_scalar_rep_i = local_scalar_rep_i.transpose(-1, -2)
        else:
            # This path might need similar treatment if it causes issues
            local_scalar_rep_i = torch.matmul(frames, vector_rep_i).transpose(-1, -2)

        # potentially enable E(3)-equivariance and, thereby, chirality-invariance
        if enable_e3_equivariance:
            # avoid corrupting gradients with an in-place operation
            local_scalar_rep_i_copy = local_scalar_rep_i.clone()
            local_scalar_rep_i_copy[:, :, 1] = torch.abs(local_scalar_rep_i[:, :, 1])
            local_scalar_rep_i = local_scalar_rep_i_copy

        # reshape frame-derived geometric scalars
        local_scalar_rep_i = local_scalar_rep_i.reshape(vector_rep_i.shape[0], 9)

        if node_inputs:
            # for node inputs, summarize all edge-wise geometric scalars using an average
            return torch_scatter.scatter(
                local_scalar_rep_i,
                # summarize according to source node indices due to the directional nature of GCP's equivariant frames
                row,
                dim=0,
                dim_size=dim_size,
                reduce="mean",
            )

        return local_scalar_rep_i

    @jaxtyped(typechecker=typechecker)
    def vectorize(
        self,
        scalar_rep: Float[torch.Tensor, "batch_num_entities merged_scalar_dim"],
        vector_hidden_rep: Float[torch.Tensor, "batch_num_entities 3 n"],
    ) -> Float[torch.Tensor, "batch_num_entities o 3"]:
        vector_rep = self.vector_up(vector_hidden_rep)
        vector_rep = vector_rep.transpose(-1, -2)

        if self.vector_gate:
            gate = self.vector_out_scale(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(
                safe_norm(vector_rep, dim=-1, keepdim=True)
            )

        return vector_rep

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        s_maybe_v: Union[
            Tuple[
                Float[torch.Tensor, "batch_num_entities scalar_dim"],
                Float[torch.Tensor, "batch_num_entities m vector_dim"],
            ],
            Float[torch.Tensor, "batch_num_entities merged_scalar_dim"],
        ],
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        node_inputs: bool = False,
        node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
    ) -> Union[
        Tuple[
            Float[torch.Tensor, "batch_num_entities new_scalar_dim"],
            Float[torch.Tensor, "batch_num_entities n vector_dim"],
        ],
        Float[torch.Tensor, "batch_num_entities new_scalar_dim"],
    ]:
        if self.vector_input_dim:
            scalar_rep, vector_rep = s_maybe_v
            v_pre = vector_rep.transpose(-1, -2)

            vector_hidden_rep = self.vector_down(v_pre)
            vector_norm = safe_norm(vector_hidden_rep, dim=-2)
            merged = torch.cat((scalar_rep, vector_norm), dim=-1)

            # curate direction-robust and (by default) chirality-aware scalar geometric features
            vector_down_frames_hidden_rep = self.vector_down_frames(v_pre)
            scalar_hidden_rep = self.scalarize(
                vector_down_frames_hidden_rep.transpose(-1, -2),
                edge_index,
                frames,
                node_inputs=node_inputs,
                enable_e3_equivariance=self.enable_e3_equivariance,
                dim_size=vector_down_frames_hidden_rep.shape[0],
                node_mask=node_mask,
            )
            merged = torch.cat((merged, scalar_hidden_rep), dim=-1)
        else:
            # bypass updating scalar features using vector information
            merged = s_maybe_v

        scalar_rep = self.scalar_out(merged)

        if not self.vector_output_dim:
            # bypass updating vector features using scalar information
            return self.scalar_nonlinearity(scalar_rep)
        elif self.vector_output_dim and not self.vector_input_dim:
            # instantiate vector features that are learnable in proceeding GCP layers
            vector_rep = self.create_zero_vector(scalar_rep)
        else:
            # update vector features using either row-wise scalar gating with complete local frames or row-wise self-scalar gating
            vector_rep = self.vectorize(scalar_rep, vector_hidden_rep)

        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        return ScalarVector(scalar_rep, vector_rep)


class GCPEmbedding(nn.Module):
    def __init__(
        self,
        edge_input_dims: ScalarVector,
        node_input_dims: ScalarVector,
        edge_hidden_dims: ScalarVector,
        node_hidden_dims: ScalarVector,
        num_atom_types: int = 0,
        nonlinearities: Tuple[Optional[str]] = ("silu", "silu"),
        cfg: DictConfig = None,
        pre_norm: bool = True,
        use_gcp_norm: bool = True,
    ):
        super().__init__()

        if num_atom_types > 0:
            self.atom_embedding = nn.Embedding(num_atom_types, num_atom_types)
        else:
            self.atom_embedding = None

        self.radial_embedding = partial(
            radial.compute_rbf, max_distance=cfg.r_max, num_rbf=cfg.num_rbf
        )

        self.pre_norm = pre_norm
        if pre_norm:
            self.edge_normalization = GCPLayerNorm(
                edge_input_dims, use_gcp_norm=use_gcp_norm
            )
            self.node_normalization = GCPLayerNorm(
                node_input_dims, use_gcp_norm=use_gcp_norm
            )
        else:
            self.edge_normalization = GCPLayerNorm(
                edge_hidden_dims, use_gcp_norm=use_gcp_norm
            )
            self.node_normalization = GCPLayerNorm(
                node_hidden_dims, use_gcp_norm=use_gcp_norm
            )

        self.edge_embedding = GCP(
            edge_input_dims,
            edge_hidden_dims,
            nonlinearities=nonlinearities,
            scalar_gate=cfg.scalar_gate,
            vector_gate=cfg.vector_gate,
            enable_e3_equivariance=cfg.enable_e3_equivariance,
        )

        self.node_embedding = GCP(
            node_input_dims,
            node_hidden_dims,
            nonlinearities=("none", "none"),
            scalar_gate=cfg.scalar_gate,
            vector_gate=cfg.vector_gate,
            enable_e3_equivariance=cfg.enable_e3_equivariance,
        )

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, batch: Union[Batch, ProteinBatch]
    ) -> Tuple[
        Union[
            Tuple[
                Float[torch.Tensor, "batch_num_nodes h_hidden_dim"],
                Float[torch.Tensor, "batch_num_nodes m chi_hidden_dim"],
            ],
            Float[torch.Tensor, "batch_num_nodes h_hidden_dim"],
        ],
        Union[
            Tuple[
                Float[torch.Tensor, "batch_num_edges e_hidden_dim"],
                Float[torch.Tensor, "batch_num_edges x xi_hidden_dim"],
            ],
            Float[torch.Tensor, "batch_num_edges e_hidden_dim"],
        ],
    ]:
        if self.atom_embedding is not None:
            node_rep = ScalarVector(self.atom_embedding(batch.h), batch.chi)
        else:
            node_rep = ScalarVector(batch.h, batch.chi)

        edge_rep = ScalarVector(batch.e, batch.xi)

        edge_vectors = (
            batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]
        )  # [n_edges, 3]
        edge_lengths = torch.linalg.norm(edge_vectors, dim=-1)  # [n_edges, 1]
        edge_rep = ScalarVector(
            torch.cat((edge_rep.scalar, self.radial_embedding(edge_lengths)), dim=-1),
            edge_rep.vector,
        )

        edge_rep = (
            edge_rep.scalar if not self.edge_embedding.vector_input_dim else edge_rep
        )
        node_rep = (
            node_rep.scalar if not self.node_embedding.vector_input_dim else node_rep
        )

        if self.pre_norm:
            edge_rep = self.edge_normalization(edge_rep)
            node_rep = self.node_normalization(node_rep)

        edge_rep = self.edge_embedding(
            edge_rep,
            batch.edge_index,
            batch.f_ij,
            node_inputs=False,
            node_mask=getattr(batch, "mask", None),
        )
        node_rep = self.node_embedding(
            node_rep,
            batch.edge_index,
            batch.f_ij,
            node_inputs=True,
            node_mask=getattr(batch, "mask", None),
        )

        if not self.pre_norm:
            edge_rep = self.edge_normalization(edge_rep)
            node_rep = self.node_normalization(node_rep)

        return node_rep, edge_rep


@typechecker
def get_GCP_with_custom_cfg(
    input_dims: Any, output_dims: Any, cfg: DictConfig, **kwargs
):
    cfg_dict = copy(OmegaConf.to_container(cfg, throw_on_missing=True))
    cfg_dict["nonlinearities"] = cfg.nonlinearities
    del cfg_dict["scalar_nonlinearity"]
    del cfg_dict["vector_nonlinearity"]

    for key in kwargs:
        cfg_dict[key] = kwargs[key]

    return GCP(input_dims, output_dims, **cfg_dict)


class GCPMessagePassing(nn.Module):
    def __init__(
        self,
        input_dims: ScalarVector,
        output_dims: ScalarVector,
        edge_dims: ScalarVector,
        cfg: DictConfig,
        mp_cfg: DictConfig,
        reduce_function: str = "sum",
        use_scalar_message_attention: bool = True,
    ):
        super().__init__()

        # hyperparameters
        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.edge_scalar_dim, self.edge_vector_dim = edge_dims
        self.conv_cfg = mp_cfg
        self.self_message = self.conv_cfg.self_message
        self.reduce_function = reduce_function
        self.use_scalar_message_attention = use_scalar_message_attention

        scalars_in_dim = 2 * self.scalar_input_dim + self.edge_scalar_dim
        vectors_in_dim = 2 * self.vector_input_dim + self.edge_vector_dim

        # config instantiations
        soft_cfg = copy(cfg)
        soft_cfg.bottleneck = cfg.default_bottleneck

        primary_cfg_GCP = partial(get_GCP_with_custom_cfg, cfg=soft_cfg)
        secondary_cfg_GCP = partial(get_GCP_with_custom_cfg, cfg=cfg)

        # PyTorch modules #
        module_list = [
            primary_cfg_GCP(
                (scalars_in_dim, vectors_in_dim),
                output_dims,
                nonlinearities=cfg.nonlinearities,
                enable_e3_equivariance=cfg.enable_e3_equivariance,
            )
        ]

        for _ in range(self.conv_cfg.num_message_layers - 2):
            module_list.append(
                secondary_cfg_GCP(
                    output_dims,
                    output_dims,
                    enable_e3_equivariance=cfg.enable_e3_equivariance,
                )
            )

        if self.conv_cfg.num_message_layers > 1:
            module_list.append(
                primary_cfg_GCP(
                    output_dims,
                    output_dims,
                    nonlinearities=cfg.nonlinearities,
                    enable_e3_equivariance=cfg.enable_e3_equivariance,
                )
            )

        self.message_fusion = nn.ModuleList(module_list)

        # learnable scalar message gating
        if use_scalar_message_attention:
            self.scalar_message_attention = nn.Sequential(
                nn.Linear(output_dims.scalar, 1), nn.Sigmoid()
            )

    @jaxtyped(typechecker=typechecker)
    def message(
        self,
        node_rep: ScalarVector,
        edge_rep: ScalarVector,
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
    ) -> Float[torch.Tensor, "batch_num_edges message_dim"]:
        row, col = edge_index
        vector = node_rep.vector.reshape(
            node_rep.vector.shape[0],
            node_rep.vector.shape[1] * node_rep.vector.shape[2],
        )
        vector_reshaped = ScalarVector(node_rep.scalar, vector)

        s_row, v_row = vector_reshaped.idx(row)
        s_col, v_col = vector_reshaped.idx(col)

        v_row = v_row.reshape(v_row.shape[0], v_row.shape[1] // 3, 3)
        v_col = v_col.reshape(v_col.shape[0], v_col.shape[1] // 3, 3)

        message = ScalarVector(s_row, v_row).concat(
            (edge_rep, ScalarVector(s_col, v_col))
        )

        message_residual = self.message_fusion[0](
            message, edge_index, frames, node_inputs=False, node_mask=node_mask
        )
        for module in self.message_fusion[1:]:
            # exchange geometric messages while maintaining residual connection to original message
            new_message = module(
                message_residual,
                edge_index,
                frames,
                node_inputs=False,
                node_mask=node_mask,
            )
            message_residual = message_residual + new_message

        # learn to gate scalar messages
        if self.use_scalar_message_attention:
            message_residual_attn = self.scalar_message_attention(
                message_residual.scalar
            )
            message_residual = ScalarVector(
                message_residual.scalar * message_residual_attn,
                message_residual.vector,
            )

        return message_residual.flatten()

    @jaxtyped(typechecker=typechecker)
    def aggregate(
        self,
        message: Float[torch.Tensor, "batch_num_edges message_dim"],
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        dim_size: int,
    ) -> Float[torch.Tensor, "batch_num_nodes aggregate_dim"]:
        row, col = edge_index
        aggregate = torch_scatter.scatter(
            message, row, dim=0, dim_size=dim_size, reduce=self.reduce_function
        )
        return aggregate

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        node_rep: ScalarVector,
        edge_rep: ScalarVector,
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
    ) -> ScalarVector:
        message = self.message(
            node_rep, edge_rep, edge_index, frames, node_mask=node_mask
        )
        aggregate = self.aggregate(
            message, edge_index, dim_size=node_rep.scalar.shape[0]
        )
        return ScalarVector.recover(aggregate, self.vector_output_dim)


class GCPInteractions(nn.Module):
    def __init__(
        self,
        node_dims: ScalarVector,
        edge_dims: ScalarVector,
        cfg: DictConfig,
        layer_cfg: DictConfig,
        dropout: float = 0.0,
        nonlinearities: Optional[Tuple[Any, Any]] = None,
    ):
        super().__init__()

        # hyperparameters #
        if nonlinearities is None:
            nonlinearities = cfg.nonlinearities
        self.pre_norm = layer_cfg.pre_norm
        self.predict_node_positions = getattr(cfg, "predict_node_positions", False)
        self.node_positions_weight = getattr(cfg, "node_positions_weight", 1.0)
        self.update_positions_with_vector_sum = getattr(
            cfg, "update_positions_with_vector_sum", False
        )
        reduce_function = "sum"

        # PyTorch modules #

        # geometry-complete message-passing neural network
        message_function = GCPMessagePassing

        self.interaction = message_function(
            node_dims,
            node_dims,
            edge_dims,
            cfg=cfg,
            mp_cfg=layer_cfg.mp_cfg,
            reduce_function=reduce_function,
            use_scalar_message_attention=layer_cfg.use_scalar_message_attention,
        )

        # config instantiations
        ff_cfg = copy(cfg)
        ff_cfg.nonlinearities = nonlinearities
        ff_GCP = partial(get_GCP_with_custom_cfg, cfg=ff_cfg)

        self.gcp_norm = nn.ModuleList(
            [GCPLayerNorm(node_dims, use_gcp_norm=layer_cfg.use_gcp_norm)]
        )
        self.gcp_dropout = nn.ModuleList(
            [GCPDropout(dropout, use_gcp_dropout=layer_cfg.use_gcp_dropout)]
        )

        # build out feedforward (FF) network modules
        hidden_dims = (
            (node_dims.scalar, node_dims.vector)
            if layer_cfg.num_feedforward_layers == 1
            else (4 * node_dims.scalar, 2 * node_dims.vector)
        )
        ff_interaction_layers = [
            ff_GCP(
                (node_dims.scalar * 2, node_dims.vector * 2),
                hidden_dims,
                nonlinearities=("none", "none")
                if layer_cfg.num_feedforward_layers == 1
                else cfg.nonlinearities,
                feedforward_out=layer_cfg.num_feedforward_layers == 1,
                enable_e3_equivariance=cfg.enable_e3_equivariance,
            )
        ]

        interaction_layers = [
            ff_GCP(
                hidden_dims,
                hidden_dims,
                enable_e3_equivariance=cfg.enable_e3_equivariance,
            )
            for _ in range(layer_cfg.num_feedforward_layers - 2)
        ]
        ff_interaction_layers.extend(interaction_layers)

        if layer_cfg.num_feedforward_layers > 1:
            ff_interaction_layers.append(
                ff_GCP(
                    hidden_dims,
                    node_dims,
                    nonlinearities=("none", "none"),
                    feedforward_out=True,
                    enable_e3_equivariance=cfg.enable_e3_equivariance,
                )
            )

        self.feedforward_network = nn.ModuleList(ff_interaction_layers)

        # potentially build out node position update modules
        if self.predict_node_positions:
            # node position update GCP(s)
            position_output_dims = (
                node_dims
                if getattr(cfg, "update_positions_with_vector_sum", False)
                else (node_dims.scalar, 1)
            )
            self.node_position_update_gcp = ff_GCP(
                node_dims,
                position_output_dims,
                nonlinearities=cfg.nonlinearities,
                enable_e3_equivariance=cfg.enable_e3_equivariance,
            )

    @jaxtyped(typechecker=typechecker)
    def derive_x_update(
        self,
        node_rep: ScalarVector,
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        f_ij: Float[torch.Tensor, "batch_num_edges 3 3"],
        node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
    ) -> Float[torch.Tensor, "batch_num_nodes 3"]:
        # use vector-valued features to derive node position updates
        node_rep_update = self.node_position_update_gcp(
            node_rep, edge_index, f_ij, node_inputs=True, node_mask=node_mask
        )
        if self.update_positions_with_vector_sum:
            x_vector_update = node_rep_update.vector.sum(1)
        else:
            x_vector_update = node_rep_update.vector.squeeze(1)

        # (up/down)weight position updates
        x_update = x_vector_update * self.node_positions_weight

        return x_update

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        node_rep: Tuple[
            Float[torch.Tensor, "batch_num_nodes node_hidden_dim"],
            Float[torch.Tensor, "batch_num_nodes m 3"],
        ],
        edge_rep: Tuple[
            Float[torch.Tensor, "batch_num_edges edge_hidden_dim"],
            Float[torch.Tensor, "batch_num_edges x 3"],
        ],
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
        node_pos: Optional[Float[torch.Tensor, "batch_num_nodes 3"]] = None,
    ) -> Tuple[
        Tuple[
            Float[torch.Tensor, "batch_num_nodes hidden_dim"],
            Float[torch.Tensor, "batch_num_nodes n 3"],
        ],
        Optional[Float[torch.Tensor, "batch_num_nodes 3"]],
    ]:
        node_rep = ScalarVector(node_rep[0], node_rep[1])
        edge_rep = ScalarVector(edge_rep[0], edge_rep[1])

        # apply GCP normalization (1)
        if self.pre_norm:
            node_rep = self.gcp_norm[0](node_rep)

        # forward propagate with interaction module
        hidden_residual = self.interaction(
            node_rep, edge_rep, edge_index, frames, node_mask=node_mask
        )

        # aggregate input and hidden features
        hidden_residual = ScalarVector(*hidden_residual.concat((node_rep,)))

        # propagate with feedforward layers
        for module in self.feedforward_network:
            hidden_residual = module(
                hidden_residual,
                edge_index,
                frames,
                node_inputs=True,
                node_mask=node_mask,
            )

        # apply GCP dropout
        node_rep = node_rep + self.gcp_dropout[0](hidden_residual)

        # apply GCP normalization (2)
        if not self.pre_norm:
            node_rep = self.gcp_norm[0](node_rep)

        # update only unmasked node representations and residuals
        if node_mask is not None:
            node_rep = node_rep.mask(node_mask.float())

        # bypass updating node positions
        if not self.predict_node_positions:
            return node_rep, node_pos

        # update node positions
        node_pos = node_pos + self.derive_x_update(
            node_rep, edge_index, frames, node_mask=node_mask
        )

        # update only unmasked node positions
        if node_mask is not None:
            node_pos = node_pos * node_mask.float().unsqueeze(-1)

        return node_rep, node_pos
