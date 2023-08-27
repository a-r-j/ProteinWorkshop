"""Spatial Attention Kinetic Networks with E(n)-Equivariance.

https://github.com/yuanqing-wang/sake
"""
import math
import random
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
EPSILON = 1e-5
INF = 1e5

# Utils


class DoubleSigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 2.0 * torch.sigmoid(x)


def get_x_minus_xt(x):
    return torch.unsqueeze(x, -3) - torch.unsqueeze(x, -2)


def get_x_minus_xt_norm(
    x_minus_xt,
    epsilon: float = EPSILON,
):
    return (
        F.relu((x_minus_xt**2).sum(dim=-1, keepdims=True)) + epsilon
    ) ** 0.5


def get_h_cat_ht(h):
    n_nodes = h.shape[-2]
    h_shape = (*h.shape[:-2], n_nodes, n_nodes, h.shape[-1])
    return torch.cat(
        [
            torch.broadcast_to(torch.unsqueeze(h, -3), h_shape),
            torch.broadcast_to(torch.unsqueeze(h, -2), h_shape),
        ],
        dim=-1,
    )


def coloring(x, mean, std):
    return std * x + mean


def cosine_cutoff(x, lower=0.0, upper=5.0):
    cutoffs = 0.5 * (
        torch.cos(math.pi * (2 * (x - lower) / (upper - lower) + 1.0)) + 1.0
    )
    # remove contributions below the cutoff radius
    x = x * (x < upper)
    x = x * (x > lower)
    return cutoffs


class ExpNormalSmearing(nn.Module):
    def __init__(
        self,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        num_rbf: int = 50,
    ):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf

        self.alpha = 5.0 / (self.cutoff_upper - self.cutoff_lower)
        means, betas = self._initial_params()
        self.out_features = self.num_rbf
        # self.means = self.param(
        #    "means",
        #    nn.initializers.constant(means),
        #    means.shape,
        # )
        # self.means = nn.Parameter(torch.Tensor(means.shape))
        # nn.init.constant_(self.means, means)
        self.means = nn.Parameter(means)

        # self.betas = self.param(
        #    "betas",
        #    nn.initializers.constant(betas),
        #    betas.shape,
        # )
        self.betas = nn.Parameter(betas)
        # self.betas = nn.Parameter(torch.Tensor(betas.shape))
        # nn.init.constant_(self.betas, betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = np.exp(-self.cutoff_upper + self.cutoff_lower)
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf,
            dtype=torch.float32,
        )
        return means, betas

    def forward(self, dist):
        return torch.exp(
            -self.betas
            * (
                torch.exp(self.alpha * (-dist + self.cutoff_lower))
                - self.means
            )
            ** 2
        )


@torch.jit.script
def mae(x, y):
    return torch.abs(x - y).mean()


# @torch.jit.script
def mae_with_replacement(x, y, seed=0):
    # key = jax.random.PRNGKey(seed)
    # idxs = jax.random.choice(
    #    key,
    #    x.shape[0],
    #    shape=(x.shape[0],),
    #    replace=True,
    # )
    # x = x[idxs]
    # y = y[idxs]
    # return mae(x, y)
    random.seed(seed)
    idxs = torch.tensor(random.choices(range(x.shape[0]), k=x.shape[0]))
    x = x[idxs]
    y = y[idxs]
    return mae(x, y)


def bootstrap_mae(x, y, n_samples=10, ci=0.95):
    original = torch.abs(x - y).mean().item()
    results = []
    for idx in range(n_samples):
        result = mae_with_replacement(x, y, idx).item()
        results.append(result)
    low = np.percentile(results, 100.0 * 0.5 * (1 - ci))
    high = np.percentile(results, (1 - ((1 - ci) * 0.5)) * 100.0)
    return original, low, high


class ContinuousFilterConvolutionWithConcatenation(nn.Module):
    def __init__(
        self,
        out_features: int,
        kernel_features: int = 50,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.out_features = out_features
        self.kernel_features = kernel_features
        self.activation = activation

        self.kernel = ExpNormalSmearing(num_rbf=self.kernel_features)
        self.mlp_in = nn.LazyLinear(self.kernel_features)
        self.mlp_out = nn.Sequential(
            nn.LazyLinear(self.out_features),
            self.activation,
            nn.LazyLinear(self.out_features),
        )

    def forward(self, h, x):
        h0 = h
        h = self.mlp_in(h)
        _x = self.kernel(x) * h

        h = self.mlp_out(torch.cat([h0, _x, x], dim=-1))

        return h


class DenseSAKELayer(nn.Module):
    def __init__(
        self,
        out_features: int,
        hidden_features: int,
        activation: nn.Module = nn.SiLU(),
        n_heads: int = 4,
        update: bool = True,
        use_semantic_attention: bool = True,
        use_euclidean_attention: bool = True,
        use_spatial_attention: bool = True,
        cutoff: Optional[Callable] = None,
    ):
        super().__init__()
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.activation = activation
        self.n_heads = n_heads
        self.update = update
        self.use_semantic_attention = use_semantic_attention
        self.use_euclidean_attention = use_euclidean_attention
        self.use_spatial_attention = use_spatial_attention
        self.cutoff = cutoff

        self.edge_model = ContinuousFilterConvolutionWithConcatenation(
            self.hidden_features
        )
        self.n_coefficients = self.n_heads * self.hidden_features

        self.node_mlp = nn.Sequential(
            # nn.LayerNorm(),
            nn.LazyLinear(self.hidden_features),
            self.activation,
            nn.LazyLinear(self.out_features),
            self.activation,
        )

        if self.update:
            self.velocity_mlp = nn.Sequential(
                nn.LazyLinear(self.hidden_features),
                self.activation,
                nn.LazyLinear(1, bias=False),
                # double_sigmoid,
                DoubleSigmoid(),
            )

        self.semantic_attention_mlp = nn.Sequential(
            nn.LazyLinear(self.n_heads),
            nn.CELU(alpha=2.0),
        )

        self.post_norm_mlp = nn.Sequential(
            nn.LazyLinear(self.hidden_features),
            self.activation,
            nn.LazyLinear(self.hidden_features),
            self.activation,
        )

        self.v_mixing = nn.LazyLinear(1, bias=False)
        self.x_mixing = nn.Sequential(
            nn.LazyLinear(self.n_coefficients, bias=False), nn.Tanh()
        )

        log_gamma = -torch.log(torch.linspace(1.0, 5.0, self.n_heads))
        if self.use_semantic_attention and self.use_euclidean_attention:
            # self.log_gamma = self.param(
            #    "log_gamma",
            #    nn.initializers.constant(log_gamma),
            #    log_gamma.shape,
            # )
            # self.log_gamma = nn.Parameter(torch.tensor(log_gamma).shape)
            # nn.init.constant_(self.log_gamma, log_gamma)
            self.log_gamma = nn.Parameter(log_gamma)
        else:
            self.log_gamma = torch.ones(self.n_heads)

    def spatial_attention(
        self, h_e_mtx, x_minus_xt, x_minus_xt_norm, mask=None
    ):
        # (batch_size, n, n, n_coefficients)
        # coefficients = self.coefficients_mlp(h_e_mtx)# .unsqueeze(-1)
        coefficients = self.x_mixing(h_e_mtx)

        # (batch_size, n, n, 3)
        # x_minus_xt = x_minus_xt * euclidean_attention.mean(dim=-1, keepdim=True) / (x_minus_xt_norm + 1e-5)
        x_minus_xt = x_minus_xt / (x_minus_xt_norm + 1e-5)  # ** 2

        # (batch_size, n, n, coefficients, 3)
        combinations = torch.unsqueeze(x_minus_xt, -2) * torch.unsqueeze(
            coefficients, -1
        )

        if mask is not None:
            _mask = torch.unsqueeze(torch.unsqueeze(mask, -1), -1)
            combinations = combinations * _mask
            combinations_sum = combinations.sum(dim=-3) / (
                _mask.sum(dim=-3) + 1e-8
            )

        else:
            # (batch_size, n, n, coefficients)
            combinations_sum = combinations.mean(dim=-3)

        combinations_norm = (combinations_sum**2).sum(-1)  # .pow(0.5)

        h_combinations = self.post_norm_mlp(combinations_norm)
        # h_combinations = self.norm(h_combinations)
        return h_combinations, combinations

    def aggregate(self, h_e_mtx, mask=None):
        # h_e_mtx = self.mask_self(h_e_mtx)
        if mask is not None:
            h_e_mtx = h_e_mtx * torch.unsqueeze(mask, -1)
        return h_e_mtx.sum(dim=-2)

    def node_model(self, h, h_e, h_combinations):
        out = torch.cat(
            [
                h,
                h_e,
                h_combinations,
            ],
            dim=-1,
        )
        out = self.node_mlp(out)
        out = h + out
        return out

    def euclidean_attention(self, x_minus_xt_norm, mask=None):
        # (batch_size, n, n, 1)
        _x_minus_xt_norm = x_minus_xt_norm + 1e5 * torch.unsqueeze(
            torch.eye(
                x_minus_xt_norm.shape[-2],
                x_minus_xt_norm.shape[-2],
                device=x_minus_xt_norm.device,
            ),
            -1,
        )

        if mask is not None:
            _x_minus_xt_norm = _x_minus_xt_norm + 1e5 * (
                1 - torch.unsqueeze(mask, -1)
            )

        return F.softmax(
            -_x_minus_xt_norm * torch.exp(self.log_gamma),
            dim=-2,
        )

    def semantic_attention(self, h_e_mtx, mask=None):
        # (batch_size, n, n, n_heads)
        att = self.semantic_attention_mlp(h_e_mtx)

        # (batch_size, n, n, n_heads)
        # att = att.view(*att.shape[:-1], self.n_heads)
        att = att - 1e5 * torch.unsqueeze(
            torch.eye(
                att.shape[-2],
                att.shape[-2],
                device=att.device,
            ),
            -1,
        )

        if mask is not None:
            att = att - 1e5 * (1 - torch.unsqueeze(mask, -1))

        return F.softmax(att, dim=-2)

    def combined_attention(self, x_minus_xt_norm, h_e_mtx, mask=None):
        euclidean_attention = self.euclidean_attention(
            x_minus_xt_norm, mask=mask
        )
        semantic_attention = self.semantic_attention(h_e_mtx, mask=mask)

        if not self.use_semantic_attention:
            semantic_attention = torch.ones_like(semantic_attention)
        if not self.use_euclidean_attention:
            euclidean_attention = torch.ones_like(euclidean_attention)

        combined_attention = euclidean_attention * semantic_attention
        if mask is not None:
            combined_attention = combined_attention - 1e5 * (
                1 - torch.unsqueeze(mask, -1)
            )
        combined_attention = F.softmax(combined_attention, dim=-2)

        if self.cutoff is not None:
            cutoff = self.cutoff(x_minus_xt_norm)
            combined_attention = combined_attention * cutoff
            combined_attention = combined_attention / combined_attention.sum(
                axis=-2, keepdims=True
            )

        return euclidean_attention, semantic_attention, combined_attention

    def velocity_model(self, v, h):
        v = self.velocity_mlp(h) * v
        return v

    def forward(
        self,
        h,
        x,
        v=None,
        mask=None,
        he=None,
    ):
        x_minus_xt = get_x_minus_xt(x)
        x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt=x_minus_xt)
        h_cat_ht = get_h_cat_ht(h)

        if he is not None:
            h_cat_ht = torch.cat([h_cat_ht, he], -1)

        h_e_mtx = self.edge_model(h_cat_ht, x_minus_xt_norm)
        (
            euclidean_attention,
            semantic_attention,
            combined_attention,
        ) = self.combined_attention(x_minus_xt_norm, h_e_mtx, mask=mask)
        h_e_att = torch.unsqueeze(h_e_mtx, -1) * torch.unsqueeze(
            combined_attention, -2
        )
        h_e_att = torch.reshape(h_e_att, h_e_att.shape[:-2] + (-1,))
        h_combinations, delta_v = self.spatial_attention(
            h_e_att, x_minus_xt, x_minus_xt_norm, mask=mask
        )

        if not self.use_spatial_attention:
            h_combinations = torch.zeros_like(h_combinations)
            delta_v = torch.zeros_like(delta_v)

        # h_e_mtx = (h_e_mtx.unsqueeze(-1) * combined_attention.unsqueeze(-2)).flatten(-2, -1)
        h_e = self.aggregate(h_e_att, mask=mask)
        h = self.node_model(h, h_e, h_combinations)

        if self.update:
            if mask is not None:
                delta_v = (
                    self.v_mixing(delta_v.swapaxes(-1, -2))
                    .swapaxes(-1, -2)
                    .sum(axis=(-2, -3))
                )
                delta_v = delta_v / (mask.sum(-1, keepdims=True) + 1e-10)
            else:
                delta_v = (
                    self.v_mixing(delta_v.swapaxes(-1, -2))
                    .swapaxes(-1, -2)
                    .mean(axis=(-2, -3))
                )

            v = (
                self.velocity_model(v, h)
                if v is not None
                else torch.zeros_like(x)
            )
            v = delta_v + v
            x = x + v

        return h, x, v


class EquivariantGraphConvolutionalLayer(nn.Module):
    def __init__(
        self,
        out_features: int,
        hidden_features: int,
        activation: nn.Module = nn.SiLU(),
        update: bool = False,
        sigmoid: bool = False,
    ):
        super().__init__()
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.activation = activation
        self.update = update
        self.sigmoid = sigmoid

        self.node_mlp = nn.Sequential(
            nn.LazyLinear(self.hidden_features),
            self.activation,
            nn.LazyLinear(self.out_features),
            self.activation,
        )

        self.scaling_mlp = nn.Sequential(
            nn.LazyLinear(self.hidden_features),
            self.activation,
            nn.LazyLinear(1, bias=False),
        )

        self.shifting_mlp = nn.Sequential(
            nn.LazyLinear(self.hidden_features),
            self.activation,
            nn.LazyLinear(1, bias=False),
        )

        if self.sigmoid:
            self.edge_model = nn.Sequential(
                nn.LazyLinear(1, bias=False),
                nn.Sigmoid(),
            )

    def aggregate(self, h_e_mtx, mask=None):
        # h_e_mtx = self.mask_self(h_e_mtx)
        if mask is not None:
            h_e_mtx = h_e_mtx * torch.unsqueeze(mask, -1)
        if self.sigmoid:
            h_e_weights = self.edge_model(h_e_mtx)
            h_e_mtx = h_e_weights * h_e_mtx
        return h_e_mtx.sum(axis=-2)  # h_e

    def node_model(self, h, h_e):
        out = torch.cat(
            [
                h,
                h_e,
            ],
            dim=-1,
        )
        out = self.node_mlp(out)
        out = h + out
        return out

    def velocity_model(self, v, h):
        v = self.velocity_mlp(h) * v
        return v

    def forward(
        self,
        h,
        x,
        v=None,
        mask=None,
    ):
        x_minus_xt = get_x_minus_xt(x)
        x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt=x_minus_xt)
        h_cat_ht = get_h_cat_ht(h)
        h_e_mtx = torch.cat([h_cat_ht, x_minus_xt_norm], dim=-1)
        h_e = self.aggregate(h_e_mtx, mask=mask)
        shift = self.shifting_mlp(h_e_mtx).sum(-2)
        scale = self.scaling_mlp(h)

        if self.update:
            v = v * scale + shift
            x = x + v
        h = self.node_model(h, h_e)
        return h, x, v


class EquivariantGraphConvolutionalLayerWithSmearing(nn.Module):
    def __init__(
        self,
        out_features: int,
        hidden_features: int,
        activation: nn.Module = nn.SiLU(),
        update: bool = False,
        sigmoid: bool = True,
    ):
        super().__init__()
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.activation = activation
        self.update = update
        self.sigmoid = sigmoid

    def setup(self):
        self.edge_model = ContinuousFilterConvolutionWithConcatenation(
            self.hidden_features
        )

        self.node_mlp = nn.Sequential(
            nn.LazyLinear(self.hidden_features),
            self.activation,
            nn.LazyLinear(self.out_features),
            self.activation,
        )

        self.scaling_mlp = nn.Sequential(
            nn.LazyLinear(self.hidden_features),
            self.activation,
            nn.LazyLinear(1, bias=False),
        )

        self.shifting_mlp = nn.Sequential(
            nn.LazyLinear(self.hidden_features),
            self.activation,
            nn.LazyLinear(1, bias=False),
        )

        if self.sigmoid:
            self.edge_att = nn.Sequential(
                nn.LazyLinear(1, bias=False),
                nn.Sigmoid(),
            )

    def aggregate(self, h_e_mtx, mask=None):
        # h_e_mtx = self.mask_self(h_e_mtx)
        if mask is not None:
            h_e_mtx = h_e_mtx * torch.unsqueeze(mask, -1)
        if self.sigmoid:
            h_e_weights = self.edge_att(h_e_mtx)
            h_e_mtx = h_e_weights * h_e_mtx
        return h_e_mtx.sum(dim=-2)

    def node_model(self, h, h_e):
        out = torch.cat(
            [
                h,
                h_e,
            ],
            dim=-1,
        )
        out = self.node_mlp(out)
        out = h + out
        return out

    def velocity_model(self, v, h):
        v = self.velocity_mlp(h) * v
        return v

    def forward(
        self,
        h,
        x,
        v=None,
        mask=None,
    ):
        x_minus_xt = get_x_minus_xt(x)
        x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt=x_minus_xt)
        h_cat_ht = get_h_cat_ht(h)
        h_e_mtx = self.edge_model(h_cat_ht, x_minus_xt_norm)
        h_e = self.aggregate(h_e_mtx, mask=mask)
        shift = self.shifting_mlp(h_e_mtx).sum(-2)
        scale = self.scaling_mlp(h)

        if self.update:
            v = v * scale + shift
            x = x + v
        h = self.node_model(h, h_e)
        return h, x, v
