"""
CDConv implementation adapted from the MIT-licensed source:
https://github.com/hehefan/Continuous-Discrete-Convolution/tree/main
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch_sparse as sparse
from torch import Tensor
from torch_geometric.nn import radius
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_scatter import scatter_max, scatter_mean


class Linear(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_norm: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
        leakyrelu_negative_slope: float = 0.1,
        momentum: float = 0.2,
    ) -> nn.Module:
        super(Linear, self).__init__()

        module = []
        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        module.append(nn.Linear(in_channels, out_channels, bias=bias))
        self.module = nn.Sequential(*module)

    def forward(self, x):
        return self.module(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        batch_norm: bool,
        dropout: float = 0.0,
        bias: bool = True,
        leakyrelu_negative_slope: float = 0.2,
        momentum: float = 0.2,
    ) -> nn.Module:
        super(MLP, self).__init__()

        module = []
        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        if mid_channels is None:
            module.append(nn.Linear(in_channels, out_channels, bias=bias))
        else:
            module.append(nn.Linear(in_channels, mid_channels, bias=bias))
        if batch_norm:
            if mid_channels is None:
                module.append(nn.BatchNorm1d(out_channels, momentum=momentum))
            else:
                module.append(nn.BatchNorm1d(mid_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        if mid_channels is None:
            module.append(nn.Dropout(dropout))
        else:
            module.append(nn.Linear(mid_channels, out_channels, bias=bias))

        self.module = nn.Sequential(*module)

    def forward(self, input):
        return self.module(input)


class BasicBlock(nn.Module):
    def __init__(
        self,
        r: float,
        l: float,
        kernel_channels: list[int],
        in_channels: int,
        out_channels: int,
        base_width: float = 16.0,
        batch_norm: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
        leakyrelu_negative_slope: float = 0.1,
        momentum: float = 0.2,
    ) -> nn.Module:
        super(BasicBlock, self).__init__()

        if in_channels != out_channels:
            self.identity = Linear(
                in_channels=in_channels,
                out_channels=out_channels,
                batch_norm=batch_norm,
                dropout=dropout,
                bias=bias,
                leakyrelu_negative_slope=leakyrelu_negative_slope,
                momentum=momentum,
            )
        else:
            self.identity = nn.Sequential()

        width = int(out_channels * (base_width / 64.0))
        self.input = MLP(
            in_channels=in_channels,
            mid_channels=None,
            out_channels=width,
            batch_norm=batch_norm,
            dropout=dropout,
            bias=bias,
            leakyrelu_negative_slope=leakyrelu_negative_slope,
            momentum=momentum,
        )
        self.conv = CDConv(
            r=r,
            l=l,
            kernel_channels=kernel_channels,
            in_channels=width,
            out_channels=width,
        )
        self.output = Linear(
            in_channels=width,
            out_channels=out_channels,
            batch_norm=batch_norm,
            dropout=dropout,
            bias=bias,
            leakyrelu_negative_slope=leakyrelu_negative_slope,
            momentum=momentum,
        )

    def forward(self, x, pos, seq, ori, batch):
        identity = self.identity(x)
        x = self.input(x)
        x = self.conv(x, pos, seq, ori, batch)
        out = self.output(x) + identity
        return out


class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, pos, seq, ori, batch):
        idx = torch.div(seq.squeeze(1), 2, rounding_mode="floor")
        idx = torch.cat([idx, idx[-1].view((1,))])

        idx = (idx[0:-1] != idx[1:]).to(torch.float32)
        idx = torch.cumsum(idx, dim=0) - idx
        idx = idx.to(torch.int64)
        x = scatter_max(src=x, index=idx, dim=0)[0]
        pos = scatter_mean(src=pos, index=idx, dim=0)
        seq = scatter_max(
            src=torch.div(seq, 2, rounding_mode="floor"), index=idx, dim=0
        )[0]
        ori = scatter_mean(src=ori, index=idx, dim=0)
        ori = torch.nn.functional.normalize(ori, 2, -1)
        batch = scatter_max(src=batch, index=idx, dim=0)[0]

        return x, pos, seq, ori, batch


class AvgPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, pos, seq, ori, batch):
        idx = torch.div(seq.squeeze(1), 2, rounding_mode="floor")
        idx = torch.cat([idx, idx[-1].view((1,))])

        idx = (idx[0:-1] != idx[1:]).to(torch.float32)
        idx = torch.cumsum(idx, dim=0) - idx
        idx = idx.to(torch.int64)
        x = scatter_mean(src=x, index=idx, dim=0)
        pos = scatter_mean(src=pos, index=idx, dim=0)
        seq = scatter_max(
            src=torch.div(seq, 2, rounding_mode="floor"), index=idx, dim=0
        )[0]
        ori = scatter_mean(src=ori, index=idx, dim=0)
        ori = torch.nn.functional.normalize(ori, 2, -1)
        batch = scatter_max(src=batch, index=idx, dim=0)[0]

        return x, pos, seq, ori, batch


def kaiming_uniform(tensor, size):
    fan = 1
    for i in range(1, len(size)):
        fan *= size[i]
    gain = math.sqrt(2.0 / (1 + math.sqrt(5) ** 2))
    std = gain / math.sqrt(fan)
    bound = (
        math.sqrt(3.0) * std
    )  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class WeightNet(nn.Module):
    def __init__(self, l: int, kernel_channels: list[int]):
        super(WeightNet, self).__init__()

        self.l = l
        self.kernel_channels = kernel_channels

        self.Ws = nn.ParameterList()
        self.bs = nn.ParameterList()

        for i, channels in enumerate(kernel_channels):
            if i == 0:
                self.Ws.append(
                    torch.nn.Parameter(torch.empty(l, 3 + 3 + 1, channels))
                )
                self.bs.append(torch.nn.Parameter(torch.empty(l, channels)))
            else:
                self.Ws.append(
                    torch.nn.Parameter(
                        torch.empty(l, kernel_channels[i - 1], channels)
                    )
                )
                self.bs.append(torch.nn.Parameter(torch.empty(l, channels)))

        self.relu = nn.LeakyReLU(0.2)

    def reset_parameters(self):
        for i, channels in enumerate(self.kernel_channels):
            if i == 0:
                kaiming_uniform(
                    self.Ws[0].data, size=[self.l, 3 + 3 + 1, channels]
                )
            else:
                kaiming_uniform(
                    self.Ws[i].data,
                    size=[self.l, self.kernel_channels[i - 1], channels],
                )
            self.bs[i].data.fill_(0.0)

    def forward(self, input, idx):
        for i in range(len(self.kernel_channels)):
            W = torch.index_select(self.Ws[i], 0, idx)
            b = torch.index_select(self.bs[i], 0, idx)
            if i == 0:
                weight = self.relu(
                    torch.bmm(input.unsqueeze(1), W).squeeze(1) + b
                )
            else:
                weight = self.relu(
                    torch.bmm(weight.unsqueeze(1), W).squeeze(1) + b
                )

        return weight


class CDConv(MessagePassing):
    def __init__(
        self,
        r: float,
        l: float,
        kernel_channels: list[int],
        in_channels: int,
        out_channels: int,
        add_self_loops: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "sum")
        super().__init__(**kwargs)
        self.r = r
        self.l = l
        self.kernel_channels = kernel_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.WeightNet = WeightNet(l, kernel_channels)
        self.W = torch.nn.Parameter(
            torch.empty(kernel_channels[-1] * in_channels, out_channels)
        )

        self.add_self_loops = add_self_loops

        self.reset_parameters()

    def reset_parameters(self):
        self.WeightNet.reset_parameters()
        kaiming_uniform(
            self.W.data,
            size=[self.kernel_channels * self.in_channels, self.out_channels],
        )

    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        seq: Tensor,
        ori: Tensor,
        batch: Tensor,
    ) -> Tensor:
        row, col = radius(
            pos, pos, self.r, batch, batch, max_num_neighbors=9999
        )
        edge_index = torch.stack([col, row], dim=0)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(
                    edge_index, num_nodes=min(pos.size(0), pos.size(0))
                )

            elif isinstance(edge_index, sparse.SparseTensor):
                edge_index = sparse.set_diag(edge_index)

        out = self.propagate(
            edge_index,
            x=(x, None),
            pos=(pos, pos),
            seq=(seq, seq),
            ori=(ori.reshape((-1, 9)), ori.reshape((-1, 9))),
            size=None,
        )
        out = torch.matmul(out, self.W)

        return out

    def message(
        self,
        x_j: Optional[Tensor],
        pos_i: Tensor,
        pos_j: Tensor,
        seq_i: Tensor,
        seq_j: Tensor,
        ori_i: Tensor,
        ori_j: Tensor,
    ) -> Tensor:
        # orientation
        pos = pos_j - pos_i
        distance = torch.norm(input=pos, p=2, dim=-1, keepdim=True)
        pos /= distance + 1e-9

        pos = torch.matmul(
            ori_i.reshape((-1, 3, 3)), pos.unsqueeze(2)
        ).squeeze(2)
        ori = torch.sum(
            input=ori_i.reshape((-1, 3, 3)) * ori_j.reshape((-1, 3, 3)),
            dim=2,
            keepdim=False,
        )

        #
        normed_distance = distance / self.r

        seq = seq_j - seq_i
        s = self.l // 2
        seq = torch.clamp(input=seq, min=-s, max=s)
        seq_idx = (seq + s).squeeze(1).to(torch.int64)
        normed_length = torch.abs(seq) / s

        # generated kernel weight: PointConv or PSTNet
        delta = torch.cat([pos, ori, distance], dim=1)
        kernel_weight = self.WeightNet(delta, seq_idx)

        # smooth: IEConv II
        smooth = (
            0.5
            - torch.tanh(normed_distance * normed_length * 16.0 - 14.0) * 0.5
        )

        # convolution
        msg = torch.matmul(
            (kernel_weight * smooth).unsqueeze(2), x_j.unsqueeze(1)
        )

        msg = msg.reshape((-1, msg.size(1) * msg.size(2)))

        return msg

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(r={self.r}, "
            f"l={self.l},"
            f"kernel_channels={self.kernel_channels},"
            f"in_channels={self.in_channels},"
            f"out_channels={self.out_channels})"
        )
