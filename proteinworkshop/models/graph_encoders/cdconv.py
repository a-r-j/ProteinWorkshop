"""
CDConv implementation adapted from the MIT-licensed source:
https://github.com/hehefan/Continuous-Discrete-Convolution/tree/main
"""

from typing import List, Set, Union

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch

from proteinworkshop.models.graph_encoders.layers.cdconv import (
    AvgPooling,
    BasicBlock,
)
from proteinworkshop.models.utils import get_aggregation
from proteinworkshop.types import EncoderOutput


class CDConvModel(nn.Module):
    def __init__(
        self,
        geometric_radius: float,
        sequential_kernel_size: float,
        kernel_channels: List[int],
        channels: List[int],
        base_width: float = 16.0,
        embedding_dim: int = 16,
        batch_norm: bool = True,
        dropout: float = 0.2,
        bias: bool = False,
        readout: str = "sum",
    ) -> nn.Module:
        super().__init__()

        geometric_radii = [
            geometric_radius * i for i in range(1, len(channels) + 1)
        ]
        assert len(geometric_radii) == len(
            channels
        ), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"

        self.embedding = torch.nn.LazyLinear(embedding_dim)

        self.local_mean_pool = AvgPooling()

        if isinstance(kernel_channels, omegaconf.ListConfig):
            kernel_channels = list(kernel_channels)

        layers = []
        in_channels = embedding_dim
        for i, radius in enumerate(geometric_radii):
            layers.append(
                BasicBlock(
                    r=radius,
                    l=sequential_kernel_size,
                    kernel_channels=kernel_channels,
                    in_channels=in_channels,
                    out_channels=channels[i],
                    base_width=base_width,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    bias=bias,
                )
            )
            layers.append(
                BasicBlock(
                    r=radius,
                    l=sequential_kernel_size,
                    kernel_channels=kernel_channels,
                    in_channels=channels[i],
                    out_channels=channels[i],
                    base_width=base_width,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    bias=bias,
                )
            )
            in_channels = channels[i]

        self.layers = nn.Sequential(*layers)
        self.readout = get_aggregation(readout)

    @property
    def required_batch_attributes(self) -> Set[str]:
        return {"x", "pos", "seq_pos", "batch"}

    @staticmethod
    def orientation(pos):
        u = F.normalize(pos[1:, :] - pos[:-1, :], p=2, dim=1)
        u1 = u[1:, :]
        u2 = u[:-1, :]
        b = F.normalize(u2 - u1, p=2, dim=1)
        n = F.normalize(torch.cross(u2, u1), p=2, dim=1)
        o = F.normalize(torch.cross(b, n), p=2, dim=1)
        ori = torch.stack([b, n, o], dim=1)
        return torch.cat(
            [torch.unsqueeze(ori[0], 0), ori, torch.unsqueeze(ori[-1], 0)],
            axis=0,
        )

    def forward(self, data: Union[ProteinBatch, Batch]) -> EncoderOutput:
        x, pos, seq, batch = (
            self.embedding(data.x),
            data.pos,
            data.seq_pos,
            data.batch,
        )

        ori = self.orientation(pos)

        for i, layer in enumerate(self.layers):
            x = layer(x, pos, seq, ori, batch)
            if i == len(self.layers) - 1:
                x = self.readout(x, batch)
            elif i % 2 == 1:
                x, pos, seq, ori, batch = self.local_mean_pool(
                    x, pos, seq, ori, batch
                )

        return EncoderOutput({"graph_embedding": x, "node_embedding": None})


if __name__ == "__main__":
    import hydra

    from proteinworkshop import constants

    cfg = omegaconf.OmegaConf.load(
        constants.SRC_PATH / "config" / "encoder" / "cdconv.yaml"
    )
    enc = hydra.utils.instantiate(cfg)
    print(enc)

    from graphein.protein.tensor.data import get_random_protein

    protein = get_random_protein()
    protein.seq_pos = torch.arange(protein.coords.shape[0])

    batch = ProteinBatch().from_data_list([protein] * 4)
    batch.x = torch.randn((batch.coords.shape[0], 16))
    batch.pos = batch.coords[:, 1, :]
    print(batch)
    print(enc.orientation(batch.pos).shape)

    # enc(batch)
