"""ConditionalUnet1D noise-prediction network.

Line-for-line port of robust-rearrangement `src/models/unet.py` (itself from
real-stanford/diffusion_policy) plus `SinusoidalPosEmb` from
`src/models/pos_embed.py`. No deviations: FiLM (per-channel scale and bias)
from the concatenation [diffusion-step embedding, flattened obs features] in
every residual block; the final Conv1dBlock uses the default n_groups=8 as in
the original.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 3, n_groups: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        # FiLM modulation https://arxiv.org/abs/1709.07871 -- per-channel scale and bias
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_channels * 2), nn.Unflatten(-1, (-1, 1)))
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """x (B, in_channels, T), cond (B, cond_dim) -> (B, out_channels, T)"""
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.out_channels, 1)
        out = embed[:, 0, ...] * out + embed[:, 1, ...]
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        """input_dim: action dim. global_cond_dim: obs_horizon * obs_dim (FiLM input
        besides the step embedding). len(down_dims) = number of UNet levels."""
        super().__init__()
        all_dims = [input_dim, *down_dims]
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        def block(dim_in: int, dim_out: int) -> ConditionalResidualBlock1D:
            return ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups)

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([block(mid_dim, mid_dim), block(mid_dim, mid_dim)])

        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList([block(dim_in, dim_out), block(dim_out, dim_out), Downsample1d(dim_out) if not is_last else nn.Identity()])
            )

        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= len(in_out) - 1
            self.up_modules.append(
                nn.ModuleList([block(dim_out * 2, dim_in), block(dim_in, dim_in), Upsample1d(dim_in) if not is_last else nn.Identity()])
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, sample: Tensor, timestep: Tensor | float | int, global_cond: Tensor | None = None) -> Tensor:
        """sample (B, T, input_dim), timestep (B,) or scalar, global_cond (B, global_cond_dim) -> (B, T, input_dim)"""
        sample = sample.moveaxis(-1, -2)
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        x = self.final_conv(x)
        return x.moveaxis(-1, -2)
