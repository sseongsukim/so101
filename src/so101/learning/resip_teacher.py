"""PyTorch inference port of a trained ResiP policy (agents/resip.py), used as
the state-based teacher for sim-to-real distillation.

Load an export made by ``scripts/export_resip_teacher.py``. The port covers
exactly what ``online.py``'s evaluation rollout does:

    every `inference_steps` env steps:
        base chunk = DDPM(20 steps) sample of DiffusionMLP(obs_norm)   (horizon, 6)
    every env step k of the chunk:
        residual = ResidualActor([obs_norm_now, base[k]]).mean          deterministic
        action   = unnormalize(base[k] + action_scale * residual)       sim radians

``tests/test_resip_teacher.py`` checks the networks and the full sampling
recursion against Flax outputs saved with the export.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def mish(x: Tensor) -> Tensor:
    return x * torch.tanh(F.softplus(x))


def _linear(params: dict[str, np.ndarray], prefix: str, bias: bool = True) -> nn.Linear:
    """Flax Dense (kernel is (in, out)) -> torch Linear."""
    kernel = params[f"{prefix}/kernel"]
    layer = nn.Linear(kernel.shape[0], kernel.shape[1], bias=bias)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(kernel.T.copy()))
        if bias:
            layer.bias.copy_(torch.from_numpy(params[f"{prefix}/bias"]))
    return layer


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        scale = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -scale)
        emb = t[:, None] * freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class DiffusionMLP(nn.Module):
    """utils/networks.DiffusionMLP: input [actions, time, obs] -> eps."""

    def __init__(self, params: dict[str, np.ndarray], time_dim: int, horizon: int, action_dim: int):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        self.time_emb = SinusoidalPosEmb(time_dim)
        self.time_1 = _linear(params, "base/Linear_0/Dense_0")
        self.time_2 = _linear(params, "base/Linear_1/Dense_0")
        names = sorted(
            {k.split("/")[2] for k in params if k.startswith("base/ResidualMLP_0/")},
            key=lambda n: int(n.split("_")[1]),
        )
        layers = [_linear(params, f"base/ResidualMLP_0/{n}/Dense_0") for n in names]
        self.input, self.blocks, self.output = layers[0], nn.ModuleList(layers[1:-1]), layers[-1]
        if len(self.blocks) % 2:
            raise ValueError("ResidualMLP expects pairs of hidden layers")

    def forward(self, obs: Tensor, actions: Tensor, t: Tensor) -> Tensor:
        time = self.time_2(mish(self.time_1(self.time_emb(t))))
        x = torch.cat([actions.reshape(len(actions), -1), time, obs.reshape(len(actions), -1)], dim=-1)
        x = self.input(x)
        for first, second in zip(self.blocks[::2], self.blocks[1::2]):
            x = x + second(mish(first(mish(x))))
        return self.output(x).reshape(actions.shape)


class ResidualActor(nn.Module):
    """utils/networks.ResidualActor mean head (ReLU MLP, bias-free output)."""

    def __init__(self, params: dict[str, np.ndarray]):
        super().__init__()
        names = sorted({k.split("/")[1] for k in params if k.startswith("actor/Dense_")},
                       key=lambda n: int(n.split("_")[1]))
        self.hidden = nn.ModuleList(_linear(params, f"actor/{n}") for n in names[:-1])
        self.mean = _linear(params, f"actor/{names[-1]}", bias=False)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.mean(x)


class ResiPTeacher(nn.Module):
    def __init__(self, export_dir: str | Path, device: str | torch.device = "cuda"):
        super().__init__()
        export_dir = Path(export_dir)
        self.config = json.loads((export_dir / "agent_config.json").read_text())
        self.norm = json.loads((export_dir / "normalization.json").read_text())
        params = dict(np.load(export_dir / "params.npz"))
        schedule = dict(np.load(export_dir / "schedule.npz"))

        self.ob_dim = len(self.norm["observation_mean"])
        self.action_dim = len(self.norm["action_min"])
        self.horizon = self.config["horizon_steps"]
        self.act_steps = self.config["inference_steps"]
        self.steps = self.config["denoising_steps"]
        self.base = DiffusionMLP(params, self.config["time_step_embed_dim"], self.horizon, self.action_dim)
        self.actor = ResidualActor(params)
        for name in ("sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod", "ddpm_mu_coef1",
                     "ddpm_mu_coef2", "ddpm_logvar_clipped"):
            self.register_buffer(name, torch.from_numpy(schedule[name].astype(np.float32)))
        f32 = lambda key: torch.tensor(self.norm[key], dtype=torch.float32)  # noqa: E731
        self.register_buffer("obs_mean", f32("observation_mean"))
        self.register_buffer("obs_std", f32("observation_std"))
        self.register_buffer("action_low", f32("action_min"))
        self.register_buffer("action_span", f32("action_max") - f32("action_min"))
        if self.norm.get("clip_actions", True):
            raise ValueError("ResiP rollouts decode actions without clipping (clip_actions=False)")
        self.to(device).eval()
        self._chunk: Tensor | None = None
        self._k = 0

    @property
    def device(self) -> torch.device:
        return self.obs_mean.device

    # -- building blocks ------------------------------------------------------

    def normalize_obs(self, obs: Tensor) -> Tensor:
        obs = (obs - self.obs_mean) / self.obs_std
        clip = self.norm.get("observation_clip")
        return obs.clamp(-clip, clip) if clip is not None else obs

    def unnormalize_action(self, action: Tensor) -> Tensor:
        # utils/datasets.Normalizer.unnormalize_actions with clip_actions=False
        span = torch.where(self.action_span.abs() < 1e-8, torch.ones_like(self.action_span), self.action_span)
        return self.action_low + (action + 1) * span / 2

    @torch.no_grad()
    def sample_base(self, obs_norm: Tensor, x_init: Tensor | None = None, noises: Tensor | None = None,
                    generator: torch.Generator | None = None) -> Tensor:
        """DDPM sampling of a normalized (B, horizon, action_dim) chunk. `x_init`
        and `noises` (steps, B, horizon, action_dim) replay given noise."""
        batch = len(obs_norm)
        shape = (batch, self.horizon, self.action_dim)
        x = x_init if x_init is not None else torch.randn(shape, device=self.device, generator=generator)
        clip_noise = self.config["randn_clip_value"]
        clip_x0 = self.config["denoised_clip_value"]
        for i in range(self.steps):
            t_index = self.steps - 1 - i
            t = torch.full((batch,), float(t_index), device=self.device)
            eps = self.base(obs_norm, x, t)
            x_recon = self.sqrt_recip_alphas_cumprod[t_index] * x - self.sqrt_recipm1_alphas_cumprod[t_index] * eps
            if clip_x0 is not None:
                x_recon = x_recon.clamp(-clip_x0, clip_x0)
            mu = self.ddpm_mu_coef1[t_index] * x_recon + self.ddpm_mu_coef2[t_index] * x
            if t_index == 0:
                x = mu
                continue
            std = torch.exp(0.5 * self.ddpm_logvar_clipped[t_index]).clamp_min(1e-3)
            noise = noises[i] if noises is not None else torch.randn(shape, device=self.device, generator=generator)
            x = mu + std * noise.clamp(-clip_noise, clip_noise)
        return x

    @torch.no_grad()
    def residual(self, obs_norm: Tensor, base_action: Tensor) -> Tensor:
        return self.actor(torch.cat([obs_norm, base_action], dim=-1))

    # -- rollout interface -----------------------------------------------------

    def reset(self) -> None:
        self._chunk, self._k = None, 0

    @torch.no_grad()
    def act(self, obs: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """(B, 36) raw teacher observation -> (B, 6) joint targets in sim radians.
        Re-plans the base chunk every `inference_steps` calls, as online.py does."""
        obs_norm = self.normalize_obs(obs.to(self.device, torch.float32))
        if self._chunk is None or self._k == self.act_steps:
            self._chunk, self._k = self.sample_base(obs_norm, generator=generator), 0
        base = self._chunk[:, self._k]
        self._k += 1
        return self.unnormalize_action(base + self.config["action_scale"] * self.residual(obs_norm, base))
