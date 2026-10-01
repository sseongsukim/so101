"""Run a `scripts/train_dp.py` checkpoint on live observations.

`DPRunner` has the same API as `so101.learning.act.inference.ACTRunner`:

    state (B, 6) sim radians + camera frames (B, H, W, 3) RGB
        -> resize with `to_uint8_rgb` to the recorded size (train_info.json)
        -> uint8 CHW -> the policy's own eval transforms (front CenterCrop 224,
           wrist Resize 224) -> ImageNet-normalized ResNet features
        -> min_max-normalized state -> warm-started DDIM chunk (pred_horizon)
        -> executes `action_horizon` actions, then re-plans
        -> (B, 6) sim-radian joint targets

Mirrors `Actor.action` / `Actor.reset` (robust-rearrangement src/behavior/base.py).
Frames must be RGB (OpenCV gives BGR: pass `frame[..., ::-1]`). Real joint
readings must first go through `LeRobotSO101Interface.real_to_sim_obs_processor`,
and targets back through `get_raw_actions_from_radians`, since the policy only
ever saw simulator joint values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from so101.learning.act.data import to_uint8_rgb
from so101.learning.dp.policy import DiffusionPolicy


class DPRunner:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device = "cuda",
        *,
        action_horizon: int | None = None,
        inference_steps: int | None = None,
        warmstart_timestep: int | None | str = "checkpoint",
        use_ema: bool = True,
    ):
        """Load a checkpoint and its sibling train_info.json.

        `action_horizon` (actions executed per plan), `inference_steps` (DDIM
        steps; the paper's sim eval used 4, its real deployment 8, the config
        16) and `warmstart_timestep` (None = pure-noise DDIM) are
        inference-only overrides. EMA weights are used when the checkpoint has them.
        """
        checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        self.policy = DiffusionPolicy.from_checkpoint(checkpoint, device=self.device, use_ema=use_ema)
        config = self.policy.config
        if action_horizon is not None:
            config.action_horizon = action_horizon
            self.policy.actions = type(self.policy.actions)(maxlen=action_horizon)
        if inference_steps is not None:
            config.inference_steps = inference_steps
        if warmstart_timestep != "checkpoint":
            config.warmstart_timestep = warmstart_timestep
        config.validate()

        info_path = checkpoint.parent / "train_info.json"
        info = json.loads(info_path.read_text()) if info_path.is_file() else {}
        self.image_keys = config.image_keys
        shapes = info.get("image_shape", {})
        self.image_sizes = {key: tuple(shapes[key][:2]) if key in shapes else tuple(config.image_shape) for key in self.image_keys}
        self.reset()

    @property
    def config(self):
        return self.policy.config

    def reset(self) -> None:
        """Drop queued actions, the observation history and the warm-start plan.
        Call at every episode start."""
        self.policy.reset()

    def preprocess(self, images: dict[str, Tensor]) -> dict[str, Tensor]:
        """images: key -> (B, H, W, 3) RGB float [0, 1] or uint8 -> key -> (B, 3, h, w) uint8."""
        out = {}
        for key in self.image_keys:
            size = self.image_sizes[key]
            frames = np.stack([to_uint8_rgb(frame, size) for frame in images[key]])
            out[key] = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().to(self.device)
        return out

    @torch.no_grad()
    def act(self, state: Tensor, images: dict[str, Tensor]) -> Tensor:
        """Next (B, action_dim) joint target in simulator radians."""
        return self.policy.select_action(state.to(self.device, torch.float32), self.preprocess(images))
