"""Run a `scripts/train_act.py` checkpoint on live observations.

`ACTRunner` owns everything between raw observations and joint targets, so the
simulator evaluation and a later real-robot loop share one code path:

    state (B, 6) sim radians + camera frames (B, H, W, 3) RGB
        -> resize with `to_uint8_rgb` (the teleop recorder's function)
        -> [0, 1] CHW, MEAN_STD-normalize with the run's stats.json
        -> ACTPolicy.select_action
        -> un-normalize with the `action` stats -> (B, 6) sim-radian targets

Frames must be RGB. OpenCV (`so101.real.cameras.Camera.read`) returns BGR;
flip with `frame[..., ::-1]` before passing it in. Real joint readings must
first go through `LeRobotSO101Interface.real_to_sim_obs_processor`, and the
returned targets through `get_raw_actions_from_radians`, since the policy
only ever saw simulator joint values.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor

from so101.learning.act.checkpoint import build_normalizer, load_stats, normalize_batch
from so101.learning.act.config import ACTION, OBS_STATE
from so101.learning.act.data import image_to_tensor, to_uint8_rgb
from so101.learning.act.policy import ACTPolicy, ACTTemporalEnsembler


class ACTRunner:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device = "cuda",
        *,
        n_action_steps: int | None = None,
        temporal_ensemble_coeff: float | None = None,
    ):
        """Load a checkpoint plus its sibling `stats.json` and `train_info.json`.

        `n_action_steps` (actions executed per policy query) and
        `temporal_ensemble_coeff` are inference-only; training bakes
        n_action_steps == chunk_size into the saved config. Temporal
        ensembling queries every step, so it forces n_action_steps=1.
        """
        checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        self.policy = ACTPolicy.from_checkpoint(checkpoint, device=self.device)
        config = self.policy.config
        if temporal_ensemble_coeff is not None:
            config.temporal_ensemble_coeff = temporal_ensemble_coeff
            config.n_action_steps = 1
            self.policy.temporal_ensembler = ACTTemporalEnsembler(temporal_ensemble_coeff, config.chunk_size)
        elif n_action_steps is not None:
            config.n_action_steps = n_action_steps
        config.validate_architecture()

        stats = load_stats(checkpoint.parent / "stats.json")
        info = json.loads((checkpoint.parent / "train_info.json").read_text())
        self.image_keys = config.image_keys
        self.image_sizes = {key: tuple(info["image_shape"][key][:2]) for key in self.image_keys}
        self.norm = build_normalizer(stats, (OBS_STATE, ACTION), self.image_keys, self.device)
        self.reset()

    def reset(self) -> None:
        """Drop queued/ensembled actions. Call at every episode start."""
        self.policy.reset()

    def preprocess(self, state: Tensor, images: dict[str, Tensor]) -> dict[str, Tensor]:
        """Build the normalized policy batch.

        state: (B, state_dim). images: feature key -> (B, H, W, 3) RGB, float
        in [0, 1] or uint8, at any source resolution.
        """
        batch = {OBS_STATE: state.to(self.device, torch.float32)}
        for key in self.image_keys:
            size = self.image_sizes[key]
            batch[key] = torch.stack(
                [image_to_tensor(to_uint8_rgb(frame, size)) for frame in images[key]]
            ).to(self.device)
        return normalize_batch(batch, self.norm)

    @torch.no_grad()
    def act(self, state: Tensor, images: dict[str, Tensor]) -> Tensor:
        """Next (B, action_dim) joint target in simulator radians."""
        action = self.policy.select_action(self.preprocess(state, images))
        mean, std = self.norm[ACTION]
        return action * std + mean
