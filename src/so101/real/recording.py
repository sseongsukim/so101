"""Write real-robot episodes in the trajectory_*.pkl format teleop_task.py uses.

Real demos (scripts/record_real_demos.py) and deployment rollouts
(scripts/deploy_policy_real.py) are stored exactly like simulated ones --
6 joint positions (Isaac radians), 6 joint targets, 240x320 front/wrist RGB --
so the student trainers co-train on them without any conversion.
"""

from __future__ import annotations

import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch

from so101.learning.act.data import to_uint8_rgb

INDEXED = re.compile(r"^trajectory_(\d+)\.pkl$")


class EpisodeRecorder:
    def __init__(self, directory: Path, image_size: tuple[int, int] = (240, 320), **metadata):
        self.directory = Path(directory)
        self.image_size = image_size
        self.metadata = metadata
        self.clear()

    def clear(self) -> None:
        self.obs, self.act, self.front, self.wrist, self.stamps = [], [], [], [], []
        self.raw_obs, self.raw_act = [], []

    def __len__(self) -> int:
        return len(self.act)

    def append(self, state: torch.Tensor, action: torch.Tensor, front_rgb: np.ndarray, wrist_rgb: np.ndarray,
               raw_state: torch.Tensor | None = None, raw_action: torch.Tensor | None = None) -> None:
        """`raw_*` are the follower's LeRobot values behind `state`/`action`;
        pass both or neither for a whole episode."""
        if raw_state is not None and raw_action is not None:
            self.raw_obs.append(raw_state.detach().cpu().numpy().astype(np.float32).copy())
            self.raw_act.append(raw_action.detach().cpu().numpy().astype(np.float32).copy())
        self.obs.append(state.detach().cpu().numpy().astype(np.float32).copy())
        self.act.append(action.detach().cpu().numpy().astype(np.float32).copy())
        # Full-size frames are kept and resized in save(): the resize stalled
        # the 30 Hz loop for ~100 ms every few ticks under CPU load (torch
        # thread-pool contention), and a control loop must not wait on it.
        self.front.append(np.array(front_rgb, copy=True))
        self.wrist.append(np.array(wrist_rgb, copy=True))
        self.stamps.append(time.perf_counter())

    def trim_idle(self, threshold_deg: float = 1.0, keep_after_motion: int = 15,
                  pause_step_deg: float = 0.3, min_pause_steps: int = 5) -> tuple[int, int, int]:
        """Remove still stretches: before the first motion, pauses in the
        middle, and the tail beyond `keep_after_motion` steps after the last
        motion.

        * start / end: motion = any commanded joint more than `threshold_deg`
          from the episode's first (resp. last) target. The synthetic teacher
          demos start moving on their first step and end 15 steps after
          success; a human's pause before starting would teach the policy to
          wait at the start pose.
        * middle: a pause is at least `min_pause_steps` consecutive ticks whose
          commanded target changes by less than `pause_step_deg`; each pause
          keeps its first frame. Shorter stops and slow motion are untouched,
          so leader jitter does not thin out ordinary movement.
        Returns (steps removed at the start, in pauses, at the end).
        """
        if len(self.act) < 2:
            return 0, 0, 0
        act = np.rad2deg(np.stack(self.act))
        moved = np.abs(act - act[0]).max(axis=1) > threshold_deg
        if not moved.any():
            return 0, 0, 0
        start = max(int(np.argmax(moved)) - 1, 0)
        moved_end = np.abs(act - act[-1]).max(axis=1) > threshold_deg
        last = len(act) - 1 - int(np.argmax(moved_end[::-1]))
        end = min(len(act), last + 1 + keep_after_motion)

        still = np.zeros(len(act), dtype=bool)
        still[1:] = np.abs(np.diff(act, axis=0)).max(axis=1) < pause_step_deg
        keep = np.zeros(len(act), dtype=bool)
        keep[start:end] = True
        t = start
        while t <= last:
            if still[t]:
                run_end = t
                while run_end + 1 <= last and still[run_end + 1]:
                    run_end += 1
                if run_end - t + 1 >= min_pause_steps:
                    keep[t + 1 : run_end + 1] = False   # keep the pause's first frame
                t = run_end + 1
            else:
                t += 1
        index = np.flatnonzero(keep)
        for name in ("obs", "act", "front", "wrist", "stamps", "raw_obs", "raw_act"):
            items = getattr(self, name)
            if items:
                setattr(self, name, [items[i] for i in index])
        removed_mid = int((~keep[start:end]).sum())
        return start, removed_mid, len(act) - end

    def next_path(self) -> Path:
        indices = [int(m.group(1)) for p in self.directory.glob("trajectory_*.pkl") if (m := INDEXED.match(p.name))]
        return self.directory / f"trajectory_{max(indices, default=-1) + 1:06d}.pkl"

    def save(self, success: bool = True, **extra) -> Path | None:
        if not self.act:
            return None
        n = len(self.act)
        obs = np.stack(self.obs)
        flags = np.zeros(n, dtype=np.bool_)
        flags[-1] = True
        data = {
            "observations": obs,
            "actions": np.stack(self.act),
            "rewards": np.zeros(n, dtype=np.float32),
            "terminals": flags.copy(),
            "successes": flags.copy() if success else np.zeros(n, dtype=np.bool_),
            "next_observations": np.concatenate([obs[1:], obs[-1:]]),
            "front_images": np.stack([to_uint8_rgb(torch.from_numpy(f), self.image_size) for f in self.front]),
            "wrist_images": np.stack([to_uint8_rgb(torch.from_numpy(f), self.image_size) for f in self.wrist]),
            "image_format": "uint8_rgb",
            "image_shape": [*self.image_size, 3],
            "image_size_source": [640, 480],
            "wall_times": np.asarray(self.stamps, dtype=np.float64),
            **({"raw_observations": np.stack(self.raw_obs), "raw_actions": np.stack(self.raw_act)}
               if len(self.raw_obs) == n else {}),
            **self.metadata,
            **extra,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.next_path()
        with path.open("xb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.clear()
        return path
