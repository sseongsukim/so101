"""Untrimmed transition recorder for operator-controlled future real sessions.

Capture immutable observations before/after each actual sent command. All
timestamps must share a monotonic clock. This class neither connects hardware
nor fabricates the final successor, and exposes no idle-trimming operation.
Real recordings do not inherit simulator rewards/value labels.
"""
from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np


class TransitionRecorder:
    def __init__(self, **metadata):
        horizon = metadata.get("task_horizon_ticks")
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("declare a positive task_horizon_ticks for the recorded control clock")
        self.metadata = metadata
        self.rows = []

    def append(self, observation, actual_sent_action, next_observation, *,
               terminated=False, truncated=False):
        if terminated and truncated:
            raise ValueError("termination and truncation must be distinct")
        current, successor = self._snapshot(observation), self._snapshot(next_observation)
        if successor["timestamp_seconds"] <= current["timestamp_seconds"]:
            raise ValueError("successor must be captured after current observation")
        if not 1 <= current["remaining_ticks"] <= self.metadata["task_horizon_ticks"]:
            raise ValueError("current clock outside declared task horizon")
        if successor["remaining_ticks"] != current["remaining_ticks"] - 1:
            raise ValueError("recorded successor clock must decrement by exactly one control tick")
        action = np.asarray(actual_sent_action, dtype=np.float32)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("actual sent action must be six finite absolute joint radians")
        if self.rows and (self.rows[-1]["terminated"] or self.rows[-1]["truncated"]):
            raise ValueError("start a new recorder after an episode boundary")
        self.rows.append({"current": current, "next": successor, "action": action.copy(),
                          "terminated": bool(terminated), "truncated": bool(truncated)})

    @staticmethod
    def _snapshot(value):
        fields = ("joints", "front", "wrist", "timestamp_seconds", "front_timestamp_seconds", "wrist_timestamp_seconds",
                  "remaining_ticks")
        missing = [k for k in fields if k not in value]
        if missing:
            raise ValueError(f"observation missing measured state/acquisition clocks: {missing}")
        result = {k: np.asarray(value[k]).copy() for k in fields}
        if result["joints"].shape != (6,) or not np.isfinite(result["joints"]).all():
            raise ValueError("measured joints must be finite six radians")
        for camera in ("front", "wrist"):
            if result[camera].dtype != np.uint8 or result[camera].ndim != 3 or result[camera].shape[-1] != 3:
                raise ValueError(f"{camera} must be uint8 HWC RGB")
        for field in fields[3:-1]:
            result[field] = float(result[field])
            if not np.isfinite(result[field]):
                raise ValueError("all capture clocks must be finite")
        remaining = float(result["remaining_ticks"])
        if not np.isfinite(remaining) or remaining < 0 or int(remaining) != remaining:
            raise ValueError("remaining control ticks must be a nonnegative integer")
        result["remaining_ticks"] = int(remaining)
        return result

    def save(self, path):
        if not self.rows:
            raise ValueError("no actual transitions to save")
        stack = lambda where, field: np.stack([r[where][field] for r in self.rows])
        n = len(self.rows)
        data = {**self.metadata,
                "observations": stack("current", "joints"), "next_observations": stack("next", "joints"),
                "front_images": stack("current", "front"), "wrist_images": stack("current", "wrist"),
                "next_front_images": stack("next", "front"), "next_wrist_images": stack("next", "wrist"),
                "actions": np.stack([r["action"] for r in self.rows]),
                "terminated": np.array([r["terminated"] for r in self.rows], bool),
                "truncated": np.array([r["truncated"] for r in self.rows], bool),
                "timestamps_seconds": stack("current", "timestamp_seconds"),
                "next_timestamps_seconds": stack("next", "timestamp_seconds"),
                "front_timestamps_seconds": stack("current", "front_timestamp_seconds"),
                "wrist_timestamps_seconds": stack("current", "wrist_timestamp_seconds"),
                "next_front_timestamps_seconds": stack("next", "front_timestamp_seconds"),
                "next_wrist_timestamps_seconds": stack("next", "wrist_timestamp_seconds"),
                "remaining_ticks": stack("current", "remaining_ticks"),
                "next_remaining_ticks": stack("next", "remaining_ticks"),
                "action_semantics": "applied_absolute_joint_target_radians",
                "reward_semantics": "unlabelled_real_observation", "reward_valid": np.zeros(n,bool),
                "value_valid": np.zeros(n,bool), "expert_policy_flag": np.zeros(n,bool),
                "recording_trimmed": False, "image_format": "uint8_rgb"}
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path
