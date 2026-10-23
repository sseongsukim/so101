"""Episode-disjoint visual windows with actual next observations and no padding."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class VisualWindowDataset(Dataset):
    def __init__(self, data, config, episodes):
        self.data, self.config, self.episodes = data, config, list(map(int, episodes))
        required = ("observations", "next_observations", "actions", "front_images", "wrist_images",
                    "next_front_images", "next_wrist_images", "rewards", "values", "value_valid",
                    "expert_policy_flag", "reward_valid", "episode_starts", "episode_lengths", "terminated", "truncated",
                    "remaining_ticks", "next_remaining_ticks", "task_horizon_ticks", "objective_terminal")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"missing SimDist contract fields {missing}; regenerate/convert aligned trajectories")
        if str(data.get("value_objective", "")) != config.value_objective:
            raise ValueError("finite-horizon value objective mismatch or missing v2 declaration")
        if not np.isclose(float(data["gamma"]), config.gamma):
            raise ValueError("dataset gamma and planner/model gamma differ")
        n = len(data["actions"])
        for k in required:
            if k in ("episode_starts", "episode_lengths"):
                continue
            if len(data[k]) != n:
                raise ValueError(f"{k} length differs from actions")
        for k in ("observations", "next_observations", "actions"):
            if data[k].shape != (n, 6) or not np.isfinite(data[k]).all():
                raise ValueError(f"{k} must be finite (N,6) radians")
        for k in ("front_images", "wrist_images", "next_front_images", "next_wrist_images"):
            if data[k].dtype != np.uint8 or data[k].ndim != 4 or data[k].shape[-1] != 3:
                raise ValueError(f"{k} must be uint8 NHWC RGB")
            if tuple(data[k].shape[1:3]) != config.stored_image_shape:
                raise ValueError(f"{k} input resolution differs from collector/checkpoint preprocessing")
        if "reward_semantics" not in data or not np.any(data["reward_valid"]):
            raise ValueError("WM requires declared dense environment rewards with reward_valid mask")
        if not np.array_equal(data["reward_valid"], data["reward_semantics"] == "dense_environment_reward"):
            raise ValueError("reward mask differs from declared dense/sparse semantics")
        if "tick_seconds" in data and not np.allclose(data["tick_seconds"], config.tick_seconds, rtol=0.01):
            raise ValueError("dataset tick duration differs from model")
        if not np.all(data["task_horizon_ticks"] == config.task_horizon_ticks):
            raise ValueError("dataset/model task horizons differ")
        if not np.array_equal(data["remaining_ticks"], config.task_horizon_ticks - data["ticks"]):
            raise ValueError("remaining ticks differ from the episode task clock")
        if not np.array_equal(data["next_remaining_ticks"], data["remaining_ticks"] - 1):
            raise ValueError("successor remaining ticks must decrement by exactly one applied tick")
        if (data["remaining_ticks"] < 1).any() or (data["next_remaining_ticks"] < 0).any():
            raise ValueError("observations outside finite task horizon")
        if not np.array_equal(data["objective_terminal"], data["terminated"] | (data["next_remaining_ticks"] == 0)):
            raise ValueError("objective boundary differs from success terminal or finite-horizon boundary")
        self.windows = []
        H, T = config.history_length, config.prediction_length
        for ep in self.episodes:
            start, length = int(data["episode_starts"][ep]), int(data["episode_lengths"][ep])
            if start < 0 or start + length > n:
                raise ValueError("episode extent outside transition arrays")
            if (data["terminated"][start:start+length-1] | data["truncated"][start:start+length-1]).any():
                raise ValueError("episode boundary flag inside a recorded episode")
            completed = bool(data["objective_terminal"][start+length-1])
            expected_valid = data["reward_valid"][start:start+length] & completed
            if not np.array_equal(data["value_valid"][start:start+length], expected_valid):
                raise ValueError("value validity differs from recorded finite objective completion")
            carry = 0.0
            for i in range(start+length-1, start-1, -1):
                expected_value = 0.0 if data["objective_terminal"][i] else carry
                if data["value_valid"][i] and not np.isclose(data["values"][i], expected_value, rtol=1e-5, atol=1e-5):
                    raise ValueError("value label does not equal the finite-horizon successor MC return")
                carry = float(data["rewards"][i]) + config.gamma * carry
            if "timestamps_seconds" in data and length > 1:
                dt = np.diff(data["timestamps_seconds"][start:start+length])
                if not np.allclose(dt, config.tick_seconds, rtol=0.01, atol=1e-5):
                    raise ValueError("timestamp gap or tick duration differs from model contract")
            for current, successor in (("observations", "next_observations"),
                                       ("front_images", "next_front_images"),
                                       ("wrist_images", "next_wrist_images")):
                if length > 1 and not np.array_equal(data[successor][start:start+length-1],
                                                      data[current][start+1:start+length]):
                    raise ValueError(f"{successor} does not match the next recorded {current}")
            # No duplicated start or terminal frames; no labels outside the recording.
            self.windows.extend(range(start + H, start + length - T + 1))
        if not self.windows:
            raise ValueError(f"no complete H={H}, T={T} windows in selected episodes")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        t = self.windows[idx]
        H, T, d = self.config.history_length, self.config.prediction_length, self.data
        h, past, future = slice(t-H, t+1), slice(t-H, t), slice(t, t+T)
        x = {"proprio_obs_hist": d["observations"][h], "acts_hist": d["actions"][past],
             "front_hist": d["front_images"][h], "wrist_hist": d["wrist_images"][h],
             "fut_acts": d["actions"][future], "remaining_ticks_hist": d["remaining_ticks"][h,None]}
        y = {"proprio_obs": d["next_observations"][future], "front": d["next_front_images"][future],
             "wrist": d["next_wrist_images"][future], "actions": d["actions"][future],
             "rewards": d["rewards"][future], "values": d["values"][future],
             "next_remaining_ticks": d["next_remaining_ticks"][future,None],
             "value_valid": d["value_valid"][future], "reward_valid": d["reward_valid"][future], "exp_pol_flags": d["expert_policy_flag"][future]}
        return {"model_in": {k: torch.as_tensor(v.copy()) for k, v in x.items()},
                "labels": {k: torch.as_tensor(v.copy()) for k, v in y.items()}}

    def stats(self):
        # Fit only the selected training episodes, never overlapping held-out windows.
        indices = np.concatenate([np.arange(int(self.data["episode_starts"][ep]),
                                             int(self.data["episode_starts"][ep]) + int(self.data["episode_lengths"][ep]))
                                  for ep in self.episodes])
        result = {}
        for group, field in (("proprio_obs", "observations"), ("actions", "actions"),
                             ("rewards", "rewards"), ("values", "values")):
            rows = indices
            if group in ("values", "rewards"):
                valid_key = "value_valid" if group == "values" else "reward_valid"
                rows = rows[self.data[valid_key][rows].astype(bool)]
            if len(rows) == 0:
                raise ValueError("no valid MC value labels; collect true-terminal episodes or train reward-only separately")
            values = self.data[field][rows].astype(np.float64)
            result[group] = {"mean": np.atleast_1d(values.mean(0)).tolist(),
                             "std": np.atleast_1d(np.maximum(values.std(0), 1e-3)).tolist()}
        return result


def split_episodes(data, seed=42, validation_fraction=0.2):
    train, validation, _ = split_episode_groups(
        data, seed=seed, validation_fraction=validation_fraction, calibration_fraction=0.0)
    return train, validation


def split_episode_groups(data, seed=42, validation_fraction=0.15, calibration_fraction=0.15):
    n = len(data["episode_starts"])
    if n < 2:
        raise ValueError("need at least two episodes for disjoint splits")
    if validation_fraction < 0 or calibration_fraction < 0 or validation_fraction + calibration_fraction >= 1:
        raise ValueError("validation and calibration fractions must be nonnegative and sum to less than one")
    # Parallel episodes from one reset round share task initialization and
    # rendering parameters. Keep their seed cluster wholly on one side.
    episode_seeds = data.get("episode_seeds")
    if episode_seeds is None:
        groups = [[i] for i in range(n)]
    else:
        if len(episode_seeds) != n:
            raise ValueError("episode_seeds must contain one seed per episode")
        by_seed = {}
        for i, episode_seed in enumerate(episode_seeds):
            by_seed.setdefault(int(episode_seed), []).append(i)
        groups = list(by_seed.values())
    required_groups = 3 if calibration_fraction else 2
    if len(groups) < required_groups:
        raise ValueError(f"need at least {required_groups} independent reset seeds for disjoint splits")
    order = np.random.default_rng(seed).permutation(len(groups))
    def take_groups(start, target, reserve):
        chosen, count = [], 0
        while start < len(order) - reserve and count < target:
            group = groups[int(order[start])]
            chosen.append(group); count += len(group); start += 1
        return chosen, start
    cursor = 0
    val_groups, cursor = take_groups(cursor, int(round(n * validation_fraction)),
                                     reserve=2 if calibration_fraction else 1)
    calibration_groups = []
    if calibration_fraction:
        calibration_groups, cursor = take_groups(cursor, int(round(n * calibration_fraction)), reserve=1)
    train_groups = [groups[int(i)] for i in order[cursor:]]
    flatten = lambda selected: sorted(i for group in selected for i in group)
    return flatten(train_groups), flatten(val_groups), flatten(calibration_groups)
