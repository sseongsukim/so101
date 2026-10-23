"""Mean/std scaler, ported from simdist's `simdist/modeling/scaler.py`.

Two differences from the original, both deliberate:

* The stats live as **buffers on the module**, so they land in `state_dict()` and
  travel inside the checkpoint. simdist kept them outside the orbax checkpoint
  and had to rebuild the model from a `scaler_params_struct` plus a dataset-side
  `scaler_params.json`; a checkpoint alone was not enough to run inference.
* `extero_obs` and `commands` groups are gone (see `config.py`).

`scale`/`unscale` map *data keys* (the schema field names) to *stat groups* via
`KEY_TO_GROUP`; unknown keys pass through untouched, which is how `latents` and
`exp_pol_flags` survive a `scale()` call.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

#: Statistic groups, i.e. what the dataset computes mean/std over.
GROUPS = ("proprio_obs", "actions", "rewards", "values")

#: Schema field name -> statistic group. Covers model inputs, training labels
#: and model outputs (they share names where they share units).
KEY_TO_GROUP = {
    "proprio_obs_hist": "proprio_obs",
    "acts_hist": "actions",
    "fut_acts": "actions",
    "proprio_obs": "proprio_obs",
    "proprio_predictions": "proprio_obs",
    "rewards": "rewards",
    "values": "values",
    "actions": "actions",
}

Stats = Mapping[str, Mapping[str, list[float]]]


class Scaler(nn.Module):
    def __init__(self, stats: Stats):
        super().__init__()
        missing = [g for g in GROUPS if g not in stats]
        if missing:
            raise KeyError(f"stats is missing group(s) {missing}; expected {list(GROUPS)}.")
        for group in GROUPS:
            mean = torch.as_tensor(stats[group]["mean"], dtype=torch.float32).reshape(-1)
            std = torch.as_tensor(stats[group]["std"], dtype=torch.float32).reshape(-1)
            if mean.shape != std.shape:
                raise ValueError(f"stats['{group}'] mean/std shape mismatch: {mean.shape} vs {std.shape}.")
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
                raise ValueError(f"invalid finite positive statistics for {group}")
            self.register_buffer(f"{group}_mean", mean)
            self.register_buffer(f"{group}_std", std)

    def forward(self, x):  # pragma: no cover - guard, mirrors the original
        raise NotImplementedError("Use Scaler.scale() / Scaler.unscale().")

    def scale(self, x: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # +1e-8 on std matches the original; it makes scale/unscale an
        # approximate rather than exact inverse (relative error ~1e-8 / std).
        return self._transform(x, lambda v, mean, std: (v - mean) / (std + 1e-8))

    def unscale(self, x: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._transform(x, lambda v, mean, std: v * std + mean)

    def stats(self) -> dict[str, dict[str, list[float]]]:
        """Round-trip of the constructor argument, e.g. to write `stats.json`."""
        return {
            group: {
                "mean": getattr(self, f"{group}_mean").tolist(),
                "std": getattr(self, f"{group}_std").tolist(),
            }
            for group in GROUPS
        }

    def _transform(self, x, func):
        out = {}
        for key, value in x.items():
            group = KEY_TO_GROUP.get(key)
            if group is None:
                out[key] = value
                continue
            mean = getattr(self, f"{group}_mean")
            std = getattr(self, f"{group}_std")
            out[key] = func(value, mean, std)
        return out
