"""Shared on-disk checkpoint format + MEAN_STD normalization helpers.

Checkpoint = `torch.save({"model": <state_dict>, "config": <dataclass __dict__>,
"step": <int>}, path)`, with a sibling `stats.json` in the same directory
holding `{feature_key: {"mean": [...], "std": [...]}}` for normalizing
observations/actions the same way at train and inference time.

This is deliberately algorithm-agnostic (no ACT-specific code): any policy's
`Config`/`nn.Module` pair can adopt this format via a `from_checkpoint`
classmethod (see `so101.learning.act.policy.ACTPolicy.from_checkpoint` for the reference
implementation), so future policies share one on-disk convention instead of
each inventing its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def save_checkpoint(path, model, config: dict, step: int) -> None:
    torch.save({"model": model.state_dict(), "config": config, "step": step}, path)


def load_checkpoint(path, device=None) -> dict:
    """Returns the raw `{"model", "config", "step"}` dict."""
    return torch.load(path, map_location=device, weights_only=False)


def save_stats(stats_path, stats: dict) -> None:
    Path(stats_path).write_text(json.dumps(stats, indent=2))


def load_stats(stats_path) -> dict:
    return json.loads(Path(stats_path).read_text())


def build_normalizer(stats: dict, vector_keys, image_keys, device) -> dict:
    """Pre-shaped (mean, std) tensors per feature for MEAN_STD normalization.
    Vector features (state/action) broadcast as-is; image features are
    per-channel, reshaped to (1,3,1,1) to broadcast over (B,C,H,W)."""
    norm = {}
    for key in vector_keys:
        mean = torch.tensor(stats[key]["mean"], dtype=torch.float32, device=device)
        std = torch.tensor(stats[key]["std"], dtype=torch.float32, device=device)
        norm[key] = (mean, std.clamp_min(1e-6))
    for key in image_keys:
        mean = torch.tensor(stats[key]["mean"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor(stats[key]["std"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        norm[key] = (mean, std.clamp_min(1e-6))
    return norm


def normalize_batch(batch: dict, norm: dict) -> dict:
    """Normalize whichever of `norm`'s keys are present in `batch` -- training
    batches carry state/action/images, inference batches omit `action`."""
    out = dict(batch)
    for key, (mean, std) in norm.items():
        if key in batch:
            out[key] = (batch[key] - mean) / std
    return out
