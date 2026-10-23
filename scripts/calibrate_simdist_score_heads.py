#!/usr/bin/env python3
"""Fit per-horizon affine reward/value output calibration on training episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from so101.learning.simdist.checkpoint import load_worldmodel
from so101.learning.simdist.dataset import VisualWindowDataset


def fit_affine(x, y):
    # Least-squares y ~= scale*x + bias; inputs are 1D CPU float64 tensors.
    xm, ym = x.mean(), y.mean()
    scale = ((x - xm) * (y - ym)).sum() / (x - xm).square().sum().clamp_min(1e-12)
    return float(scale), float(ym - scale * xm)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--training-data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4)
    a = p.parse_args()
    if a.device == "cpu":
        torch.set_num_threads(1)
    model, saved = load_worldmodel(a.checkpoint, a.device)
    manifest = dict(saved["manifest"])
    if Path(manifest["source"]).resolve() != a.training_data.resolve():
        raise ValueError("training data must match checkpoint source")
    if manifest.get("score_head_calibration"):
        raise ValueError("checkpoint already has score-head calibration")
    with np.load(a.training_data, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    calibration_episodes = list(map(int, manifest.get("calibration_episodes", manifest["train_episodes"])))
    if not calibration_episodes:
        raise ValueError("checkpoint manifest has no calibration episodes")
    reserved_calibration = bool(manifest.get("calibration_episodes"))
    dataset = VisualWindowDataset(data, model.config, calibration_episodes)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False)
    pred = {name: [[] for _ in range(model.config.prediction_length)] for name in ("rewards", "values")}
    target = {name: [[] for _ in range(model.config.prediction_length)] for name in ("rewards", "values")}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = {k: v.to(a.device) for k, v in batch["model_in"].items()}
            y = {k: v.to(a.device) for k, v in batch["labels"].items()}
            output = model(x)
            scaled_y = model.scaler.scale(y)
            for name in ("rewards", "values"):
                valid = y["reward_valid" if name == "rewards" else "value_valid"].bool()
                for h in range(model.config.prediction_length):
                    mask = valid[:, h]
                    if mask.any():
                        pred[name][h].append(output[name][mask, h].double().cpu())
                        target[name][h].append(scaled_y[name][mask, h].double().cpu())
    fitted = {}
    for name in ("rewards", "values"):
        scales, biases, counts = [], [], []
        for h in range(model.config.prediction_length):
            if not pred[name][h]:
                scales.append(1.0); biases.append(0.0); counts.append(0)
                continue
            x, y = torch.cat(pred[name][h]), torch.cat(target[name][h])
            scale, bias = fit_affine(x, y)
            scales.append(scale); biases.append(bias); counts.append(len(x))
        key = "reward" if name == "rewards" else "value"
        getattr(model, key + "_calibration_scale").copy_(torch.tensor(scales, device=a.device))
        getattr(model, key + "_calibration_bias").copy_(torch.tensor(biases, device=a.device))
        fitted[name] = {"scale": scales, "bias": biases, "valid_targets_per_horizon": counts}
    output = dict(saved); output["model"] = model.cpu().state_dict()
    manifest["score_head_calibration"] = {
        "method": "per-horizon affine least-squares in normalized head units on reserved reset-seed calibration windows"
                  if reserved_calibration else
                  "per-horizon affine least-squares in normalized head units on legacy training windows",
        "windows": len(dataset), "heads": fitted, "validation_or_holdout_used_for_fit": False,
        "calibration_episodes": calibration_episodes,
        "optimizer_used_calibration_episodes": not reserved_calibration,
    }
    output["manifest"] = manifest
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(a.out.suffix + ".tmp")
    torch.save(output, tmp); tmp.replace(a.out)
    print(json.dumps(manifest["score_head_calibration"], indent=2))


if __name__ == "__main__":
    main()
