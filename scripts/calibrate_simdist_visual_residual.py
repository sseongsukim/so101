#!/usr/bin/env python3
"""Fit train-only per-horizon/camera shrinkage for visual feature deltas."""
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
    if Path(saved["manifest"]["source"]).resolve() != a.training_data.resolve():
        raise ValueError("training data must match checkpoint manifest source")
    if not model.config.visual_dynamics:
        raise ValueError("checkpoint must have visual dynamics enabled")
    if saved["manifest"].get("visual_residual_calibration"):
        raise ValueError("checkpoint already has visual residual calibration")
    with np.load(a.training_data, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    dataset = VisualWindowDataset(data, model.config, saved["manifest"]["train_episodes"])
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False)
    cross = torch.zeros(model.config.prediction_length, 2, dtype=torch.float64)
    denominator = torch.zeros_like(cross)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = {k: v.to(a.device) for k, v in batch["model_in"].items()}
            y = {k: v.to(a.device) for k, v in batch["labels"].items()}
            current = model.current_visual_features(x)
            predicted = model.inference(x)["visual_predictions"]
            target = {"front": model.encoder.front(y["front"]),
                      "wrist": model.encoder.wrist(y["wrist"])}
            for camera_idx, camera in enumerate(("front", "wrist")):
                d = predicted[camera] - current[camera][:, None]
                dy = target[camera] - current[camera][:, None]
                cross[:, camera_idx] += (d.double() * dy.double()).sum((0, 2)).cpu()
                denominator[:, camera_idx] += d.double().square().sum((0, 2)).cpu()
    raw = cross / denominator.clamp_min(1e-12)
    scale = raw.clamp(0, 2)
    model.visual_residual_scale.copy_(scale.to(model.visual_residual_scale.device,
                                                  model.visual_residual_scale.dtype))
    output = dict(saved)
    output["model"] = model.cpu().state_dict()
    manifest = dict(saved["manifest"])
    manifest["visual_residual_calibration"] = {
        "method": "least-squares scalar shrinkage per camera and horizon, training episodes only",
        "windows": len(dataset), "raw_scale": raw.tolist(), "applied_scale_0_2": scale.tolist(),
        "validation_or_holdout_used_for_fit": False,
    }
    output["manifest"] = manifest
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(a.out.suffix + ".tmp")
    torch.save(output, tmp); tmp.replace(a.out)
    print(json.dumps(manifest["visual_residual_calibration"], indent=2))


if __name__ == "__main__":
    main()
