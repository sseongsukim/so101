#!/usr/bin/env python3
"""Fit per-horizon joint-residual shrinkage on training episodes only."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(1)
    model, saved = load_worldmodel(args.checkpoint, args.device)
    if Path(saved["manifest"]["source"]).resolve() != args.training_data.resolve():
        raise ValueError("training data must exactly match the checkpoint manifest source")
    if not model.config.joint_dynamics:
        raise ValueError("checkpoint must include the measured-joint dynamics head")
    if saved["manifest"].get("joint_residual_calibration"):
        raise ValueError("checkpoint already has a joint-residual calibration; use the uncalibrated source")
    with np.load(args.training_data, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    episodes = list(map(int, saved["manifest"].get(
        "calibration_episodes", saved["manifest"]["train_episodes"])))
    if not episodes:
        raise ValueError("checkpoint manifest has no calibration episodes")
    reserved_calibration = bool(saved["manifest"].get("calibration_episodes"))
    dataset = VisualWindowDataset(data, model.config, episodes)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    cross = denominator = None
    windows = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = {key: value.to(args.device) for key, value in batch["model_in"].items()}
            y = {key: value.to(args.device) for key, value in batch["labels"].items()}
            prediction = model.inference(x)["proprio_predictions"].double()
            target = y["proprio_obs"].double()
            current = x["proprio_obs_hist"][:, -1].double()[:, None, :]
            delta = prediction - current
            target_delta = target - current
            batch_cross = (delta * target_delta).sum(dim=(0, 2)).cpu().numpy()
            batch_denominator = delta.square().sum(dim=(0, 2)).cpu().numpy()
            cross = batch_cross if cross is None else cross + batch_cross
            denominator = batch_denominator if denominator is None else denominator + batch_denominator
            windows += len(prediction)
    alpha_raw = cross / np.maximum(denominator, 1e-12)
    alpha = np.clip(alpha_raw, 0.0, 2.0)
    model.joint_residual_scale.copy_(torch.as_tensor(alpha, dtype=model.joint_residual_scale.dtype,
                                                      device=model.joint_residual_scale.device))
    output = dict(saved)
    output["model"] = model.cpu().state_dict()
    manifest = dict(saved["manifest"])
    manifest["joint_residual_calibration"] = {
        "method": "least-squares scalar shrinkage per horizon on reserved reset-seed calibration windows"
                  if reserved_calibration else
                  "least-squares scalar shrinkage per horizon on legacy training windows",
        "objective": "mean raw joint squared error in radians^2 across six joints",
        "windows": windows, "raw_alpha": alpha_raw.tolist(), "applied_alpha_clipped_0_2": alpha.tolist(),
        "calibration_episodes": episodes,
        "optimizer_used_calibration_episodes": not reserved_calibration,
        "external_holdout_used_for_fit": False,
    }
    output["manifest"] = manifest
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    torch.save(output, temporary)
    temporary.replace(args.out)
    print(json.dumps(manifest["joint_residual_calibration"], indent=2))


if __name__ == "__main__":
    main()
