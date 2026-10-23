#!/usr/bin/env python3
"""Read-only decomposition of joint and visual contributions to future latents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from so101.learning.simdist.checkpoint import load_worldmodel
from so101.learning.simdist.dataset import VisualWindowDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--training-data", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--windows-per-episode", type=int, default=2)
    args = p.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(1)
    model, saved = load_worldmodel(args.checkpoint, args.device)
    if Path(saved["manifest"]["source"]).resolve() != args.training_data.resolve():
        raise ValueError("training-data must match checkpoint source")
    if not model.config.joint_conditioned_latents:
        raise ValueError("checkpoint must use joint-conditioned latents")
    with np.load(args.data, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    dataset = VisualWindowDataset(data, model.config, range(len(data["episode_starts"])))
    selected = []
    for ep, (start, length) in enumerate(zip(data["episode_starts"], data["episode_lengths"])):
        ids = [i for i, tick in enumerate(dataset.windows) if start <= tick < start + length]
        if not ids:
            continue
        pos = np.linspace(0, len(ids)-1, min(args.windows_per_episode, len(ids))).astype(int)
        selected.extend(ids[i] for i in pos)
    loader = DataLoader(Subset(dataset, selected), batch_size=args.batch_size, shuffle=False)
    errors = {k: [] for k in ("A_current_q_current_rgb", "B_pred_q_current_rgb",
                              "C_true_q_current_rgb", "D_pred_q_future_rgb", "T_true_q_future_rgb",
                              "learned_joint_conditioned", "front_feature_current", "front_feature_predicted",
                              "wrist_feature_current", "wrist_feature_predicted")}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = {k: v.to(args.device) for k, v in batch["model_in"].items()}
            y = {k: v.to(args.device) for k, v in batch["labels"].items()}
            future_time = y["next_remaining_ticks"]
            q_current = model.scaler.scale({"proprio_obs_hist": x["proprio_obs_hist"]})[
                "proprio_obs_hist"][:, -1]
            q_true = model.scaler.scale({"proprio_obs": y["proprio_obs"]})["proprio_obs"]
            inferred = model.inference(x)
            q_pred_raw = inferred["proprio_predictions"]
            q_pred = model.scaler.scale({"proprio_predictions": q_pred_raw})["proprio_predictions"]
            visual_now = model.current_visual_features(x)
            visual_future = {"front": model.encoder.front(y["front"]),
                             "wrist": model.encoder.wrist(y["wrist"])}
            def encode(q, visual):
                front = visual["front"]
                wrist = visual["wrist"]
                if front.ndim == 2:
                    front = front[:, None].expand(-1, q.shape[1], -1)
                    wrist = wrist[:, None].expand(-1, q.shape[1], -1)
                content = torch.cat((model.encoder.joints(q), front, wrist), -1)
                return model.encoder.latent_from_content(content, future_time)
            target = encode(q_true, visual_future)
            variants = {
                "A_current_q_current_rgb": encode(q_current[:, None].expand_as(q_true), visual_now),
                "B_pred_q_current_rgb": encode(q_pred, visual_now),
                "C_true_q_current_rgb": encode(q_true, visual_now),
                "D_pred_q_future_rgb": encode(q_pred, visual_future),
                "T_true_q_future_rgb": target,
                "learned_joint_conditioned": inferred["latents"],
            }
            for name, prediction in variants.items():
                errors[name].append((prediction - target).square().mean(-1).cpu().numpy())
            if "visual_predictions" in inferred:
                for camera in ("front", "wrist"):
                    target_camera = visual_future[camera]
                    current_camera = visual_now[camera][:, None].expand_as(target_camera)
                    predicted_camera = inferred["visual_predictions"][camera]
                    errors[camera + "_feature_current"].append(
                        (current_camera - target_camera).square().mean(-1).cpu().numpy())
                    errors[camera + "_feature_predicted"].append(
                        (predicted_camera - target_camera).square().mean(-1).cpu().numpy())
    report = {"checkpoint": str(args.checkpoint.resolve()), "dataset": str(args.data.resolve()),
              "windows": len(selected), "per_horizon_mse": {
                  name: np.concatenate(rows).mean(0).tolist() for name, rows in errors.items()},
              "interpretation": "oracle substitutions are diagnostic only; future image/joint inputs are not deployable"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
