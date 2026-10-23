#!/usr/bin/env python3
"""Evaluate a cube-pose checkpoint on full, in-region and lost-tail strata.

The table-support region is a diagnostic state partition only. It is neither a
visibility annotation nor a training mask and does not alter source datasets.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so101.learning.dp.config import DPConfig
from so101.learning.dp.policy import DiffusionPolicy
from so101.learning.simdist.cube_pose import CubeSpatialPoseHead
from so101.learning.simdist.cube_pose_spatial import CubePoseNormalization, FrozenDPSpatialEncoder

TABLE_X = (-0.0377, 0.70 - 0.0377)
TABLE_Y = (-0.3967, 1.20 - 0.3967)
TABLE_TOP_Z = 0.0300814467
HELD_HALF_SIZE = 0.0125


def region_prefix(pose: np.ndarray) -> tuple[np.ndarray, int | None]:
    held = pose[:, 0]
    supported = ((held[:, 0] >= TABLE_X[0]) & (held[:, 0] <= TABLE_X[1])
                 & (held[:, 1] >= TABLE_Y[0]) & (held[:, 1] <= TABLE_Y[1])
                 & (held[:, 2] >= TABLE_TOP_Z - HELD_HALF_SIZE))
    lost = np.flatnonzero(~supported)
    prefix_end = int(lost[0]) if len(lost) else len(pose)
    prefix = np.arange(len(pose)) < prefix_end
    return prefix, (int(lost[0]) if len(lost) else None)


def pose_stats(rows: list[np.ndarray]) -> dict:
    values = np.concatenate(rows, axis=0).reshape(-1, 6)
    return {
        "rows": len(values), "mean_m": values.mean(0).tolist(), "std_m": values.std(0).tolist(),
        "p01_p50_p99_m": np.quantile(values, [.01, .5, .99], axis=0).tolist(),
    }


def metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    pred, target = pred.reshape(-1, 2, 3), target.reshape(-1, 2, 3)
    z = np.abs(pred[:, 0, 2] - target[:, 0, 2]) * 1000
    xy = np.abs((pred[:, 0, :2] - pred[:, 1, :2]) - (target[:, 0, :2] - target[:, 1, :2])) * 1000
    return {"samples": len(target), "held_z_abs_mm_p50_p95": np.quantile(z, [.5, .95]).tolist(),
            "relative_xy_abs_mm_p95": np.quantile(xy, .95, axis=0).tolist()}


def load_fit_targets(data_dirs: list[Path], archives: list[Path], fit_seeds: set[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    full, prefix = [], []
    for directory in data_dirs:
        for path in sorted(directory.glob("trajectory_*.pkl")):
            with path.open("rb") as stream:
                ep = pickle.load(stream)
            if int(ep["seed"]) not in fit_seeds:
                continue
            ticks = np.asarray(ep["tick"])
            keep = ticks < int(ep["task_horizon_ticks"])
            pose = np.asarray(ep["cube_positions_robot_relative_m"], dtype=np.float32)[keep]
            mask, _ = region_prefix(pose)
            full.append(pose)
            if mask.any():
                prefix.append(pose[mask])
    for path in archives:
        with np.load(path, allow_pickle=False) as data:
            starts = data["episode_starts"]
            lengths = data["episode_lengths"]
            seeds = data["episode_seeds"]
            positions = data["cube_positions_robot_relative_m"]
            for start, length, seed in zip(starts, lengths, seeds):
                if int(seed) not in fit_seeds:
                    continue
                start, stop = int(start), int(start + length)
                pose = np.asarray(positions[start:stop], dtype=np.float32)
                mask, _ = region_prefix(pose)
                full.append(pose)
                if mask.any():
                    prefix.append(pose[mask])
    if not full or not prefix:
        raise ValueError("fit data must contain both full and in-table-support prefix rows")
    return full, prefix


def evaluate(checkpoint_path: Path, dp_path: Path, archives: list[Path], seeds: set[int], device: torch.device,
             batch_size: int) -> dict:
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved.get("representation") not in ("spatial", "spatial-mean"):
        raise ValueError("regional audit currently requires a spatial-pose checkpoint")
    dp_saved = torch.load(dp_path, map_location="cpu", weights_only=True)
    config_values = dict(dp_saved["config"])
    config_values["pretrained_backbone"] = False
    dp = DiffusionPolicy(DPConfig.from_dict(config_values), dp_saved.get("stats"))
    dp.load_state_dict(dp_saved["model"], strict=True)
    encoder = FrozenDPSpatialEncoder(dp).to(device).eval()
    normalization = CubePoseNormalization(saved["normalization"]).to(device)
    head = CubeSpatialPoseHead(saved["feature_shape"][0], attention=bool(saved["spatial_attention"])).to(device)
    head.load_state_dict(saved["model"], strict=True)
    head.eval()
    predictions, targets, prefix_masks = [], [], []
    episode_metrics = []
    for path in archives:
        with np.load(path, allow_pickle=False) as data:
            starts = data["episode_starts"]
            lengths = data["episode_lengths"]
            episode_seeds = data["episode_seeds"]
            front_all, wrist_all = data["front_images"], data["wrist_images"]
            q_all = data["observations"]
            pose_all = data["cube_positions_robot_relative_m"]
            for index, (start, length, seed) in enumerate(zip(starts, lengths, episode_seeds)):
                if int(seed) not in seeds:
                    continue
                start, length = int(start), int(length)
                stop = start + length
                q = np.asarray(q_all[start:stop], dtype=np.float32)
                target = np.asarray(pose_all[start:stop], dtype=np.float32).reshape(-1, 6)
                prefix, first_lost = region_prefix(target.reshape(-1, 2, 3))
                predicted = []
                with torch.inference_mode():
                    for begin in range(0, length, batch_size):
                        end = min(begin + batch_size, length)
                        cameras = []
                        for array in (front_all, wrist_all):
                            image = torch.as_tensor(array[start + begin:start + end], device=device)
                            cameras.append(image.permute(0, 3, 1, 2).contiguous())
                        features = encoder(*cameras)
                        joints = normalization.normalize_joints(
                            torch.as_tensor(q[begin:end], dtype=torch.float32, device=device)
                        )
                        normalized = head(features[:, 0], features[:, 1], joints)
                        predicted.append(normalization.denormalize_poses(normalized).cpu().numpy())
                predicted = np.concatenate(predicted)
                predictions.append(predicted)
                targets.append(target)
                prefix_masks.append(prefix)
                episode_metrics.append({
                    "seed": int(seed), "episode_index": index, "samples": length,
                    "first_lost_support_tick_index": first_lost,
                    "all": metrics(predicted, target),
                    "pre_lost_prefix": metrics(predicted[prefix], target[prefix]) if prefix.any() else None,
                })
    if not predictions:
        raise ValueError("selection seeds not found in input archives")
    pred = np.concatenate(predictions)
    target = np.concatenate(targets)
    prefix = np.concatenate(prefix_masks)
    return {
        "checkpoint": str(checkpoint_path), "selection_seeds": sorted(seeds),
        "definition": "held cube remains in tabletop XY and above tabletop support; diagnostic state partition, not visibility",
        "all": metrics(pred, target),
        "GT_region_prefix": metrics(pred[prefix], target[prefix]) if prefix.any() else None,
        "GT_lost_tail": metrics(pred[~prefix], target[~prefix]) if (~prefix).any() else None,
        "GT_region_coverage_fraction": float(prefix.mean()),
        "episodes": episode_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dp-checkpoint", type=Path, required=True)
    parser.add_argument("--fit-data-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--data-archive", type=Path, nargs="+", required=True)
    parser.add_argument("--fit-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--selection-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    full_fit, region_fit = load_fit_targets(args.fit_data_dir, args.data_archive, set(args.fit_seeds))
    report = {
        "diagnostic_only": True,
        "fit_pose_distribution": {
            "all_fit_rows": pose_stats(full_fit),
            "pre_lost_table_support_prefix": pose_stats(region_fit),
            "note": "prefix partition is defined by simulator GT crossing table XY/support, not image visibility or a training filter",
        },
        "selection_predictions": evaluate(args.checkpoint, args.dp_checkpoint, args.data_archive,
                                           set(args.selection_seeds), torch.device(args.device), args.batch_size),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "all_fit_rows": report["fit_pose_distribution"]["all_fit_rows"]["rows"],
        "pre_lost_prefix_rows": report["fit_pose_distribution"]["pre_lost_table_support_prefix"]["rows"],
        "selection": {k: v for k, v in report["selection_predictions"].items() if k != "episodes"},
    }, indent=2))


if __name__ == "__main__":
    main()
