#!/usr/bin/env python3
"""Train/evaluate the bounded simulator RGB-to-cube-pose baseline.

Uses frozen image-DP visual features and current measured q. The optional
spatial/spatial-mean pair bypasses global pooling/projection/LayerNorm with a
matched trainable pose head; spatial uses learned object attention and the
mean control fixes it uniform. Camera trunks and preprocessing stay frozen.
Spatial maps are streamed in chunks through a bounded CPU cache, never packed
as a complete dataset or written to disk.
Privileged cube labels are targets only. Reset-round seeds split fit,
calibration/early-stop and development selection; H2 remains reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so101.learning.dp.config import DPConfig
from so101.learning.dp.policy import DiffusionPolicy
from so101.learning.simdist.cube_pose import CubePoseHead, CubeSpatialPoseHead
from so101.learning.simdist.cube_pose_features import SpatialFeatureStream
from so101.learning.simdist.cube_pose_spatial import CubePoseNormalization, FrozenDPSpatialEncoder
from so101.learning.simdist.cube_pose_region import (
    CubeTaskRegion, REGION_LOSS_WEIGHT, conditional_pose_loss, current_successor_regions,
    fit_region_class_weights, fit_region_normalization, masked_xyz_terms,
    neutral_pose_targets, region_pose_metrics,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_features(dp, episode: dict, device: torch.device, batch_size: int) -> np.ndarray:
    """The legacy small pooled cache; spatial modes use SpatialFeatureStream."""
    n = len(episode["actions"])
    output = np.empty((n, 2, 128), dtype=np.float32)
    # DP image order is wrist, front; output order is explicitly front, wrist.
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        for out_index, (key, dp_index) in enumerate((("front_images", 1), ("wrist_images", 0))):
            image = torch.as_tensor(episode[key][start:stop], device=device).permute(0, 3, 1, 2).contiguous()
            with torch.inference_mode():
                transformed = dp.transforms[dp_index](image)
                feature = dp.layernorms[dp_index](
                    dp.projections[dp_index](dp.encoders[dp_index](transformed))
                )
            output[start:stop, out_index] = feature.float().cpu().numpy()
    if not n:
        raise ValueError("cannot extract features from an empty episode")
    if not np.isfinite(output).all():
        raise ValueError("visual feature cache contains non-finite values")
    return output


def _metrics(pred: np.ndarray, target: np.ndarray, moved: np.ndarray, phases: np.ndarray) -> dict:
    pred = pred.reshape(-1, 2, 3)
    target = target.reshape(-1, 2, 3)
    held_z_abs = np.abs(pred[:, 0, 2] - target[:, 0, 2])
    rel_xy_abs = np.abs((pred[:, 0, :2] - pred[:, 1, :2]) -
                        (target[:, 0, :2] - target[:, 1, :2]))
    report = {
        "samples": len(target),
        "held_height_abs_mm_p50_p95": (np.quantile(held_z_abs, [.5, .95]) * 1000).tolist(),
        "relative_xy_abs_mm_p95_per_axis": (np.quantile(rel_xy_abs, .95, axis=0) * 1000).tolist(),
        "held_xyz_rmse_mm": (np.sqrt(np.mean((pred[:, 0] - target[:, 0]) ** 2, axis=0)) * 1000).tolist(),
    }
    strata = {"moving_held_cube": moved, "static_held_cube": ~moved}
    for phase in sorted(set(phases.tolist())):
        strata[f"phase:{phase}"] = phases == phase
    report["strata"] = {}
    for name, mask in strata.items():
        count = int(mask.sum())
        if count:
            report["strata"][name] = {
                "samples": count,
                "held_height_abs_mm_p95": float(np.quantile(held_z_abs[mask], .95) * 1000),
                "relative_xy_abs_mm_p95_per_axis": (np.quantile(rel_xy_abs[mask], .95, axis=0) * 1000).tolist(),
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, nargs="+",
                        help="directories with collector trajectory_*.pkl episodes")
    parser.add_argument("--data-archive", type=Path, nargs="+",
                        help="aligned NPZ shards; current RGB arrays are decoded into host RAM")
    parser.add_argument("--dp-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--fit-seeds", type=int, nargs="+")
    parser.add_argument("--calibration-seeds", type=int, nargs="+")
    parser.add_argument("--selection-seeds", type=int, nargs="+")
    parser.add_argument("--vision-mode", choices=("both", "front", "wrist", "none"), default="both")
    parser.add_argument("--representation", choices=("pooled", "spatial", "spatial-mean", "pixel"), default="pooled",
                        help="frozen post-LN features or frozen pre-pool spatial maps with learned/uniform attention")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--pixel-centre-weight", type=float, default=.1,
                        help="front projected-centre auxiliary weight; pixel representation only")
    parser.add_argument("--pixel-mask-weight", type=float, default=.1,
                        help="both-camera mask auxiliary weight; pixel representation only")
    parser.add_argument("--spatial-cache-mib", type=float, default=2048,
                        help="bounded CPU map cache for spatial modes; 0 streams without caching")
    parser.add_argument("--region-conditioned", action="store_true",
                        help="GT task-domain conditional XYZ plus per-object inside logits; preserves all outcomes")
    parser.add_argument("--region-table-height-m", type=float, default=0.0300814467)
    parser.add_argument("--region-height-source", default=CubeTaskRegion().height_source,
                        help="declared static scene/calibration source, never fitted from cube positions")
    args = parser.parse_args()
    if (min(args.feature_batch_size, args.train_batch_size, args.epochs) < 1
            or not np.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError("batch sizes, epochs and learning rate must be positive")
    if not np.isfinite(args.spatial_cache_mib) or args.spatial_cache_mib < 0:
        raise ValueError("spatial cache size must be finite and nonnegative")
    if args.device == "cpu":
        torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    domain = (CubeTaskRegion(args.region_table_height_m, args.region_height_source)
              if args.region_conditioned else None)

    def region_record(current, successor, table_height=None):
        if domain is None:
            return {}
        if table_height is not None:
            domain.validate_height(table_height)
        inside, next_inside = current_successor_regions(current, successor, domain)
        return {"region_current": inside, "region_next": next_inside,
                "recorded_height_cross_checked": table_height is not None}

    checkpoint = torch.load(args.dp_checkpoint, map_location="cpu", weights_only=True)
    dp_config_values = dict(checkpoint["config"])
    dp_config_values["pretrained_backbone"] = False
    dp = DiffusionPolicy(DPConfig.from_dict(dp_config_values), checkpoint.get("stats"))
    dp.load_state_dict(checkpoint["model"], strict=True)
    if args.representation == "pixel":
        from so101.learning.simdist.cube_pose_pixel_training import run_pixel_training
        # Keep the unused source actor/layer4 on CPU. The pixel wrapper moves
        # only checkpoint conv1..layer3 and trainable FPN/head to args.device.
        dp.eval()
        run_pixel_training(args, dp, checkpoint)
        return
    dp = dp.to(device).eval()
    for parameter in dp.parameters():
        parameter.requires_grad_(False)
    spatial_encoder = None
    if args.representation != "pooled":
        spatial_encoder = FrozenDPSpatialEncoder(dp)
        # The wrapper retains only source transforms/trunks, releasing the actor.
        del dp

    records = []
    paths = sorted(path for directory in (args.data_dir or []) for path in directory.glob("trajectory_*.pkl"))
    if not paths and not args.data_archive:
        raise ValueError("provide at least one --data-dir or --data-archive")
    for path in paths:
        with path.open("rb") as stream:
            episode = pickle.load(stream)
        if 1900 <= int(episode["seed"]) <= 1903:
            raise ValueError("H2 reset seeds 1900–1903 are reserved; development training must not use them")
        # Match the declared finite task objective while keeping only real
        # observed transitions; post-horizon success tails are outside it.
        horizon = int(episode["task_horizon_ticks"])
        ticks = np.asarray(episode["tick"], dtype=np.int64)
        keep = int(np.count_nonzero(ticks < horizon))
        if keep < 1:
            raise ValueError(f"{path}: no transitions inside the declared task horizon")
        episode = {
            **episode,
            **{key: np.asarray(episode[key])[:keep] for key in (
                "actions", "observations", "front_images", "wrist_images",
                "cube_positions_robot_relative_m", "next_cube_positions_robot_relative_m",
                "successes", "terminated", "phase_labels",
                "table_height_robot_relative_m",
            ) if key in episode},
        }
        if episode.get("cube_position_label_semantics") != "simulator_xyz_robot_root_relative_training_only":
            raise ValueError(f"{path}: missing training-only robot-relative cube labels")
        if episode.get("cube_position_order") != ["held_cube", "fixed_cube"]:
            raise ValueError(f"{path}: unexpected cube order")
        current = np.asarray(episode["cube_positions_robot_relative_m"], dtype=np.float32)
        successor = np.asarray(episode["next_cube_positions_robot_relative_m"], dtype=np.float32)
        q = np.asarray(episode["observations"], dtype=np.float32)
        if current.shape != (len(q), 2, 3) or successor.shape != current.shape or q.shape != (len(q), 6):
            raise ValueError(f"{path}: malformed q/cube label shapes")
        if len(q) > 1 and not np.allclose(successor[:-1], current[1:], atol=2e-6, rtol=0):
            raise ValueError(f"{path}: cube successor labels cross a discontinuity")
        feature_record = ({"features": _extract_features(dp, episode, device, args.feature_batch_size)}
                          if args.representation == "pooled" else {})
        displacement = np.linalg.norm(successor[:, 0] - current[:, 0], axis=-1)
        records.append({
            "seed": int(episode["seed"]), **feature_record, "q": q,
            **region_record(current, successor, episode.get("table_height_robot_relative_m")),
            "target": current.reshape(-1, 6),
            "moved": displacement > .002,
            "phase": np.asarray(episode.get("phase_labels", ["unknown"] * len(q))).astype(str),
            "quality": str(episode.get("trajectory_quality", "unknown")),
            "success_frames": int(np.asarray(episode.get("successes", np.zeros(len(q)))).sum()),
            "success_episode": bool(np.asarray(episode.get("successes", np.zeros(len(q)))).any()),
            "stable_success": bool(np.asarray(episode.get("terminated", [False]))[-1]),
            "length": len(q), "source": str(path), "image_source": path,
        })
    if paths:
        del episode
    # Sharded aligned datasets preserve the same per-episode labels while
    # avoiding simultaneous raw-pickle storage. NPZ is compressed on disk;
    # only current RGB and the compact pose/q targets needed here are loaded.
    for archive_path in (args.data_archive or []):
        with np.load(archive_path, allow_pickle=False) as archive:
            if str(archive["cube_position_label_semantics"]) != "simulator_xyz_robot_root_relative_training_only":
                raise ValueError(f"{archive_path}: unexpected cube-label semantics")
            if archive["cube_position_order"].tolist() != ["held_cube", "fixed_cube"]:
                raise ValueError(f"{archive_path}: unexpected cube order")
            starts = archive["episode_starts"].astype(np.int64)
            lengths = archive["episode_lengths"].astype(np.int64)
            seeds_in_archive = archive["episode_seeds"].astype(np.int64)
            front_all = archive["front_images"]
            wrist_all = archive["wrist_images"]
            q_all = archive["observations"].astype(np.float32)
            current_all = archive["cube_positions_robot_relative_m"].astype(np.float32)
            successor_all = archive["next_cube_positions_robot_relative_m"].astype(np.float32)
            moved_all = np.linalg.norm(successor_all[:, 0] - current_all[:, 0], axis=-1) > .002
            phase_all = archive["phase_labels"].astype(str)
            quality_all = archive["trajectory_quality"].astype(str)
            terminal_all = archive["terminated"].astype(bool)
            table_all = archive["table_height_robot_relative_m"] if "table_height_robot_relative_m" in archive else None
            for episode_index, (start, length, seed_value) in enumerate(zip(starts, lengths, seeds_in_archive)):
                seed_value, start, length = int(seed_value), int(start), int(length)
                stop = start + length
                if 1900 <= seed_value <= 1903:
                    raise ValueError("H2 reset seeds 1900–1903 are reserved; development training must not use them")
                q = q_all[start:stop]
                current = current_all[start:stop]
                successor = successor_all[start:stop]
                if (current.shape != (length, 2, 3) or successor.shape != current.shape
                        or q.shape != (length, 6)):
                    raise ValueError(f"{archive_path}: malformed episode {episode_index} labels")
                if length > 1 and not np.allclose(successor[:-1], current[1:], atol=2e-6, rtol=0):
                    raise ValueError(f"{archive_path}: cube successor discontinuity in episode {episode_index}")
                image_source = (front_all[start:stop], wrist_all[start:stop])
                feature_record = {}
                if args.representation == "pooled":
                    episode_data = {"front_images": image_source[0], "wrist_images": image_source[1],
                                    "actions": np.zeros((length, 6), dtype=np.float32)}
                    feature_record = {"features": _extract_features(dp, episode_data, device,
                                                                       args.feature_batch_size)}
                records.append({
                    "seed": seed_value, **feature_record, "q": q,
                    **region_record(current, successor, None if table_all is None else table_all[start:stop]),
                    "target": current.reshape(-1, 6), "moved": moved_all[start:stop],
                    "phase": phase_all[start:stop], "quality": str(quality_all[start]),
                    # The aligned packer currently omits transient success
                    # labels; do not misreport terminal success as any-frame.
                    "success_frames": None,
                    "success_episode": None,
                    "stable_success": bool(terminal_all[stop - 1]), "length": length,
                    "source": f"{archive_path}#episode={episode_index}", "image_source": image_source,
                })
    feature_stream = None
    if spatial_encoder is not None:
        feature_stream = SpatialFeatureStream(
            spatial_encoder, [(row["image_source"], row["length"]) for row in records], device,
            args.feature_batch_size, args.spatial_cache_mib,
        )
        offset = 0
        for row in records:
            row["sample_ids"] = np.arange(offset, offset + row["length"], dtype=np.int64)
            offset += row["length"]
    seeds = sorted({row["seed"] for row in records})
    explicit_split = any((args.fit_seeds, args.calibration_seeds, args.selection_seeds))
    if explicit_split:
        if not (args.fit_seeds and args.calibration_seeds and args.selection_seeds):
            raise ValueError("explicit splitting requires --fit-seeds, --calibration-seeds and --selection-seeds")
        fit_seeds = sorted(set(args.fit_seeds))
        calibration_seeds = sorted(set(args.calibration_seeds))
        selection_seeds = sorted(set(args.selection_seeds))
        groups = [set(fit_seeds), set(calibration_seeds), set(selection_seeds)]
        if any(not group for group in groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("reset-seed split groups must be nonempty and disjoint")
        if set.union(*groups) != set(seeds):
            raise ValueError(f"split groups must cover exactly dataset seeds {seeds}")
    else:
        if len(seeds) != 4:
            raise ValueError(f"default split expects exactly four reset-round seed groups, got {seeds}")
        fit_seeds, calibration_seeds, selection_seeds = seeds[:2], seeds[2:3], seeds[3:4]

    def combine(seed_group):
        selected = [row for row in records if row["seed"] in seed_group]
        feature_key = "features" if feature_stream is None else "sample_ids"
        output = {
            key: np.concatenate([row[key] for row in selected], axis=0)
            for key in (feature_key, "q", "target", "moved", "phase") +
            (("region_current", "region_next") if domain is not None else ())
        }
        if domain is not None:
            output.update({
                "seed_frames": np.concatenate([np.full(row["length"], row["seed"], dtype=np.int64) for row in selected]),
                "episode_frames": np.concatenate([np.full(row["length"], row["source"]) for row in selected]),
                "quality_frames": np.concatenate([np.full(row["length"], row["quality"]) for row in selected]),
            })
        return output

    fit, calibration, selection = map(combine, (fit_seeds, calibration_seeds, selection_seeds))
    if feature_stream is None:
        feature_shape = list(fit["features"].shape[2:])
        cached_feature_bytes = sum(row["features"].nbytes for row in records)
        for row in records:
            del row["features"]
    else:
        # Probe one fit sample, retaining at most one extraction chunk.
        feature_shape = list(feature_stream.get(fit["sample_ids"][:1]).shape[2:])
        prefilled = feature_stream.prefill_if_fits()
        print(json.dumps({"spatial_cache_prefilled": prefilled,
                          "spatial_cache": feature_stream.statistics()}, indent=2), file=sys.stderr)
    region_contract = {}
    class_weights = None
    if domain is not None:
        stats, norm_contract = fit_region_normalization(fit["q"], fit["target"], fit["region_current"])
        fit_weights, class_contract = fit_region_class_weights(fit["region_current"])
        class_weights = torch.as_tensor(fit_weights, device=device)
        if np.any(calibration["region_current"].sum(0) == 0):
            raise ValueError("calibration must contain a GT-inside pose for each object to select XYZ epochs")
        region_contract = {
            "region_domain": domain.metadata(), "region_normalization": norm_contract,
            "region_class_weighting": class_contract, "region_loss_weight": REGION_LOSS_WEIGHT,
            "region_decision_threshold": .5, "epoch_selection": "calibration_masked_XYZ_only",
            "recorded_height_cross_checked_episodes": sum(r["recorded_height_cross_checked"] for r in records),
            "region_labels": "actual_current_and_successor_XYZ_GT_per_object_not_model_inputs",
            "region_label_counts_by_split": {
                name: {"frames": len(split["q"]),
                       "current_inside_per_object": split["region_current"].sum(0).tolist(),
                       "successor_inside_per_object": split["region_next"].sum(0).tolist()}
                for name, split in (("fit", fit), ("calibration", calibration), ("selection", selection))
            },
            "termination_value_semantics": "unchanged_no_absorbing_outside_or_new_zero_value",
        }
    else:
        stats = {"q_mean": fit["q"].mean(0), "q_std": np.maximum(fit["q"].std(0), 1e-3),
                 "pose_mean": fit["target"].mean(0), "pose_std": np.maximum(fit["target"].std(0), 1e-3)}
    q_mean, q_std, y_mean, y_std = [stats[k] for k in ("q_mean", "q_std", "pose_mean", "pose_std")]
    normalization = CubePoseNormalization(stats).to(device)

    def tensors(split):
        q = normalization.normalize_joints(torch.as_tensor(split["q"], dtype=torch.float32, device=device))
        targets = torch.as_tensor(split["target"], dtype=torch.float32, device=device)
        if domain is not None:
            split["inside_tensor"] = torch.as_tensor(split["region_current"], dtype=torch.bool, device=device)
            targets = neutral_pose_targets(targets, split["inside_tensor"], normalization.pose_mean)
        targets = normalization.normalize_targets(targets)
        if feature_stream is not None:
            return None, q, targets
        features = torch.as_tensor(split["features"], dtype=torch.float32, device=device).clone()
        if args.vision_mode in ("wrist", "none"):
            features[:, 0] = 0
        if args.vision_mode in ("front", "none"):
            features[:, 1] = 0
        return features, q, targets

    x_fit, q_fit, y_fit = tensors(fit)
    x_cal, q_cal, y_cal = tensors(calibration)
    x_sel, q_sel, y_sel = tensors(selection)
    if feature_stream is None:
        for split in (fit, calibration, selection):
            del split["features"]
    model = (CubePoseHead(region_conditioned=domain is not None) if args.representation == "pooled" else
             CubeSpatialPoseHead(feature_shape[0], attention=args.representation == "spatial",
                                 region_conditioned=domain is not None)).to(device)

    def batch_features(x, split, ids, permutation=None):
        feature_ids = ids if permutation is None else permutation[ids]
        if feature_stream is None:
            return x[feature_ids]
        features = feature_stream.get(split["sample_ids"][feature_ids.detach().cpu().numpy()]).to(device)
        if args.vision_mode in ("wrist", "none"):
            features[:, 0] = 0
        if args.vision_mode in ("front", "none"):
            features[:, 1] = 0
        return features

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.learning_rate, weight_decay=1e-4)
    best_calibration = float("inf")
    best_state = None
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(y_fit), device=device)
        for ids in order.split(args.train_batch_size):
            optimizer.zero_grad(set_to_none=True)
            features = batch_features(x_fit, fit, ids)
            pred = model(features[:, 0], features[:, 1], q_fit[ids])
            loss = (conditional_pose_loss(pred.normalized_xyz, y_fit[ids], pred.region_logits,
                                          fit["inside_tensor"][ids], class_weights)["loss"]
                    if domain is not None else nn.functional.mse_loss(pred, y_fit[ids]))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            cal_sum = 0.0
            region_sums = torch.zeros(2, device=device)
            region_counts = torch.zeros(2, device=device)
            for ids in torch.arange(len(y_cal), device=device).split(args.train_batch_size):
                features = batch_features(x_cal, calibration, ids)
                pred = model(features[:, 0], features[:, 1], q_cal[ids])
                if domain is not None:
                    sums, counts = masked_xyz_terms(pred.normalized_xyz, y_cal[ids], calibration["inside_tensor"][ids])
                    region_sums += sums
                    region_counts += counts
                else:
                    cal_sum += float(nn.functional.mse_loss(pred, y_cal[ids], reduction="sum").item())
            cal_loss = (float((.5 * (region_sums / region_counts).sum()).item())
                        if domain is not None else cal_sum / y_cal.numel())
        if cal_loss < best_calibration:
            best_calibration = cal_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no calibration-selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()

    def predict(x, q, split, permutation=None):
        parts, logits = [], []
        with torch.no_grad():
            for ids in torch.arange(len(q), device=device).split(args.train_batch_size):
                features = batch_features(x, split, ids, permutation)
                pred = model(features[:, 0], features[:, 1], q[ids])
                if domain is not None:
                    parts.append(pred.normalized_xyz.reshape(-1, 6).cpu().numpy())
                    logits.append(pred.region_logits.cpu().numpy())
                else:
                    parts.append(pred.cpu().numpy())
        normalized = np.concatenate(parts)
        return normalized * y_std + y_mean, np.concatenate(logits) if logits else None

    cal_pred, cal_logits = predict(x_cal, q_cal, calibration)
    sel_pred, sel_logits = predict(x_sel, q_sel, selection)
    mean_baseline_cal = np.broadcast_to(y_mean, calibration["target"].shape)
    mean_baseline_sel = np.broadcast_to(y_mean, selection["target"].shape)
    cal_target = calibration["target"]
    sel_target = selection["target"]
    def split_metrics(pred, logits, split):
        if domain is None:
            return _metrics(pred, split["target"], split["moved"], split["phase"])
        def evaluate(mask):
            return region_pose_metrics(pred[mask], split["target"][mask], split["region_current"][mask],
                                       None if logits is None else logits[mask], seeds=split["seed_frames"][mask],
                                       episode_ids=split["episode_frames"][mask])
        report = evaluate(np.ones(len(pred), dtype=bool))
        report["full_raw_diagnostic"] = _metrics(pred, split["target"], split["moved"], split["phase"])
        report["strata"] = {"moving_held_cube": evaluate(split["moved"]),
                            "static_held_cube": evaluate(~split["moved"])}
        for prefix, key in (("phase", "phase"), ("quality", "quality_frames"),
                            ("round", "seed_frames"), ("episode", "episode_frames")):
            for value in np.unique(split[key]):
                report["strata"][f"{prefix}:{value}"] = evaluate(split[key] == value)
        report["every_round_pose_gate_passed"] = all(
            value["pose_gate"]["passed"] for name, value in report["strata"].items() if name.startswith("round:"))
        return report
    calibration_report = split_metrics(cal_pred, cal_logits, calibration)
    selection_report = split_metrics(sel_pred, sel_logits, selection)
    fit_pred, fit_logits = predict(x_fit, q_fit, fit)
    fit_report = split_metrics(fit_pred, fit_logits, fit)
    # Diagnostic only: jointly swap the two camera maps across selection
    # samples while leaving q/targets intact. No fitting or epoch selection.
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    swapped = torch.randperm(len(q_sel), generator=generator, device=device)
    swap_pred, swap_logits = predict(x_sel, q_sel, selection, swapped)
    swap_report = split_metrics(swap_pred, swap_logits, selection)
    calibration_report["fit_mean_baseline"] = split_metrics(mean_baseline_cal, None, calibration)
    selection_report["fit_mean_baseline"] = split_metrics(mean_baseline_sel, None, selection)
    gate_pass = (selection_report["pose_gate"]["passed"] and selection_report["every_round_pose_gate_passed"]
                 if domain is not None else (
                     selection_report["held_height_abs_mm_p50_p95"][1] < 2.5 and
                     all(v < 9.0 for v in selection_report["relative_xy_abs_mm_p95_per_axis"])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    spatial_contract = {} if feature_stream is None else {
        "spatial_cache_statistics": feature_stream.statistics(),
        "spatial_feature_contract": {
            "camera_order": ["front", "wrist"], "dp_camera_indices": list(spatial_encoder.dp_camera_indices),
            "source": "exact DP ResNet18 layer4 before avgpool/projection/LayerNorm",
            "nominal_input_stride": 32,
            "soft_coordinates": "transformed camera-map cell centers in [-1,1]; u right, v down",
            "dp_image_shape": list(spatial_encoder.image_shape), "dp_crop_size": spatial_encoder.crop_size,
            "transforms": "loaded DP per-camera deterministic eval crop/resize and ImageNet normalization",
            "backbone_frozen": True, "feature_cache_limit_mib": args.spatial_cache_mib,
            "feature_batch_size": args.feature_batch_size, "train_batch_size": args.train_batch_size,
            "cache_storage": "bounded CPU float32 chunks; no feature-map disk files",
        },
    }
    torch.save({
        "model": best_state,
        "normalization": {k: torch.as_tensor(v) for k, v in stats.items()},
        "region_conditioned": domain is not None, **region_contract,
        **spatial_contract,
        "model_inputs": ["current_front_rgb", "current_wrist_rgb", "current_measured_q"],
        "pose_output_contract": "held_xyz_then_fixed_xyz_robot_root_relative_metres_after_fit_only_denormalization",
        "pose_frame_axes": "simulator_world_xyz_minus_robot_root_world_translation; no root rotation applied",
        "architecture": ("region_conditioned_" if domain is not None else "") + (
                         "frozen_dp_post_layernorm_front_wrist_plus_current_q_mlp_262_128_64_6"
                         if args.representation == "pooled" else
                         "frozen_dp_layer4_object_attention_front_wrist_plus_current_q_mlp"),
        "representation": args.representation,
        "feature_shape": feature_shape,
        "spatial_attention": args.representation == "spatial",
        "spatial_cache_dtype": "float32",
        "vision_mode": args.vision_mode,
        "cube_position_order": ["held_cube", "fixed_cube"],
        "label_semantics": "simulator_xyz_robot_root_relative_training_only",
        "dp_checkpoint": str(args.dp_checkpoint.resolve()),
        "dp_checkpoint_sha256": _sha256(args.dp_checkpoint),
        "dp_training_step": int(checkpoint.get("step", -1)),
        "dp_vision_source_has_real_data": True,
        "fit_seeds": fit_seeds,
        "calibration_seeds": calibration_seeds,
        "selection_seeds": selection_seeds,
        "best_epoch_by_calibration_mse": best_epoch,
        "calibration_normalized_mse": best_calibration,
        "fit_only_normalization": True,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "source_episodes": [r["source"] for r in records],
    }, args.out)
    report = {
        "region_conditioned": domain is not None, **region_contract,
        **spatial_contract,
        "checkpoint": str(args.out), "episodes": len(records), "transitions": int(sum(r["length"] for r in records)),
        "reset_rounds": {"fit": fit_seeds, "calibration": calibration_seeds, "selection": selection_seeds},
        "fit_episodes": sum(r["seed"] in fit_seeds for r in records),
        "calibration_episodes": sum(r["seed"] in calibration_seeds for r in records),
        "selection_episodes": sum(r["seed"] in selection_seeds for r in records),
        "success_frames": (None if any(r["success_frames"] is None for r in records)
                           else sum(r["success_frames"] for r in records)),
        "success_episodes_any_frame": (None if any(r["success_episode"] is None for r in records)
                                       else sum(r["success_episode"] for r in records)),
        "success_label_coverage": "partial; aligned archives omit per-frame successes"
        if any(r["success_frames"] is None for r in records) else "all episodes",
        "stable_success_episodes": sum(r["stable_success"] for r in records),
        "best_epoch": best_epoch,
        "representation": args.representation,
        "feature_shape": feature_shape,
        "feature_cache_bytes": cached_feature_bytes if feature_stream is None else feature_stream.cached_bytes,
        "feature_cache_peak_bytes": (cached_feature_bytes if feature_stream is None
                                     else feature_stream.peak_cached_bytes),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "fit": fit_report,
        "calibration": calibration_report,
        "selection": selection_report,
        "selection_image_swap_diagnostic": swap_report,
        "pose_gate": {"thresholds_mm": {"held_height_p95": 2.5, "relative_xy_axis_p95": 9.0},
                      "passed": bool(gate_pass),
                      "claim_scope": f"development selection over {len(selection_seeds)} reset-round groups; no efficacy claim"},
        "limitations": [f"selection contains {len(selection_seeds)} reset-round seed groups and remains development data",
                        "DP visual features include real+sim pretraining", "no occlusion labels",
                        "learned attention is not a supervised cube keypoint", "front DP centre crop unchanged",
                        "no transition or reward/planner evaluation"],
    }
    report_path = args.out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
