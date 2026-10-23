#!/usr/bin/env python3
"""Bounded, read-only probes for a frozen SO101 SimDist checkpoint."""
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
from so101.learning.simdist.dynamics_audit import time_aware_persistence


def sample_window_indices(dataset, data, episodes, per_episode=4):
    if per_episode < 1:
        raise ValueError("per_episode must be positive")
    selected = []
    for episode in map(int, episodes):
        start = int(data["episode_starts"][episode])
        stop = start + int(data["episode_lengths"][episode])
        local = [index for index, tick in enumerate(dataset.windows) if start <= tick < stop]
        if not local:
            raise ValueError(f"episode {episode} has no complete forecast windows")
        count = min(per_episode, len(local))
        fractions = np.linspace(0.2, 0.8, count)
        positions = np.floor((len(local) - 1) * fractions).astype(int)
        selected.extend(local[position] for position in positions)
    return selected


def _copy_with(x, replacements):
    result = dict(x)
    result.update(replacements)
    return result


@torch.no_grad()
def collect_split(model, dataset, selected, *, device, batch_size, seed=42):
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(Subset(dataset, selected), batch_size=batch_size,
                        shuffle=False, generator=generator)
    outputs = {name: {"prediction": [], "mse": []} for name in (
        "learned", "action_shuffle", "action_repeat_last", "history_repeat",
        "clock_plus_30", "action_plus_005")}
    target_rows, persistence_rows = [], []
    joint_prediction_rows, joint_target_rows, joint_current_rows = [], [], []
    suffix_early, suffix_late = [], []
    branch_rows = {name: [] for name in ("joints", "front", "wrist")}
    model.eval()
    for batch in loader:
        x = {key: value.to(device) for key, value in batch["model_in"].items()}
        y = {key: value.to(device) for key, value in batch["labels"].items()}
        scaled_y = model.scaler.scale(y)
        target = model.encode_latent(scaled_y["proprio_obs"], y["front"], y["wrist"],
                                     y["next_remaining_ticks"])
        learned_outputs = model.inference(x)
        learned = learned_outputs["latents"]
        if "proprio_predictions" in learned_outputs:
            joint_prediction_rows.append(learned_outputs["proprio_predictions"].cpu().numpy())
            joint_target_rows.append(y["proprio_obs"].cpu().numpy())
            joint_current_rows.append(x["proprio_obs_hist"][:, -1].cpu().numpy())
        persistence = time_aware_persistence(model, x, y["next_remaining_ticks"])
        target_rows.append(target.cpu().numpy())
        persistence_rows.append(persistence.cpu().numpy())

        predictions = {"learned": learned}
        n = len(learned)
        action_permutation = torch.arange(n - 1, -1, -1, device=learned.device)
        predictions["action_shuffle"] = model.inference(_copy_with(
            x, {"fut_acts": x["fut_acts"][action_permutation]}))["latents"]
        repeated_action = x["fut_acts"][:, -1:, :].expand_as(x["fut_acts"])
        predictions["action_repeat_last"] = model.inference(_copy_with(
            x, {"fut_acts": repeated_action}))["latents"]
        history_replacements = {}
        for key in ("proprio_obs_hist", "front_hist", "wrist_hist", "acts_hist"):
            history_replacements[key] = x[key][:, -1:, ...].expand_as(x[key])
        predictions["history_repeat"] = model.inference(_copy_with(
            x, history_replacements))["latents"]
        clock = x["remaining_ticks_hist"].clone()
        clock[:, -1, 0] = (clock[:, -1, 0] + 30).clamp_max(model.config.task_horizon_ticks)
        predictions["clock_plus_30"] = model.inference(_copy_with(
            x, {"remaining_ticks_hist": clock}))["latents"]
        predictions["action_plus_005"] = model.inference(_copy_with(
            x, {"fut_acts": x["fut_acts"] + 0.05}))["latents"]

        suffix_actions = x["fut_acts"].clone()
        suffix_index = model.config.prediction_length // 2
        suffix_actions[:, suffix_index:, :] += 0.05
        suffix_pred = model.inference(_copy_with(x, {"fut_acts": suffix_actions}))["latents"]
        suffix_delta = (suffix_pred - learned).square().mean(-1).sqrt()
        suffix_early.append(suffix_delta[:, :suffix_index].cpu().numpy())
        suffix_late.append(suffix_delta[:, suffix_index:].cpu().numpy())

        scaled_x = model.scaler.scale(x)
        joints = scaled_x["proprio_obs_hist"][:, -1]
        branch_rows["joints"].append(model.encoder.joints(joints).cpu().numpy())
        branch_rows["front"].append(model.encoder.front(x["front_hist"][:, -1]).cpu().numpy())
        branch_rows["wrist"].append(model.encoder.wrist(x["wrist_hist"][:, -1]).cpu().numpy())
        for name, prediction in predictions.items():
            outputs[name]["prediction"].append(prediction.cpu().numpy())

    target = np.concatenate(target_rows)
    persistence = np.concatenate(persistence_rows)
    result = {"windows": len(target), "horizons": target.shape[1]}
    for name, items in outputs.items():
        prediction = np.concatenate(items["prediction"])
        mse = np.square(prediction - target).mean(axis=-1).mean(axis=0)
        result[name + "_mse_by_horizon"] = mse.tolist()
        result[name + "_prediction"] = prediction
    result["persistence_mse_by_horizon"] = np.square(persistence - target).mean(axis=-1).mean(axis=0).tolist()
    result["suffix_action_delta_rms_early"] = float(np.concatenate(suffix_early).mean())
    result["suffix_action_delta_rms_late"] = float(np.concatenate(suffix_late).mean())
    for name, values in branch_rows.items():
        branch = np.concatenate(values)
        result["branch_" + name] = {"rms": float(np.mean(np.square(branch)) ** 0.5),
                                    "feature_std": float(np.std(branch))}
    result["target"] = target
    result["persistence"] = persistence
    if joint_prediction_rows:
        joint_prediction = np.concatenate(joint_prediction_rows)
        joint_target = np.concatenate(joint_target_rows)
        joint_current = np.concatenate(joint_current_rows)
        joint_baseline = np.broadcast_to(joint_current[:, None, :], joint_target.shape)
        result["joint_prediction_mse_by_horizon"] = _mse(joint_prediction, joint_target).tolist()
        result["joint_persistence_mse_by_horizon"] = _mse(joint_baseline, joint_target).tolist()
        result["joint_predictions"] = joint_prediction
        result["joint_targets"] = joint_target
        result["joint_current"] = joint_current
    return result


def latent_geometry(target, prediction, final_norm):
    gamma = final_norm.weight.detach().float().cpu().numpy()
    beta = final_norm.bias.detach().float().cpu().numpy()
    q = (target - beta) / gamma
    q_mean = q.mean(axis=-1)
    q_std = q.std(axis=-1)
    g_min = float(np.min(np.abs(gamma)))
    row_bound = g_min ** 2 * (q_mean ** 2 + np.maximum(q_std - 1.0, 0.0) ** 2)
    strict_lower = float(row_bound.mean())
    descriptive_unit_sphere = g_min ** 2 * np.mean(q_mean ** 2 + (q_std - 1.0) ** 2)
    predicted_q = (prediction - beta) / gamma
    return {
        "target_row_mean_std": float(target.mean(axis=-1).std()),
        "target_row_std_mean": float(target.std(axis=-1).mean()),
        "target_row_std_std": float(target.std(axis=-1).std()),
        "prediction_row_mean_std": float(prediction.mean(axis=-1).std()),
        "prediction_row_std_mean": float(prediction.std(axis=-1).mean()),
        "prediction_row_std_std": float(prediction.std(axis=-1).std()),
        "inverse_layernorm_target_row_mean_abs": float(np.abs(q_mean).mean()),
        "inverse_layernorm_target_row_std_mean": float(q_std.mean()),
        "inverse_layernorm_target_row_std_std": float(q_std.std()),
        "inverse_layernorm_prediction_row_mean_abs": float(np.abs(predicted_q.mean(axis=-1)).mean()),
        "inverse_layernorm_prediction_row_std_mean": float(predicted_q.std(axis=-1).mean()),
        "layernorm_min_abs_gamma": g_min,
        "conservative_normalized_manifold_mse_lower_bound": float(strict_lower),
        "conservative_normalized_manifold_mse_lower_bound_by_horizon": row_bound.mean(axis=0).tolist(),
        "descriptive_unit_sphere_distance_not_a_strict_bound": float(descriptive_unit_sphere),
    }


def _mse(prediction, target):
    return np.square(prediction - target).mean(axis=-1).mean(axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--holdout-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--windows-per-episode", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model, saved = load_worldmodel(args.checkpoint, args.device)
    if Path(saved["manifest"]["source"]).resolve() != args.training_data.resolve():
        raise ValueError("training-data must exactly match the checkpoint manifest source")
    with np.load(args.training_data, allow_pickle=False) as archive:
        train_data = {key: archive[key] for key in archive.files}
    with np.load(args.holdout_data, allow_pickle=False) as archive:
        holdout_data = {key: archive[key] for key in archive.files}
    train_eps = list(map(int, saved["manifest"]["train_episodes"]))
    val_eps = list(map(int, saved["manifest"]["validation_episodes"]))
    train_dataset = VisualWindowDataset(train_data, model.config, train_eps)
    val_dataset = VisualWindowDataset(train_data, model.config, val_eps)
    holdout_dataset = VisualWindowDataset(holdout_data, model.config,
                                          range(len(holdout_data["episode_starts"])))
    splits = {}
    for name, data, dataset, episodes in (
        ("train", train_data, train_dataset, train_eps),
        ("original_validation", train_data, val_dataset, val_eps),
        ("new_holdout", holdout_data, holdout_dataset, range(len(holdout_data["episode_starts"]))),
    ):
        selected = sample_window_indices(dataset, data, episodes, args.windows_per_episode)
        splits[name] = collect_split(model, dataset, selected, device=args.device,
                                     batch_size=args.batch_size, seed=args.seed)

    train_target = splits["train"]["target"]
    target_mean = train_target.mean(axis=0, keepdims=True)
    # Diagnostic only: fit one shared affine output calibration on the bounded
    # train probe, then check its transfer to untouched validation/holdout.
    train_prediction = splits["train"]["learned_prediction"]
    train_y = train_target.reshape(-1, train_target.shape[-1]).astype(np.float64)
    train_z = train_prediction.reshape(-1, train_prediction.shape[-1]).astype(np.float64)
    train_design = np.concatenate([train_z, np.ones((len(train_z), 1))], axis=1)
    projection_coef = np.linalg.lstsq(train_design, train_y, rcond=1e-8)[0]
    train_base = splits["train"]["persistence"].astype(np.float64)
    train_target64 = train_target.astype(np.float64)
    train_delta = splits["train"]["learned_prediction"].astype(np.float64) - train_base
    target_delta = train_target64 - train_base
    residual_denominator = np.square(train_delta).mean(axis=(0, 2))
    residual_shrinkage = (train_delta * target_delta).mean(axis=(0, 2)) / np.maximum(
        residual_denominator, 1e-12)
    joint_shrinkage = None
    if "joint_predictions" in splits["train"]:
        current = splits["train"]["joint_current"].astype(np.float64)
        delta = splits["train"]["joint_predictions"].astype(np.float64) - current[:, None]
        target_delta = splits["train"]["joint_targets"].astype(np.float64) - current[:, None]
        denominator = np.square(delta).mean(axis=(0, 2))
        joint_shrinkage = (delta * target_delta).mean(axis=(0, 2)) / np.maximum(denominator, 1e-12)
    stats = saved["manifest"]["stats"]
    scaler_report = {}
    for name, raw, group in (("observations", holdout_data["observations"], "proprio_obs"),
                             ("actions", holdout_data["actions"], "actions")):
        z = (raw - np.asarray(stats[group]["mean"])) / np.asarray(stats[group]["std"])
        scaler_report[name] = {"standardized_mean_by_joint": z.mean(axis=0).tolist(),
                               "standardized_std_by_joint": z.std(axis=0).tolist(),
                               "fraction_abs_z_gt_3_by_joint": (np.abs(z) > 3).mean(axis=0).tolist()}
    norm = model.dynamics.layers[-1].layernorm_3
    result = {"checkpoint": str(args.checkpoint.resolve()), "training_data": str(args.training_data.resolve()),
              "holdout_data": str(args.holdout_data.resolve()), "read_only_no_training": True,
              "sampling": {"windows_per_episode": args.windows_per_episode,
                           "fractions": [0.2, 0.4, 0.6, 0.8], "batch_size": args.batch_size,
                           "seed": args.seed, "selection": "evenly spaced local complete windows per episode"},
              "output_projection_probe": {
                  "method": "single shared affine least-squares map after frozen decoder output",
                  "fit_split": "bounded train probe only",
                  "fit_windows": int(len(train_target)),
                  "fit_tokens": int(len(train_z)),
                  "diagnostic_only_not_checkpoint_training": True,
              },
              "hypotheses": {
                  "final_layernorm_target_geometry": "A target-geometry mismatch is supported if latent targets vary outside the final decoder LayerNorm affine manifold.",
                  "action_conditioning": "Compare actual actions to batch-shuffled, repeated-last, and +0.05 rad action sequences.",
                  "history_and_clock": "Compare repeated-current history and +30 remaining-tick controls.",
                  "data_shift": "Check held-out standardized joint/action tails against training-only normalizers.",
              },
              "splits": {}}
    for name, split in splits.items():
        target = split.pop("target")
        prediction = split.pop("learned_prediction")
        persistence = split.pop("persistence")
        design = np.concatenate([
            prediction.reshape(-1, prediction.shape[-1]).astype(np.float64),
            np.ones((prediction.size // prediction.shape[-1], 1)),
        ], axis=1)
        affine_prediction = (design @ projection_coef).reshape(target.shape)
        split["train_probe_affine_projection_mse_by_horizon"] = _mse(
            affine_prediction, target).tolist()
        split["train_probe_affine_projection_vs_persistence_ratio_by_horizon"] = (
            _mse(affine_prediction, target) /
            np.maximum(_mse(persistence, target), 1e-12)
        ).tolist()
        base64 = persistence.astype(np.float64)
        delta64 = prediction.astype(np.float64) - base64
        shrunk = base64 + residual_shrinkage[None, :, None] * delta64
        split["train_probe_residual_shrinkage_alpha_by_horizon"] = residual_shrinkage.tolist()
        split["train_probe_residual_shrinkage_mse_by_horizon"] = _mse(shrunk, target).tolist()
        split["train_probe_residual_shrinkage_vs_persistence_ratio_by_horizon"] = (
            _mse(shrunk, target) / np.maximum(_mse(persistence, target), 1e-12)
        ).tolist()
        if joint_shrinkage is not None:
            joint_prediction = split.pop("joint_predictions").astype(np.float64)
            joint_target = split.pop("joint_targets").astype(np.float64)
            joint_current = split.pop("joint_current").astype(np.float64)
            joint_baseline = np.broadcast_to(joint_current[:, None, :], joint_target.shape)
            joint_delta = joint_prediction - joint_baseline
            joint_shrunk = joint_baseline + joint_shrinkage[None, :, None] * joint_delta
            split["joint_prediction_mse_by_horizon_rad2"] = _mse(joint_prediction, joint_target).tolist()
            split["joint_persistence_mse_by_horizon_rad2"] = _mse(joint_baseline, joint_target).tolist()
            split["train_probe_joint_residual_shrinkage_alpha_by_horizon"] = joint_shrinkage.tolist()
            split["train_probe_joint_residual_shrinkage_mse_by_horizon_rad2"] = _mse(
                joint_shrunk, joint_target).tolist()
            split["train_probe_joint_residual_shrinkage_ratio_by_horizon"] = (
                _mse(joint_shrunk, joint_target) / np.maximum(_mse(joint_baseline, joint_target), 1e-12)
            ).tolist()
        mean_pred = np.broadcast_to(target_mean, target.shape)
        geometry = latent_geometry(target, prediction, norm)
        split["global_mean_train_target_mse_by_horizon"] = _mse(mean_pred, target).tolist()
        split["target_latent_global_std"] = float(target.std())
        split["prediction_latent_global_std"] = float(prediction.std())
        split["target_vs_prediction_geometry"] = geometry
        split["layernorm_geometry_lower_bound_to_learned_error_ratio"] = (
            geometry["conservative_normalized_manifold_mse_lower_bound"] /
            max(float(np.square(prediction - target).mean()), 1e-12))
        split["horizon_1_and_8"] = {
            "learned_mse": [split["learned_mse_by_horizon"][0], split["learned_mse_by_horizon"][-1]],
            "persistence_mse": [split["persistence_mse_by_horizon"][0], split["persistence_mse_by_horizon"][-1]],
            "action_shuffle_mse": [split["action_shuffle_mse_by_horizon"][0], split["action_shuffle_mse_by_horizon"][-1]],
            "action_repeat_last_mse": [split["action_repeat_last_mse_by_horizon"][0], split["action_repeat_last_mse_by_horizon"][-1]],
            "history_repeat_mse": [split["history_repeat_mse_by_horizon"][0], split["history_repeat_mse_by_horizon"][-1]],
            "clock_plus_30_mse": [split["clock_plus_30_mse_by_horizon"][0], split["clock_plus_30_mse_by_horizon"][-1]],
            "action_plus_005_mse": [split["action_plus_005_mse_by_horizon"][0], split["action_plus_005_mse_by_horizon"][-1]],
        }
        # JSON artifact omits raw vectors; they are retained only for calculations.
        for key in list(split):
            if key.endswith("_prediction") or key in ("persistence", "target"):
                split.pop(key, None)
        result["splits"][name] = split
    result["holdout_normalization_tail_check"] = scaler_report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: report["horizon_1_and_8"] for name, report in result["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
