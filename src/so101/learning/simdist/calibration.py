"""Held-out raw-unit calibration; finite-horizon targets include failed episodes."""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def calibration(model, loader, device="cpu"):
    model.eval()
    totals = {name: 0.0 for name in ("reward_count", "reward_absolute_error", "reward_squared_error",
              "value_count", "value_absolute_error", "value_squared_error", "value_target_sum",
              "value_target_square_sum", "value_prediction_sum", "value_prediction_square_sum",
              "horizon_zero_count", "horizon_zero_absolute_prediction")}
    latent_sum = latent_square = None
    latent_count = 0
    horizon_error = persistence_error = None
    horizon_batch_count = 0
    sensitivity = None
    for batch in loader:
        x = {k: v.to(device) for k,v in batch["model_in"].items()}
        y = {k: v.to(device) for k,v in batch["labels"].items()}
        raw = model.inference(x)
        for head in ("reward", "value"):
            key = head + "s"
            valid = y[head + "_valid"].bool()
            error = raw[key][valid] - y[key][valid]
            totals[head+"_count"] += int(valid.sum())
            totals[head+"_absolute_error"] += float(error.abs().sum())
            totals[head+"_squared_error"] += float(error.square().sum())
        valid = y["value_valid"].bool()
        for name, tensor in (("target", y["values"]), ("prediction", raw["values"])):
            rows = tensor[valid]
            totals["value_"+name+"_sum"] += float(rows.sum())
            totals["value_"+name+"_square_sum"] += float(rows.square().sum())
        zeros = y["next_remaining_ticks"].squeeze(-1) == 0
        totals["horizon_zero_count"] += int(zeros.sum())
        totals["horizon_zero_absolute_prediction"] += float(raw["values"][zeros].abs().sum())
        target_sequence = model.encode_latent(model.scaler.scale(y)["proprio_obs"], y["front"], y["wrist"],
                                             y["next_remaining_ticks"])
        _, current = model.context(x)
        error = (raw["latents"]-target_sequence).square().mean(-1).sum(0).cpu()
        persistence = (current[:,None]-target_sequence).square().mean(-1).sum(0).cpu()
        horizon_error = error if horizon_error is None else horizon_error+error
        persistence_error = persistence if persistence_error is None else persistence_error+persistence
        horizon_batch_count += len(current)
        if sensitivity is None:
            sensitivity = {}
            for camera in ("front", "wrist"):
                altered = dict(x); altered[camera+"_hist"] = torch.zeros_like(x[camera+"_hist"])
                _, changed = model.context(altered)
                sensitivity[camera+"_blackout_latent_rms"] = float((changed-current).square().mean().sqrt())
            altered = dict(x); altered["fut_acts"] = x["fut_acts"] + 0.05
            changed = model.inference(altered)
            sensitivity["future_action_plus_005_rad_latent_rms"] = float((changed["latents"]-raw["latents"]).square().mean().sqrt())
            sensitivity["future_action_plus_005_rad_reward_mae"] = float((changed["rewards"]-raw["rewards"]).abs().mean())
        target = target_sequence.reshape(-1, model.config.latent_dim)
        summed, squared = target.sum(0).cpu(), target.square().sum(0).cpu()
        latent_sum = summed if latent_sum is None else latent_sum+summed
        latent_square = squared if latent_square is None else latent_square+squared
        latent_count += len(target)
    result = {"counts_include_overlapping_windows": True,
              "research_status": "held-out diagnostics; no quality threshold or real robustness conclusion"}
    for head in ("reward", "value"):
        count = totals[head+"_count"]
        result[head+"_targets"] = int(count)
        result[head+"_mae_raw"] = totals[head+"_absolute_error"]/count if count else None
        result[head+"_rmse_raw"] = math.sqrt(totals[head+"_squared_error"]/count) if count else None
    count = totals["value_count"]
    for name in ("target", "prediction"):
        mean = totals["value_"+name+"_sum"]/count if count else None
        variance = max(totals["value_"+name+"_square_sum"]/count - mean**2,0) if count else None
        result["value_"+name+"_mean"] = mean
        result["value_"+name+"_std"] = math.sqrt(variance) if count else None
    result["known_horizon_zero_targets"] = int(totals["horizon_zero_count"])
    result["known_horizon_zero_prediction_abs_sum"] = totals["horizon_zero_absolute_prediction"]
    result["latent_feature_std_mean"] = float((latent_square/latent_count - (latent_sum/latent_count).square()).clamp_min(0).sqrt().mean()) if latent_count else None
    result["latent_mse_per_horizon"] = (horizon_error/horizon_batch_count).tolist() if horizon_batch_count else None
    result["current_latent_persistence_mse_per_horizon"] = (persistence_error/horizon_batch_count).tolist() if horizon_batch_count else None
    result["persistence_baseline_keeps_current_time_tag"] = True
    result["first_validation_batch_sensitivity_diagnostics"] = sensitivity
    return result
