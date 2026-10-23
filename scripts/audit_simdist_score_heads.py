#!/usr/bin/env python3
"""Compare held-out reward/value heads with a training-only time-mean baseline."""
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


def time_bin(remaining, horizon, bins):
    return np.minimum(np.maximum(remaining - 1, 0) * bins // horizon, bins - 1).astype(np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261008)
    parser.add_argument("--time-bins", type=int, default=6)
    parser.add_argument("--external-holdout", action="store_true")
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(1)
    model, saved = load_worldmodel(args.checkpoint, args.device)
    manifest = saved["manifest"]
    source = Path(manifest["source"]).resolve()
    if source == args.data.resolve() and args.external_holdout:
        raise ValueError("external holdout path must differ from checkpoint source")
    if source != args.data.resolve() and not args.external_holdout:
        raise ValueError("pass --external-holdout for data outside the checkpoint source")
    with np.load(source, allow_pickle=False) as f:
        train_data = {k: f[k] for k in f.files}
    with np.load(args.data, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    train_eps = list(map(int, manifest["train_episodes"]))
    eval_eps = (list(range(len(data["episode_starts"]))) if args.external_holdout
                else list(map(int, manifest["validation_episodes"])))
    if args.external_holdout and set(map(int, train_data["episode_seeds"])) & set(map(int, data["episode_seeds"])):
        raise ValueError("external score holdout reset seeds overlap development data")
    train = VisualWindowDataset(train_data, model.config, train_eps)
    evaluation = VisualWindowDataset(data, model.config, eval_eps)
    H = model.config.prediction_length
    bins = args.time_bins
    heads = {"rewards": {"mask": "reward_valid", "episodes": {}},
             "values": {"mask": "value_valid", "episodes": {}}}
    baseline_sum = {name: np.zeros((H, bins), np.float64) for name in heads}
    baseline_count = {name: np.zeros((H, bins), np.int64) for name in heads}

    # Fit clock-conditioned means on optimizer episodes only, using the same
    # complete windows and masks as the forecast model.
    for ds in (train,):
        for h in range(H):
            indices = np.asarray(ds.windows, np.int64) + h
            b = time_bin(train_data["remaining_ticks"][indices], model.config.task_horizon_ticks, bins)
            for name, spec in heads.items():
                valid = train_data[spec["mask"]][indices].astype(bool)
                target = train_data[name][indices].astype(np.float64)
                for bi in range(bins):
                    m = valid & (b == bi)
                    baseline_sum[name][h, bi] += target[m].sum()
                    baseline_count[name][h, bi] += int(m.sum())
    baseline_mean = {}
    for name in heads:
        overall_count = baseline_count[name].sum(axis=1)
        overall_mean = baseline_sum[name].sum(axis=1) / np.maximum(overall_count, 1)
        baseline_mean[name] = np.where(baseline_count[name] > 0,
                                       baseline_sum[name] / np.maximum(baseline_count[name], 1),
                                       overall_mean[:, None])

    starts, lengths = data["episode_starts"], data["episode_lengths"]
    ep_terminal = []
    for ep, (start, length) in enumerate(zip(starts, lengths)):
        last = int(start + length - 1)
        ep_terminal.append("success" if bool(data["terminated"][last]) else
                           "horizon" if int(data["next_remaining_ticks"][last]) == 0 else "truncated")
        for name in heads:
            heads[name]["episodes"][ep] = {"count": np.zeros(H, np.int64),
                                             "sq": np.zeros(H), "base_sq": np.zeros(H),
                                             "abs": np.zeros(H), "bias": np.zeros(H)}
    window_ep = np.searchsorted(starts, np.asarray(evaluation.windows), side="right") - 1
    eval_set = set(eval_eps)
    if not set(map(int, window_ep)) <= eval_set:
        raise ValueError("evaluation windows crossed their episode partition")
    model.eval()
    with torch.no_grad():
        loader = DataLoader(evaluation, batch_size=args.batch_size, shuffle=False)
        offset = 0
        for batch in loader:
            x = {k: v.to(args.device) for k, v in batch["model_in"].items()}
            output = model.inference(x)
            count_batch = len(batch["labels"]["rewards"])
            eps = window_ep[offset:offset + count_batch]
            offset += count_batch
            for name, spec in heads.items():
                pred = output[name].double().cpu().numpy()
                labels = batch["labels"][name].double().numpy()
                valid = batch["labels"][spec["mask"]].bool().numpy()
                target_indices = np.asarray(evaluation.windows[offset-count_batch:offset])[:, None] + np.arange(H)[None, :]
                remain = data["remaining_ticks"][target_indices]
                tb = time_bin(remain, model.config.task_horizon_ticks, bins)
                base = baseline_mean[name][np.arange(H)[None, :], tb]
                for row, ep in enumerate(eps):
                    record = heads[name]["episodes"][int(ep)]
                    mask = valid[row]
                    error = pred[row] - labels[row]
                    residual = base[row] - labels[row]
                    record["count"] += mask
                    record["sq"] += error ** 2 * mask
                    record["base_sq"] += residual ** 2 * mask
                    record["abs"] += np.abs(error) * mask
                    record["bias"] += error * mask

    rng = np.random.default_rng(args.bootstrap_seed)
    result_heads = {}
    for name in heads:
        episodes = heads[name]["episodes"]
        result_groups = {}
        for group_name, group_eps in (("all", eval_eps),
                                      ("success", [e for e in eval_eps if ep_terminal[e] == "success"]),
                                      ("horizon", [e for e in eval_eps if ep_terminal[e] == "horizon"]),
                                      ("truncated", [e for e in eval_eps if ep_terminal[e] == "truncated"])):
            horizons = []
            for h in range(H):
                valid_eps = [e for e in group_eps if episodes[e]["count"][h] > 0]
                n = int(sum(episodes[e]["count"][h] for e in valid_eps))
                if n == 0:
                    horizons.append({"horizon": h+1, "episodes": 0, "valid_targets": 0,
                                     "verdict": "inconclusive_no_targets"})
                    continue
                sq = sum(episodes[e]["sq"][h] for e in valid_eps)
                base_sq = sum(episodes[e]["base_sq"][h] for e in valid_eps)
                abs_error = sum(episodes[e]["abs"][h] for e in valid_eps)
                bias = sum(episodes[e]["bias"][h] for e in valid_eps)
                boot = []
                if len(valid_eps) >= 2:
                    for _ in range(args.bootstrap_repetitions):
                        draw = rng.choice(valid_eps, size=len(valid_eps), replace=True)
                        num = sum(episodes[int(e)]["sq"][h] for e in draw)
                        den = sum(episodes[int(e)]["base_sq"][h] for e in draw)
                        if den > 0:
                            boot.append(num / den)
                ci = np.percentile(boot, [2.5, 97.5]).tolist() if boot else None
                horizons.append({"horizon": h+1, "episodes": len(valid_eps), "valid_targets": n,
                                 "mae": float(abs_error/n), "rmse": float(np.sqrt(sq/n)),
                                 "signed_bias": float(bias/n), "time_mean_baseline_rmse": float(np.sqrt(base_sq/n)),
                                 "mse_ratio_to_time_mean": float(sq/base_sq) if base_sq else None,
                                 "ratio_ci95_episode_bootstrap": ci,
                                 "verdict": "pass" if ci and ci[1] < 1 else
                                            "fail" if ci and ci[0] > 1 else "inconclusive"})
            result_groups[group_name] = {"episodes": len(group_eps), "horizons": horizons}
        result_heads[name] = result_groups
    result = {"checkpoint": str(args.checkpoint.resolve()), "data": str(args.data.resolve()),
              "external_holdout": args.external_holdout, "seed_disjoint": args.external_holdout,
              "training_episodes": train_eps, "evaluation_episodes": eval_eps,
              "baseline": "training-window mean conditioned on prediction horizon and six equal bins of remaining task ticks",
              "value_objective": model.config.value_objective, "bootstrap_repetitions": args.bootstrap_repetitions,
              "heads": result_heads,
              "research_status": "score prediction diagnostic; does not validate candidate action ranking or planner return"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
