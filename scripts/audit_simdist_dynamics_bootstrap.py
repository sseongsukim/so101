#!/usr/bin/env python3
"""Frozen-checkpoint held-out latent forecast audit with episode bootstrap."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from so101.learning.simdist.checkpoint import load_worldmodel
from so101.learning.simdist.dataset import VisualWindowDataset
from so101.learning.simdist.dynamics_audit import (
    aggregate_by_cluster,
    bootstrap_ratio,
    episode_forecast_mse,
    packed_episode_fingerprints,
    summarize_groups,
    training_seed_inventory,
    validate_episode_keys,
    validate_fingerprint_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True,
                        help="one or more disjoint packed holdout datasets; multiple inputs are pooled in memory")
    parser.add_argument("--training-episode-dirs", type=Path, nargs="+",
                        help="optional raw-source cross-check; packed source fingerprints are always checked")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--min-episodes", type=int, default=16)
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(1)
    before = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    before_stat = args.checkpoint.stat().st_mtime_ns, args.checkpoint.stat().st_size
    model, saved = load_worldmodel(args.checkpoint, args.device)
    manifest = saved["manifest"]
    train_source = Path(manifest["source"]).resolve()
    if any(train_source == path.resolve() for path in args.data):
        raise ValueError("held-out audit data must differ from checkpoint training dataset")
    if not train_source.is_file():
        raise ValueError(f"checkpoint training dataset source does not exist: {train_source}")
    with np.load(train_source, allow_pickle=False) as archive:
        packed_training = {key: archive[key] for key in (
            "episode_seeds", "episode_fingerprints", "episode_starts", "episode_lengths",
            "observations", "next_observations", "actions", "ticks", "task_horizon_ticks")}
    training_required = ("episode_seeds", "episode_fingerprints", "episode_starts", "episode_lengths",
                        "observations", "next_observations", "actions", "ticks", "task_horizon_ticks")
    missing_train = [key for key in training_required if key not in packed_training]
    if missing_train:
        raise ValueError(f"checkpoint source lacks packed training provenance fields {missing_train}")
    training_fingerprints = packed_episode_fingerprints(packed_training)
    declared_training_fingerprints = list(map(str, packed_training["episode_fingerprints"].tolist()))
    if training_fingerprints != declared_training_fingerprints:
        raise ValueError("training fingerprint metadata does not match packed transitions")
    if len(set(training_fingerprints)) != len(training_fingerprints):
        raise ValueError("duplicate episode fingerprints in packed training source")
    train = sorted(set(map(int, packed_training["episode_seeds"].tolist())))
    training_episode_count = len(packed_training["episode_starts"])
    if len(packed_training["episode_seeds"]) != training_episode_count or any(seed < 0 for seed in train):
        raise ValueError("packed training seed inventory is incomplete")
    if args.training_episode_dirs:
        raw_seeds, raw_count, raw_fingerprints = training_seed_inventory(
            args.training_episode_dirs, model.config.task_horizon_ticks)
        if raw_seeds != train or raw_count != training_episode_count or set(raw_fingerprints) != set(training_fingerprints):
            raise ValueError("raw training episode directories do not match the packed checkpoint source")
    rows = []
    all_keys, all_holdout_seeds, dataset_summaries = [], [], []
    for data_path in args.data:
        with np.load(data_path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        required = ("episode_seeds", "episode_env_indices", "episode_quality", "episode_noise_std",
                    "episode_sources", "episode_fingerprints", "terminated", "truncated")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"holdout data {data_path} lacks provenance fields {missing}; regenerate with current converter")
        seeds = list(map(int, data["episode_seeds"].tolist()))
        env_ids = list(map(int, data["episode_env_indices"].tolist()))
        keys = validate_episode_keys(train, seeds, env_ids, len(data["episode_starts"]))
        all_keys.extend(keys)
        all_holdout_seeds.extend(seeds)
        actual_fingerprints = packed_episode_fingerprints(data)
        declared_fingerprints = list(map(str, data["episode_fingerprints"].tolist()))
        if actual_fingerprints != declared_fingerprints:
            raise ValueError(f"holdout fingerprint metadata mismatch: {data_path}")
        validate_fingerprint_split(training_fingerprints, training_fingerprints, actual_fingerprints)
        dataset_summaries.append({"path": str(data_path.resolve()), "episodes": len(seeds),
                                  "seeds": sorted(set(seeds)), "fingerprint_count": len(actual_fingerprints)})
        episode_offset = len(rows)
        for ep in range(len(data["episode_starts"])):
            dataset = VisualWindowDataset(data, model.config, [ep])
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            mse = episode_forecast_mse(model, loader, args.device)
            start, length = int(data["episode_starts"][ep]), int(data["episode_lengths"][ep])
            stop = start + length
            terminal = ("success" if bool(data["terminated"][stop-1]) else
                        "horizon" if int(data["next_remaining_ticks"][stop-1]) == 0 else
                        "truncated" if bool(data["truncated"][stop-1]) else "censored")
            rows.append({"episode": episode_offset + ep, "seed": int(data["episode_seeds"][ep]),
                         "env_index": int(data["episode_env_indices"][ep]),
                         "quality": str(data["episode_quality"][ep]),
                         "noise_std": float(data["episode_noise_std"][ep]),
                         "terminal_class": terminal, "source": str(data["episode_sources"][ep]),
                         **mse})
        del data
    if len(set(all_keys)) != len(all_keys):
        raise ValueError("pooled holdout datasets contain duplicate (seed, env_index) keys")
    groups = summarize_groups(rows, min_episodes=args.min_episodes,
                              repetitions=args.bootstrap_repetitions, seed=args.bootstrap_seed)
    round_clusters = aggregate_by_cluster(rows, "seed")
    round_horizons = bootstrap_ratio(
        [row["learned_mse"] for row in round_clusters],
        [row["persistence_mse"] for row in round_clusters],
        repetitions=args.bootstrap_repetitions, seed=args.bootstrap_seed)
    joint_audit = None
    if rows and "joint_learned_mse" in rows[0]:
        joint_rows = [{**row, "learned_mse": row["joint_learned_mse"],
                       "persistence_mse": row["joint_persistence_mse"]} for row in rows]
        joint_groups = summarize_groups(joint_rows, min_episodes=args.min_episodes,
                                        repetitions=args.bootstrap_repetitions, seed=args.bootstrap_seed)
        joint_clusters = aggregate_by_cluster(joint_rows, "seed")
        joint_round_horizons = bootstrap_ratio(
            [row["learned_mse"] for row in joint_clusters],
            [row["persistence_mse"] for row in joint_clusters],
            repetitions=args.bootstrap_repetitions, seed=args.bootstrap_seed)
        joint_audit = {"baseline": "repeat current measured joint position",
                       "bootstrap_unit": "episode; aggregate overlapping windows within episode before resampling",
                       "acceptance_gate": "all horizon upper 95% CI of learned/persistence joint MSE ratio < 1",
                       "groups": joint_groups,
                       "round_seed_cluster_sensitivity": {
                           "clusters": len(joint_clusters),
                           "verdict": "pass" if all(row["verdict"] == "pass" for row in joint_round_horizons)
                           else "fail" if any(row["verdict"] == "fail" for row in joint_round_horizons)
                           else "inconclusive",
                           "horizons": joint_round_horizons}}
    result = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": before,
              "datasets": dataset_summaries, "training_seeds": train,
              "training_episodes_in_inventory": training_episode_count,
              "training_inventory_source": "packed_source_npz" if not args.training_episode_dirs else
              "packed_source_npz_plus_raw_trajectory_crosscheck",
              "training_dataset_source": str(train_source),
              "training_fingerprint_count": len(training_fingerprints),
              "holdout_seeds": sorted(set(all_holdout_seeds)), "holdout_episode_keys": [list(key) for key in all_keys], "seed_disjoint": True,
              "baseline": "current measured RGB/joints content held fixed; only future remaining-time embedding changes",
              "bootstrap_unit": "episode; aggregate overlapping windows within episode before resampling",
              "bootstrap_repetitions": args.bootstrap_repetitions, "bootstrap_seed": args.bootstrap_seed,
              "minimum_aggregate_episodes": args.min_episodes, "minimum_group_episodes": 4,
              "primary_bootstrap_unit": "episode",
              "episode_bootstrap_assumption": "episodes sampled in parallel within a reset round may be dependent; round-seed cluster sensitivity is also reported",
              "round_seed_cluster_sensitivity": {"clusters": len(round_clusters), "episodes_per_cluster": [row["episodes"] for row in round_clusters],
                                                  "verdict": "pass" if all(row["verdict"] == "pass" for row in round_horizons) else "fail" if any(row["verdict"] == "fail" for row in round_horizons) else "inconclusive",
                                                  "horizons": round_horizons},
              "acceptance_gate": "all horizon upper 95% CI of learned/persistence ratio < 1",
              "groups": groups, "joint_dynamics_audit": joint_audit, "episodes": rows}
    after = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    after_stat = args.checkpoint.stat().st_mtime_ns, args.checkpoint.stat().st_size
    if before != after or before_stat != after_stat:
        raise RuntimeError("frozen checkpoint changed during audit")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"groups": groups, "episodes": len(rows), "seed_disjoint": True}, indent=2))


if __name__ == "__main__":
    main()
