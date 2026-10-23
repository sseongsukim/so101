"""Episode-macro forecast audit and paired bootstrap, without model updates."""
from __future__ import annotations

import pickle
import hashlib
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def time_aware_persistence(model, x, successor_remaining_ticks):
    """Keep current RGB/joint features; change only the explicit future clock.

The fusion MLP is nonlinear. 'Keep content' means identical inputs to its
joint/front/wrist branches, not subtracting a learned time vector from an
already fused latent. No future measured state/image is read by this baseline.
"""
    scaled = model.scaler.scale(x)
    content = model.encoder.content(scaled["proprio_obs_hist"][:, -1],
                                    x["front_hist"][:, -1], x["wrist_hist"][:, -1])
    T = successor_remaining_ticks.shape[1]
    return model.encoder.latent_from_content(content[:,None].expand(-1,T,-1), successor_remaining_ticks)


def bootstrap_ratio(learned, persistence, *, repetitions=2000, seed=42):
    """Percentile 95% CI of ratio of episode mean MSEs, paired by episode.

    The estimate is mean_e L_e / mean_e P_e, never mean_e (L_e/P_e).
    Both methods receive exactly the same resampled episode indices.
    Undefined denominator draws make that horizon inconclusive.
    """
    learned, persistence = np.asarray(learned,np.float64), np.asarray(persistence,np.float64)
    if learned.ndim != 2 or learned.shape != persistence.shape or len(learned) < 2:
        raise ValueError("paired (episodes,horizons) arrays with at least two episodes required")
    if not np.isfinite(learned).all() or not np.isfinite(persistence).all() or (learned<0).any() or (persistence<0).any():
        raise ValueError("finite nonnegative episode MSEs required")
    if repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions required")
    n = len(learned)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0,n,size=(repetitions,n))
    boot_l, boot_p = learned[indices].mean(1), persistence[indices].mean(1)
    center_l, center_p = learned.mean(0), persistence.mean(0)
    rows = []
    for h in range(learned.shape[1]):
        undefined = int(np.count_nonzero(boot_p[:,h] <= 0))
        if center_p[h] <= 0 or undefined:
            ratio, bounds, verdict = None, [None,None], "inconclusive_zero_persistence_error"
        else:
            ratio = float(center_l[h]/center_p[h])
            bounds = np.percentile(boot_l[:,h]/boot_p[:,h],[2.5,97.5]).tolist()
            verdict = "pass" if bounds[1] < 1.0 else "fail" if bounds[0] >= 1.0 else "inconclusive_ci_crosses_one"
        rows.append({"horizon": h+1, "episodes": n, "learned_mean_mse": float(center_l[h]),
                     "persistence_mean_mse": float(center_p[h]), "ratio_of_means": ratio,
                     "ratio_ci95_percentile": bounds, "undefined_bootstrap_draws": undefined,
                     "verdict": verdict})
    return rows


def summarize_groups(episodes, *, min_episodes=16, min_group_episodes=4, repetitions=2000, seed=42):
    if min_episodes < 2 or min_group_episodes < 2:
        raise ValueError("episode bootstrap needs minimum counts >=2")
    if len(episodes) < min_episodes:
        raise ValueError(f"insufficient complete episodes: {len(episodes)} < required {min_episodes}")
    groups = {"all": episodes}
    for field in ("quality", "noise_std", "terminal_class"):
        for label in sorted(set(e[field] for e in episodes)):
            groups[field+":"+str(label)] = [e for e in episodes if e[field]==label]
    out = {}
    for label, rows in groups.items():
        count = len(rows)
        window_count = sum(r["windows"] for r in rows)
        if label != "all" and count < min_group_episodes:
            out[label] = {"episodes": count, "windows": window_count,
                          "verdict": "inconclusive_insufficient_group_episodes", "horizons": []}
            continue
        horizons = bootstrap_ratio([r["learned_mse"] for r in rows], [r["persistence_mse"] for r in rows],
                                   repetitions=repetitions,seed=seed)
        verdicts = [r["verdict"] for r in horizons]
        verdict = "pass" if all(v=="pass" for v in verdicts) else "fail" if "fail" in verdicts else "inconclusive"
        out[label] = {"episodes": count, "windows": window_count, "verdict": verdict, "horizons": horizons}
    return out


def trajectory_fingerprint(episode, horizon):
    """Fingerprint the exact transition rows retained by the finite task horizon."""
    count = len(episode["actions"])
    ticks = np.asarray(episode.get("tick", np.arange(count)))
    if ticks.shape != (count,):
        raise ValueError("training episode tick vector does not match its actions")
    retained = np.flatnonzero(ticks < horizon)
    kept_ticks = ticks[:len(retained)]
    if (not len(retained) or not np.array_equal(retained, np.arange(len(retained)))
            or (len(kept_ticks) > 1 and not np.all(np.diff(kept_ticks) == 1))):
        raise ValueError("training episode ticks are outside or non-contiguous within the task horizon")
    h = hashlib.sha256()
    for key in ("observations", "next_observations", "actions", "tick"):
        values = ticks[:len(retained)] if key == "tick" else np.asarray(episode[key])[:len(retained)]
        h.update(key.encode())
        h.update(str(values.shape).encode())
        h.update(values.astype(np.int64 if key == "tick" else np.float32).tobytes())
    return h.hexdigest()


def packed_episode_fingerprints(data):
    result = []
    for start, length in zip(data["episode_starts"], data["episode_lengths"]):
        start, stop = int(start), int(start + length)
        episode = {key: data[key][start:stop] for key in ("observations", "next_observations", "actions", "ticks")}
        episode["tick"] = episode.pop("ticks")
        result.append(trajectory_fingerprint(episode, int(data["task_horizon_ticks"][start])))
    return result


def training_seed_inventory(directories, horizon=300):
    """Fail closed unless every declared training source contains seeded episodes."""
    if not directories:
        raise ValueError("at least one training episode directory is required")
    seeds, fingerprints, episode_count = set(), set(), 0
    for value in directories:
        directory = Path(value)
        if not directory.is_dir():
            raise ValueError(f"training episode directory does not exist: {directory}")
        files = sorted(directory.glob("trajectory_*.pkl"))
        if not files:
            raise ValueError(f"training episode directory has no trajectory pickles: {directory}")
        for path in files:
            with path.open("rb") as stream:
                episode = pickle.load(stream)
            seed = int(episode.get("seed", -1))
            if seed < 0:
                raise ValueError(f"training episode has missing/invalid seed metadata: {path}")
            seeds.add(seed)
            fingerprints.add(trajectory_fingerprint(episode, horizon))
            episode_count += 1
    if not seeds:
        raise ValueError("training episode seed inventory is empty")
    if len(fingerprints) != episode_count:
        raise ValueError("training episode directories contain duplicate trajectories")
    return sorted(seeds), episode_count, fingerprints


def aggregate_by_cluster(episodes, field):
    """Average episode-macro vectors inside a shared seed/cluster."""
    labels = sorted(set(row[field] for row in episodes))
    result = []
    for label in labels:
        rows = [row for row in episodes if row[field] == label]
        result.append({"cluster": label, "episodes": len(rows),
                       "windows": sum(int(row["windows"]) for row in rows),
                       "learned_mse": np.mean([row["learned_mse"] for row in rows], axis=0).tolist(),
                       "persistence_mse": np.mean([row["persistence_mse"] for row in rows], axis=0).tolist()})
    return result


def validate_episode_keys(training_seeds, holdout_seeds, env_ids, episode_count):
    if len(holdout_seeds) != episode_count or len(env_ids) != episode_count:
        raise ValueError("holdout seed/env provenance count differs from episode count")
    if any(seed < 0 for seed in holdout_seeds) or any(env_id < 0 for env_id in env_ids):
        raise ValueError("holdout episodes contain unknown seed or env index")
    overlap = set(training_seeds).intersection(holdout_seeds)
    if overlap:
        raise ValueError(f"training/holdout seed overlap: {sorted(overlap)}")
    keys = list(zip(holdout_seeds, env_ids))
    if len(set(keys)) != episode_count:
        raise ValueError("holdout episodes must have unique (round seed, env index) identifiers")
    return keys


def validate_fingerprint_split(training_directory_fingerprints, packed_training_fingerprints,
                               holdout_fingerprints):
    directories = list(training_directory_fingerprints)
    packed = list(packed_training_fingerprints)
    holdout = list(holdout_fingerprints)
    if not directories or not packed or not holdout:
        raise ValueError("training and holdout fingerprint inventories must be nonempty")
    if len(set(directories)) != len(directories) or len(set(packed)) != len(packed):
        raise ValueError("duplicate episode fingerprints in training inventory")
    if len(set(holdout)) != len(holdout):
        raise ValueError("duplicate episode fingerprints in holdout inventory")
    if set(directories) != set(packed):
        raise ValueError("training episode directories do not match checkpoint manifest dataset")
    overlap = set(packed).intersection(holdout)
    if overlap:
        raise ValueError("training/holdout transition fingerprint overlap")


@torch.no_grad()
def episode_forecast_mse(model, loader, device="cpu"):
    """Average overlapping-window errors only inside one episode."""
    model.eval()
    learned_sum = persistence_sum = None
    joint_learned_sum = joint_persistence_sum = None
    windows = 0
    for batch in loader:
        x = {k:v.to(device) for k,v in batch["model_in"].items()}
        y = {k:v.to(device) for k,v in batch["labels"].items()}
        outputs = model.inference(x)
        prediction = outputs["latents"]
        target = model.encode_latent(model.scaler.scale(y)["proprio_obs"], y["front"], y["wrist"],
                                     y["next_remaining_ticks"])
        baseline = time_aware_persistence(model,x,y["next_remaining_ticks"])
        learned = (prediction-target).square().mean(-1).sum(0).double().cpu().numpy()
        persistence = (baseline-target).square().mean(-1).sum(0).double().cpu().numpy()
        if not np.isfinite(learned).all() or not np.isfinite(persistence).all():
            raise ValueError("non-finite forecast errors")
        learned_sum = learned if learned_sum is None else learned_sum+learned
        persistence_sum = persistence if persistence_sum is None else persistence_sum+persistence
        if "proprio_predictions" in outputs:
            joint_prediction = (outputs["proprio_predictions"] - y["proprio_obs"]).square().mean(-1).sum(0).double().cpu().numpy()
            current_joints = x["proprio_obs_hist"][:, -1]
            joint_baseline = (current_joints[:, None] - y["proprio_obs"]).square().mean(-1).sum(0).double().cpu().numpy()
            joint_learned_sum = joint_prediction if joint_learned_sum is None else joint_learned_sum + joint_prediction
            joint_persistence_sum = joint_baseline if joint_persistence_sum is None else joint_persistence_sum + joint_baseline
        windows += len(prediction)
    if windows == 0:
        raise ValueError("episode has no full forecast windows")
    result = {"windows": windows, "learned_mse": (learned_sum/windows).tolist(),
              "persistence_mse": (persistence_sum/windows).tolist()}
    if joint_learned_sum is not None:
        result["joint_learned_mse"] = (joint_learned_sum / windows).tolist()
        result["joint_persistence_mse"] = (joint_persistence_sum / windows).tolist()
    return result
