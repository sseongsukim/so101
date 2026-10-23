#!/usr/bin/env python3
"""Audit v2 labels and episode-disjoint model calibration in raw reward units."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from so101.learning.simdist.checkpoint import load_worldmodel
from so101.learning.simdist.dataset import VisualWindowDataset
from so101.learning.simdist.calibration import calibration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--external-holdout", action="store_true",
                   help="audit a separate dataset after verifying its episode seeds do not overlap training seeds")
    a = p.parse_args()
    if a.device == "cpu":
        torch.set_num_threads(1)
    model, saved = load_worldmodel(a.checkpoint, a.device)
    manifest = saved["manifest"]
    train_eps, val_eps = manifest["train_episodes"], manifest["validation_episodes"]
    if set(train_eps) & set(val_eps):
        raise ValueError("training/validation episode leakage")
    source_matches = Path(manifest["source"]).resolve() == a.data.resolve()
    if source_matches == a.external_holdout:
        raise ValueError("pass --external-holdout only for a dataset different from checkpoint training source")
    if a.external_holdout:
        with np.load(manifest["source"], allow_pickle=False) as train_archive:
            training_seeds = set(map(int, train_archive["episode_seeds"]))
    else:
        training_seeds = set()
    with np.load(a.data, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    if a.external_holdout:
        holdout_seeds = set(map(int, data["episode_seeds"]))
        if training_seeds & holdout_seeds:
            raise ValueError(f"external holdout seed overlap: {sorted(training_seeds & holdout_seeds)}")
    # Validate every episode's time/mask/MC contracts, fit no new normalizers.
    VisualWindowDataset(data, model.config, range(len(data["episode_starts"])))
    val = VisualWindowDataset(data, model.config,
                              range(len(data["episode_starts"])) if a.external_holdout else val_eps)
    diagnostics = calibration(model, DataLoader(val,batch_size=a.batch_size), a.device)
    details = []
    for ep in range(len(data["episode_starts"])):
        start, n = int(data["episode_starts"][ep]), int(data["episode_lengths"][ep])
        stop = start+n
        details.append({"episode": ep, "split": "external_holdout" if a.external_holdout else
                        "train" if ep in train_eps else "validation",
                        "transitions": n, "true_success_terminal": bool(data["terminated"][stop-1]),
                        "environment_truncated": bool(data["truncated"][stop-1]),
                        "finite_horizon_completed": bool(data["next_remaining_ticks"][stop-1]==0),
                        "valid_value_targets": int(data["value_valid"][start:stop].sum()),
                        "dense_reward_targets": int(data["reward_valid"][start:stop].sum()),
                        "MC_successor_return_mean": float(data["values"][start:stop].mean())})
    result = {"checkpoint": str(a.checkpoint.resolve()), "data": str(a.data.resolve()),
              "episode_disjoint": True, "seed_disjoint_external_holdout": a.external_holdout,
              "label_contract_valid": True, "value_objective": model.config.value_objective,
              "task_horizon_ticks": model.config.task_horizon_ticks, "episodes": details,
              "validation_calibration": diagnostics}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
