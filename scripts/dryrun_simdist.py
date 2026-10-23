#!/usr/bin/env python3
"""Replay recorded observations through a planner without opening any hardware.

Commands are saved for inspection and never sent. The history follows recorded
applied commands, since candidate commands did not produce recorded successors.
Missing camera acquisition timestamps remain explicitly unverified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from so101.learning.simdist.checkpoint import load_worldmodel
from so101.learning.simdist.mppi import MppiConfig, MppiController
from so101.real.constants import SO101_SIM_JOINT_ORDER
from so101.learning.act.data import to_uint8_rgb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--recording", type=Path, required=True, help="aligned NPZ or trusted local episode pickle")
    p.add_argument("--limits-json", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-plans", type=int, default=8)
    p.add_argument("--value-weight", type=float, default=1)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--elites", type=int, default=8)
    a = p.parse_args()
    if a.max_plans < 1:
        p.error("positive max plans required")
    if a.device == "cpu":
        torch.set_num_threads(1)
    model, _ = load_worldmodel(a.checkpoint, a.device)
    limits = json.loads(a.limits_json.read_text())
    if limits["joint_order"] != SO101_SIM_JOINT_ORDER:
        raise ValueError("limits joint order differs from model convention")
    ctrl = MppiController(model, limits["lower_radians"], limits["upper_radians"],
                          limits["max_velocity_rad_s"], MppiConfig(num_samples=a.samples,
                          num_elites=a.elites, iterations=a.iterations, value_weight=a.value_weight))
    if a.recording.suffix == ".npz":
        with np.load(a.recording, allow_pickle=False) as f:
            data = {k: f[k] for k in f.files}
    else:
        with a.recording.open("rb") as f:
            data = pickle.load(f)
    if data.get("action_semantics", "") != "applied_absolute_joint_target_radians":
        raise ValueError("recording must explicitly declare applied absolute joint radians")
    if "timestamps_seconds" in data:
        stamps = data["timestamps_seconds"]
    elif "wall_times" in data:
        stamps = data["wall_times"]
    else:
        raise ValueError("recorded timestamps required; never guess timing across trimmed pauses")
    starts = set(map(int, data.get("episode_starts", [0])))
    rows, rejected = [], []
    for i in range(len(data["actions"])):
        observation = {"joints": data["observations"][i], "front": to_uint8_rgb(torch.as_tensor(data["front_images"][i]), model.config.stored_image_shape),
                       "wrist": to_uint8_rgb(torch.as_tensor(data["wrist_images"][i]), model.config.stored_image_shape), "timestamp_seconds": float(stamps[i]),
                       "remaining_ticks": int(data["remaining_ticks"][i])}
        if i in starts:
            # Startup reference only; no plan until H true recorded actions.
            ctrl.reset(observation, data["observations"][i])
        else:
            try:
                ctrl.update(observation, data["actions"][i-1])
            except ValueError as e:
                rejected.append({"row": i, "reason": str(e)})
                ctrl.reset(observation, data["actions"][i-1])
                continue
        if not ctrl.ready:
            continue
        freshness = {}
        for camera in ("front", "wrist"):
            key = camera + "_timestamps_seconds"
            freshness[camera] = (float(stamps[i]-data[key][i]) if key in data else None)
        if any(age is not None and (age < -0.001 or age > model.config.tick_seconds) for age in freshness.values()):
            rejected.append({"row": i, "reason": "camera frame stale or timestamp clock mismatch", "camera_age_seconds": freshness})
            continue
        plan = ctrl.plan()
        rows.append({"row": i, "proposed_command": plan[0].tolist(), "plan": plan.tolist(),
                     "recorded_command": data["actions"][i].tolist(), **ctrl.last_report,
                     "camera_age_seconds": freshness, "frame_freshness_verified": all(v is not None for v in freshness.values()),
                     "actuated": False})
        if len(rows) >= a.max_plans:
            break
    result = {"recording": str(a.recording.resolve()), "checkpoint": str(a.checkpoint.resolve()),
              "actuated": False, "commands": rows, "rejected_history_rows": rejected,
              "research_status": "offline replay; neither closed-loop success nor real robustness evaluation"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"plans": len(rows), "rejected_history_rows": len(rejected), "actuated": False, "out": str(a.out)}))
    if not rows:
        raise SystemExit("no temporally aligned planning windows; inspect rejected_history_rows")


if __name__ == "__main__":
    main()
