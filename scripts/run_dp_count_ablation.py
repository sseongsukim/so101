#!/usr/bin/env python3
"""Run the nested DP sim/real dataset-size ablation sequentially.

Run from the activated so101-teleop environment on the GPU workstation:
    python -u scripts/run_dp_count_ablation.py

Phase 1 compares 200/400/600 sim episodes with 40 real demos. The best sim
count (mean nominal + rendering-DR success, tie -> fewer episodes) is then
used for the 20/30/40 real-demo comparison. All trials start from scratch.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT = ROOT / "outputs/dp_count_ablation_v1"
SIM_DIR = ROOT / "outputs/synthetic/resip_v3_slow2"
REAL_DIR = ROOT / "outputs/real_demos/v2_remapped"
SEED = 20261002
TRAIN_STEPS = 100_000
SIM_COUNTS = (200, 400, 600)
REAL_COUNTS = (20, 30, 40)
# Preserve the actual v3_slow2 sample mix (102,689 sim / 110,554 total samples).
SIM_FRACTION = 102_689 / 110_554
REAL_FRACTION = 1.0 - SIM_FRACTION


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def inventory(directory: Path, expected: int) -> list[str]:
    names = sorted(p.name for p in directory.glob("trajectory_*.pkl"))
    if len(names) != expected:
        raise RuntimeError(f"{directory}: expected {expected} episodes, found {len(names)}")
    return names


def select() -> dict[str, dict[str, list[str]]]:
    sim = inventory(SIM_DIR, 600)
    real = inventory(REAL_DIR, 40)
    rng = np.random.default_rng(SEED)
    sim = [sim[i] for i in rng.permutation(len(sim))]
    real = [real[i] for i in rng.permutation(len(real))]
    selections: dict[str, dict[str, list[str]]] = {}
    for n in SIM_COUNTS:
        selections[f"sim{n}_real40"] = {"sim": sim[:n], "real": real[:40]}
    # For the second phase, the same nested sim winner and real subsets are used.
    return selections | {f"sim{{winner}}_real{n}": {"sim": [], "real": real[:n]} for n in REAL_COUNTS}


def wait_for_existing_training() -> None:
    """Avoid competing with an already-running training/evaluation process."""
    known = ("scripts/train_dp.py", "scripts/train_act_cotrain.py", "scripts/train_act.py",
             "scripts/eval_dp_sim.py", "scripts/eval_act_sim.py")
    while True:
        result = subprocess.run(["ps", "-eo", "args="], text=True, capture_output=True, check=True)
        active = [line for line in result.stdout.splitlines()
                  if any(token in line for token in known) and "run_dp_count_ablation.py" not in line]
        if not active:
            return
        print(f"[WAIT] found {len(active)} existing train/eval process(es); retry in 5 min", flush=True)
        time.sleep(300)


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("[RUN] " + " ".join(command), flush=True)
    with log.open("a", buffering=1) as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def valid_eval(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text())
        if result.get("episodes") == 40 and "success_rate" in result:
            return result
    except (json.JSONDecodeError, OSError):
        pass
    return None


def trial(name: str, sim_names: list[str], real_names: list[str]) -> dict:
    trial_dir = OUT / name
    trial_dir.mkdir(parents=True, exist_ok=True)
    selection_path = OUT / "selections" / f"{name}.json"
    dump(selection_path, {"sim": sim_names, "real": real_names})
    ckpt = trial_dir / "dp_so101.pt"
    if not ckpt.is_file():
        command = [sys.executable, "-u", "scripts/train_dp.py",
             "--data", f"sim={SIM_DIR}", "--data", f"real={REAL_DIR}",
             "--out", str(trial_dir), "--steps", str(TRAIN_STEPS),
             "--batch-size", "128", "--num-workers", "6", "--amp", "--save-freq", "10000",
             "--seed", str(SEED), "--episode-selection", str(selection_path),
             "--source-fractions", f"sim={SIM_FRACTION:.12f},real={REAL_FRACTION:.12f}"]
        resume = trial_dir / "resume.pt"
        if resume.is_file():
            print(f"[RESUME] {name} from {resume}", flush=True)
            command.extend(["--init-from", str(resume)])
        run(command, trial_dir / "train.log")
    result: dict = {"name": name, "checkpoint": str(ckpt), "steps": TRAIN_STEPS,
                    "sim_episodes": len(sim_names), "real_episodes": len(real_names),
                    "sim_fraction": SIM_FRACTION, "real_fraction": REAL_FRACTION}
    for tag, dr in (("nominal", False), ("render_dr", True)):
        eval_path = trial_dir / f"eval_{tag}_40.json"
        metrics = valid_eval(eval_path)
        if metrics is None:
            cmd = [sys.executable, "-u", "scripts/eval_dp_sim.py", "--headless",
                   "--checkpoint", str(ckpt), "--num-envs", "4", "--num-rounds", "10",
                   "--max-steps", "300", "--seed", str(SEED), "--out", str(eval_path)]
            if dr:
                cmd.append("--render-randomization")
            run(cmd, trial_dir / f"eval_{tag}.log")
            metrics = valid_eval(eval_path)
        if metrics is None:
            raise RuntimeError(f"invalid or incomplete evaluation result: {eval_path}")
        result[tag] = metrics["success_rate"]
        result[tag + "_stacked"] = metrics.get("stacked_rate")
    result["selection_score"] = (result["nominal"] + result["render_dr"]) / 2
    dump(trial_dir / "result.json", result)
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this from the host's activated so101-teleop environment.")
    if not SIM_DIR.is_dir() or not REAL_DIR.is_dir():
        raise FileNotFoundError(f"Missing input directory: {SIM_DIR} or {REAL_DIR}")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another ablation runner holds {OUT / '.lock'}") from exc
        print(f"[INFO] CUDA={torch.cuda.get_device_name(0)}; output={OUT}", flush=True)
        wait_for_existing_training()
        selections = select()
        results = []
        state_path = OUT / "summary.json"
        for n in SIM_COUNTS:
            name = f"sim{n}_real40"
            results.append(trial(name, selections[name]["sim"], selections[name]["real"]))
            dump(state_path, {"phase": "sim_count", "results": results})
        best = sorted(results, key=lambda r: (-r["selection_score"], r["sim_episodes"]))[0]
        winner_n = best["sim_episodes"]
        winner_sim = selections[f"sim{winner_n}_real40"]["sim"]
        real_all = selections[f"sim{winner_n}_real40"]["real"]
        real_results = [best]
        for n in REAL_COUNTS[:2]:
            name = f"sim{winner_n}_real{n}"
            real_results.append(trial(name, winner_sim, real_all[:n]))
            dump(state_path, {"phase": "real_count", "best_sim_count": winner_n,
                              "sim_count_results": results, "real_count_results": real_results})
        # Report 20/30/40 as a consistent nested real-data comparison.
        real_results = sorted(real_results, key=lambda r: r["real_episodes"])
        summary = {
            "status": "complete", "seed": SEED, "training_steps": TRAIN_STEPS,
            "eval_episodes_each": 40, "eval_max_steps": 300,
            "source_sampling_fractions": {"sim": SIM_FRACTION, "real": REAL_FRACTION},
            "best_sim_count_rule": "highest mean of nominal and rendering-DR success; tie -> fewer sim episodes",
            "best_sim_count": winner_n, "sim_count_results": results,
            "real_count_results": real_results,
            "note": "One fixed evaluation seed; use as an ablation screen, not a confidence interval.",
        }
        dump(state_path, summary)
        print("[COMPLETE] " + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
