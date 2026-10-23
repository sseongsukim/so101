#!/usr/bin/env python3
"""Check analytic StackCube reward against recorded simulator rewards."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so101.learning.simdist.geometry import stack_cube_dense_reward


def audit(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        required = (
            "next_cube_positions_robot_relative_m", "next_grasp_points_robot_relative_m",
            "table_height_robot_relative_m", "rewards", "reward_valid",
            "geometry_reward_config_json", "episode_starts", "episode_lengths",
        )
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"archive lacks geometry parity fields: {missing}")
        config = json.loads(archive["geometry_reward_config_json"].item())
        predicted, components = stack_cube_dense_reward(
            archive["next_cube_positions_robot_relative_m"],
            archive["next_grasp_points_robot_relative_m"],
            archive["table_height_robot_relative_m"], config,
        )
        target = np.asarray(archive["rewards"], np.float32)
        valid = np.asarray(archive["reward_valid"], bool)
        errors = np.abs(predicted[valid] - target[valid])
        if not len(errors):
            raise ValueError("archive has no valid dense rewards")
        return {
            "archive": str(path.resolve()), "transitions": int(len(target)),
            "valid_dense_rewards": int(valid.sum()),
            "max_abs_error": float(errors.max()), "mae": float(errors.mean()),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "recomputed_success_count": int(components["success"][valid].sum()),
            "recorded_success_reward_count": int(np.isclose(target[valid], config["reward_success"]).sum()),
            "tolerance": 1e-4, "passed": bool(errors.max() <= 1e-4),
            "success_semantics": "analytic reward parity from actual successor simulator cube and grasp-point labels",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
