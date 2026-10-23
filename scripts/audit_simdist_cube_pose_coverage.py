#!/usr/bin/env python3
"""Report per-episode cube-pose range and motion diagnostics from aligned shards.

This is descriptive only: it never filters or edits trajectories. The 1 m
coordinate flag is deliberately a gross-anomaly sentinel, not a task validity
gate or a training mask.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Scene contract from src/so101/scenes/tabletop.py (translation-only robot frame).
TABLE_X = (-0.0377, 0.70 - 0.0377)
TABLE_Y = (-0.3967, 1.20 - 0.3967)
TABLE_TOP_Z = 0.0300814467
DT = 1.0 / 30.0
HALF_SIZES = np.array([0.0125, 0.0200], dtype=np.float64)


def audit(path: Path, gross_coordinate_m: float = 1.0) -> list[dict]:
    with np.load(path, allow_pickle=False) as data:
        positions = data["cube_positions_robot_relative_m"]
        successors = data["next_cube_positions_robot_relative_m"]
        result = []
        for index, (start, length, seed, env) in enumerate(zip(
                data["episode_starts"], data["episode_lengths"],
                data["episode_seeds"], data["episode_env_indices"])):
            start, length = int(start), int(length)
            stop = start + length
            current = positions[start:stop]
            next_pose = successors[start:stop]
            magnitude = np.maximum(np.max(np.abs(current), axis=(1, 2)),
                                   np.max(np.abs(next_pose), axis=(1, 2)))
            held_step = np.linalg.norm(next_pose[:, 0] - current[:, 0], axis=-1)
            velocity = (next_pose - current) / DT
            acceleration = np.diff(velocity, axis=0) / DT if length > 1 else np.zeros((0, 2, 3))
            held_xy = current[:, 0, :2]
            outside = ((held_xy[:, 0] < TABLE_X[0]) | (held_xy[:, 0] > TABLE_X[1])
                       | (held_xy[:, 1] < TABLE_Y[0]) | (held_xy[:, 1] > TABLE_Y[1]))
            penetrating = current[:, 0, 2] < TABLE_TOP_Z - HALF_SIZES[0]
            table_event = np.flatnonzero(outside | penetrating)
            freefall_fit = None
            if len(table_event) and length - int(table_event[0]) >= 6:
                event_index = int(table_event[0])
                t = np.arange(length - event_index, dtype=np.float64) * DT
                z = current[event_index:, 0, 2].astype(np.float64)
                coefficients = np.polyfit(t, z, 2)
                residual = z - np.polyval(coefficients, t)
                freefall_fit = {
                    "samples": len(t), "estimated_g_m_s2": float(-2 * coefficients[0]),
                    "z_fit_rmse_mm": float(np.sqrt(np.mean(residual ** 2)) * 1000),
                    "z_fit_p95_abs_residual_mm": float(np.quantile(np.abs(residual), .95) * 1000),
                    "xy_speed_m_s_p50_p95": np.quantile(
                        np.linalg.norm(velocity[event_index:, 0, :2], axis=-1), [.5, .95]
                    ).tolist(),
                }
            offending = np.flatnonzero(magnitude[:length] > gross_coordinate_m)
            result.append({
                "archive": str(path), "episode_index": index,
                "seed": int(seed), "env_index": int(env), "transitions": length,
                "cube_xyz_min_m": current.min(axis=(0, 1)).tolist(),
                "cube_xyz_max_m": current.max(axis=(0, 1)).tolist(),
                "max_abs_cube_coordinate_m": float(magnitude.max()),
                "first_gross_coordinate_tick": (int(data["ticks"][start + offending[0]])
                                                  if len(offending) else None),
                "gross_coordinate_transition_count": int(len(offending)),
                "first_outside_table_xy_tick": (int(data["ticks"][start + np.flatnonzero(outside)[0]])
                                                 if outside.any() else None),
                "first_below_table_support_tick": (
                    int(data["ticks"][start + np.flatnonzero(penetrating)[0]])
                    if penetrating.any() else None
                ),
                "held_cube_speed_m_s_p50_p95_max": [
                    float(v) for v in np.quantile(np.linalg.norm(velocity[:, 0], axis=-1), [.5, .95])
                ] + [float(np.linalg.norm(velocity[:, 0], axis=-1).max())],
                "held_cube_acceleration_m_s2_p50_p95_max": (
                    [float(v) for v in np.quantile(np.linalg.norm(acceleration[:, 0], axis=-1), [.5, .95])]
                    + [float(np.linalg.norm(acceleration[:, 0], axis=-1).max())]
                    if len(acceleration) else [0.0, 0.0, 0.0]
                ),
                "post_table_event_z_quadratic_fit": freefall_fit,
                "held_cube_step_mm_p50_p95_max":
                    (np.quantile(held_step, [.5, .95]) * 1000).tolist()
                    + [float(held_step.max() * 1000)],
                "fixed_cube_xyz_range_m": [
                    current[:, 1].min(axis=0).tolist(), current[:, 1].max(axis=0).tolist()
                ],
            })
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True, help="aligned NPZ shards")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gross-coordinate-m", type=float, default=1.0,
                        help="descriptive anomaly sentinel; does not filter data")
    args = parser.parse_args()
    if not np.isfinite(args.gross_coordinate_m) or args.gross_coordinate_m <= 0:
        raise ValueError("gross-coordinate-m must be finite and positive")
    episodes = [row for path in args.data for row in audit(path, args.gross_coordinate_m)]
    report = {
        "diagnostic_only": True,
        "gross_coordinate_sentinel_m": args.gross_coordinate_m,
        "episodes": episodes,
        "gross_coordinate_episode_count": sum(row["first_gross_coordinate_tick"] is not None
                                               for row in episodes),
        "total_episodes": len(episodes),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
