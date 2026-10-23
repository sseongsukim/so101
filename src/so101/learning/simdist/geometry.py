"""Pure geometry reward used by SO101 StackCube reward parity audits."""
from __future__ import annotations

import numpy as np


def stack_cube_dense_reward(
    cube_positions: np.ndarray,
    grasp_points: np.ndarray,
    table_height: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Recompute task reward from successor cube/fingertip geometry.

    `cube_positions` order is [held, fixed], `grasp_points` is [midpoint,
    fixed fingertip, moving fingertip], and every XYZ uses robot-root frame.
    `grasp_points` must come from FK for the deployed model; simulator labels
    can first be used to validate the task geometry independently.
    """
    cubes = np.asarray(cube_positions, dtype=np.float64)
    grasp = np.asarray(grasp_points, dtype=np.float64)
    table = np.asarray(table_height, dtype=np.float64)
    if cubes.ndim != 3 or cubes.shape[1:] != (2, 3):
        raise ValueError("cube_positions must have shape (N, 2, 3)")
    if grasp.shape != (len(cubes), 3, 3) or table.shape != (len(cubes),):
        raise ValueError("grasp_points/table_height must have shapes (N,3,3)/(N,)")
    if not np.isfinite(cubes).all() or not np.isfinite(grasp).all() or not np.isfinite(table).all():
        raise ValueError("geometry inputs must be finite")

    held, fixed = cubes[:, 0], cubes[:, 1]
    midpoint, fixed_tip, moving_tip = grasp[:, 0], grasp[:, 1], grasp[:, 2]
    size_a = float(config["small_cube_size_m"])
    size_b = float(config["large_cube_size_m"])
    gain = float(config["distance_gain"])
    held_height = held[:, 2] - table
    d = np.linalg.norm(held - midpoint, axis=-1)
    d_lf = np.linalg.norm(held - fixed_tip, axis=-1)
    d_rf = np.linalg.norm(held - moving_tip, axis=-1)
    distance_reward = 1.0 - np.tanh(gain * (d + d_lf + d_rf) / 3.0)
    lifted = (held_height - size_a) > float(config["lift_clearance_m"])
    cube_delta = fixed - held
    target_offset = np.zeros_like(cube_delta)
    target_offset[:, 2] = (size_a + size_b) / 2.0
    target_distance = np.linalg.norm(cube_delta + target_offset, axis=-1)
    align_reward = (1.0 - np.tanh(gain * target_distance)) * lifted.astype(np.float64)
    distance_reward = np.maximum(distance_reward, align_reward)
    target_height = size_b + size_a / 2.0
    aligned = np.all(np.abs(cube_delta[:, :2]) < float(config["success_xy_threshold_m"]), axis=-1)
    on_top = np.abs(held_height - target_height) < float(config["success_height_threshold_m"])
    gripper_away = d > float(config["gripper_away_threshold_m"])
    success = aligned & on_top & gripper_away
    reward = np.where(
        success,
        float(config["reward_success"]),
        float(config["reward_distance_weight"]) * distance_reward
        + float(config["reward_lift_weight"]) * lifted.astype(np.float64)
        + float(config["reward_align_weight"]) * align_reward,
    )
    return reward.astype(np.float32), {
        "distance_reward": distance_reward.astype(np.float32),
        "lifted": lifted,
        "align_reward": align_reward.astype(np.float32),
        "success": success,
    }
