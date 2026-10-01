"""Fit the wrist pitch/roll readback mapping from an existing hand-eye capture.

The table ChArUco board is fixed.  For every candidate mapping from the
follower's normalised readback values to Isaac radians, this script replays all
recorded poses through Isaac FK, solves hand-eye, and measures how tightly the
resulting board poses agree.  It never writes robot calibration, camera
extrinsics, or the capture records.

This is deliberately limited to wrist pitch and wrist roll: they are the two
large-angle joints that showed a visible sim/real mismatch at capture pose 17.

Example:
    python -u scripts/fit_wrist_joint_mapping.py \\
        --capture-dir outputs/handeye/wrist_fixed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "handeye" / "wrist_fixed",
    )
    parser.add_argument(
        "--coarse-steps",
        type=int,
        default=9,
        help="grid resolution per joint for the initial fit",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = False

    app = AppLauncher(args).app
    try:
        import gymnasium as gym
        import torch

        import so101.tasks  # noqa: F401
        from so101.camera_calibration import load_calibration, quat_wxyz_to_matrix
        from so101.charuco import board_object_points, charuco_board, detect_board
        from so101.configs import make_env_cfg
        from so101.handeye import pose_matrix, solve_hand_eye

        records_path = args.capture_dir / "records.json"
        records = json.loads(records_path.read_text(encoding="utf-8"))["records"]
        calibration = load_calibration("wrist")
        board = charuco_board()

        raw_rows: list[np.ndarray] = []
        base_joint_rows: list[np.ndarray] = []
        target_poses: list[np.ndarray] = []
        observations: list[tuple[np.ndarray, np.ndarray]] = []
        for record in records:
            image = cv2.imread(str(args.capture_dir / record["image"]))
            if image is None:
                continue
            detection = detect_board(image, board)
            if not detection.usable(minimum=8):
                continue
            object_points = board_object_points(board, detection.charuco_ids)
            image_points = detection.charuco_corners.reshape(-1, 2).astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(
                object_points, image_points,
                calibration.camera_matrix, calibration.distortion,
            )
            if not ok:
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            raw_rows.append(np.asarray(record["raw_values"], dtype=np.float64))
            base_joint_rows.append(
                np.asarray(record["joint_positions_rad"], dtype=np.float64)
            )
            target_poses.append(pose_matrix(rotation, tvec))
            observations.append((object_points, image_points))

        if len(target_poses) < 8:
            raise SystemExit(f"need at least 8 usable records, got {len(target_poses)}")
        raw = np.stack(raw_rows)
        base_joints = np.stack(base_joint_rows)

        cfg = make_env_cfg("so101-StackCube-v0", num_envs=1, device=args.device)
        env = gym.make("so101-StackCube-v0", cfg=cfg).unwrapped
        try:
            env.reset()
            robot = env.scene["robot"]
            device = robot.data.joint_pos.device
            env_origin = env.scene.env_origins[0].cpu().numpy()
            gripper_index = list(robot.data.body_names).index("gripper")

            def fk(joints: np.ndarray) -> list[np.ndarray]:
                poses: list[np.ndarray] = []
                for q in joints:
                    tensor = torch.tensor(q, dtype=torch.float32, device=device).unsqueeze(0)
                    robot.write_joint_state_to_sim(tensor, torch.zeros_like(tensor))
                    robot.set_joint_position_target(tensor)
                    robot.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(dt=env.physics_dt)
                    position = (
                        robot.data.body_pos_w[0, gripper_index].cpu().numpy() - env_origin
                    )
                    rotation = quat_wxyz_to_matrix(
                        robot.data.body_quat_w[0, gripper_index].cpu().numpy()
                    )
                    poses.append(pose_matrix(rotation, position))
                return poses

            def evaluate(pitch_scale: float, roll_scale: float):
                joints = base_joints.copy()
                # raw values are LeRobot RANGE_M100_100 values, not degrees.
                # These two coefficients are Isaac degrees per raw unit.
                joints[:, 3] = np.deg2rad(pitch_scale * raw[:, 3])
                joints[:, 4] = np.deg2rad(roll_scale * raw[:, 4])
                results = solve_hand_eye(
                    fk(joints), target_poses, observations,
                    calibration.camera_matrix, calibration.distortion, eye_in_hand=True,
                )
                park = next((item for item in results if item.method == "PARK"), None)
                if park is None:
                    return float("inf"), None
                # Board rigidity is the physical invariant.  Pixel error is a
                # tie-breaker only because the intrinsics are already precise.
                return park.rig_scatter_mm + 0.05 * park.reprojection_rmse_px, park

            current_pitch_scale, current_roll_scale = 0.95, 1.60
            baseline_score, baseline = evaluate(current_pitch_scale, current_roll_scale)
            print("\n=== wrist mapping fit (no files are changed) ===")
            print(f"usable captures: {len(target_poses)}")
            print(
                "current mapping: pitch=%.4f, roll=%.4f deg/raw  |  "
                "PARK scatter=%.2f mm, reproj=%.2f px"
                % (
                    current_pitch_scale, current_roll_scale,
                    baseline.rig_scatter_mm, baseline.reprojection_rmse_px,
                )
            )

            def grid(pitch_values: np.ndarray, roll_values: np.ndarray):
                best = (float("inf"), None, None, None)
                for pitch_scale in pitch_values:
                    for roll_scale in roll_values:
                        score, result = evaluate(float(pitch_scale), float(roll_scale))
                        if score < best[0]:
                            best = (score, float(pitch_scale), float(roll_scale), result)
                return best

            steps = max(3, args.coarse_steps)
            coarse = grid(np.linspace(0.70, 1.20, steps), np.linspace(1.00, 2.10, steps))
            pitch_step = 0.50 / (steps - 1)
            roll_step = 1.10 / (steps - 1)
            refined = grid(
                np.linspace(coarse[1] - pitch_step, coarse[1] + pitch_step, steps),
                np.linspace(coarse[2] - roll_step, coarse[2] + roll_step, steps),
            )
            _, pitch_scale, roll_scale, result = refined
            print(
                "fit candidate:  pitch=%.4f, roll=%.4f deg/raw  |  "
                "PARK scatter=%.2f mm, reproj=%.2f px"
                % (pitch_scale, roll_scale, result.rig_scatter_mm, result.reprojection_rmse_px)
            )
            print(
                "\nDo not apply this automatically. Verify pose 17 and one mid-range "
                "pose visually before changing the project mapping."
            )
        finally:
            env.close()
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
