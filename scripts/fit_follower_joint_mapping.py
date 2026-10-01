"""Measure how the real follower's joint readings map onto Isaac joint angles.

Compares three follower mappings (see src/so101/real/joint_mapping.py) on a
wrist-camera hand-eye capture of the fixed table board:

    linear     the current interface mapping (calibrated range stretched onto
               the USD joint limits)
    physical   motor ticks -> physical degrees (LeRobot's use_degrees=True
               scale), range midpoint placed at the USD midpoint
    fitted     physical, plus per-joint offsets fitted here

Score = PARK hand-eye board scatter (mm) + 0.05 * reprojection RMSE (px), the
same physical invariant fit_wrist_joint_mapping.py uses: the table board does
not move, so FK that matches the real arm makes every capture agree on where
it is. Hand-eye is re-solved for every candidate, which also means offsets it
can absorb are not observable here: a constant shoulder_pan offset (a yaw of
the whole arm about the base) and a constant wrist_roll offset (a spin of the
camera about the gripper) are soaked up by the solved transforms. Only
shoulder_lift / elbow_flex / wrist_flex offsets are fitted by default; the
gripper jaw is not in the camera's kinematic chain at all.

`fitted` is also cross-validated -- fitted on even-indexed captures, scored on
odd ones -- so a gain that is only overfitting shows up as such.

No robot is needed; everything comes from records.json + the images. Nothing
is written unless --write is given.

Example:
    python -u scripts/fit_follower_joint_mapping.py --headless
    python -u scripts/fit_follower_joint_mapping.py --headless --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
# simulation_app.close() ends the process without flushing a redirected stdout.
sys.stdout.reconfigure(line_buffering=True)


def _default_calibration() -> Path:
    project = REPO_ROOT / "calibration/robots/so_follower/my_follower.json"
    cache = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json"
    return project if project.is_file() else cache


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--capture-dir", type=Path, default=REPO_ROOT / "outputs/handeye/wrist_fixed")
    parser.add_argument("--calibration", type=Path, default=_default_calibration(), help="follower LeRobot calibration JSON")
    parser.add_argument(
        "--fit-joints",
        default="shoulder_lift,elbow_flex,wrist_flex",
        help="comma-separated joints whose offsets are fitted (pan/roll are not observable here)",
    )
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--initial-step", type=float, default=2.0, help="optimizer's first offset step (deg)")
    parser.add_argument("--stability-deg", type=float, default=2.0, help="max full-vs-half fit disagreement treated as stable")
    parser.add_argument("--write", action="store_true", help="save the fitted mapping to --out")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "calibration/joint_mapping/follower.yaml")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = False

    app = AppLauncher(args).app
    try:
        import cv2
        import gymnasium as gym
        import numpy as np
        import torch
        from scipy.optimize import minimize

        import so101.tasks  # noqa: F401
        from so101.camera_calibration import load_calibration, quat_wxyz_to_matrix
        from so101.charuco import board_object_points, charuco_board, detect_board
        from so101.configs import make_env_cfg
        from so101.handeye import pose_matrix, solve_hand_eye
        from so101.real.joint_mapping import JOINTS, JointMapping, calibrated_sweep_deg, load_lerobot_calibration

        fit_joints = [j.strip() for j in args.fit_joints.split(",") if j.strip()]
        unknown = set(fit_joints) - set(JOINTS)
        if unknown:
            raise SystemExit(f"unknown joints in --fit-joints: {sorted(unknown)}")
        fit_index = [JOINTS.index(j) for j in fit_joints]

        capture = json.loads((args.capture_dir / "records.json").read_text(encoding="utf-8"))
        if capture.get("camera") != "wrist":
            raise SystemExit(
                f"{args.capture_dir} is a {capture.get('camera')!r} capture; this fit needs the wrist "
                "camera looking at the fixed table board (eye-in-hand)"
            )
        camera = load_calibration("wrist")
        board = charuco_board()
        calibration = load_lerobot_calibration(args.calibration)

        raw_rows, target_poses, observations = [], [], []
        for record in capture["records"]:
            image = cv2.imread(str(args.capture_dir / record["image"]))
            if image is None:
                continue
            detection = detect_board(image, board)
            if not detection.usable(minimum=8):
                continue
            object_points = board_object_points(board, detection.charuco_ids)
            image_points = detection.charuco_corners.reshape(-1, 2).astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera.camera_matrix, camera.distortion)
            if not ok:
                continue
            raw_rows.append(np.asarray(record["raw_values"], dtype=np.float64))
            target_poses.append(pose_matrix(cv2.Rodrigues(rvec)[0], tvec))
            observations.append((object_points, image_points))
        if len(target_poses) < 12:
            raise SystemExit(f"need at least 12 usable captures, got {len(target_poses)}")
        raw = np.stack(raw_rows)
        if np.abs(raw[:, :5]).max() > 100.0 + 1e-6:
            raise SystemExit("raw_values exceed +-100: this capture was not recorded with use_degrees=False")

        cfg = make_env_cfg("so101-StackCube-v0", num_envs=1, device=args.device)
        env = gym.make("so101-StackCube-v0", cfg=cfg).unwrapped
        try:
            env.reset()
            robot = env.scene["robot"]
            device = robot.data.joint_pos.device
            env_origin = env.scene.env_origins[0].cpu().numpy()
            gripper_index = list(robot.data.body_names).index("gripper")

            def fk(joints: np.ndarray) -> list[np.ndarray]:
                poses = []
                for q in joints:
                    tensor = torch.tensor(q, dtype=torch.float32, device=device).unsqueeze(0)
                    robot.write_joint_state_to_sim(tensor, torch.zeros_like(tensor))
                    robot.set_joint_position_target(tensor)
                    robot.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(dt=env.physics_dt)
                    position = robot.data.body_pos_w[0, gripper_index].cpu().numpy() - env_origin
                    rotation = quat_wxyz_to_matrix(robot.data.body_quat_w[0, gripper_index].cpu().numpy())
                    poses.append(pose_matrix(rotation, position))
                return poses

            def score(mapping: JointMapping, rows: np.ndarray):
                results = solve_hand_eye(
                    fk(mapping.to_sim(raw[rows])),
                    [target_poses[i] for i in rows],
                    [observations[i] for i in rows],
                    camera.camera_matrix,
                    camera.distortion,
                    eye_in_hand=True,
                )
                park = next((r for r in results if r.method == "PARK"), None)
                if park is None:
                    return float("inf"), None
                return park.rig_scatter_mm + 0.05 * park.reprojection_rmse_px, park

            def fit(rows: np.ndarray) -> np.ndarray:
                """Offset deltas (deg) from the nominal physical mapping."""

                def objective(delta: np.ndarray) -> float:
                    offsets = nominal.offset_deg.copy()
                    offsets[fit_index] += delta
                    return score(JointMapping.physical(calibration, offsets), rows)[0]

                # scipy's default simplex around 0 steps by 0.00025, which is
                # already inside xatol: it would "converge" without moving.
                n = len(fit_index)
                simplex = np.vstack([np.zeros(n), np.eye(n) * args.initial_step])
                result = minimize(
                    objective,
                    np.zeros(n),
                    method="Nelder-Mead",
                    options={"maxiter": args.max_iter, "xatol": 0.05, "fatol": 1e-3, "initial_simplex": simplex},
                )
                return result.x

            def with_delta(delta: np.ndarray) -> JointMapping:
                offsets = nominal.offset_deg.copy()
                offsets[fit_index] += delta
                return JointMapping.physical(calibration, offsets)

            everything = np.arange(len(raw))
            even, odd = everything[::2], everything[1::2]
            linear = JointMapping.linear()
            nominal = JointMapping.physical(calibration)

            print(f"\n=== follower joint mapping ({len(raw)} usable captures from {args.capture_dir}) ===")
            print(f"calibration: {args.calibration}")
            print("calibrated sweep vs USD range (linear mapping scale error):")
            for j, sweep, lin, phys in zip(JOINTS, calibrated_sweep_deg(calibration), linear.scale_deg, nominal.scale_deg):
                print(f"  {j:13s} {sweep:6.1f} deg physical, linear mapping scales it by {lin / phys:5.3f}")

            print(f"\nfitting offsets for {fit_joints} ...")
            delta_all = fit(everything)
            delta_even = fit(even)

            rows = []
            for name, mapping in (("linear", linear), ("physical", nominal), ("fitted", with_delta(delta_all))):
                s, park = score(mapping, everything)
                rows.append((name, s, park))
            print("\nmapping    score   PARK scatter   reproj     (all captures)")
            for name, s, park in rows:
                print(f"  {name:9s} {s:6.2f}   {park.rig_scatter_mm:8.2f} mm  {park.reprojection_rmse_px:6.2f} px")

            print("\ncross-validation: fitted on even captures, scored on odd ones")
            for name, mapping in (("linear", linear), ("physical", nominal), ("fitted(even)", with_delta(delta_even))):
                s, park = score(mapping, odd)
                print(f"  {name:13s} {s:6.2f}   {park.rig_scatter_mm:8.2f} mm  {park.reprojection_rmse_px:6.2f} px")

            fitted = with_delta(delta_all)
            print("\nfitted offsets relative to the nominal physical mapping:")
            unstable = []
            for j, d_all, d_even in zip(fit_joints, delta_all, delta_even):
                flag = abs(d_all - d_even) > args.stability_deg
                if flag:
                    unstable.append(j)
                print(f"  {j:13s} {d_all:+6.2f} deg   (even-half fit {d_even:+6.2f} deg){'   <- UNSTABLE' if flag else ''}")
            if unstable:
                print(
                    f"  warning: {unstable} change by more than {args.stability_deg} deg between the full and "
                    "half-data fits, so these captures do not pin them down. Capture more poses that sweep "
                    "those joints (both directions, board still in view) before trusting --write."
                )
            difference = np.rad2deg(np.abs(fitted.to_sim(raw) - linear.to_sim(raw))).max(axis=0)
            print("\nlargest |fitted - linear| over the captured poses (deg):")
            print("  " + "  ".join(f"{j} {d:.1f}" for j, d in zip(JOINTS, difference)))

            if args.write:
                fitted.save(
                    args.out,
                    fitted_joints=fit_joints,
                    source={"capture_dir": str(args.capture_dir), "calibration": str(args.calibration)},
                    score={name: round(float(s), 3) for name, s, _ in rows},
                )
                print(f"\nwrote {args.out}")
            else:
                print("\nnothing written (pass --write to save the fitted mapping)")
        finally:
            env.close()
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
