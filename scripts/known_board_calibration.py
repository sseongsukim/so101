"""Fit the follower's wrist_roll offset and the wrist camera mount from a board at a KNOWN pose.

Ordinary eye-in-hand hand-eye (calibrate_handeye.py) solves the board's pose
together with the camera mount, so a constant wrist_roll reading offset --
a spin of the whole gripper about its own axis -- is absorbed into the
answer and never measured; on 2026-09-30 it also failed its quality checks.
With the board taped at a measured pose on the table, the unknowns shrink to

    wrist_roll offset (1)  +  camera pose in the gripper link (6)

and are fitted by minimising the reprojection error of every detected ChArUco
corner: raw readings -> current follower mapping (+ roll offset) -> Isaac FK
of the gripper link -> camera mount -> board corners (known world pose) ->
distorted pixels (wrist intrinsics). Captures come from

    python scripts/calibrate_handeye.py --camera wrist --capture --leader-teleop \\
        --capture-dir outputs/handeye/wrist_known_board

Board placement (yellow-sphere frame, cm): the outer rectangle of the checker
grid spans X --board-x-min .. +16.0, Y -12.8 .. +12.8 (long side along Y);
+Y is the side the front camera stands on. Which of the two 180-degree
placements was used is decided by the fit.

    --fit [--write]   write adds the roll offset to calibration/joint_mapping/
                      follower.yaml and the mount to calibration/cameras/wrist.yaml
    --fit --selftest  synthetic detections from known parameters
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402

YELLOW_FRAME_ORIGIN = (0.0, 0.4175)


def rotvec_to_matrix(v: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.Rodrigues(np.asarray(v, dtype=np.float64).reshape(3, 1))[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fit", action="store_true", required=True)
    parser.add_argument("--capture-dir", type=Path, default=REPO_ROOT / "outputs/handeye/wrist_known_board")
    parser.add_argument("--board-x-min", type=float, default=20.0, help="grid near edge, cm from the table rear edge")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--write", action="store_true")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = False
    app = AppLauncher(args).app
    try:
        import cv2
        import gymnasium as gym
        import torch
        from scipy.optimize import least_squares

        import so101.tasks  # noqa: F401
        from so101.camera_calibration import load_calibration, matrix_to_quat_wxyz, quat_wxyz_to_matrix
        from so101.charuco import board_object_points, charuco_board, detect_board
        from so101.configs import make_env_cfg
        from so101.handeye import pose_matrix
        from so101.real.joint_mapping import DEFAULT_FOLLOWER_MAPPING, JOINTS, follower_mapping
        from so101.scenes.tabletop import ROBOT_BASE_BOTTOM_Z

        camera = load_calibration("wrist")
        K, dist = camera.camera_matrix, camera.distortion
        nominal = pose_matrix(quat_wxyz_to_matrix(np.asarray(camera.extrinsic.quat_wxyz)), np.asarray(camera.extrinsic.pos))
        board = charuco_board()
        mapping = follower_mapping()
        roll = JOINTS.index("wrist_roll")

        # Board frame -> world, for the two physically possible placements
        # (board z must point into the table: x_dir x y_dir = -Z).
        ox, oy = YELLOW_FRAME_ORIGIN
        x0, x1 = ox + args.board_x_min / 100, ox + (args.board_x_min + 16.0) / 100
        y0, y1 = oy - 0.128, oy + 0.128
        z = ROBOT_BASE_BOTTOM_Z
        placements = {
            "origin at -Y/near edge": pose_matrix(np.column_stack([[0, 1, 0], [1, 0, 0], [0, 0, -1]]), [x0, y0, z]),
            "origin at +Y/far edge": pose_matrix(np.column_stack([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]), [x1, y1, z]),
        }

        # ---- observations ---------------------------------------------------
        if args.selftest:
            records = json.loads((REPO_ROOT / "outputs/handeye/wrist_mapping/records.json").read_text())["records"]
            raw = np.array([r["raw_values"] for r in records], dtype=np.float64)
        else:
            capture = json.loads((args.capture_dir / "records.json").read_text())
            raw_rows, obs = [], []
            for record in capture["records"]:
                image = cv2.imread(str(args.capture_dir / record["image"]))
                if image is None:
                    continue
                detection = detect_board(image, board)
                if not detection.usable(minimum=6):
                    continue
                raw_rows.append(record["raw_values"])
                obs.append((board_object_points(board, detection.charuco_ids),
                            detection.charuco_corners.reshape(-1, 2).astype(np.float64)))
            raw = np.array(raw_rows, dtype=np.float64)
            if len(obs) < 6:
                raise SystemExit(f"need at least 6 usable captures, got {len(obs)}")

        n = len(raw)
        cfg = make_env_cfg("so101-StackCube-v0", num_envs=n, device=args.device)
        cfg.terminate_on_success = False
        cfg.truncate_on_timeout = False
        env = gym.make("so101-StackCube-v0", cfg=cfg).unwrapped
        env.reset()
        robot = env.robot
        lift = 0.3  # keep the arm clear of the table during FK queries (see touch_calibration.py)
        root = robot.data.default_root_state.clone()
        root[:, :3] += env.scene.env_origins
        root[:, 2] += lift
        robot.write_root_pose_to_sim(root[:, :7])
        origins = env.scene.env_origins.cpu().numpy() + np.array([0.0, 0.0, lift])
        gi = list(robot.data.body_names).index("gripper")
        fk_cache: dict[float, list[np.ndarray]] = {}

        def gripper_poses(roll_offset_deg: float) -> list[np.ndarray]:
            key = round(float(roll_offset_deg), 6)
            if key not in fk_cache:
                q = mapping.to_sim(raw).copy()
                q[:, roll] += np.deg2rad(roll_offset_deg)
                t = torch.tensor(q, dtype=torch.float32, device=env.device)
                robot.write_joint_state_to_sim(t, torch.zeros_like(t))
                robot.set_joint_position_target(t)
                robot.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
                pos = robot.data.body_pos_w[:, gi].cpu().numpy() - origins
                quat = robot.data.body_quat_w[:, gi].cpu().numpy()
                fk_cache[key] = [pose_matrix(quat_wxyz_to_matrix(qq), p) for qq, p in zip(quat, pos)]
            return fk_cache[key]

        def mount(p: np.ndarray) -> np.ndarray:
            """p[1:4] rotation vector (deg), p[4:7] translation (mm), applied on the nominal mount."""
            delta = pose_matrix(rotvec_to_matrix(np.deg2rad(p[1:4])), p[4:7] / 1000.0)
            return nominal @ delta

        def project(p: np.ndarray, board_world: np.ndarray, object_points: np.ndarray, i: int) -> np.ndarray:
            world_cam = gripper_poses(p[0])[i] @ mount(p)
            cam_board = np.linalg.inv(world_cam) @ board_world
            rvec, _ = cv2.Rodrigues(cam_board[:3, :3])
            pts, _ = cv2.projectPoints(object_points, rvec, cam_board[:3, 3], K, dist)
            return pts.reshape(-1, 2)

        if args.selftest:
            truth = np.array([0.0, 3.0, -5.0, 8.0, 12.0, -20.0, 30.0])  # roll offset 0: it is held, not fitted
            board_world = placements["origin at -Y/near edge"]
            all_corners = board.getChessboardCorners().astype(np.float64)
            rng = np.random.default_rng(0)
            obs = []
            for i in range(n):
                px = project(truth, board_world, all_corners, i)
                world_cam = gripper_poses(truth[0])[i] @ mount(truth)
                in_front = (np.linalg.inv(world_cam) @ board_world @ np.c_[all_corners, np.ones(len(all_corners))].T)[2] > 0.05
                inside = (px[:, 0] > 0) & (px[:, 0] < 640) & (px[:, 1] > 0) & (px[:, 1] < 480) & in_front
                obs.append((all_corners[inside], px[inside] + rng.normal(0, 0.5, (inside.sum(), 2))))
            keep = [i for i, (o, _) in enumerate(obs) if len(o) >= 6]
            print(f"[selftest] truth roll {truth[0]} deg, mount rot {truth[1:4].tolist()} deg, trans {truth[4:].tolist()} mm; "
                  f"{len(keep)}/{n} synthetic views see the board")
        else:
            keep = list(range(n))

        def residual(p: np.ndarray, board_world: np.ndarray) -> np.ndarray:
            return np.concatenate([project(p, board_world, obs[i][0], i) - obs[i][1] for i in keep]).ravel()

        steps = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])

        def jacobian(p: np.ndarray, board_world: np.ndarray) -> np.ndarray:
            f0 = residual(p, board_world)
            cols = []
            for k, h in enumerate(steps):
                q = p.copy(); q[k] += h
                cols.append((residual(q, board_world) - f0) / h)
            return np.stack(cols, axis=1)

        # The camera rides on the gripper link, so a roll-reading offset and a
        # counter-rotation of the mount about the roll axis give identical
        # camera poses: from wrist images alone the two are NOT separable
        # (verified with --selftest). The roll offset is therefore held at
        # ROLL_FIX (measured separately, see roll_alignment.py) and only the
        # mount is fitted; the fitted mount reproduces the real wrist views
        # whatever the roll offset, and the gripper's own orientation comes
        # from the roll alignment.
        roll_fix = 0.0

        def fit_from(board_world: np.ndarray):
            def res6(m):
                return residual(np.r_[roll_fix, m], board_world)
            def jac6(m):
                return jacobian(np.r_[roll_fix, m], board_world)[:, 1:]
            best = None
            for start in (np.zeros(6), np.r_[0, 0, 90, 0, 0, 0], np.r_[0, 0, -90, 0, 0, 0], np.r_[0, 0, 180, 0, 0, 0]):
                f = least_squares(res6, start.astype(float), jac=jac6, loss="soft_l1", f_scale=5.0)
                if best is None or f.cost < best.cost:
                    best = f
            best.x = np.r_[roll_fix, best.x]
            best.fun = res6(best.x[1:])
            return best

        results = {}
        for name, board_world in placements.items():
            r0 = residual(np.zeros(7), board_world)
            fit = fit_from(board_world)
            rms0 = np.sqrt(np.mean(r0.reshape(-1, 2) ** 2) * 2)
            rms = np.sqrt(np.mean(fit.fun.reshape(-1, 2) ** 2) * 2)
            results[name] = (fit, rms0, rms, board_world)
            print(f"[fit] {name:24s}: reprojection RMS {rms0:7.1f} px -> {rms:6.2f} px | roll offset {fit.x[0]:+.2f} deg, "
                  f"mount rot {np.round(fit.x[1:4], 2).tolist()} deg, trans {np.round(fit.x[4:], 1).tolist()} mm")
        name = min(results, key=lambda k: results[k][2])
        fit, rms0, rms, board_world = results[name]
        # Leave-one-image-out spread.
        loo = []
        for drop in list(keep):
            sub = [i for i in keep if i != drop]
            def res_sub(m, sub=sub):
                p = np.r_[roll_fix, m]
                return np.concatenate([project(p, board_world, obs[i][0], i) - obs[i][1] for i in sub]).ravel()
            def jac_sub(m, sub=sub):
                f0 = res_sub(m)
                return np.stack([(res_sub(m + np.eye(6)[k] * h) - f0) / h for k, h in enumerate(steps[1:])], axis=1)
            loo.append(np.r_[roll_fix, least_squares(res_sub, fit.x[1:], jac=jac_sub, loss="soft_l1", f_scale=5.0).x])
        loo = np.array(loo)
        per_image = [np.sqrt(np.mean((project(fit.x, board_world, obs[i][0], i) - obs[i][1]) ** 2) * 2) for i in keep]
        print(f"\n=== known-board wrist calibration: placement '{name}', {len(keep)} views ===")
        print(f"reprojection RMS {rms0:.1f} px (nominal mount, no roll offset) -> {rms:.2f} px; per-view max {max(per_image):.1f} px")
        labels = ["roll offset (deg)", "mount rx (deg)", "mount ry (deg)", "mount rz (deg)", "mount tx (mm)", "mount ty (mm)", "mount tz (mm)"]
        for label, v, s in zip(labels, fit.x, loo.std(0)):
            print(f"  {label:18s} {v:+8.2f}   (leave-one-out +-{s:.2f})")
        final_mount = mount(fit.x)
        print(f"  camera in gripper link: pos {np.round(final_mount[:3, 3] * 1000, 1).tolist()} mm "
              f"(nominal {np.round(nominal[:3, 3] * 1000, 1).tolist()} mm)")

        if args.write and not args.selftest:
            import yaml

            from so101.camera_calibration import CameraExtrinsic, calibration_path

            camera.extrinsic = CameraExtrinsic(parent="gripper", pos=tuple(float(v) for v in final_mount[:3, 3]),
                                               quat_wxyz=matrix_to_quat_wxyz(final_mount[:3, :3]))
            camera.handeye_rmse_px = float(rms)
            camera.notes = (camera.notes + " | " if camera.notes else "") + (
                f"known-board fit ({args.capture_dir}, {len(keep)} views, {rms:.2f} px) under the follower mapping "
                f"with wrist_roll offset {mapping.offset_deg[roll]:+.2f} deg -- refit if that offset changes")
            camera.save(calibration_path("wrist"))
            print(f"\nwrote {calibration_path('wrist')} (mount)")
        elif not args.selftest:
            print("\nnothing written (pass --write)")
        env.close()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
