"""Fit the follower's per-joint angle offsets by touching known points on the table.

The wrist-camera fit (fit_follower_joint_mapping.py) cannot see a constant
shoulder_pan offset and cannot separate lift/elbow/flex offsets, and a
fixed-layout replay (fixed_layout_check.py, 2026-09-30) showed the real gripper
~2 cm away from the simulated one at identical joint readings. Touching
points whose table coordinates are known pins those offsets down: every touch
says "the fingertip is HERE", in the robot's own frame.

    --record  (on site) teleoperate with the leader; close the gripper fully and
              put the meeting point of the two fingertips on each marked point,
              then press SPACE (u = redo the last point, q = finish early).
              Raw follower readings are saved, so the fit can be redone with
              any mapping.
    --fit     (sim, Isaac FK) find offsets for shoulder_pan, shoulder_lift,
              elbow_flex, wrist_flex, plus the height of the simulated
              fingertip reference above the real contact point, minimising the
              distance between the simulated grasp point and each touched point.
              --write adds the offsets to calibration/joint_mapping/follower.yaml.

Points are in the yellow-sphere frame of show_robot_base_frame.py
(origin: table rear edge x robot centre line; +X into the table, +Y along the
width = the robot's left), centimetres, on the table surface.

Example:
    python scripts/touch_calibration.py --record                 # on site
    python -u scripts/touch_calibration.py --fit --headless
    python -u scripts/touch_calibration.py --fit --headless --write
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402

POINTS_CM = [(x, y) for x in (18.0, 25.0, 32.0) for y in (-14.0, 0.0, 14.0)]
YELLOW_FRAME_ORIGIN = (0.0, 0.4175)
OUT = REPO_ROOT / "outputs/touch_calibration"
FIT_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")


def run_record(args) -> None:
    import torch

    torch.set_num_threads(1)
    from pynput.keyboard import Key, Listener

    from so101.real.constants import SO101_HOME_POSE, SO101_JOINT_ORDER
    from so101.real.follower import FollowerArm, Rate, SafetyLimits
    from so101.real.interface import LeRobotSO101Interface

    keys: queue.SimpleQueue[str] = queue.SimpleQueue()

    def on_press(key):
        if key == Key.space:
            keys.put("space")
        elif getattr(key, "char", None) in ("u", "q"):
            keys.put(key.char)

    calib = Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/so_leader"
    project = REPO_ROOT / "calibration/teleoperators/so_leader"
    leader = LeRobotSO101Interface("cpu", args.leader_port, "my_leader", {}, 30, kind="leader",
                                   calibration_dir=project if (project / "my_leader.json").is_file() else calib)
    arm = FollowerArm(limits=SafetyLimits(max_step_rad=0.15))
    listener = Listener(on_press=on_press)
    records: list[dict] = []
    try:
        leader.init_device()
        leader.connect()
        arm.connect()
        listener.start()
        _, target = leader.real_to_sim_obs_processor(leader.robot.get_action())
        arm.move_to(target.float(), seconds=3.0)
        print("[touch] close the gripper fully; touch each point with the tip where the two fingers meet.")
        print("[touch] SPACE = record, u = redo previous, q = finish")
        i = 0
        print(f"[touch] point {i + 1}/{len(POINTS_CM)}: X {POINTS_CM[i][0]:.0f} cm, Y {POINTS_CM[i][1]:+.0f} cm")
        rate = Rate(30.0)
        while i < len(POINTS_CM):
            _, target = leader.real_to_sim_obs_processor(leader.robot.get_action())
            arm.send(target.float())
            arm.read()
            try:
                key = keys.get_nowait()
            except queue.Empty:
                key = None
            if key == "space":
                raw = arm.last_raw_read.numpy().tolist()
                records.append({"point_cm": POINTS_CM[i], "raw_values": raw,
                                "joints_rad": arm.interface.get_mapped_actions_vectorized(arm.last_raw_read).tolist()})
                print(f"[touch]   recorded point {i + 1}")
                i += 1
            elif key == "u" and records:
                records.pop()
                i -= 1
                print(f"[touch]   removed; redo point {i + 1}")
            elif key == "q":
                break
            else:
                rate.sleep()
                continue
            if i < len(POINTS_CM):
                print(f"[touch] point {i + 1}/{len(POINTS_CM)}: X {POINTS_CM[i][0]:.0f} cm, Y {POINTS_CM[i][1]:+.0f} cm")
            rate.sleep()
    finally:
        listener.stop()
        if arm.connected:
            arm.go_home()
            arm.close()
        leader.robot.disconnect()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"touches_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"frame": "yellow sphere (show_robot_base_frame.py)", "records": records}, indent=2))
    print(f"[touch] {len(records)} points -> {path}")


def run_fit(args) -> None:
    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app
    try:
        import gymnasium as gym
        import torch
        from scipy.optimize import least_squares

        import so101.tasks  # noqa: F401
        from so101.configs import make_env_cfg
        from so101.real.joint_mapping import DEFAULT_FOLLOWER_MAPPING, JOINTS, follower_mapping
        from so101.scenes.tabletop import ROBOT_BASE_BOTTOM_Z

        if args.selftest:
            path = "selftest"
            records = None
        else:
            path = args.touches or max(OUT.glob("touches_*.json"))
            records = json.loads(Path(path).read_text())["records"]
        mapping = follower_mapping()
        idx = [JOINTS.index(j) for j in FIT_JOINTS]
        n = 9 if records is None else len(records)
        if n < 5:
            raise SystemExit(f"need at least 5 touches, got {n}")
        cfg = make_env_cfg("so101-StackCube-v0", num_envs=n, device=args.device)
        cfg.terminate_on_success = False
        cfg.truncate_on_timeout = False
        env = gym.make("so101-StackCube-v0", cfg=cfg).unwrapped
        env.reset()
        robot = env.robot
        # Lift the (fixed-base) robot clear of the table for FK queries: at the
        # touch poses the simulated fingers reach into the tabletop, and the
        # contact impulse of the physics step that refreshes the link poses
        # would push the arm by an amount that depends on the previous query.
        lift = 0.3
        root = robot.data.default_root_state.clone()
        root[:, :3] += env.scene.env_origins
        root[:, 2] += lift
        robot.write_root_pose_to_sim(root[:, :7])
        origins = env.scene.env_origins.cpu().numpy() + np.array([0.0, 0.0, lift])

        tip_index = {"midpoint": 0, "fixed": 1, "moving": 2}
        state = {"tip": "midpoint"}

        def tips(q: np.ndarray) -> np.ndarray:
            """Chosen fingertip reference (N, 3) in env coordinates for joints q (N, 6)."""
            tensor = torch.tensor(q, dtype=torch.float32, device=env.device)
            robot.write_joint_state_to_sim(tensor, torch.zeros_like(tensor))
            robot.set_joint_position_target(tensor)
            robot.write_data_to_sim()
            # A physics step (with the pose held as the target) is what refreshes
            # the link poses on the GPU pipeline; sim.forward() alone does not.
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            point = env._get_grasp_points()[tip_index[state["tip"]]]
            return point.cpu().numpy() - origins

        if records is None:
            # Known answer: joint configurations along the teacher's grasp, the
            # grasp points they produce, and readings skewed by known offsets.
            plan = np.load(REPO_ROOT / "outputs/layout_check/plan.npz")
            rng = np.random.default_rng(0)
            q_true = plan["states"][rng.choice(len(plan["states"]), n, replace=False)].astype(np.float64)
            q_true[:, 0] += rng.uniform(-0.4, 0.4, n)          # spread over pan too
            points = tips(q_true)
            true_delta = np.array([2.0, -3.0, 4.0, -2.0])
            q_read = q_true.copy()
            q_read[:, idx] -= np.deg2rad(true_delta)           # what a miscalibrated arm would report
            raw_read = mapping.to_lerobot(q_read)
            records = [{"point_cm": [(p[0] - YELLOW_FRAME_ORIGIN[0]) * 100, (p[1] - YELLOW_FRAME_ORIGIN[1]) * 100],
                        "z_m": float(p[2]), "raw_values": r.tolist()} for p, r in zip(points, raw_read)]
            print(f"[selftest] true offsets {true_delta.tolist()} deg, tip height 0 mm")

        if args.flip_y:
            records = [{**r, "point_cm": [r["point_cm"][0], -r["point_cm"][1]]} for r in records]
            print("[fit] --flip-y: point Y signs mirrored")
        raw = np.array([r["raw_values"] for r in records], dtype=np.float64)
        base_q = mapping.to_sim(raw)  # (N, 6) sim radians under the current mapping
        ox, oy = YELLOW_FRAME_ORIGIN
        targets = np.array([[ox + r["point_cm"][0] / 100, oy + r["point_cm"][1] / 100, r.get("z_m", ROBOT_BASE_BOTTOM_Z)]
                            for r in records])

        def residual(p: np.ndarray) -> np.ndarray:
            q = base_q.copy()
            q[:, idx] += np.deg2rad(p[:4])
            t = targets.copy()
            t[:, 2] += p[4] / 1000.0          # fingertip reference height above contact, mm
            return ((tips(q) - t) * 1000.0).ravel()  # mm

        steps = np.array([0.5, 0.5, 0.5, 0.5, 1.0])  # deg x4, mm

        def jacobian(p: np.ndarray, rows=None) -> np.ndarray:
            # Explicit forward differences: scipy's automatic steps are far
            # below what the float32 physics FK resolves, which reads as a
            # zero gradient for every joint.
            f0 = residual(p) if rows is None else residual(p).reshape(-1, 3)[rows].ravel()
            cols = []
            for k, h in enumerate(steps):
                q = p.copy(); q[k] += h
                f = residual(q) if rows is None else residual(q).reshape(-1, 3)[rows].ravel()
                cols.append((f - f0) / h)
            return np.stack(cols, axis=1)

        results = {}
        for tip in ([args.tip] if args.tip else ["fixed", "moving", "midpoint"]):
            state["tip"] = tip
            before = residual(np.zeros(5)).reshape(-1, 3)
            fit = least_squares(residual, np.zeros(5), jac=jacobian)
            after = fit.fun.reshape(-1, 3)
            results[tip] = (fit, before, after)
            print(f"[fit] tip={tip:8s}: mean |error| {np.linalg.norm(before, axis=1).mean():5.1f} -> "
                  f"{np.linalg.norm(after, axis=1).mean():5.1f} mm, offsets {np.round(fit.x[:4], 2).tolist()} deg, "
                  f"tip height {fit.x[4]:.1f} mm")
        tip = min(results, key=lambda k: np.linalg.norm(results[k][2], axis=1).mean())
        state["tip"] = tip
        fit, before, after = results[tip]
        # Stability: refit on every leave-one-out subset.
        loo = []
        for k in range(len(records)):
            keep = [i for i in range(len(records)) if i != k]
            def res_k(p, keep=keep):
                return residual(p).reshape(-1, 3)[keep].ravel()
            loo.append(least_squares(res_k, fit.x, jac=lambda p, keep=keep: jacobian(p, keep)).x)
        loo = np.array(loo)

        print(f"\n=== touch calibration ({len(records)} points from {path}), best tip reference: {tip} ===")
        print("per-point error (mm, x/y/z) before -> after:")
        for r, b, a in zip(records, before, after):
            print(f"  X {r['point_cm'][0]:4.0f} Y {r['point_cm'][1]:+4.0f}:  {np.round(b, 1).tolist()} (|{np.linalg.norm(b):5.1f}|)  ->  "
                  f"{np.round(a, 1).tolist()} (|{np.linalg.norm(a):5.1f}|)")
        print(f"mean |error|: {np.linalg.norm(before, axis=1).mean():.1f} mm -> {np.linalg.norm(after, axis=1).mean():.1f} mm")
        print("fitted offsets (added to the current mapping), with leave-one-out spread:")
        for name, v, s_ in zip((*FIT_JOINTS, "tip height (mm)"), fit.x, loo.std(0)):
            print(f"  {name:16s} {v:+7.2f}   (+-{s_:.2f})")

        if args.write:
            new_offsets = mapping.offset_deg.copy()
            for j, v in zip(idx, fit.x[:4]):
                new_offsets[j] += v
            new = type(mapping).physical(mapping.calibration, new_offsets)
            new.save(DEFAULT_FOLLOWER_MAPPING, fitted_joints=list(FIT_JOINTS),
                     note=(f"Physical scale; offsets from touch calibration ({len(records)} points, {path}): "
                           f"grasp-point error {np.linalg.norm(before, axis=1).mean():.1f} -> "
                           f"{np.linalg.norm(after, axis=1).mean():.1f} mm. Fitted tip height {fit.x[4]:.1f} mm."),
                     source={"touches": str(path)},
                     score={"mean_error_mm_before": float(np.linalg.norm(before, axis=1).mean()),
                            "mean_error_mm_after": float(np.linalg.norm(after, axis=1).mean())})
            print(f"\nwrote {DEFAULT_FOLLOWER_MAPPING}")
        else:
            print("\nnothing written (pass --write)")
        env.close()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", action="store_true")
    mode.add_argument("--fit", action="store_true")
    parser.add_argument("--touches", type=Path, default=None, help="--fit: touches_*.json (default: newest)")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--flip-y", action="store_true", help="mirror the points' Y (marked on the other side)")
    parser.add_argument("--selftest", action="store_true", help="--fit on synthetic touches with known offsets")
    parser.add_argument("--tip", choices=["fixed", "moving", "midpoint"], default=None,
                        help="fingertip reference that touched the points (default: try all, keep the best fit)")
    parser.add_argument("--leader-port", default="/dev/so101-leader")
    if "--fit" in sys.argv:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    run_record(args) if args.record else run_fit(args)


if __name__ == "__main__":
    main()
