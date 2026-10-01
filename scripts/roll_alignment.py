"""Measure the follower's wrist_roll reading offset by aligning the jaw with the table edge.

Neither the wrist-camera board fits nor the fingertip touches can observe a
constant wrist_roll offset (the camera and fingers turn with the roll joint;
the touched fingertip sits almost on the roll axis). The gripper's orientation
has to be observed from outside, and the table edge is the reference:

    --record  (on site) teleoperate with the leader; point the gripper roughly
              straight down, open it a little, and turn the roll until the line
              through the two fingertips is parallel to the table's rear edge
              (the Y axis of the yellow-sphere frame) seen from above, with the
              FIXED finger on the front camera's side (+Y); press SPACE.
              Repeat at a few different places (default 3). q = finish.
    --fit     (sim, Isaac FK) find the roll offset that makes the XY projection
              of the fixed->moving fingertip vector parallel to world Y at every
              recorded pose (modulo 180 deg). --write adds it to
              calibration/joint_mapping/follower.yaml.

After changing the roll offset, refit the wrist camera mount
(known_board_calibration.py --fit --write): the mount is fitted under the
roll offset in force.
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

OUT = REPO_ROOT / "outputs/roll_alignment"


def run_record(args) -> None:
    import torch

    torch.set_num_threads(1)
    from pynput.keyboard import Key, Listener

    from so101.real.follower import FollowerArm, Rate, SafetyLimits
    from so101.real.interface import LeRobotSO101Interface

    keys: queue.SimpleQueue[str] = queue.SimpleQueue()

    def on_press(key):
        if key == Key.space:
            keys.put("space")
        elif getattr(key, "char", None) == "q":
            keys.put("q")

    calib = Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/so_leader"
    project = REPO_ROOT / "calibration/teleoperators/so_leader"
    leader = LeRobotSO101Interface("cpu", args.leader_port, "my_leader", {}, 30, kind="leader",
                                   calibration_dir=project if (project / "my_leader.json").is_file() else calib)
    arm = FollowerArm(limits=SafetyLimits(max_step_rad=0.15))
    listener = Listener(on_press=on_press)
    records = []
    try:
        leader.init_device()
        leader.connect()
        arm.connect()
        listener.start()
        _, target = leader.real_to_sim_obs_processor(leader.robot.get_action())
        arm.move_to(target.float(), seconds=3.0)
        print("[roll] gripper roughly straight down, slightly open; fingertip line parallel to the table rear edge,")
        print("[roll] FIXED finger on the front-camera side.")
        print(f"[roll] SPACE = record ({args.count} poses, at different places), q = finish")
        rate = Rate(30.0)
        while len(records) < args.count:
            _, target = leader.real_to_sim_obs_processor(leader.robot.get_action())
            arm.send(target.float())
            arm.read()
            try:
                key = keys.get_nowait()
            except queue.Empty:
                key = None
            if key == "space":
                records.append({"raw_values": arm.last_raw_read.numpy().tolist()})
                print(f"[roll]   recorded {len(records)}/{args.count}")
            elif key == "q":
                break
            rate.sleep()
    finally:
        listener.stop()
        if arm.connected:
            arm.go_home()
            arm.close()
        leader.robot.disconnect()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"roll_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"criterion": "fingertip line parallel to world Y (table rear edge)", "records": records}, indent=2))
    print(f"[roll] {len(records)} poses -> {path}")


def run_fit(args) -> None:
    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app
    try:
        import gymnasium as gym
        import torch
        import yaml

        import so101.tasks  # noqa: F401
        from so101.configs import make_env_cfg
        from so101.real.joint_mapping import DEFAULT_FOLLOWER_MAPPING, JOINTS, follower_mapping

        path = args.records or max(OUT.glob("roll_*.json"))
        records = json.loads(Path(path).read_text())["records"]
        mapping = follower_mapping()
        roll = JOINTS.index("wrist_roll")
        raw = np.array([r["raw_values"] for r in records], dtype=np.float64)
        n = len(raw)
        cfg = make_env_cfg("so101-StackCube-v0", num_envs=n, device=args.device)
        cfg.terminate_on_success = False
        cfg.truncate_on_timeout = False
        env = gym.make("so101-StackCube-v0", cfg=cfg).unwrapped
        env.reset()
        robot = env.robot
        lift = 0.3
        root = robot.data.default_root_state.clone()
        root[:, :3] += env.scene.env_origins
        root[:, 2] += lift
        robot.write_root_pose_to_sim(root[:, :7])

        def angle_errors(offset_deg: float) -> np.ndarray:
            q = mapping.to_sim(raw).copy()
            q[:, roll] += np.deg2rad(offset_deg)
            t = torch.tensor(q, dtype=torch.float32, device=env.device)
            robot.write_joint_state_to_sim(t, torch.zeros_like(t))
            robot.set_joint_position_target(t)
            robot.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            _, fixed, moving = env._get_grasp_points()
            v = (moving - fixed).cpu().numpy()[:, :2]
            # Fixed finger on the +Y (front camera) side: fixed -> moving points
            # along -Y. The full circle is used, so the 180-deg flip that a mere
            # "parallel" criterion cannot tell apart is excluded.
            angle = np.degrees(np.arctan2(v[:, 0], -v[:, 1]))
            return (angle + 180.0) % 360.0 - 180.0

        grid = np.arange(-180.0, 180.0, 0.5)
        cost = [np.sum(angle_errors(g) ** 2) for g in grid]
        best = grid[int(np.argmin(cost))]
        fine = np.arange(best - 1.0, best + 1.0, 0.05)
        best = fine[int(np.argmin([np.sum(angle_errors(g) ** 2) for g in fine]))]
        before, after = angle_errors(0.0), angle_errors(best)
        print(f"\n=== roll alignment ({n} poses from {path}) ===")
        print(f"fingertip-line angle to the table edge, per pose (deg): {np.round(before, 1).tolist()} -> {np.round(after, 1).tolist()}")
        print(f"wrist_roll offset {best:+.2f} deg (per-pose estimates {np.round(-before, 1).tolist()})")
        if args.write:
            offsets = mapping.offset_deg.copy()
            offsets[roll] += best
            meta = yaml.safe_load(open(DEFAULT_FOLLOWER_MAPPING))
            type(mapping).physical(mapping.calibration, offsets).save(
                DEFAULT_FOLLOWER_MAPPING, fitted_joints=sorted(set(meta.get("fitted_joints", []) + ["wrist_roll"])),
                note=meta.get("note", "") + f" wrist_roll offset {best:+.2f} deg from jaw/table-edge alignment ({path}).",
                source={**meta.get("source", {}), "roll_alignment": str(path)}, score=meta.get("score", {}))
            print(f"wrote {DEFAULT_FOLLOWER_MAPPING} -- now refit the wrist mount (known_board_calibration.py --fit --write)")
        else:
            print("nothing written (pass --write)")
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
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--records", type=Path, default=None)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--leader-port", default="/dev/so101-leader")
    if "--fit" in sys.argv:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    run_record(args) if args.record else run_fit(args)


if __name__ == "__main__":
    main()
