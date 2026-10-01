"""Separate kinematic from visual sim-to-real error with one known cube layout.

Put both cubes at measured positions (yellow-sphere frame of
show_robot_base_frame.py: origin on the table's rear edge at the robot's
centre line, +X into the table, +Y along the table width; centimetres; yaw 0
= cube edges parallel to the table edges) and run, in order:

  --plan    (sim)  the ResiP teacher solves that exact layout; its joint-target
                   trajectory is saved (first successful attempt of a batch).
  --replay  (real) the same targets are replayed open loop on the follower,
                   no policy, no images in the loop. If the real gripper closes
                   on the cube, the joint mapping / robot geometry are
                   consistent with the simulator; if it misses, the miss is
                   kinematic. Real joints and camera frames are recorded.
  --render  (sim)  the simulator is rendered with the robot at the joints the
                   REAL arm reported during the replay and the cubes at the
                   layout: real | sim | 50% blend per camera, per chosen tick.
                   Misaligned cube/finger positions at equal joints mean a
                   camera (extrinsic) error; aligned ones mean the policy's
                   failure is about what it sees, not where.

Example:
    python -u scripts/fixed_layout_check.py --plan --headless
    python scripts/fixed_layout_check.py --replay                  # on site
    python -u scripts/fixed_layout_check.py --render --headless
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402

YELLOW_FRAME_ORIGIN = (0.0, 0.4175)   # env xy of show_robot_base_frame.py's frame
OUT = REPO_ROOT / "outputs/layout_check"


def layout_world(args) -> dict[str, tuple[float, float, float]]:
    """Cube -> (env x, env y, yaw rad)."""
    ox, oy = YELLOW_FRAME_ORIGIN
    return {
        "HeldAsset": (ox + args.small[0] / 100, oy + args.small[1] / 100, np.deg2rad(args.small[2])),
        "FixedAsset": (ox + args.large[0] / 100, oy + args.large[1] / 100, np.deg2rad(args.large[2])),
    }


def place_cubes(env, layout) -> None:
    import torch

    for name, asset in (("HeldAsset", env.held_asset), ("FixedAsset", env.fixed_asset)):
        x, y, yaw = layout[name]
        state = asset.data.default_root_state.clone()
        state[:, 0] = x
        state[:, 1] = y
        state[:, 3:7] = torch.tensor([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], device=env.device)
        state[:, 7:] = 0.0
        state[:, :3] += env.scene.env_origins
        asset.write_root_pose_to_sim(state[:, :7])
        asset.write_root_velocity_to_sim(state[:, 7:])
        asset.reset()


def run_plan(args) -> None:
    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app
    try:
        import gymnasium as gym
        import torch

        import so101.tasks  # noqa: F401
        from so101.configs import make_env_cfg
        from so101.learning.resip_teacher import ResiPTeacher
        from so101.tasks.teacher import teacher_observation

        cfg = make_env_cfg("so101-StackCube-v0", num_envs=args.num_envs, device=args.device)
        cfg.terminate_on_success = False
        cfg.truncate_on_timeout = False
        env = gym.make("so101-StackCube-v0", cfg=cfg).unwrapped
        teacher = ResiPTeacher(args.teacher, device=env.device)
        layout = layout_world(args)
        generator = torch.Generator(device=env.device).manual_seed(args.seed)
        with torch.inference_mode():
            env.reset(seed=args.seed)
            place_cubes(env, layout)
            hold = env.robot.data.default_joint_pos.clone()
            for _ in range(3):
                env.step(hold)
            teacher.reset()
            actions, states = [], []
            first_success = torch.full((env.num_envs,), -1, device=env.device)
            for step in range(env.max_episode_length):
                states.append(env.robot.data.joint_pos.cpu().numpy().copy())
                action = teacher.act(teacher_observation(env), generator=generator)
                actions.append(action.cpu().numpy().copy())
                _, _, _, _, info = env.step(action)
                new = info["success"].bool() & (first_success < 0)
                first_success[new] = step
        first = first_success.cpu().numpy()
        ok = np.flatnonzero(first >= 0)
        print(f"[plan] teacher solved the layout in {len(ok)}/{env.num_envs} attempts")
        if not len(ok):
            raise SystemExit("no successful attempt; move the cubes or raise --num-envs")
        i = int(ok[np.argmin(first[ok])])
        end = int(first[i]) + args.post_success
        acts = np.stack(actions)[: end + 1, i]
        jaw = np.rad2deg(acts[:, 5])
        close = int(np.argmax((jaw[1:] < 0) & (jaw[:-1] >= 0)) + 1) if (jaw < 0).any() else -1
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / "plan.npz", actions=acts, states=np.stack(states)[: end + 1, i],
                 success_step=int(first[i]), close_step=close, layout=json.dumps({k: list(v) for k, v in layout.items()}),
                 args=json.dumps({"small": args.small, "large": args.large}))
        print(f"[plan] attempt {i}: success at step {first[i]}, gripper closes at step {close}, "
              f"{len(acts)} steps saved -> {OUT / 'plan.npz'}")
        env.close()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


def run_replay(args) -> None:
    import torch

    torch.set_num_threads(1)
    from so101.real.cameras import METER_TARGETS
    from so101.real.constants import SO101_HOME_POSE, SO101_JOINT_ORDER
    from so101.real.follower import START_POSE, CameraThread, FollowerArm, Rate

    plan = np.load(OUT / "plan.npz")
    acts = torch.from_numpy(plan["actions"]).float()
    print(f"[replay] {len(acts)} steps; small cube at X {args.small[0]} cm, Y {args.small[1]:+} cm; "
          f"large at X {args.large[0]} cm, Y {args.large[1]:+} cm (yellow frame)")
    if json.loads(str(plan["args"])) != {"small": args.small, "large": args.large}:
        raise SystemExit("the plan was made for a different layout; re-run --plan with the same --small/--large")
    if not args.yes and input("Cubes placed as above and the area clear? Type MOVE: ").strip() != "MOVE":
        print("aborted")
        return
    arm = FollowerArm()
    cams = {}
    joints, frames_front, frames_wrist, sent, raw = [], [], [], [], []
    try:
        arm.connect()
        arm.move_to(START_POSE, seconds=3.0)
        cams = {n: CameraThread(n, METER_TARGETS[n]) for n in ("front", "wrist")}
        arm.command.reset(arm.read())
        rate = Rate(30.0)
        for target in acts:
            joints.append(arm.read().numpy())
            raw.append(arm.last_raw_read.numpy())
            frames_front.append(cams["front"].latest())
            frames_wrist.append(cams["wrist"].latest())
            sent.append(arm.send(target).numpy())
            rate.sleep()
        for _ in range(30):  # hold the final target briefly
            arm.send(acts[-1])
            rate.sleep()
        print(f"[replay] loop overruns {rate.overruns}")
        grasped = input("Did the gripper pick the small cube up and put it on the large one? [y/N]: ").strip().lower() == "y"
    finally:
        for c in cams.values():
            c.close()
        if arm.connected:
            arm.go_home()
            arm.close()
    joints = np.stack(joints)
    np.savez(OUT / "real_replay.npz", joints=joints, raw=np.stack(raw), sent=np.stack(sent), front=np.stack(frames_front),
             wrist=np.stack(frames_wrist), grasped=grasped)
    close = int(plan["close_step"])
    err = np.rad2deg(joints - plan["actions"])
    print(f"[replay] saved {OUT / 'real_replay.npz'}; grasp by eye: {'yes' if grasped else 'no'}")
    print(f"[replay] real joint - planned target at the close step ({close}): "
          + "  ".join(f"{n} {v:+.1f}" for n, v in zip(("pan", "lift", "elbow", "flex", "roll", "jaw"), err[close])))


def run_render(args) -> None:
    from isaaclab.app import AppLauncher

    args.enable_cameras = True
    app = AppLauncher(args).app
    try:
        import cv2
        import gymnasium as gym
        import torch

        import so101.tasks  # noqa: F401
        from so101.configs import make_env_cfg

        plan = np.load(OUT / "plan.npz")
        real = dict(np.load(OUT / "real_replay.npz"))
        from so101.real.joint_mapping import JointMapping, follower_calibration_path, follower_mapping, load_lerobot_calibration

        if "raw" not in real:
            # Replays recorded before raw values were saved were read through
            # the nominal physical mapping; invert it to get the raw readings.
            real["raw"] = JointMapping.physical(load_lerobot_calibration(follower_calibration_path())).to_lerobot(
                real["joints"].astype(np.float64))
            print("[render] raw readings reconstructed through the nominal physical mapping")
        real["joints"] = follower_mapping().to_sim(real["raw"].astype(np.float64)).astype(np.float32)
        layout = {k: tuple(v) for k, v in json.loads(str(plan["layout"])).items()}
        close = int(plan["close_step"])
        ticks = sorted({0, *range(0, close + 1, max(close // 4, 1)), max(close - 3, 0), close})
        cfg = make_env_cfg("so101-visual-StackCube-v0", num_envs=1, device=args.device)
        cfg.terminate_on_success = False
        cfg.truncate_on_timeout = False
        cfg.scene.wrist_camera.data_types = ["rgb"]
        cfg.scene.external_camera.data_types = ["rgb"]
        env = gym.make("so101-visual-StackCube-v0", cfg=cfg).unwrapped
        with torch.inference_mode():
            env.reset()
            rows = []
            for t in ticks:
                place_cubes(env, layout)  # before contact, the cubes are where they were placed
                q = torch.tensor(real["joints"][t], dtype=torch.float32, device=env.device).unsqueeze(0)
                for _ in range(3):
                    env.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
                    observation, *_ = env.step(q)
                tiles = []
                for key, name in (("front_image", "front"), ("wrist_image", "wrist")):
                    sim = observation[key][0].clamp(0, 1).mul(255).byte().cpu().numpy()
                    rl = real[name][t]
                    blend = (0.5 * rl.astype(np.float32) + 0.5 * sim.astype(np.float32)).astype(np.uint8)
                    tiles.append(np.concatenate([rl, sim, blend], axis=1))
                row = np.concatenate(tiles, axis=1)
                cv2.putText(row, f"tick {t}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                rows.append(cv2.resize(row, None, fx=0.5, fy=0.5))
                cv2.imwrite(str(OUT / f"render_tick{t:03d}.png"), row[..., ::-1])
            cv2.imwrite(str(OUT / "render_summary.png"), np.concatenate(rows, axis=0)[..., ::-1])
        print(f"[render] ticks {ticks} -> {OUT}/render_*.png (real | sim | blend: front, then wrist)")
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
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--replay", action="store_true")
    mode.add_argument("--render", action="store_true")
    parser.add_argument("--small", nargs=3, type=float, default=[28.0, 8.0, 0.0], metavar=("X_CM", "Y_CM", "YAW_DEG"))
    parser.add_argument("--large", nargs=3, type=float, default=[28.0, -12.0, 0.0], metavar=("X_CM", "Y_CM", "YAW_DEG"))
    parser.add_argument("--teacher", type=Path, default=REPO_ROOT / "outputs/teachers/resip_sd042_20260924_210715_e2000")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--post-success", type=int, default=20)
    parser.add_argument("--yes", action="store_true")
    if "--plan" in sys.argv or "--render" in sys.argv:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    run_plan(args) if args.plan else run_replay(args) if args.replay else run_render(args)


if __name__ == "__main__":
    main()
