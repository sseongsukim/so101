"""Put the real and simulated arm in the same pose and compare what the cameras see.

    --real   (on site) ease the follower to the StackCube start pose, grab
             rectified front/wrist frames and the measured joints, then fold
             to the home pose and release torque. Writes <out>/real.npz + PNGs.
    --sim    render the nominal (unrandomized) visual scene with the arm
             teleported to the joints measured by --real; writes a
             real | sim composite per camera.

This is the direct check of the image gap the student has to bridge --
exposure/white balance, table appearance, arm pose -- before training on
synthetic data.

Example:
    python scripts/compare_start_pose_views.py --real --out outputs/start_pose_views
    python -u scripts/compare_start_pose_views.py --sim --headless --out outputs/start_pose_views
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402


def run_real(args) -> None:
    import cv2
    import torch

    from so101.real.constants import SO101_HOME_POSE, SO101_JOINT_ORDER
    from so101.real.follower import START_POSE, CameraThread, FollowerArm

    arm = FollowerArm(args.port, args.robot_id)
    cameras = {}
    try:
        arm.connect()
        cameras = {name: CameraThread(name) for name in ("front", "wrist")}
        print(f"[info] current joints {arm.read().numpy().round(3).tolist()}")
        print(f"[info] moving to start pose {START_POSE.numpy().round(3).tolist()} over {args.seconds} s")
        arm.move_to(START_POSE, seconds=args.seconds)
        # Let the servos settle and the auto-exposure adapt to the new view.
        for _ in range(60):
            arm.send(START_POSE)
            time.sleep(1 / 30)
        joints = arm.read()
        frames = {name: cam.latest() for name, cam in cameras.items()}
        args.out.mkdir(parents=True, exist_ok=True)
        np.savez(args.out / "real.npz", joints=joints.numpy(), raw=arm.last_raw_read.numpy(),
                 command=START_POSE.numpy(), **{name: f for name, f in frames.items()})
        for name, frame in frames.items():
            cv2.imwrite(str(args.out / f"real_{name}.png"), frame[..., ::-1])
        err = np.rad2deg((joints - START_POSE).abs().numpy())
        print(f"[info] measured joints {joints.numpy().round(3).tolist()} (|error| deg {err.round(1).tolist()})")
        print(f"[info] wrote {args.out / 'real.npz'}")
    finally:
        for cam in cameras.values():
            cam.close()
        if arm.connected:
            print("[info] folding to home pose and releasing torque")
            arm.go_home(seconds=args.seconds)
            arm.close()


def run_sim(args) -> None:
    from isaaclab.app import AppLauncher

    args.enable_cameras = True
    app = AppLauncher(args).app
    try:
        import cv2
        import gymnasium as gym
        import torch

        import so101.tasks  # noqa: F401
        from so101.configs import make_env_cfg
        from so101.tasks.render_randomization import add_backdrop

        real = np.load(args.out / "real.npz")
        cfg = make_env_cfg("so101-visual-StackCube-v0", num_envs=1, device=args.device)
        cfg.terminate_on_success = False
        cfg.truncate_on_timeout = False
        cfg.scene.wrist_camera.data_types = ["rgb"]
        cfg.scene.external_camera.data_types = ["rgb"]
        if args.backdrop:
            add_backdrop(cfg.scene)
        env = gym.make("so101-visual-StackCube-v0", cfg=cfg).unwrapped
        env.reset()
        q = torch.tensor(real["joints"], dtype=torch.float32, device=env.device).unsqueeze(0)
        with torch.inference_mode():
            for _ in range(5):
                env.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
                observation, *_ = env.step(q)
        for name, key in (("front", "front_image"), ("wrist", "wrist_image")):
            sim = observation[key][0].clamp(0, 1).mul(255).byte().cpu().numpy()
            both = np.concatenate([real[name], sim], axis=1)
            cv2.imwrite(str(args.out / f"compare_{name}.png"), both[..., ::-1])
            print(f"[info] {name}: real mean RGB {real[name].reshape(-1, 3).mean(0).round(1).tolist()}, "
                  f"sim {sim.reshape(-1, 3).mean(0).round(1).tolist()} -> {args.out / f'compare_{name}.png'}")
        env.close()
    except BaseException:
        # app.close() ends the process before Python would print this.
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--real", action="store_true")
    mode.add_argument("--sim", action="store_true")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/start_pose_views")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--backdrop", action="store_true", help="render with the (gray) backdrop")
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--robot-id", default="my_follower")
    if "--sim" in sys.argv:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    run_real(args) if args.real else run_sim(args)


if __name__ == "__main__":
    main()
