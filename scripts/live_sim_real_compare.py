"""Live side-by-side of the real cameras and the simulated ones at the real arm's pose.

Teleoperate the follower with the leader; every refresh, the follower's
readings go through the follower joint mapping in force
(calibration/joint_mapping/follower.yaml) onto the simulated robot, both
simulated cameras are rendered, and one window shows

    front:  real | sim | 50% blend
    wrist:  real | sim | 50% blend

so camera and kinematic calibration can be judged by eye at any pose. The
cubes are placed in the simulation at --small / --large (yellow-sphere frame,
cm; +Y = the front camera's side); put the real cubes at the same spots.

Keys (in the window): s = save a snapshot, q/Esc = quit. The arm is folded to
its rest pose on exit.

Run from the desktop session (the window needs a display):
    python -u scripts/live_sim_real_compare.py --headless
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)

YELLOW_FRAME_ORIGIN = (0.0, 0.4175)


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--small", nargs=3, type=float, default=[28.0, 8.0, 0.0], metavar=("X_CM", "Y_CM", "YAW_DEG"))
    parser.add_argument("--large", nargs=3, type=float, default=[28.0, -12.0, 0.0], metavar=("X_CM", "Y_CM", "YAW_DEG"))
    parser.add_argument("--no-cubes", action="store_true", help="leave the simulated cubes out of view")
    parser.add_argument("--meter", action="store_true", help="meter the real cameras to the training brightness")
    parser.add_argument("--hz", type=float, default=10.0, help="refresh rate of the comparison")
    parser.add_argument("--leader-port", default="/dev/so101-leader")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/live_compare")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app
    arm = leader = None
    cams = {}
    window = None
    try:
        import cv2
        import gymnasium as gym
        import numpy as np
        import torch

        import so101.tasks  # noqa: F401
        from so101.configs import make_env_cfg
        from so101.real.cameras import METER_TARGETS
        from so101.real.follower import CameraThread, FollowerArm, Rate, SafetyLimits
        from so101.real.interface import LeRobotSO101Interface
        from so101.real.preview import PreviewWindow

        cfg = make_env_cfg("so101-visual-StackCube-v0", num_envs=1, device=args.device)
        cfg.terminate_on_success = False
        cfg.truncate_on_timeout = False
        cfg.scene.wrist_camera.data_types = ["rgb"]
        cfg.scene.external_camera.data_types = ["rgb"]
        env = gym.make("so101-visual-StackCube-v0", cfg=cfg).unwrapped
        env.reset()

        def place_cubes() -> None:
            ox, oy = YELLOW_FRAME_ORIGIN
            for spec, asset in ((args.small, env.held_asset), (args.large, env.fixed_asset)):
                state = asset.data.default_root_state.clone()
                if args.no_cubes:
                    state[:, 0] -= 5.0  # far outside every camera's view
                else:
                    state[:, 0] = ox + spec[0] / 100
                    state[:, 1] = oy + spec[1] / 100
                    yaw = np.deg2rad(spec[2])
                    state[:, 3:7] = torch.tensor([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], device=env.device)
                state[:, 7:] = 0.0
                state[:, :3] += env.scene.env_origins
                asset.write_root_pose_to_sim(state[:, :7])
                asset.write_root_velocity_to_sim(state[:, 7:])

        calib = Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/so_leader"
        project = REPO_ROOT / "calibration/teleoperators/so_leader"
        leader = LeRobotSO101Interface("cpu", args.leader_port, "my_leader", {}, 30, kind="leader",
                                       calibration_dir=project if (project / "my_leader.json").is_file() else calib)
        leader.init_device()
        leader.connect()
        arm = FollowerArm(limits=SafetyLimits(max_step_rad=0.15))
        arm.connect()
        _, target = leader.real_to_sim_obs_processor(leader.robot.get_action())
        arm.move_to(target.float(), seconds=3.0)
        cams = {n: CameraThread(n, METER_TARGETS[n] if args.meter else None) for n in ("front", "wrist")}
        window = PreviewWindow("real | sim | blend  (s = snapshot, q = quit)", size=(1440, 720))
        args.out.mkdir(parents=True, exist_ok=True)
        print("[live] teleoperate with the leader; s = snapshot, q = quit")

        control = Rate(30.0)
        render_every = max(int(round(30.0 / args.hz)), 1)
        tick = 0
        while True:
            _, target = leader.real_to_sim_obs_processor(leader.robot.get_action())
            arm.send(target.float())
            joints = arm.read()
            if tick % render_every == 0:
                with torch.inference_mode():
                    place_cubes()  # cubes stay where the real ones are, whatever the sim arm touches
                    q = joints.to(env.device).unsqueeze(0)
                    for _ in range(2):
                        env.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
                        observation, *_ = env.step(q)
                rows = []
                for key, name in (("front_image", "front"), ("wrist_image", "wrist")):
                    sim = observation[key][0].clamp(0, 1).mul(255).byte().cpu().numpy()
                    real = cams[name].latest()
                    blend = (0.5 * real.astype(np.float32) + 0.5 * sim.astype(np.float32)).astype(np.uint8)
                    rows.append(np.concatenate([real, sim, blend], axis=1))
                frame = np.concatenate(rows, axis=0)                       # RGB
                deg = np.rad2deg(joints.numpy())
                label = "joints (deg): " + "  ".join(f"{v:+6.1f}" for v in deg)
                shown = np.ascontiguousarray(frame[..., ::-1])              # BGR for drawing/showing
                cv2.putText(shown, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                window.show(shown)
                key = window.poll_key()
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    path = args.out / f"compare_{time.strftime('%H%M%S')}.png"
                    cv2.imwrite(str(path), shown)
                    print(f"[live] saved {path}")
            tick += 1
            control.sleep()
        env.close()
    except KeyboardInterrupt:
        pass
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        for c in cams.values():
            c.close()
        if window is not None:
            window.close()
        if arm is not None and arm.connected:
            arm.go_home()
            arm.close()
        if leader is not None:
            try:
                leader.robot.disconnect()
            except Exception:
                pass
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
