"""Compare an Isaac SO-101 pose with the real follower.

The pose supplied on the command line is always Isaac/robot radians in this
order: Rotation Pitch Elbow Wrist_Pitch Wrist_Roll Jaw.  With ``--send`` the
same pose is converted to LeRobot raw values, sent to the follower, and read
back after settling.  Without ``--send`` this is a simulation-only FK check.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SIM_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


def pose_text(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:+.5f}" for value in values) + "]"


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pose", nargs=6, type=float, required=True, metavar=("ROT", "PITCH", "ELBOW", "WPITCH", "WROLL", "JAW"),
        help="six Isaac joint positions in radians",
    )
    parser.add_argument("--send", action="store_true", help="also command the real follower")
    parser.add_argument("--yes", action="store_true", help="skip the real-motion confirmation")
    parser.add_argument("--settle", type=float, default=3.0)
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--robot-id", default="my_follower")
    parser.add_argument("--view", action="store_true", help="show the Isaac viewport")
    parser.add_argument("--view-seconds", type=float, default=20.0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = False
    args.headless = not args.view
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import so101.tasks  # noqa: F401
    from so101.camera_calibration import quat_wxyz_to_matrix
    from so101.configs import make_env_cfg

    target = np.asarray(args.pose, dtype=np.float64)
    cfg = make_env_cfg("so101-StackCube-v0", num_envs=1, device=args.device)
    env = gym.make("so101-StackCube-v0", cfg=cfg).unwrapped
    try:
        env.reset()
        robot = env.scene["robot"]
        device = robot.data.joint_pos.device
        gripper_index = list(robot.data.body_names).index("gripper")
        env_origin = env.scene.env_origins[0].cpu().numpy()
        joints = torch.tensor(target, dtype=torch.float32, device=device).unsqueeze(0)
        robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        robot.set_joint_position_target(joints)
        robot.write_data_to_sim()
        if args.view:
            # Isaac's default editor camera is often zoomed into the robot
            # base.  Set a stable overview programmatically so the gripper is
            # visible without requiring mouse navigation.
            try:
                from omni.kit.viewport.utility import get_active_viewport

                viewport = get_active_viewport()
                viewport.set_camera_view(
                    eye=np.array([0.95, -1.05, 0.75]),
                    target=np.array([0.22, 0.0, 0.25]),
                )
            except Exception as error:  # noqa: BLE001
                print(f"[warn] could not set overview viewport: {error}")
        env.sim.step(render=args.view)
        env.scene.update(dt=env.physics_dt)
        position = robot.data.body_pos_w[0, gripper_index].cpu().numpy() - env_origin
        rotation = quat_wxyz_to_matrix(
            robot.data.body_quat_w[0, gripper_index].cpu().numpy()
        )

        print("\n=== Isaac FK ===")
        print("joint radians:", pose_text(target))
        print("gripper position [m]:", pose_text(position))
        print("gripper rotation:\n", np.array2string(rotation, precision=5, suppress_small=True))

        if args.view:
            print(f"\nIsaac viewport is showing the pose for {args.view_seconds:.1f} seconds.")
            deadline = time.monotonic() + args.view_seconds
            while time.monotonic() < deadline:
                env.sim.step(render=True)
                env.scene.update(dt=env.physics_dt)
                time.sleep(1.0 / 30.0)

        if not args.send:
            print("\nSimulation only. Add --send to command the real follower.")
            return 0

        from so101.real.constants import SO101_JOINT_ORDER
        from so101.real.interface import LeRobotSO101Interface

        if not args.yes:
            answer = input("\nThe real arm will move. Type 'move' to continue: ").strip().lower()
            if answer != "move":
                print("Aborted before commanding the real arm.")
                return 1

        interface = LeRobotSO101Interface(
            device="cpu", port=args.port, id=args.robot_id,
            cameras={}, fps=30, kind="follower",
        )
        interface.init_device()
        interface.connect()
        try:
            raw = interface.get_raw_actions_from_radians(
                torch.tensor(target, dtype=torch.float32)
            )
            action = {
                joint: float(value)
                for joint, value in zip(SO101_JOINT_ORDER, raw.tolist())
            }
            print("raw command:", pose_text(raw.numpy()))
            interface.robot.send_action(action)
            time.sleep(args.settle)
            observation = interface.robot.get_observation()
            actual_raw = torch.tensor(
                [float(observation[joint]) for joint in SO101_JOINT_ORDER],
                dtype=torch.float32,
            )
            actual = interface.get_mapped_actions_vectorized(actual_raw).numpy()
            error = actual - target
            print("\n=== Real observation ===")
            print("actual radians:", pose_text(actual))
            print("error radians :", pose_text(error))
            print("error degrees :", pose_text(np.degrees(error)))
            print("raw observed :", pose_text(actual_raw.numpy()))
        finally:
            interface.robot.disconnect()
        return 0
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
