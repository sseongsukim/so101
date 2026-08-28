"""Drive an SO-101 simulation task with a physical SO-101 leader arm.

Examples:

    python scripts/teleop_task.py so101-PegInsert-v0 --port /dev/ttyACM0
    python scripts/teleop_task.py --task so101-GearMesh-v0 --print-every 1
    python scripts/teleop_task.py so101-StackCube-v0 --parallel-gripper
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from isaaclab.app import AppLauncher

TASK_NAMES = (
    "so101-PegInsert-v0",
    "so101-GearMesh-v0",
    "so101-NutThread-v0",
    "so101-StackCube-v0",
    "so101-visual-PegInsert-v0",
    "so101-visual-GearMesh-v0",
    "so101-visual-NutThread-v0",
    "so101-visual-StackCube-v0",
)

parser = argparse.ArgumentParser(
    description="Teleoperate an SO-101 task with a physical SO-101 leader arm."
)
parser.add_argument(
    "--parallel-gripper",
    action="store_true",
    help="Teleoperate the parallel-gripper robot using the leader's six-value action.",
)
parser.add_argument(
    "task_name",
    nargs="?",
    choices=TASK_NAMES,
    help="Registered SO-101 environment name.",
)
parser.add_argument(
    "--task",
    "--task-name",
    "--task_name",
    dest="task_option",
    choices=TASK_NAMES,
    help="Registered SO-101 environment name (alternative to the positional argument).",
)
parser.add_argument(
    "--port",
    default=os.getenv("TELEOP_PORT", "/dev/ttyACM0"),
    help="Serial port of the SO-101 leader arm.",
)
parser.add_argument(
    "--robot-id",
    default="my_leader",
    help="LeRobot calibration ID of the leader arm.",
)
parser.add_argument(
    "--print-every",
    type=int,
    default=30,
    help="Print diagnostics every N environment steps (default: 30).",
)
parser.add_argument(
    "--rate",
    type=float,
    default=30.0,
    help="Maximum wall-clock control rate in Hz; 0 disables pacing (default: 30).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

selected_task = args_cli.task_option or args_cli.task_name
if selected_task is None:
    parser.error("a task name is required (positional or via --task)")
if args_cli.print_every < 1:
    parser.error("--print-every must be at least 1")
if args_cli.rate < 0.0:
    parser.error("--rate must be non-negative")

args_cli.enable_cameras = "-visual-" in selected_task
if args_cli.parallel_gripper and args_cli.enable_cameras:
    parser.error("--parallel-gripper currently supports non-visual task scenes only")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

import so101.tasks  # noqa: E402,F401  (registers environments)
from so101.assets import SO101_PARALLEL_CFG, logical_to_parallel_joint_pos  # noqa: E402
from so101.assets.materials import spawn_so101_parallel_viewer_usd  # noqa: E402
from so101.configs import make_env_cfg  # noqa: E402
from so101.real.interface import LeRobotSO101Interface  # noqa: E402


def _first_value(value: Any, default: Any = None) -> Any:
    """Convert a scalar or first vectorized-environment value for logging."""
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return value.reshape(-1)[0].item()
    return value


def _print_diagnostics(step: int, reward: torch.Tensor, info: dict[str, Any]) -> None:
    """Print reward, success, and filtered jaw-to-held contact diagnostics."""
    reward_value = float(_first_value(reward, 0.0))
    phase = int(_first_value(info.get("reward_phase"), 0))
    success = bool(_first_value(info.get("success"), False))
    reward_success = float(_first_value(info.get("reward_success"), 0.0))
    contact = bool(_first_value(info.get("jaw_contact"), False))
    contact_force = float(_first_value(info.get("jaw_contact_force"), 0.0))
    lift_height = float(_first_value(info.get("lift_height"), 0.0))
    has_lifted = bool(_first_value(info.get("has_lifted"), False))
    reach_distance = float(_first_value(info.get("reach_distance"), float("nan")))
    target_distance = float(
        _first_value(info.get("held_target_distance"), float("nan"))
    )

    print(
        f"[TELEOP] step={step:06d} reward={reward_value:7.4f} phase={phase} "
        f"success={success} reward_success={reward_success:.0f} "
        f"jaw_contact={contact} contact_force={contact_force:7.3f}N "
        f"lift={lift_height * 1000.0:7.2f}mm has_lifted={has_lifted} "
        f"reach_dist={reach_distance * 1000.0:7.2f}mm "
        f"target_dist={target_distance * 1000.0:7.2f}mm",
        flush=True,
    )


def teleop_parallel_gripper() -> None:
    """Kinematically teleoperate the parallel robot in the selected task scene."""
    env_cfg = make_env_cfg(selected_task, num_envs=1, device=args_cli.device)
    scene_cfg = env_cfg.scene
    scene_cfg.robot = SO101_PARALLEL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    scene_cfg.robot.spawn.func = spawn_so101_parallel_viewer_usd
    # Existing task sensors target /Robot/jaw, which does not exist on the
    # parallel asset. This path visualizes leader control without evaluating
    # the old single-jaw contact reward.
    scene_cfg.jaw_contact = None

    sim = sim_utils.SimulationContext(env_cfg.sim)
    sim.set_camera_view(eye=(0.82, -0.68, 0.50), target=(0.27, 0.0, 0.10))
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]

    expected_joint_order = [
        "base_link_to_link1",
        "link1_to_link2",
        "link2_to_link3",
        "link3_to_link4",
        "link4_to_link5",
        "left_clamp",
        "right_clamp",
    ]
    if robot.joint_names != expected_joint_order:
        raise RuntimeError(
            "Unexpected parallel joint order. "
            f"Expected {expected_joint_order}, got {robot.joint_names}."
        )

    leader = LeRobotSO101Interface(
        device=robot.device,
        port=args_cli.port,
        id=args_cli.robot_id,
        cameras={},
        fps=30,
        kind="leader",
    )
    leader_connected = False
    try:
        leader.init_device()
        leader.connect()
        leader_connected = True
        print(f"[INFO] Parallel-gripper teleoperation: {selected_task}")
        print("[INFO] Leader gripper -10..100 deg maps linearly to 0..0.037 m.")
        print("[INFO] Press Ctrl+C or close the Isaac Sim window to stop.")

        step = 0
        zero_velocity = torch.zeros((1, 7), device=robot.device)
        control_period = 1.0 / args_cli.rate if args_cli.rate > 0.0 else 0.0
        while simulation_app.is_running():
            step_started = time.perf_counter()
            with torch.inference_mode():
                leader_action = leader.robot.get_action()
                _, logical_action = leader.real_to_sim_obs_processor(leader_action)
                if logical_action.ndim == 1:
                    logical_action = logical_action.unsqueeze(0)
                parallel_action = logical_to_parallel_joint_pos(logical_action)
                robot.write_joint_state_to_sim(parallel_action, zero_velocity)
                sim.forward()
                sim.render()

            step += 1
            if step == 1 or step % args_cli.print_every == 0:
                leader_gripper_deg = float(logical_action[0, 5] * 180.0 / torch.pi)
                opening = float(parallel_action[0, 6])
                print(
                    f"[PARALLEL] step={step:06d} "
                    f"leader_gripper={leader_gripper_deg:7.2f}deg "
                    f"left={-opening:7.4f}m right={opening:7.4f}m",
                    flush=True,
                )

            remaining = control_period - (time.perf_counter() - step_started)
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n[INFO] Parallel-gripper teleoperation stopped by user.")
    finally:
        if leader_connected:
            leader.robot.disconnect()


def teleop_task() -> None:
    env_cfg = make_env_cfg(selected_task, num_envs=1, device=args_cli.device)

    env = gym.make(selected_task, cfg=env_cfg, render_mode=None)
    leader = LeRobotSO101Interface(
        device=env.unwrapped.device,
        port=args_cli.port,
        id=args_cli.robot_id,
        cameras={},
        fps=30,
        kind="leader",
    )
    leader_connected = False

    try:
        env.reset()
        leader.init_device()
        leader.connect()
        leader_connected = True

        actions = env.unwrapped.robot.data.default_joint_pos.clone()
        print(f"[INFO] Teleoperating {selected_task} with leader {args_cli.robot_id}")
        print(f"[INFO] Leader serial port: {args_cli.port}")
        print("[INFO] Press Ctrl+C or close the Isaac Sim window to stop.")
        print(
            "[INFO] Diagnostics: reward/phase/success and jaw-to-held contact/lift/target distance."
        )

        step = 0
        control_period = 1.0 / args_cli.rate if args_cli.rate > 0.0 else 0.0
        while simulation_app.is_running():
            step_started = time.perf_counter()
            with torch.inference_mode():
                leader_action = leader.robot.get_action()
                _, mapped_action = leader.real_to_sim_obs_processor(leader_action)

                actions[:] = mapped_action
                _, reward, _, _, info = env.step(actions)

            step += 1
            if step == 1 or step % args_cli.print_every == 0:
                _print_diagnostics(step, reward, info)

            remaining = control_period - (time.perf_counter() - step_started)
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n[INFO] Teleoperation stopped by user.")
    finally:
        if leader_connected:
            leader.robot.disconnect()
        env.close()


def main() -> None:
    if args_cli.parallel_gripper:
        teleop_parallel_gripper()
    else:
        teleop_task()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
