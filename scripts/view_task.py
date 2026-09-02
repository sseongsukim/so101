"""View one of the SO-101 contact-rich manipulation environments.

Examples:

    python scripts/view_task.py --task so101-StackCube-v0
    python scripts/view_task.py --task so101-visual-StackCube-v0
"""

from __future__ import annotations

import argparse
import math
from isaaclab.app import AppLauncher

TASK_NAMES = ("so101-StackCube-v0", "so101-visual-StackCube-v0")

parser = argparse.ArgumentParser(
    description="Render an SO-101 contact-rich manipulation environment."
)
parser.add_argument(
    "task_name",
    nargs="?",
    choices=TASK_NAMES,
    help="Registered SO-101 environment name.",
    default="so101-StackCube-v0",
)
parser.add_argument(
    "--task",
    dest="task_option",
    choices=TASK_NAMES,
    help="Registered SO-101 environment name.",
)
parser.add_argument(
    "--steps",
    type=int,
    default=0,
    help="Exit after this many environment steps; 0 keeps the viewer open.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
selected_task = args_cli.task_option or args_cli.task_name
if selected_task is None:
    parser.error("an environment name is required (positional or via --task)")
args_cli.enable_cameras = selected_task == "so101-visual-StackCube-v0"
# if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
#     args_cli.headless = True
#     print("[INFO] No desktop display detected; starting Isaac Sim in headless mode.")


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import so101.tasks  # noqa: E402,F401  (registers environments)
from so101.configs import make_env_cfg  # noqa: E402


STATE_LAYOUT = (
    ("held_cube_quat_w", 0, 4, "small cube quaternion (w, x, y, z)"),
    ("held_cube_pos_w", 4, 7, "small cube world position (x, y, z)"),
    ("target_cube_quat_w", 7, 11, "large cube quaternion (w, x, y, z)"),
    ("target_cube_pos_w", 11, 14, "large cube world position (x, y, z)"),
    ("target_minus_held_pos_w", 14, 17, "large-cube position minus small-cube position"),
    ("grasp_midpoint_pos_w", 17, 20, "midpoint between the two fingertips"),
    ("gripper_quat_w", 20, 24, "gripper quaternion (w, x, y, z)"),
    ("joint_pos", 24, 30, "six joint positions in radians"),
    ("joint_vel", 30, 36, "six joint velocities in radians/second"),
)


def print_observation_info(observations: dict[str, object], joint_names: list[str]) -> None:
    """Print the keys, runtime shapes, and dtypes returned by the environment."""
    print("[INFO] Observation entries:")
    for name, value in observations.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {name}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            shape = getattr(value, "shape", None)
            dtype = getattr(value, "dtype", type(value).__name__)
            print(f"  - {name}: shape={shape}, dtype={dtype}")

    state = observations.get("state")
    if not isinstance(state, torch.Tensor):
        return

    if state.shape[-1] == 6:
        values = state[0].detach().cpu().tolist()
        print("[INFO] 6-D deployable visual state (values shown for env_0):")
        print(f"  - joint_pos: shape=(6,), values={values}")
        print("      six joint positions in radians")
        print(f"      joint order={joint_names}")
        return
    if state.shape[-1] != 36:
        return

    print("[INFO] 36-D state layout (values shown for env_0):")
    for name, start, end, description in STATE_LAYOUT:
        values = state[0, start:end].detach().cpu().tolist()
        print(
            f"  - state[{start:02d}:{end:02d}] {name}: "
            f"shape=({end - start},), values={values}"
        )
        print(f"      {description}")
        if name in ("joint_pos", "joint_vel"):
            print(f"      joint order={joint_names}")


def view_task() -> None:
    env_cfg = make_env_cfg(selected_task, num_envs=1, device=args_cli.device)

    env = gym.make(
        selected_task,
        cfg=env_cfg,
        render_mode=None,
    )
    reset_result = env.reset()
    observations = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    print(f"[INFO] Loaded environment: {selected_task}")
    print("[INFO] Cubes are exact 2.5 cm and 4 cm Isaac Lab primitives.")
    action_dim = env.unwrapped.cfg.action_space
    state_dim = env.unwrapped.cfg.observation_space["state"]
    print(f"[INFO] Actions are absolute joint-position targets ({action_dim} joints).")
    print(f"[INFO] Observations contain one {state_dim}-D state tensor.")
    if args_cli.enable_cameras:
        print("[INFO] Camera observations: wrist_image and front_image (480x640 RGB).")
    print_observation_info(observations, env.unwrapped.robot.joint_names)

    step_count = 0
    reset_every_steps = (
        max(1, round(2.0 / env.unwrapped.step_dt))
        if "StackCube" in selected_task
        else None
    )
    if reset_every_steps is not None:
        print("[INFO] View-only StackCube pose reset interval: 2 seconds.")
    robot = env.unwrapped.robot
    hold_action = robot.data.default_joint_pos.clone()
    jaw_ids, jaw_names = robot.find_joints("Jaw")
    if len(jaw_ids) != 1:
        raise RuntimeError(
            f"Expected exactly one Jaw joint, found {jaw_names} in {robot.joint_names}."
        )
    jaw_id = jaw_ids[0]
    jaw_lower = float(robot.data.soft_joint_pos_limits[0, jaw_id, 0])
    jaw_upper = float(robot.data.soft_joint_pos_limits[0, jaw_id, 1])
    gripper_period_s = 4.0
    print(
        "[INFO] Jaw repeatedly closes/opens over "
        f"{math.degrees(jaw_lower):.1f}..{math.degrees(jaw_upper):.1f} deg "
        f"with a {gripper_period_s:.1f} s period."
    )
    try:
        while simulation_app.is_running() and (
            args_cli.steps <= 0 or step_count < args_cli.steps
        ):
            elapsed_s = step_count * env.unwrapped.step_dt
            open_fraction = 0.5 * (
                1.0 - math.cos(2.0 * math.pi * elapsed_s / gripper_period_s)
            )
            hold_action[:, jaw_id] = jaw_lower + open_fraction * (jaw_upper - jaw_lower)
            env.step(hold_action)
            step_count += 1
            if reset_every_steps is not None and step_count % reset_every_steps == 0:
                env.reset()
    finally:
        env.close()


def main() -> None:
    view_task()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
