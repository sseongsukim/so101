"""View one of the SO-101 contact-rich manipulation environments.

Examples:

    python scripts/view_task.py --task so101-PegInsert-v0
    python scripts/view_task.py so101-GearMesh-v0
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

TASK_NAMES = (
    "so101-PegInsert-v0",
    "so101-GearMesh-v0",
    "so101-NutThread-v0",
    "so101-visual-PegInsert-v0",
    "so101-visual-GearMesh-v0",
    "so101-visual-NutThread-v0",
)

parser = argparse.ArgumentParser(
    description="Render an SO-101 contact-rich manipulation environment."
)
parser.add_argument(
    "task_name",
    nargs="?",
    choices=TASK_NAMES,
    help="Registered SO-101 environment name.",
    default="so101-PegInsert-v0",
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
args_cli.enable_cameras = "-visual-" in selected_task
# if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
#     args_cli.headless = True
#     print("[INFO] No desktop display detected; starting Isaac Sim in headless mode.")


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import so101.tasks  # noqa: E402,F401  (registers environments)


def main() -> None:
    cfg_entry = gym.spec(selected_task).kwargs["env_cfg_entry_point"]
    env_cfg = cfg_entry()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device

    env = gym.make(
        selected_task,
        cfg=env_cfg,
        # AppLauncher controls GUI/headless mode; no Gym render output is needed.
        render_mode=None,
    )

    ob = env.reset()
    print(f"[INFO] Loaded environment: {selected_task}")
    if "PegInsert" in selected_task or "NutThread" in selected_task:
        print("[INFO] Task assets use their original Isaac Factory scale (100%).")
    else:
        print("[INFO] Factory assets are uniformly scaled to 75%.")
    print(
        "[INFO] Actions are absolute SO-101 joint-position targets in radians (6 joints)."
    )
    if "-visual-" in selected_task:
        print(
            "[INFO] Policy observations include proprio, rgb_wrist, and rgb_external."
        )
    else:
        print(
            "[INFO] Policy and critic observations share the 38-D task state; "
            "no camera sensors are spawned."
        )

    step_count = 0
    hold_action = env.unwrapped.robot.data.default_joint_pos.clone()
    try:
        while simulation_app.is_running() and (
            args_cli.steps <= 0 or step_count < args_cli.steps
        ):
            env.step(hold_action)
            step_count += 1
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
