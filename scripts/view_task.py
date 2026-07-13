"""View one of the SO-101 contact-rich manipulation environments.

Examples:

    python scripts/view_task.py --task so101-PegInsert-v0
    python scripts/view_task.py so101-GearMesh-v0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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
parser.add_argument(
    "--save-render",
    type=Path,
    default=None,
    metavar="PATH",
    help="Save the final task-evaluation viewport frame as a PNG image.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
selected_task = args_cli.task_option or args_cli.task_name
if selected_task is None:
    parser.error("an environment name is required (positional or via --task)")
args_cli.enable_cameras = "-visual-" in selected_task or args_cli.save_render is not None
# if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
#     args_cli.headless = True
#     print("[INFO] No desktop display detected; starting Isaac Sim in headless mode.")


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import so101.tasks  # noqa: E402,F401  (registers environments)
from so101.configs import make_env_cfg  # noqa: E402
from so101.tasks.scenes import SO101RenderStackCubeSceneCfg  # noqa: E402


def main() -> None:
    env_cfg = make_env_cfg(selected_task, num_envs=1, device=args_cli.device)
    if args_cli.save_render is not None:
        if selected_task != "so101-StackCube-v0":
            raise ValueError("--save-render currently supports so101-StackCube-v0 only.")
        env_cfg.scene = SO101RenderStackCubeSceneCfg(
            num_envs=1, env_spacing=1.0, clone_in_fabric=False
        )

    env = gym.make(
        selected_task,
        cfg=env_cfg,
        render_mode=None,
    )

    ob = env.reset()
    print(f"[INFO] Loaded environment: {selected_task}")
    if "StackCube" in selected_task:
        print("[INFO] Cubes are exact 2.5 cm and 4 cm Isaac Lab primitives.")
    elif "PegInsert" in selected_task or "NutThread" in selected_task:
        print("[INFO] Task assets use their original Isaac Factory scale (100%).")
    else:
        print("[INFO] Factory assets are uniformly scaled to 75%.")
    print(
        "[INFO] Actions are absolute SO-101 joint-position targets in radians (6 joints)."
    )
    if "-visual-" in selected_task:
        print(
            "[INFO] Observations include state, wrist_image, and front_image."
        )
    else:
        state_dim = 29 if "StackCube" in selected_task else 38
        print(
            f"[INFO] Observations contain one {state_dim}-D state tensor; "
            "no camera sensors are spawned."
        )

    step_count = 0
    render_camera = (
        env.unwrapped.scene["render_camera"]
        if args_cli.save_render is not None
        else None
    )
    reset_every_steps = (
        max(1, round(5.0 / env.unwrapped.step_dt))
        if "StackCube" in selected_task
        else None
    )
    if reset_every_steps is not None:
        print("[INFO] View-only StackCube pose reset interval: 5 seconds.")
    hold_action = env.unwrapped.robot.data.default_joint_pos.clone()
    try:
        while simulation_app.is_running() and (
            args_cli.steps <= 0 or step_count < args_cli.steps
        ):
            env.step(hold_action)
            step_count += 1
            if reset_every_steps is not None and step_count % reset_every_steps == 0:
                env.reset()
        if args_cli.save_render is not None:
            from PIL import Image

            if render_camera is None:
                raise RuntimeError("Render camera was not initialized.")
            render_image = render_camera.data.output["rgb"][0, ..., :3]
            render_image = render_image.to(dtype=torch.uint8).cpu().numpy()
            args_cli.save_render.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(render_image).save(args_cli.save_render)
            print(f"[INFO] Saved render image: {args_cli.save_render.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
