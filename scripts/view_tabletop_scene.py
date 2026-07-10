"""Launch an interactive viewer for the SO-101 tabletop scene.

Run after installing this package in the Isaac Lab environment:

    python scripts/view_tabletop_scene.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View the SO-101 on its fixed 50 cm tabletop.")
parser.add_argument(
    "--steps",
    type=int,
    default=0,
    help="Exit after this many simulation steps; 0 keeps the viewer open.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab simulation modules must be imported after AppLauncher starts Kit.
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from so101.scenes.tabletop import ROBOT_BASE_BOTTOM_Z, SO101TabletopSceneCfg  # noqa: E402


def main() -> None:
    """Create the scene and keep rendering until the viewer is closed."""
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device))
    sim.set_camera_view(eye=(0.85, -0.75, 0.55), target=(0.22, 0.0, 0.12))

    scene_cfg = SO101TabletopSceneCfg(num_envs=1, env_spacing=1.5)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    robot.set_joint_position_target(robot.data.default_joint_pos)
    scene.write_data_to_sim()

    print("[INFO] SO-101 tabletop scene is ready.")
    print("[INFO] Robot base: (0.0, 0.0, 0.0)")
    print(
        "[INFO] Tabletop: x=[0.0, 0.5], y=[-0.25, 0.25], "
        f"top z={ROBOT_BASE_BOTTOM_Z:.6f}"
    )

    step_count = 0
    while simulation_app.is_running() and (args_cli.steps <= 0 or step_count < args_cli.steps):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        step_count += 1


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
