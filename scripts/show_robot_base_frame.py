"""Show the Isaac SO-101 scene with the robot-base coordinate frame.

The visual axes are ROS-style/world axes used by this project:

* red   = +X
* green = +Y
* blue  = +Z

The practical measurement frame is the robot-centre frame, placed at the
measured distance from the table's rear edge.  +X points into the table, +Y
follows the table width, and +Z points upward.  The table rear edge is shown
as a separate yellow line.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument(
        "--dump-bounds",
        action="store_true",
        help="print USD prim bounds containing 'base' or 'mount'",
    )
    parser.add_argument(
        "--robot-x",
        type=float,
        default=0.0377,
        help="robot USD-root x position after the final -9 mm adjustment (m)",
    )
    parser.add_argument(
        "--robot-y",
        type=float,
        default=0.3967,
        help="robot USD-root y position; base centre lands at y=0.4175 m",
    )
    parser.add_argument(
        "--frame-x",
        type=float,
        default=0.0,
        help="base-frame x position on the table rear edge (m)",
    )
    parser.add_argument(
        "--frame-y",
        type=float,
        default=0.4175,
        help="measured robot-centre y position from the table reference edge (m)",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = False
    args.enable_cameras = False

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab.sim as sim_utils

    import so101.tasks  # noqa: F401
    from so101.configs import make_env_cfg
    from so101.scenes.tabletop import ROBOT_BASE_BOTTOM_Z, TABLETOP_WIDTH

    env_cfg = make_env_cfg("so101-StackCube-v0", num_envs=1, device=args.device)
    env_cfg.scene.robot.init_state.pos = (args.robot_x, args.robot_y, 0.0)
    env = gym.make(
        "so101-StackCube-v0",
        cfg=env_cfg,
    ).unwrapped

    try:
        env.reset()

        if args.dump_bounds:
            from omni.usd import get_context
            from pxr import Usd, UsdGeom

            stage = get_context().get_stage()
            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
            )
            print("\n=== ROBOT USD BOUNDS (world frame) ===")
            for prim in stage.Traverse():
                path = str(prim.GetPath()).lower()
                if "/robot" not in path:
                    continue
                box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
                if box.IsEmpty():
                    continue
                lo, hi = box.GetMin(), box.GetMax()
                print(
                    f"{prim.GetPath()}: "
                    f"min=({lo[0]:+.4f},{lo[1]:+.4f},{lo[2]:+.4f}) "
                    f"max=({hi[0]:+.4f},{hi[1]:+.4f},{hi[2]:+.4f})"
                )

        # The USD base footprint bounds are x=[0.0000, 0.0956],
        # y=[0.3828, 0.4938], z=[0.0301, 0.1021] m for the current root pose.
        # Use their centre so the frame is centred through the fixed mount,
        # rather than sitting on the mount's rear edge.
        measurement_origin = np.array(
            [args.frame_x, args.frame_y, ROBOT_BASE_BOTTOM_Z]
        )
        usd_root_origin = np.array([args.robot_x, args.robot_y, 0.0])
        axis_length = 0.30
        thickness = 0.012

        rods = (
            ("X_plus_RED", (axis_length, thickness, thickness),
             (measurement_origin[0] + axis_length / 2.0,
              measurement_origin[1], measurement_origin[2]),
             (1.0, 0.0, 0.0)),
            ("Y_plus_GREEN", (thickness, axis_length, thickness),
             (measurement_origin[0],
              measurement_origin[1] + axis_length / 2.0,
              measurement_origin[2]),
             (0.0, 1.0, 0.0)),
            ("Z_plus_BLUE", (thickness, thickness, axis_length),
             (measurement_origin[0], measurement_origin[1],
              measurement_origin[2] + axis_length / 2.0),
             (0.0, 0.0, 1.0)),
        )
        for name, size, translation, color in rods:
            cfg = sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color,
                    roughness=0.5,
                ),
            )
            cfg.func(f"/World/BaseFrame/{name}", cfg, translation=translation)

        origin_cfg = sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 0.0),
                roughness=0.4,
            ),
        )
        origin_cfg.func(
            "/World/BaseFrame/TableRearEdgeCenter",
            origin_cfg,
            translation=tuple(measurement_origin),
        )

        try:
            from omni.kit.viewport.utility import get_active_viewport

            get_active_viewport().set_camera_view(
                eye=np.array([0.95, -1.05, 0.75]),
                target=np.array([0.18, 0.0, 0.20]),
            )
        except Exception as error:  # noqa: BLE001
            print(f"[warn] could not set overview viewport: {error}")

        print("\n=== SO-101 PRACTICAL ROBOT-CENTRE FRAME ===")
        print(
            "robot-centre frame: "
            f"({measurement_origin[0]:.6f}, {measurement_origin[1]:.6f}, "
            f"{measurement_origin[2]:.6f}) m"
        )
        print("+X: red   (into the table)")
        print("+Y: green (across the table width)")
        print("+Z: blue  (up)")
        print("table boundaries are visible; yellow sphere marks the measured robot-centre frame")
        print("table dimensions: X=0.700 m, Y=1.200 m")
        print(
            "USD articulation root: "
            f"({usd_root_origin[0]:.6f}, {usd_root_origin[1]:.6f}, "
            f"{usd_root_origin[2]:.6f}) m"
        )
        print(f"viewer active for {args.seconds:.1f} seconds; close the window or press Ctrl+C to stop")

        deadline = time.monotonic() + args.seconds
        while simulation_app.is_running() and time.monotonic() < deadline:
            env.sim.step(render=True)
            env.scene.update(dt=env.physics_dt)
            time.sleep(1.0 / 30.0)
    finally:
        env.close()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
