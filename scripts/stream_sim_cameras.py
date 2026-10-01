"""Stream Isaac's calibrated front and wrist camera views at one joint pose.

The robot root is placed at the final physical layout used for calibration,
then both simulated cameras are rendered from exactly the same robot pose.
Press ``q`` in either image window or Ctrl+C in the terminal to stop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Physical layout measured from the yellow robot-centre/base-frame marker.
# The board centre is 327.5 mm forward (+X) and 57.0 mm to the left (+Y)
# from that marker.  Keep these as offsets so they cannot be confused with
# absolute world coordinates again.
BASE_FRAME_X = 0.0
BASE_FRAME_Y = 0.4175
BOARD_OFFSET_X = 0.3275
BOARD_OFFSET_Y = 0.0570


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pose",
        type=float,
        nargs=6,
        metavar=("ROT", "PITCH", "ELBOW", "WRIST_P", "WRIST_R", "JAW"),
        default=[0.0, -0.6, 0.9, 1.2, 0.0, 0.0],
        help="joint radians in Rotation Pitch Elbow Wrist_Pitch Wrist_Roll Jaw order",
    )
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "sim_camera_stream",
        help="directory for exact 640x480 sensor frames",
    )
    parser.add_argument(
        "--no-board",
        action="store_true",
        help="do not spawn the table ChArUco board",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = False
    args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import omni.usd
    import torch

    import so101.tasks  # noqa: F401
    from so101.configs import make_env_cfg
    from so101.camera_calibration import matrix_to_quat_wxyz
    from so101.charuco import load_board_spec
    from so101.scenes.tabletop import ROBOT_BASE_BOTTOM_Z, ROBOT_ROOT_POS
    from so101.sim_board import board_size_m, spawn_board, write_board_texture

    # Final physical layout: the yellow frame stays at y=0.4175 m; the USD
    # base centre is brought there by placing the articulation root at y=0.3967.
    cfg = make_env_cfg("so101-visual-StackCube-v0", num_envs=1, device=args.device)
    cfg.scene.robot.init_state.pos = ROBOT_ROOT_POS
    env = gym.make("so101-visual-StackCube-v0", cfg=cfg).unwrapped

    try:
        env.reset()
        robot = env.scene["robot"]
        pose = torch.tensor(args.pose, dtype=torch.float32, device=robot.device).unsqueeze(0)
        if len(robot.data.joint_names) != 6:
            raise RuntimeError(f"expected 6 robot joints, got {robot.data.joint_names}")
        robot.write_joint_state_to_sim(pose, torch.zeros_like(pose))
        robot.set_joint_position_target(pose)
        robot.write_data_to_sim()

        if not args.no_board:
            board_spec = load_board_spec()
            # The physical board is on the tabletop at ROBOT_BASE_BOTTOM_Z.
            # Lift only the visual mesh by 0.5 mm to avoid z-fighting with the
            # tabletop; its calibrated XY position and orientation are unchanged.
            board_visual_clearance = 0.0005
            board_center = np.array(
                [
                    BASE_FRAME_X + BOARD_OFFSET_X,
                    BASE_FRAME_Y + BOARD_OFFSET_Y,
                    ROBOT_BASE_BOTTOM_Z + board_visual_clearance,
                ],
                dtype=float,
            )
            # Face-up table board.  The physical board's long axis is world Y,
            # so retain the 90-degree in-plane rotation used by the rig.
            board_rotation = np.array(
                [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            ) @ np.diag([1.0, -1.0, -1.0])
            board_half_size = np.array(
                [board_spec.width_mm / 2000.0, board_spec.height_mm / 2000.0, 0.0]
            )
            board_origin = board_center - board_rotation @ board_half_size
            texture = write_board_texture(
                REPO_ROOT / "outputs" / "sim_camera_stream" / "table_charuco.png"
            )
            spawn_board(
                "/World/CalibrationBoard",
                texture,
                board_origin,
                np.asarray(matrix_to_quat_wxyz(board_rotation)),
            )

            # Visual measurement aids.  Create plain USD display-color prims
            # directly; spawning SphereCfg/CuboidCfg after the scene starts
            # can terminate Isaac without a Python traceback.
            from pxr import Gf, Sdf, UsdGeom, UsdShade

            stage = omni.usd.get_context().get_stage()
            base_center = np.array(
                [BASE_FRAME_X, BASE_FRAME_Y, ROBOT_BASE_BOTTOM_Z + 0.002],
                dtype=float,
            )
            board_marker = board_center + np.array([0.0, 0.0, 0.0008])
            # Keep the marker almost on the board plane.  Raising a large
            # sphere by 2 mm creates a visible perspective parallax and makes
            # an otherwise correct 3-D centre look offset in the viewport.

            def marker_material(path, color):
                material = UsdShade.Material.Define(stage, path)
                shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(*[float(value) for value in color])
                )
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
                output = material.CreateSurfaceOutput("surface")
                output.ConnectToSource(shader.ConnectableAPI(), "surface")
                return material

            def bind_material(prim, material):
                UsdShade.MaterialBindingAPI(prim).Bind(material)

            def sphere(path, position, radius, color, material_name):
                prim = UsdGeom.Sphere.Define(stage, path).GetPrim()
                UsdGeom.Sphere(prim).GetRadiusAttr().Set(float(radius))
                UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*position))
                bind_material(prim, marker_material(material_name, color))

            def cube(path, position, size, color, material_name):
                prim = UsdGeom.Cube.Define(stage, path).GetPrim()
                UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*position))
                UsdGeom.XformCommonAPI(prim).SetScale(
                    Gf.Vec3f(*(float(value) / 2.0 for value in size))
                )
                bind_material(prim, marker_material(material_name, color))

            sphere("/World/CalibrationMarkers/BaseFrame", base_center, 0.018, (1.0, 1.0, 0.0), "/World/CalibrationMarkers/BaseMaterial")
            sphere("/World/CalibrationMarkers/BoardCenter", board_marker, 0.008, (1.0, 0.0, 1.0), "/World/CalibrationMarkers/CenterMaterial")

            # Cyan points are the actual four mesh corners; their diagonal
            # intersection is the magenta board centre.
            half = np.array([board_spec.width_mm / 2000.0, board_spec.height_mm / 2000.0, 0.0])
            for index, signs in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
                corner = board_center + board_rotation @ (half * np.array([signs[0], signs[1], 1.0])) + np.array([0.0, 0.0, 0.0008])
                sphere(f"/World/CalibrationMarkers/BoardCorner{index}", corner, 0.005, (0.0, 1.0, 1.0), "/World/CalibrationMarkers/CornerMaterial")
            axis_length = 0.16
            axis_thickness = 0.006
            for name, axis, color in (
                ("X", np.array([1.0, 0.0, 0.0]), (1.0, 0.0, 0.0)),
                ("Y", np.array([0.0, 1.0, 0.0]), (0.0, 1.0, 0.0)),
                ("Z", np.array([0.0, 0.0, 1.0]), (0.0, 0.0, 1.0)),
            ):
                size = np.abs(axis * axis_length) + axis_thickness * (1.0 - np.abs(axis))
                cube(
                    f"/World/CalibrationMarkers/BaseFrameAxis{name}",
                    base_center + axis * axis_length / 2.0,
                    size,
                    color,
                    f"/World/CalibrationMarkers/{name}Material",
                )

            print(
                "table board: 256x160 mm, centre="
                f"{np.round(board_center, 6).tolist()} m, face-up "
                f"(visual clearance={board_visual_clearance * 1000:.1f} mm)"
            )
            print(
                "board offset from yellow base frame: "
                f"(+{BOARD_OFFSET_X * 1000:.1f}, +{BOARD_OFFSET_Y * 1000:.1f}) mm"
            )
            print("markers: yellow=base frame, magenta=board centre, RGB=+X/+Y/+Z")

        # Let the articulation and both sensors settle at the requested pose.
        for _ in range(30):
            env.sim.step(render=True)
            env.scene.update(dt=env.physics_dt)

        args.out.mkdir(parents=True, exist_ok=True)

        def save_sensor_frames() -> None:
            """Save sensor pixels without viewport panel scaling or cropping."""
            for name, sensor_name in (
                ("front", "external_camera"),
                ("wrist", "wrist_camera"),
            ):
                rgb = env.scene[sensor_name].data.output["rgb"][0].cpu().numpy()
                bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(args.out / f"latest_{name}.png"), bgr)

        save_sensor_frames()

        print("\n=== SIM CAMERA STREAM ===")
        print(f"joint pose rad: {np.round(args.pose, 5).tolist()}")
        print(f"robot root: {ROBOT_ROOT_POS} m")
        print("front and wrist camera windows show the same simulated robot pose")
        print(f"exact 640x480 sensor frames: {args.out}")
        print("press q in either window or Ctrl+C to stop")

        from omni.kit.viewport.utility import create_viewport_window
        from pxr import Sdf

        front_viewport = create_viewport_window(
            "SIM FRONT CAMERA",
            width=640,
            height=480,
            position_x=20,
            position_y=60,
            camera_path=Sdf.Path("/World/envs/env_0/ExternalCamera"),
        )
        wrist_viewport = create_viewport_window(
            "SIM WRIST CAMERA",
            width=640,
            height=480,
            position_x=680,
            position_y=60,
            camera_path=Sdf.Path("/World/envs/env_0/Robot/gripper/gripper_cam"),
        )
        if front_viewport is None or wrist_viewport is None:
            raise RuntimeError("Isaac viewport creation failed")

        deadline = time.monotonic() + args.seconds
        next_save = time.monotonic() + 0.5
        while simulation_app.is_running() and time.monotonic() < deadline:
            env.sim.step(render=True)
            env.scene.update(dt=env.physics_dt)
            if time.monotonic() >= next_save:
                save_sensor_frames()
                next_save = time.monotonic() + 0.5
    finally:
        env.close()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
