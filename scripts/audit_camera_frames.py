"""Report the frames and framing the camera calibration depends on.

Two things have to be known before anyone stands at the rig, and both can be
answered from simulation alone:

1. **What "base" means.**  Hand-eye calibration solves in whatever frame
   forward kinematics reports.  ``CameraCfg.OffsetCfg`` is applied relative to
   the environment origin.  If those differ, the difference becomes a constant
   error in every measured camera pose -- and a very hard one to trace later.
   The robot prim sits at ``(0, 0, 0)``, but ``ROBOT_BASE_BOTTOM_Z =
   0.0300814467`` shows the USD carries a non-zero internal offset, so this is
   measured rather than assumed.

2. **Whether the cube workspace stays in frame.**  If a camera cannot see the
   spawn region, that is a framing problem, not an alignment problem, and no
   amount of calibration fixes it.  Catching it here means the camera can be
   repositioned during the same site visit rather than after it.

Examples:

    python scripts/audit_camera_frames.py
    python scripts/audit_camera_frames.py --save-renders outputs/frame_audit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path

# so101 is installed editable from a sibling checkout, so an unqualified
# import silently resolves there instead of to this working tree.  The .pth
# only appends to sys.path, so putting this repo's src first wins.
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Audit SO-101 camera frames and workspace visibility."
)
parser.add_argument(
    "--task",
    default="so101-visual-StackCube-v0",
    help="task to inspect (default: so101-visual-StackCube-v0)",
)
parser.add_argument(
    "--save-renders",
    type=Path,
    default=None,
    help="directory to write one RGB frame per camera, with the spawn region marked",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import so101.tasks  # noqa: E402,F401  (registers environments)
from so101.configs import make_env_cfg  # noqa: E402
from so101.configs.tasks import SO101VisualStackCubeEnvCfg  # noqa: E402
from so101.scenes.tabletop import ROBOT_BASE_BOTTOM_Z  # noqa: E402
from so101.tasks.assets import LARGE_CUBE_SIZE, SMALL_CUBE_SIZE  # noqa: E402


def _fmt(values) -> str:
    return "[" + ", ".join(f"{float(v):+.6f}" for v in values) + "]"


def spawn_region_points(cfg: SO101VisualStackCubeEnvCfg) -> dict[str, np.ndarray]:
    """Corners of the cube spawn region, at the table and at stack height.

    A camera that sees the tabletop corners but clips the top of a completed
    stack is still a failure, so both heights are checked.
    """
    x_min, x_max = cfg.asset_spawn_x_range
    y_min_abs, y_max_abs = cfg.asset_spawn_y_abs_range
    table_z = ROBOT_BASE_BOTTOM_Z
    stack_z = ROBOT_BASE_BOTTOM_Z + LARGE_CUBE_SIZE + SMALL_CUBE_SIZE

    points: dict[str, np.ndarray] = {}
    for x_name, x in (("xmin", x_min), ("xmax", x_max)):
        for y_name, y in (
            ("y-out", -y_max_abs),
            ("y-in", -y_min_abs),
            ("y+in", y_min_abs),
            ("y+out", y_max_abs),
        ):
            for z_name, z in (("table", table_z), ("stack", stack_z)):
                points[f"{x_name}/{y_name}/{z_name}"] = np.array([x, y, z])
    points["centre/table"] = np.array(
        [(x_min + x_max) / 2.0, 0.0, table_z]
    )
    return points


def project(
    point_w: np.ndarray,
    cam_pos_w: np.ndarray,
    cam_rot_w_ros: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Project a world point into a camera using the ROS convention.

    ``cam_rot_w_ros`` is the camera-to-world rotation with +Z along the optical
    axis and +Y down the image rows, which is what ``Camera.data.quat_w_ros``
    provides.
    """
    point_cam = cam_rot_w_ros.T @ (point_w - cam_pos_w)
    depth = float(point_cam[2])
    if depth <= 1e-6:
        return np.array([np.nan, np.nan]), depth
    pixel = intrinsics @ (point_cam / depth)
    return pixel[:2], depth


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def main() -> None:
    cfg = make_env_cfg(args_cli.task, num_envs=1, device=args_cli.device)
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    env.reset()
    # One step so the render pipeline has populated the sensor buffers.
    env.sim.step()
    env.scene.update(dt=env.physics_dt)

    env_origin = env.scene.env_origins[0].cpu().numpy()
    robot = env.scene["robot"]
    root_pos_w = robot.data.root_pos_w[0].cpu().numpy()
    root_quat_w = robot.data.root_quat_w[0].cpu().numpy()

    print("\n" + "=" * 72)
    print("1. FRAMES")
    print("=" * 72)
    print(f"env origin (world)          : {_fmt(env_origin)}")
    print(f"robot root  (world)         : {_fmt(root_pos_w)}")
    print(f"robot root quat (w,x,y,z)   : {_fmt(root_quat_w)}")
    t_env_base = root_pos_w - env_origin
    rot_env_base = quat_wxyz_to_matrix(root_quat_w)
    yaw_deg = np.degrees(np.arctan2(rot_env_base[1, 0], rot_env_base[0, 0]))
    print(f"T_env_base translation      : {_fmt(t_env_base)}")
    print(f"T_env_base rotation (yaw)   : {yaw_deg:+.4f} deg about Z")
    print("T_env_base rotation matrix  :")
    for row in rot_env_base:
        print(f"    {_fmt(row)}")

    translation_zero = np.allclose(t_env_base, 0.0, atol=1e-6)
    rotation_identity = np.allclose(rot_env_base, np.eye(3), atol=1e-6)
    if translation_zero and rotation_identity:
        print("  -> env origin and the robot root coincide; camera offsets and")
        print("     hand-eye results share one frame, no correction needed.")
    else:
        print("  -> env origin and the robot root DIFFER.")
        if translation_zero and not rotation_identity:
            print("     The translation is zero but the ROTATION is not identity, so")
            print("     'the prim sits at (0,0,0)' is true and still misleading:")
            print("     a hand-eye result expressed in the robot base frame must be")
            print("     rotated by this transform before it becomes an OffsetCfg.")
        print("     Record T_env_base in the calibration YAML and apply it.")

    body_names = list(robot.data.body_names)
    print(f"\nbody frames available       : {body_names}")
    for index, name in enumerate(body_names):
        pos = robot.data.body_pos_w[0, index].cpu().numpy() - env_origin
        print(f"  {name:<24} pos rel. env origin {_fmt(pos)}")
    print(
        f"\nROBOT_BASE_BOTTOM_Z constant: {ROBOT_BASE_BOTTOM_Z:.10f}"
        "  (tabletop surface height)"
    )

    print("\n" + "=" * 72)
    print("2. CAMERAS")
    print("=" * 72)
    cameras = {
        "wrist": env.scene["wrist_camera"],
        "front": env.scene["external_camera"],
    }
    camera_state: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, camera in cameras.items():
        pos_w = camera.data.pos_w[0].cpu().numpy()
        quat_ros = camera.data.quat_w_ros[0].cpu().numpy()
        intrinsics = camera.data.intrinsic_matrices[0].cpu().numpy()
        rot = quat_wxyz_to_matrix(quat_ros)
        camera_state[name] = (pos_w, rot, intrinsics)

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        width, height = camera.image_shape[1], camera.image_shape[0]
        hfov = np.degrees(2 * np.arctan(width / (2 * fx)))
        print(f"\n[{name}]")
        print(f"  position rel. env origin : {_fmt(pos_w - env_origin)}")
        print(f"  quat_w_ros (w,x,y,z)     : {_fmt(quat_ros)}")
        print(f"  resolution               : {width}x{height}")
        print(f"  fx={fx:.3f}  fy={fy:.3f}  cx={cx:.1f}  cy={cy:.1f}  hFOV={hfov:.2f} deg")
        print(f"  optical axis (world)     : {_fmt(rot[:, 2])}")

    print("\n" + "=" * 72)
    print("3. SPAWN REGION VISIBILITY")
    print("=" * 72)
    points = spawn_region_points(cfg)
    # Only the fixed-mount camera is judged here.  The wrist camera rides the
    # gripper, so what it frames is a function of the arm's pose, not of a
    # mount that could be repositioned -- holding it to the same standard at
    # the rest pose would report a failure that means nothing.
    fixed_mount = {"front"}
    verdict_failures: list[str] = []
    for name, camera in cameras.items():
        pos_w, rot, intrinsics = camera_state[name]
        height, width = camera.image_shape[0], camera.image_shape[1]
        judged = name in fixed_mount
        role = "fixed mount" if judged else "arm-mounted, pose-dependent (informational)"
        print(f"\n[{name}]  frame is {width}x{height}  -- {role}")
        misses = []
        for label, point_base in points.items():
            point_w = point_base + env_origin
            pixel, depth = project(point_w, pos_w, rot, intrinsics)
            if np.isnan(pixel).any():
                status, detail = "BEHIND", "point is behind the camera"
            else:
                inside = 0 <= pixel[0] < width and 0 <= pixel[1] < height
                status = "ok" if inside else "OUT"
                detail = f"u={pixel[0]:7.1f} v={pixel[1]:7.1f} depth={depth:.3f} m"
            if status != "ok":
                misses.append(label)
            print(f"  {status:<7} {label:<26} {detail}")
        if not misses:
            print("  -> entire spawn region, including stack height, is in frame")
        elif judged:
            verdict_failures.append(name)
            print(f"  -> {len(misses)}/{len(points)} workspace points are NOT visible")
        else:
            print(
                f"  -> {len(misses)}/{len(points)} points out of frame at this arm "
                "pose; expected, and not a mount problem"
            )

    print("\n" + "=" * 72)
    if not verdict_failures:
        print("RESULT: the fixed-mount camera frames the workspace.")
    else:
        print(f"RESULT: {', '.join(verdict_failures)} clips the workspace.")
        print("Per the alignment plan this is a framing problem: reposition the")
        print("physical camera and recalibrate, rather than shrinking the spawn")
        print("range or letting the policy cope with occlusion.")
    print("=" * 72 + "\n")

    if args_cli.save_renders is not None:
        args_cli.save_renders.mkdir(parents=True, exist_ok=True)
        import imageio.v3 as iio

        for name, camera in cameras.items():
            rgb = camera.data.output["rgb"][0].cpu().numpy()
            path = args_cli.save_renders / f"{name}.png"
            iio.imwrite(path, rgb.astype(np.uint8))
            print(f"wrote {path}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
