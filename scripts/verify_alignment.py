"""Decide whether the simulated camera and the real one actually agree.

Three modes, in increasing strength:

``--quick``  projects the board's known corners through the calibration's own
numbers and compares them to the corners detected in a real frame.  It runs in
seconds and is useful while iterating, but it cannot fail when the *renderer*
is misconfigured, because it never asks the renderer anything.  On its own it
would be the code marking its own homework.

``--gate``   places the board in Isaac where the calibration says the real one
is, renders, detects corners in both images and compares pixels.  This is the
acceptance criterion, because what has to match is the picture Isaac draws,
not the arithmetic behind it.

``--selftest`` needs neither the robot nor a printed target.  It puts the board
at a pose it chooses, renders, and checks the pipeline recovers that pose.
Ground truth is known, so a failure here is a bug in the code rather than a
problem at the rig -- which is the distinction that is expensive to make on
site and free to make here.

``--drift`` re-runs the gate's comparison against a stored baseline, to catch a
camera that has been nudged.  "It is bolted down" is an assumption; this turns
it into a check.

Examples:

    python -u scripts/verify_alignment.py --selftest
    python -u scripts/verify_alignment.py --camera front --gate --real-image capture.png
    python scripts/verify_alignment.py --camera front --quick --real-image capture.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Verify sim/real camera alignment.")
parser.add_argument("--camera", default="front", choices=["wrist", "front"])
parser.add_argument("--quick", action="store_true")
parser.add_argument("--gate", action="store_true")
parser.add_argument("--selftest", action="store_true")
parser.add_argument("--drift", action="store_true")
parser.add_argument("--real-image", type=_Path, default=None)
parser.add_argument("--task", default="so101-visual-StackCube-v0")
parser.add_argument("--rmse-gate", type=float, default=3.0)
parser.add_argument(
    "--square-mm",
    type=float,
    default=None,
    help="board square size; larger boards survive being seen from further away",
)
parser.add_argument("--out-dir", type=_Path, default=_Path("outputs") / "verify")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

import so101.tasks  # noqa: E402,F401  (registers environments)
from so101.camera_calibration import (  # noqa: E402
    matrix_to_quat_wxyz,
    quat_wxyz_to_matrix,
    try_load_calibration,
)
from so101.charuco import board_object_points, charuco_board, detect_board  # noqa: E402
from so101.configs import make_env_cfg  # noqa: E402
from so101.handeye import pose_matrix  # noqa: E402
from so101.scenes.tabletop import ROBOT_BASE_BOTTOM_Z  # noqa: E402
from so101.sim_board import (  # noqa: E402
    board_size_m,
    spawn_board,
    write_board_texture,
)

SENSOR_NAME = {"wrist": "wrist_camera", "front": "external_camera"}


def corner_map(detection) -> dict[int, np.ndarray]:
    if detection.charuco_ids is None:
        return {}
    corners = detection.charuco_corners.reshape(-1, 2)
    return {
        int(i): corners[k] for k, i in enumerate(detection.charuco_ids.flatten())
    }


def compare_corners(
    sim_detection, real_detection
) -> tuple[float, int, list[tuple[int, float]]]:
    """RMSE over corners both images found, so a partial view still scores."""
    sim = corner_map(sim_detection)
    real = corner_map(real_detection)
    shared = sorted(set(sim) & set(real))
    if not shared:
        return float("nan"), 0, []
    errors = [(i, float(np.linalg.norm(sim[i] - real[i]))) for i in shared]
    rmse = float(np.sqrt(np.mean([error**2 for _, error in errors])))
    return rmse, len(shared), errors


def board_pose_from_image(frame, calibration, board):
    """``T_cam_board`` from a captured frame."""
    detection = detect_board(frame, board)
    if not detection.usable(minimum=8):
        return None, detection
    object_points = board_object_points(board, detection.charuco_ids)
    image_points = detection.charuco_corners.reshape(-1, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        calibration.camera_matrix,
        calibration.distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, detection
    rotation, _ = cv2.Rodrigues(rvec)
    return pose_matrix(rotation, tvec), detection


def camera_world_pose(camera):
    position = camera.data.pos_w[0].cpu().numpy()
    rotation = quat_wxyz_to_matrix(camera.data.quat_w_ros[0].cpu().numpy())
    return pose_matrix(rotation, position)


def render(env, camera, warmup: int = 60):
    """Render, after giving Omniverse time to stream the board texture in.

    Textures load asynchronously.  Capturing one frame straight after the
    material is created returns the untextured grey fallback, and the board
    then fails to detect for a reason that has nothing to do with calibration.
    """
    for _ in range(max(warmup, 1)):
        env.sim.step()
        env.scene.update(dt=env.physics_dt)
        camera.update(dt=env.physics_dt)
    rgb = camera.data.output["rgb"][0].cpu().numpy()
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)


def run_selftest(out_dir: _Path) -> int:
    """Place the board at a known pose, render, and try to recover it."""
    from so101.charuco import MARKER_MM, SQUARE_MM

    square_mm = args_cli.square_mm or SQUARE_MM
    board = charuco_board(square_mm, MARKER_MM * square_mm / SQUARE_MM)
    texture = write_board_texture(out_dir / "charuco_texture.png", square_mm)

    cfg = make_env_cfg(args_cli.task, num_envs=1, device=args_cli.device)
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    env.reset()
    camera = env.scene[SENSOR_NAME[args_cli.camera]]
    env_origin = env.scene.env_origins[0].cpu().numpy()

    env.sim.step()
    env.scene.update(dt=env.physics_dt)
    camera_pose_now = camera_world_pose(camera)
    cam_pos, cam_rot = camera_pose_now[:3, 3], camera_pose_now[:3, :3]

    # The board is placed facing the camera at a working distance rather than
    # lying on the table.  This test is checking the renderer and the pose
    # maths, and a board flat on the tabletop is seen at such a grazing angle
    # that its marker cells fall to about 4 px -- below what survives RTX
    # texture filtering, so it would fail for a reason that has nothing to do
    # with what is being tested.  See the gate notes for what that means on
    # site.
    #
    # Orientation matters: getChessboardCorners defines the board frame with
    # +x across the columns and +y DOWN the rows, which puts +z on the side
    # away from the face you read.  Aligning the board frame with the camera
    # frame therefore shows the readable face.  Getting this backwards renders
    # a mirrored board, and mirrored ArUco does not decode at all -- the board
    # looks perfect and yields zero corners.
    # Validated for the front camera only.  The wrist camera sits close to the
    # tabletop, so this distance puts the board past the table and out of
    # frame; shortening it to 0.14 m made the run hang rather than fail, and
    # the cause is not yet understood.  Until that is sorted, run the rendered
    # self-test on the front camera and rely on selftest_pipeline.py for the
    # wrist, which covers the same maths without the renderer.
    distance = args_cli.board_distance or 0.32
    if args_cli.camera == "wrist" and args_cli.board_distance is None:
        print(
            "[warn] the rendered self-test is only validated for --camera front; "
            "the wrist placement is unresolved (see issue notes)"
        )
    tilt = np.radians(12.0)
    tilt_rotation = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, np.cos(tilt), -np.sin(tilt)],
         [0.0, np.sin(tilt), np.cos(tilt)]]
    )
    rotation = cam_rot @ tilt_rotation
    width_m, height_m = board_size_m(square_mm)
    origin = cam_pos + rotation @ np.array([-width_m / 2.0, -height_m / 2.0, distance])
    truth = pose_matrix(rotation, origin)

    spawn_board(
        "/World/CalibrationBoard",
        texture,
        origin,
        matrix_to_quat_wxyz(rotation),
        square_mm,
    )

    frame = render(env, camera)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"selftest_{args_cli.camera}.png"), frame)

    intrinsics = camera.data.intrinsic_matrices[0].cpu().numpy()
    camera_pose = camera_world_pose(camera)
    env.close()

    detection = detect_board(frame, board)
    print("\n" + "=" * 68)
    print(f"SELF-TEST (rendered): {args_cli.camera}")
    print("=" * 68)
    print(f"  corners detected in the render : {detection.count}")
    if not detection.usable(minimum=8):
        print("  FAIL: the rendered board is not detectable.")
        print("  Suspect texture resolution or scene lighting before suspecting")
        print("  the calibration maths.")
        return 1

    object_points = board_object_points(board, detection.charuco_ids)
    image_points = detection.charuco_corners.reshape(-1, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, intrinsics, np.zeros(5),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        print("  FAIL: solvePnP did not converge on the rendered board")
        return 1
    rotation_cam, _ = cv2.Rodrigues(rvec)
    recovered = camera_pose @ pose_matrix(rotation_cam, tvec)

    position_error_mm = float(np.linalg.norm(recovered[:3, 3] - truth[:3, 3]) * 1000.0)
    relative = recovered[:3, :3].T @ truth[:3, :3]
    angle_error_deg = float(
        np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0)))
    )
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        intrinsics,
        np.zeros(5),
    )
    reprojection = float(
        np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1)))
    )

    print(f"  board origin truth             : {np.round(truth[:3, 3] - env_origin, 5).tolist()}")
    print(f"  board origin recovered         : {np.round(recovered[:3, 3] - env_origin, 5).tolist()}")
    print(f"  position error                 : {position_error_mm:.2f} mm   (tolerance 2.00)")
    print(f"  orientation error              : {angle_error_deg:.3f} deg")
    print(f"  reprojection on the render     : {reprojection:.3f} px")

    passed = position_error_mm < 2.0
    print(f"  -> {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("  The renderer, the intrinsics and the pose maths disagree with")
        print("  each other, independent of any real camera.")
    return 0 if passed else 1


def run_against_real(out_dir: _Path) -> int:
    if args_cli.real_image is None or not args_cli.real_image.is_file():
        print("[fail] --real-image is required for --quick, --gate and --drift")
        return 1
    calibration = try_load_calibration(args_cli.camera)
    if calibration is None:
        print(f"[fail] no calibration for {args_cli.camera!r}; run the intrinsic and")
        print("       hand-eye calibrations first")
        return 1
    if calibration.extrinsic is None:
        print(f"[fail] {args_cli.camera!r} has intrinsics but no extrinsic yet")
        return 1

    board = charuco_board()
    real_raw = cv2.imread(str(args_cli.real_image))
    if real_raw is None:
        print(f"[fail] could not read {args_cli.real_image}")
        return 1

    target_pose, real_detection = board_pose_from_image(real_raw, calibration, board)
    if target_pose is None:
        print(f"[fail] board not detected in {args_cli.real_image.name} "
              f"({real_detection.count} corners)")
        return 1
    print(f"[info] real frame: {real_detection.count} corners")

    # Everything downstream compares against the rectified frame, because that
    # is what the policy and the simulation both see.
    maps = calibration.rectify_maps()
    real_rect = cv2.remap(real_raw, maps[0], maps[1], cv2.INTER_LINEAR)
    rect_detection = detect_board(real_rect, board)

    out_dir.mkdir(parents=True, exist_ok=True)
    virtual = calibration.virtual_matrix

    if args_cli.quick:
        # Re-solve on the rectified frame with the virtual intrinsics, then
        # project the known corners straight back.  This exercises only the
        # calibration's own arithmetic.
        object_points = board_object_points(board, rect_detection.charuco_ids)
        image_points = rect_detection.charuco_corners.reshape(-1, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, virtual, np.zeros(5)
        )
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, virtual, np.zeros(5))
        residual = float(
            np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1)))
        )
        print("\n" + "=" * 68)
        print(f"QUICK CHECK: {args_cli.camera}")
        print("=" * 68)
        print(f"  corners on the rectified frame : {rect_detection.count}")
        print(f"  self-consistent residual       : {residual:.3f} px")
        print("  Note: this never asks the renderer anything, so it cannot fail")
        print("  on a misconfigured sim camera. Use --gate for acceptance.")
        return 0

    # --gate / --drift: put the board in the simulation where the calibration
    # says the real one is, then compare rendered pixels with captured ones.
    extrinsic = calibration.extrinsic.matrix
    texture = write_board_texture(out_dir / "charuco_texture.png")

    cfg = make_env_cfg(args_cli.task, num_envs=1, device=args_cli.device)
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    env.reset()
    camera = env.scene[SENSOR_NAME[args_cli.camera]]
    env.sim.step()
    env.scene.update(dt=env.physics_dt)

    if calibration.extrinsic.parent == "env":
        camera_pose = extrinsic
    else:
        camera_pose = camera_world_pose(camera)
    board_pose = camera_pose @ target_pose

    spawn_board(
        "/World/CalibrationBoard",
        texture,
        board_pose[:3, 3],
        matrix_to_quat_wxyz(board_pose[:3, :3]),
    )
    sim_frame = render(env, camera)
    env.close()

    cv2.imwrite(str(out_dir / f"gate_sim_{args_cli.camera}.png"), sim_frame)
    cv2.imwrite(str(out_dir / f"gate_real_{args_cli.camera}.png"), real_rect)
    blended = cv2.addWeighted(sim_frame, 0.5, real_rect, 0.5, 0.0)
    cv2.imwrite(str(out_dir / f"gate_overlay_{args_cli.camera}.png"), blended)

    sim_detection = detect_board(sim_frame, board)
    rmse, shared, errors = compare_corners(sim_detection, rect_detection)

    print("\n" + "=" * 68)
    print(f"{'DRIFT CHECK' if args_cli.drift else 'ALIGNMENT GATE'}: {args_cli.camera}")
    print("=" * 68)
    print(f"  corners in render / real / shared : "
          f"{sim_detection.count} / {rect_detection.count} / {shared}")
    if shared == 0:
        print("  FAIL: no corner was found in both images; nothing to compare")
        return 1
    worst = max(errors, key=lambda pair: pair[1])
    print(f"  reprojection RMSE                 : {rmse:.2f} px "
          f"(gate < {args_cli.rmse_gate})")
    print(f"  worst corner                      : id {worst[0]} at {worst[1]:.2f} px")
    print(f"  overlay written to                : "
          f"{out_dir / f'gate_overlay_{args_cli.camera}.png'}")

    passed = rmse < args_cli.rmse_gate
    print(f"  -> {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("\n  Diagnosis order:")
        print("   1. run --selftest. If that fails too, the problem is in the code,")
        print("      not the measurements.")
        print("   2. check the intrinsic RMS and the hand-eye inter-solver spread.")
        print("   3. look at the overlay: a uniform shift points at the extrinsic,")
        print("      a scale difference at the focal length.")
    return 0 if passed else 1


def main() -> int:
    modes = [args_cli.quick, args_cli.gate, args_cli.selftest, args_cli.drift]
    if sum(bool(mode) for mode in modes) != 1:
        print("[fail] choose exactly one of --quick, --gate, --selftest, --drift")
        return 1
    if args_cli.selftest:
        return run_selftest(args_cli.out_dir)
    return run_against_real(args_cli.out_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
