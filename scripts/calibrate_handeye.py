"""Measure where a camera sits, by hand-eye calibration.

Split into two phases on purpose:

``--capture`` talks to the real follower and writes ``(joint angles, image)``
pairs to disk.  No simulator, no solving -- so the time spent standing at the
rig is as short as possible, and a bad solve never costs a re-shoot.

``--solve`` launches Isaac, replays the recorded joint angles through the
articulation to get forward kinematics, detects the target in each image, and
runs every hand-eye solver OpenCV offers.

Two details decide whether the answer is right:

*Forward kinematics comes from Isaac*, not from a separate URDF chain.  The
simulation's kinematics is the thing the policy will be trained against, so it
is the definition that matters, and any disagreement with the real arm shows up
honestly as hand-eye residual instead of being absorbed silently.  It also
sidesteps a trap: the robot base frame is rotated 90 degrees about Z from the
environment frame (see scripts/audit_camera_frames.py), so a solve done in
robot-base coordinates would need that correction before it could become a
CameraCfg.OffsetCfg.  Isaac reports link poses in the environment frame
already, so the result drops straight in.

*The front camera is eye-to-hand.*  ``cv2.calibrateHandEye`` is written for
eye-in-hand, so the front camera's transforms are inverted before being passed
in, which turns the same call into a solve for the camera's pose in the
environment frame.  Getting this backwards produces a plausible, wrong answer.

``--via-board`` is the preferred route for the front camera and needs nothing
attached to the gripper.  The wrist camera, once calibrated, measures where the
static table board sits; the front camera is then solved from its own view of
that same board.  Because the board does not move, the two cameras need not
even see it at the same moment.  The cost is that the result inherits the wrist
hand-eye error on top of two PnP solves -- measured at roughly 5.8 mm for a
2 mm error in the board pose -- so the wrist solve has to be good first.

Examples:

    python scripts/calibrate_handeye.py --camera wrist --capture
    python -u scripts/calibrate_handeye.py --camera wrist --solve
    python -u scripts/calibrate_handeye.py --camera front --solve --via-board \
        --front-image outputs/camera_views/front_raw_000.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

REPO_ROOT = _Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE_ROOT = REPO_ROOT / "outputs" / "handeye"
DEFAULT_POSE_DIR = REPO_ROOT / "calibration" / "handeye_poses"

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hand-eye calibration for one SO-101 camera."
    )
    parser.add_argument("--camera", required=True, choices=["wrist", "front"])
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--capture-dir", type=_Path, default=None)
    parser.add_argument("--pose-file", type=_Path, default=None)
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--robot-id", default="so101_follower")
    parser.add_argument("--square-mm", type=float, default=None)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="command the arm through the pose sequence instead of waiting for "
        "the operator to place it",
    )
    parser.add_argument("--settle", type=float, default=1.5)
    parser.add_argument(
        "--via-board",
        action="store_true",
        help="front only: derive the pose from the table board that the already "
        "calibrated wrist camera localises, instead of a gripper-mounted tag",
    )
    parser.add_argument(
        "--front-image",
        type=_Path,
        nargs="+",
        default=None,
        help="--via-board: one or more front-camera frames showing the board",
    )
    return parser


def capture(args) -> int:
    """Record (joint angles, image) pairs from the real robot."""
    import logging

    import cv2
    import numpy as np
    import torch

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from so101.real.cameras import DEFAULT_SPECS, Camera
    from so101.real.constants import SO101_JOINT_ORDER
    from so101.real.interface import LeRobotSO101Interface

    capture_dir = args.capture_dir or (DEFAULT_CAPTURE_ROOT / args.camera)
    capture_dir.mkdir(parents=True, exist_ok=True)
    pose_file = args.pose_file or (DEFAULT_POSE_DIR / f"{args.camera}.json")

    poses = None
    if args.replay:
        if not pose_file.is_file():
            print(f"[fail] no pose sequence at {pose_file}; run generate_handeye_poses.py")
            return 1
        poses = json.loads(pose_file.read_text())["poses"]
        print(
            f"\n!! Replay will MOVE the arm through {len(poses)} poses.\n"
            "!! Simulation does not model cables, clamps or the front camera "
            "mount.\n!! Keep a hand on the stop and watch the first run.\n"
        )
        if input("Type 'move' to continue: ").strip() != "move":
            print("aborted")
            return 1

    interface = LeRobotSO101Interface(
        device="cpu",
        port=args.port,
        id=args.robot_id,
        cameras={},
        fps=30,
        kind="follower",
    )
    interface.init_device()
    interface.connect()

    spec = DEFAULT_SPECS[args.camera]
    records: list[dict] = []
    # Raw frames: the target's apparent position must carry the lens
    # distortion that the intrinsic calibration measured, since solvePnP is
    # given that same distortion model.
    with Camera(spec, calibration=None, rectify=False) as camera:
        print(f"[info] {args.camera} on {spec.device}, exposure "
              f"{camera.exposure_lock.as_dict()}")
        index = 0
        while True:
            if poses is not None:
                if index >= len(poses):
                    break
                target = torch.tensor(poses[index]["joint_positions"], dtype=torch.float32)
                raw = interface.get_raw_actions_from_radians(target)
                action = {
                    joint: float(value)
                    for joint, value in zip(SO101_JOINT_ORDER, raw.tolist())
                }
                interface.robot.send_action(action)
                import time

                time.sleep(args.settle)
            else:
                answer = input(
                    f"[{index}] place the arm, then [Enter] to capture, 'q' to finish: "
                ).strip()
                if answer.lower() == "q":
                    break

            observation = interface.robot.get_observation()
            raw_values = torch.tensor(
                [float(observation[joint]) for joint in SO101_JOINT_ORDER],
                dtype=torch.float32,
            )
            radians = interface.get_mapped_actions_vectorized(raw_values)
            frame = camera.read_fresh(raw=True)

            image_path = capture_dir / f"pose_{index:03d}.png"
            cv2.imwrite(str(image_path), frame)
            records.append(
                {
                    "index": index,
                    "image": image_path.name,
                    "joint_positions_rad": [float(v) for v in radians.tolist()],
                    "raw_values": [float(v) for v in raw_values.tolist()],
                }
            )
            print(f"  captured {image_path.name}  joints(rad)="
                  f"{np.round(radians.numpy(), 4).tolist()}")
            index += 1

    interface.robot.disconnect()
    record_path = capture_dir / "records.json"
    record_path.write_text(
        json.dumps({"camera": args.camera, "records": records}, indent=2),
        encoding="utf-8",
    )
    print(f"\n[info] wrote {len(records)} records to {record_path}")
    if len(records) < 8:
        print("[warn] fewer than 8 poses; the solve will be poorly conditioned")
    return 0


def _target_pose_in_camera(args, frame, calibration, board, tag_id):
    """``T_cam_target`` for one image, or ``None`` if the target is not usable."""
    import cv2
    import numpy as np

    from so101.charuco import (
        board_object_points,
        detect_board,
        detect_gripper_tags,
        tag_object_points,
    )
    from so101.handeye import pose_matrix

    if args.camera == "wrist":
        detection = detect_board(frame, board)
        if not detection.usable(minimum=8):
            return None, detection.count
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
            return None, detection.count
        rotation, _ = cv2.Rodrigues(rvec)
        return (pose_matrix(rotation, tvec), object_points, image_points), detection.count

    tags = detect_gripper_tags(frame)
    if tag_id not in tags:
        return None, len(tags)
    image_points = tags[tag_id].astype(np.float64)
    object_points = tag_object_points()
    # IPPE_SQUARE is the estimator meant for a single square marker; it returns
    # the better-supported of the two solutions the planar ambiguity allows.
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        calibration.camera_matrix,
        calibration.distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None, len(tags)
    rotation, _ = cv2.Rodrigues(rvec)
    return (pose_matrix(rotation, tvec), object_points, image_points), len(tags)


def _board_pose_in_env(args, board):
    """Where the table board sits, according to the calibrated wrist camera.

    This is what makes the shared-board route work: the board does not move, so
    the wrist camera can measure its pose in the environment frame at leisure,
    and the front camera can then be solved from a single view of that same
    board -- with nothing attached to the gripper.

    Returns the pose, its scatter across the wrist poses, and how many were
    usable.  The scatter is the honest error bar: it folds in the wrist
    hand-eye error and every PnP error along the way.
    """
    import cv2
    import gymnasium as gym
    import numpy as np
    import torch

    import so101.tasks  # noqa: F401  (registers environments)
    from so101.camera_calibration import load_calibration, quat_wxyz_to_matrix
    from so101.charuco import board_object_points, detect_board
    from so101.configs import make_env_cfg
    from so101.handeye import board_pose_in_env, consensus, pose_matrix

    wrist = load_calibration("wrist")
    if wrist.extrinsic is None:
        raise SystemExit(
            "the wrist camera has no extrinsic yet. The shared-board route "
            "stands on the wrist hand-eye result, so solve that first."
        )
    if wrist.extrinsic.parent != "gripper":
        raise SystemExit(
            f"expected the wrist extrinsic to be relative to 'gripper', "
            f"got {wrist.extrinsic.parent!r}"
        )

    wrist_dir = DEFAULT_CAPTURE_ROOT / "wrist"
    record_path = wrist_dir / "records.json"
    if not record_path.is_file():
        raise SystemExit(f"no wrist capture at {record_path}")
    records = json.loads(record_path.read_text())["records"]

    cfg = make_env_cfg("so101-visual-StackCube-v0", num_envs=1, device=args.device)
    env = gym.make("so101-visual-StackCube-v0", cfg=cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    device = robot.data.joint_pos.device
    env_origin = env.scene.env_origins[0].cpu().numpy()
    gripper_index = list(robot.data.body_names).index("gripper")
    gripper_camera = wrist.extrinsic.matrix

    poses = []
    for record in records:
        frame = cv2.imread(str(wrist_dir / record["image"]))
        if frame is None:
            continue
        detection = detect_board(frame, board)
        if not detection.usable(minimum=8):
            continue
        object_points = board_object_points(board, detection.charuco_ids)
        image_points = detection.charuco_corners.reshape(-1, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, wrist.camera_matrix, wrist.distortion
        )
        if not ok:
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        camera_board = pose_matrix(rotation, tvec)

        joints = torch.tensor(
            record["joint_positions_rad"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        env_gripper = pose_matrix(
            quat_wxyz_to_matrix(robot.data.body_quat_w[0, gripper_index].cpu().numpy()),
            robot.data.body_pos_w[0, gripper_index].cpu().numpy() - env_origin,
        )
        poses.extend(board_pose_in_env([env_gripper], gripper_camera, [camera_board]))

    env.close()
    if len(poses) < 3:
        raise SystemExit(
            f"only {len(poses)} wrist frames localised the board; need at least 3"
        )
    best, scatter_mm, spread_deg = consensus(poses)
    return best, scatter_mm, spread_deg, len(poses)


def _solve_via_board(args) -> int:
    """Front camera pose from the board, with nothing on the gripper."""
    import cv2
    import numpy as np

    from so101.camera_calibration import (
        CameraExtrinsic,
        calibration_path,
        load_calibration,
        matrix_to_quat_wxyz,
    )
    from so101.charuco import (
        MARKER_MM,
        SQUARE_MM,
        board_object_points,
        charuco_board,
        detect_board,
    )
    from so101.handeye import camera_pose_from_board, consensus, pose_matrix

    if not args.front_image:
        print("[fail] --via-board needs --front-image with at least one frame")
        return 1
    try:
        front = load_calibration("front")
    except Exception as error:  # noqa: BLE001 - surfaced verbatim
        print(f"[fail] front intrinsics are required first: {error}")
        return 1

    square_mm = args.square_mm
    board = charuco_board(square_mm) if square_mm else charuco_board()

    env_board, board_scatter_mm, board_spread_deg, used = _board_pose_in_env(args, board)

    print("\n" + "=" * 68)
    print("SHARED BOARD: front camera from the wrist camera's view of the table")
    print("=" * 68)
    print(f"  wrist frames that localised the board : {used}")
    print(f"  board position scatter                : {board_scatter_mm:.2f} mm")
    print(f"  board orientation spread              : {board_spread_deg:.2f} deg")
    if board_scatter_mm > 10.0:
        print("  WARNING: the board's measured pose disagrees between wrist frames")
        print("  by more than 10 mm. That is the wrist hand-eye result showing")
        print("  through -- fix it before trusting anything derived from it.")

    solutions = []
    for path in args.front_image:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  skipping {path}: unreadable")
            continue
        detection = detect_board(frame, board)
        if not detection.usable(minimum=8):
            print(f"  skipping {path.name}: {detection.count} corners")
            continue
        object_points = board_object_points(board, detection.charuco_ids)
        image_points = detection.charuco_corners.reshape(-1, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, front.camera_matrix, front.distortion
        )
        if not ok:
            print(f"  skipping {path.name}: solvePnP did not converge")
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        camera_board = pose_matrix(rotation, tvec)
        solutions.append(camera_pose_from_board(env_board, camera_board))
        print(f"  {path.name}: {detection.count} corners")

    if not solutions:
        print("[fail] no front frame localised the board")
        return 1

    best, scatter_mm, spread_deg = consensus(solutions)

    print(f"\n  front frames used         : {len(solutions)}")
    print(f"  position across frames    : {scatter_mm:.2f} mm scatter")
    print(f"  orientation across frames : {spread_deg:.2f} deg spread")
    print(f"  camera position (env)     : {np.round(best[:3, 3], 4).tolist()} m")
    print(
        "\n  Note: this pose inherits the wrist hand-eye error on top of two PnP "
        "solves,\n  so expect it to be roughly twice as uncertain as a direct "
        "eye-to-hand result.\n  The alignment gate is what decides whether that is "
        "good enough."
    )

    front.extrinsic = CameraExtrinsic(
        parent="env",
        pos=tuple(float(v) for v in best[:3, 3]),
        quat_wxyz=matrix_to_quat_wxyz(best[:3, :3]),
    )
    front.notes = (front.notes + " | " if front.notes else "") + (
        f"shared board via wrist, {used} wrist frames, {len(solutions)} front "
        f"frames, board scatter {board_scatter_mm:.2f} mm"
    )
    output = calibration_path("front")
    front.save(output)
    print(f"  wrote extrinsic (parent=env, convention=ros) -> {output}")
    return 0


def _solve_inner(args) -> int:
    import cv2
    import gymnasium as gym
    import numpy as np
    import torch

    import so101.tasks  # noqa: F401  (registers environments)
    from so101.camera_calibration import (
        CameraExtrinsic,
        calibration_path,
        load_calibration,
        matrix_to_quat_wxyz,
        quat_wxyz_to_matrix,
    )
    from so101.charuco import MARKER_MM, SQUARE_MM, charuco_board, detect_gripper_tags
    from so101.configs import make_env_cfg
    from so101.handeye import inter_solver_spread, pose_matrix, solve_hand_eye

    capture_dir = args.capture_dir or (DEFAULT_CAPTURE_ROOT / args.camera)
    record_path = capture_dir / "records.json"
    if not record_path.is_file():
        print(f"[fail] no capture at {record_path}; run --capture first")
        return 1
    records = json.loads(record_path.read_text())["records"]

    try:
        calibration = load_calibration(args.camera)
    except Exception as error:  # noqa: BLE001 - surfaced verbatim below
        print(f"[fail] intrinsics for {args.camera!r} are required first: {error}")
        return 1

    square_mm = args.square_mm or SQUARE_MM
    board = charuco_board(square_mm, MARKER_MM * square_mm / SQUARE_MM)

    # The hand-eye target must be one rigid frame across every pose, so for the
    # gripper rig pick the single tag seen most often rather than mixing tags
    # whose relative placement is unknown.
    tag_id = None
    if args.camera == "front":
        counts: dict[int, int] = {}
        for record in records:
            frame = cv2.imread(str(capture_dir / record["image"]))
            if frame is None:
                continue
            for found in detect_gripper_tags(frame):
                counts[found] = counts.get(found, 0) + 1
        if not counts:
            print("[fail] no gripper tags detected in any capture")
            return 1
        tag_id = max(counts, key=lambda key: counts[key])
        print(f"[info] gripper tag usage {counts}; using id {tag_id}")

    cfg = make_env_cfg("so101-visual-StackCube-v0", num_envs=1, device=args.device)
    env = gym.make("so101-visual-StackCube-v0", cfg=cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    device = robot.data.joint_pos.device
    env_origin = env.scene.env_origins[0].cpu().numpy()
    gripper_index = list(robot.data.body_names).index("gripper")

    gripper_poses: list[np.ndarray] = []
    target_poses: list[np.ndarray] = []
    observations: list[tuple[np.ndarray, np.ndarray]] = []
    skipped = 0

    for record in records:
        frame = cv2.imread(str(capture_dir / record["image"]))
        if frame is None:
            skipped += 1
            continue
        result, count = _target_pose_in_camera(args, frame, calibration, board, tag_id)
        if result is None:
            print(f"  skipping {record['image']}: target not usable ({count} found)")
            skipped += 1
            continue
        target_pose, object_points, image_points = result

        joints = torch.tensor(
            record["joint_positions_rad"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

        position = robot.data.body_pos_w[0, gripper_index].cpu().numpy() - env_origin
        rotation = quat_wxyz_to_matrix(
            robot.data.body_quat_w[0, gripper_index].cpu().numpy()
        )
        gripper_poses.append(pose_matrix(rotation, position))
        target_poses.append(target_pose)
        observations.append((object_points, image_points))

    env.close()

    usable = len(gripper_poses)
    print(f"\n[info] {usable} usable poses ({skipped} skipped)")
    if usable < 6:
        print("[fail] need at least 6 usable poses")
        return 1

    eye_in_hand = args.camera == "wrist"

    print("\n" + "=" * 68)
    print(f"HAND-EYE: {args.camera}  ({'eye-in-hand' if eye_in_hand else 'eye-to-hand'})")
    print("=" * 68)

    results = solve_hand_eye(
        gripper_poses,
        target_poses,
        observations,
        calibration.camera_matrix,
        calibration.distortion,
        eye_in_hand,
    )
    if not results:
        print("[fail] every solver failed")
        return 1
    for result in results:
        print(
            f"  {result.method:<11} pos "
            f"{np.round(result.transform[:3, 3] * 1000, 2).tolist()} mm   "
            f"reproj {result.reprojection_rmse_px:6.2f} px   "
            f"rig scatter {result.rig_scatter_mm:5.2f} mm"
        )

    spread_mm, spread_deg = inter_solver_spread(results)
    print(
        f"\n  inter-solver spread: position {spread_mm:.2f} mm max, "
        f"rotation {spread_deg:.2f} deg max"
    )
    if spread_mm > 10.0:
        print(
            "  WARNING: solvers disagree by more than 10 mm. That points at "
            "pose diversity or kinematics, not at the solver -- recapture with "
            "more varied rotations before trusting any of these."
        )

    chosen = min(results, key=lambda r: r.reprojection_rmse_px)
    best, rmse, best_name = chosen.transform, chosen.reprojection_rmse_px, chosen.method
    print(f"\n  best by reprojection: {best_name}  ({rmse:.2f} px)")

    parent = "gripper" if eye_in_hand else "env"
    calibration.extrinsic = CameraExtrinsic(
        parent=parent,
        pos=tuple(float(v) for v in best[:3, 3]),
        quat_wxyz=matrix_to_quat_wxyz(best[:3, :3]),
    )
    calibration.handeye_rmse_px = float(rmse)
    notes = (calibration.notes + " | " if calibration.notes else "") + (
        f"hand-eye {best_name}, {usable} poses, "
        f"inter-solver spread {spread_mm:.2f} mm"
    )
    calibration.notes = notes
    output = calibration_path(args.camera)
    calibration.save(output)
    print(f"  wrote extrinsic (parent={parent}, convention=ros) -> {output}")
    return 0


def solve(args) -> int:
    from isaaclab.app import AppLauncher

    args.enable_cameras = False
    args.headless = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        if args.via_board:
            if args.camera != "front":
                print("[fail] --via-board applies to the front camera only")
                return 1
            return _solve_via_board(args)
        return _solve_inner(args)
    finally:
        simulation_app.close()


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.capture == args.solve:
        print("[fail] choose exactly one of --capture or --solve")
        return 1
    return capture(args) if args.capture else solve(args)


if __name__ == "__main__":
    raise SystemExit(main())
