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

The default way to solve the front camera is a ChArUco board taped flat to
the gripper (``--camera front --capture`` / ``--solve``, no extra flag):
each camera is then solved independently, from its own images, with no
dependency between them.  Tape is enough -- it does not need to be a
permanent mount.  It is a full board, not a single marker glued to an
angled mount: a board's many corners resist the head-on planar-pose
ambiguity far better than one 4-corner marker would, and hand-eye capture
already spans a wide range of angles across poses (see
generate_handeye_poses.py), so a flat board is enough.  Both cameras'
targets share one ArUco dictionary but disjoint id ranges (table board:
0-19, gripper board: 20-29), so either can be in frame for the other's
capture without being confused for it.

``--via-board`` is the fallback for when nothing can be attached to the
gripper.  It needs the wrist camera calibrated first: the wrist camera
measures where the static table board sits, and the front camera is then
solved from its own view of that same board.  Because the board does not
move, the two cameras need not even see it at the same moment.  The cost is
that the result inherits the wrist hand-eye error on top of two PnP solves --
measured at roughly 5.8 mm for a 2 mm error in the board pose -- so this route
is only as good as the wrist solve underneath it.

Examples:

    python scripts/calibrate_handeye.py --camera wrist --capture
    python -u scripts/calibrate_handeye.py --camera wrist --solve

    python scripts/calibrate_handeye.py --camera front --capture
    python -u scripts/calibrate_handeye.py --camera front --solve

    # fallback, no tag on the gripper:
    python -u scripts/calibrate_handeye.py --camera front --solve --via-board \
        --front-image outputs/camera_views/front_raw_000.png
"""

from __future__ import annotations

import argparse
import atexit
import json
import sys
import threading
import time
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
    parser.add_argument("--robot-id", default="my_follower")
    parser.add_argument("--leader-port", default="/dev/so101-leader")
    parser.add_argument("--leader-id", default="my_leader")
    parser.add_argument(
        "--leader-teleop",
        action="store_true",
        help="mirror a physical leader arm into the follower during manual capture",
    )
    parser.add_argument(
        "--square-mm",
        type=float,
        default=None,
        help="measured printed square size, for whichever board --camera "
        "implies (table for wrist, gripper for front); scales the marker "
        "size with it, correcting for a printer that missed 100%% scale",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="command the arm through the pose sequence instead of waiting for "
        "the operator to place it",
    )
    parser.add_argument("--settle", type=float, default=1.5)
    parser.add_argument("--force", action="store_true",
                        help="--solve: write the best solution even when the quality checks fail (verify it visually)")
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
    parser.add_argument(
        "--front-calibration",
        default=None,
        help="--via-board: which front intrinsics the frames were taken with, "
        "e.g. front_1280x720. Defaults to matching the image size.",
    )
    parser.add_argument(
        "--known-board-center-env",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="--via-board: known geometric centre of the fixed table board in "
        "the env frame, metres; bypasses the wrist extrinsic",
    )
    parser.add_argument(
        "--known-board-yaw-deg",
        type=float,
        default=0.0,
        help="--via-board: yaw of the table board frame in env degrees",
    )
    return parser


def capture(args) -> int:
    """Record (joint angles, image) pairs from the real robot."""
    import logging

    import cv2
    import numpy as np
    import torch

    from so101.charuco import charuco_board, detect_board, gripper_board
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from so101.real.cameras import DEFAULT_SPECS, Camera
    from so101.real.constants import SO101_JOINT_ORDER, SO101_USD_MAPPING
    from so101.real.interface import LeRobotSO101Interface

    # Manual/replay capture needs a live preview with key events: OpenCV's GUI
    # when this build has one, otherwise a matplotlib window (LeRobot pins the
    # headless OpenCV wheel). Check before opening serial devices or releasing
    # torque.
    from so101.real.preview import PreviewWindow, preview_available

    if not preview_available():
        capture_dir = args.capture_dir or (DEFAULT_CAPTURE_ROOT / args.camera)
        print("[fail] no preview possible: headless OpenCV and no DISPLAY.")
        print("       Run this from the robot PC's desktop session.")
        print("       No robot command was sent and torque was not changed.")
        print("       Or use the browser capture tool instead:")
        print(
            "       python scripts/web_handeye_capture.py "
            f"--camera {args.camera} --capture-dir {capture_dir}"
        )
        return 1

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

    if args.leader_teleop and args.replay:
        print("[fail] --leader-teleop cannot be combined with --replay")
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
    leader = None
    robot_stop = threading.Event()
    robot_thread = None
    robot_state_lock = threading.Lock()
    latest_observation = None
    robot_error = None

    def cleanup_connections():
        robot_stop.set()
        if robot_thread is not None:
            robot_thread.join(timeout=2.0)
        try:
            interface.robot.bus.disable_torque()
        except Exception:
            pass
        try:
            interface.robot.disconnect()
        except Exception:
            pass
        if leader is not None:
            try:
                leader.robot.disconnect()
            except Exception:
                pass

    atexit.register(cleanup_connections)

    manual_mode = poses is None
    if manual_mode and args.leader_teleop:
        leader = LeRobotSO101Interface(
            device="cpu",
            port=args.leader_port,
            id=args.leader_id,
            cameras={},
            fps=30,
            kind="leader",
        )
        leader.init_device()
        leader.connect()

        interface.robot.bus.enable_torque()

        def robot_io_loop():
            """Own both serial ports; no other thread may touch robot objects."""
            nonlocal latest_observation, robot_error
            control_period = 1.0 / 30.0
            observation_period = 1.0 / 10.0
            last_observation_at = 0.0
            have_observation = False
            while not robot_stop.is_set():
                started = time.monotonic()
                try:
                    leader_action = leader.robot.get_action()
                    follower_action = {
                        joint: float(leader_action[joint])
                        for joint in SO101_JOINT_ORDER
                    }
                    interface.robot.send_action(follower_action)
                    now = time.monotonic()
                    if (
                        not have_observation
                        or now - last_observation_at >= observation_period
                    ):
                        observation = interface.robot.get_observation()
                        with robot_state_lock:
                            latest_observation = dict(observation)
                        last_observation_at = now
                        have_observation = True
                except Exception as error:  # noqa: BLE001
                    robot_error = error
                    robot_stop.set()
                    print(f"\n[teleop] stopped: {error}", flush=True)
                    break
                remaining = control_period - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)

        robot_thread = threading.Thread(target=robot_io_loop, daemon=True)
        robot_thread.start()
        print(
            "\n[leader teleop] Move the leader arm; the follower mirrors it. "
            "Hold the leader still, then press ENTER/SPACE to capture."
        )
    elif manual_mode:
        print(
            "\n[manual capture] The table board must stay fixed. You will "
            "hand-guide the arm."
        )
        input("Support the arm, then press ENTER to release torque: ")
        interface.robot.bus.disable_torque()
        print("Torque OFF. Move the arm while supporting it.")

    spec = DEFAULT_SPECS[args.camera]
    target_board = charuco_board() if args.camera == "wrist" else gripper_board()
    joint_names = [joint.split(".")[0] for joint in SO101_JOINT_ORDER]
    joint_limits_deg = np.array(
        [
            [SO101_USD_MAPPING[name]["joint_min"], SO101_USD_MAPPING[name]["joint_max"]]
            for name in joint_names
        ],
        dtype=np.float32,
    )

    def read_joint_status():
        if args.leader_teleop:
            with robot_state_lock:
                observation = None if latest_observation is None else dict(latest_observation)
            if robot_error is not None:
                raise ConnectionError(f"leader teleop stopped: {robot_error}")
            if observation is None:
                return np.zeros(len(SO101_JOINT_ORDER)), np.zeros(
                    len(SO101_JOINT_ORDER), dtype=bool
                )
        else:
            observation = interface.robot.get_observation()
        raw = torch.tensor(
            [float(observation[joint]) for joint in SO101_JOINT_ORDER],
            dtype=torch.float32,
        )
        radians = interface.get_mapped_actions_vectorized(raw).numpy()
        degrees = np.degrees(radians)
        valid = np.logical_and(
            degrees >= joint_limits_deg[:, 0], degrees <= joint_limits_deg[:, 1]
        )
        return degrees, valid

    def draw_joint_status(image, degrees, valid):
        overall = "SIM JOINTS: OK" if bool(np.all(valid)) else "SIM JOINTS: OUT OF LIMIT"
        cv2.putText(
            image, overall, (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            (0, 255, 0) if np.all(valid) else (0, 0, 255), 2, cv2.LINE_AA,
        )
        for index, (name, value, good) in enumerate(zip(joint_names, degrees, valid)):
            y = 116 + index * 24
            text = f"{name:<12} {value:+6.1f} deg  [{joint_limits_deg[index, 0]:+.0f},{joint_limits_deg[index, 1]:+.0f}]"
            cv2.putText(
                image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                (0, 220, 0) if good else (0, 0, 255), 1, cv2.LINE_AA,
            )

    records: list[dict] = []
    # Raw frames: the target's apparent position must carry the lens
    # distortion that the intrinsic calibration measured, since solvePnP is
    # given that same distortion model.
    with Camera(spec, calibration=None, rectify=False) as camera:
        print(f"[info] {args.camera} on {spec.device}, exposure "
              f"{camera.exposure_lock.as_dict()}")
        preview_name = f"hand-eye capture ({args.camera})"
        window = PreviewWindow(preview_name)

        def wait_for_preview_key(message: str) -> str:
            """Keep streaming while waiting for Enter or q/Esc."""
            print(message)
            while True:
                live = camera.read_fresh(raw=True)
                live_detection = detect_board(live, target_board)
                live_degrees, live_valid = read_joint_status()
                cv2.putText(
                    live,
                    f"ChArUco corners: {live_detection.count}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 0) if live_detection.usable() else (0, 180, 255),
                    2, cv2.LINE_AA,
                )
                cv2.putText(
                    live, message, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )
                draw_joint_status(live, live_degrees, live_valid)
                window.show(live)
                key = window.poll_key()
                if key in (10, 13, 32):
                    return "enter"
                if key in (ord("q"), ord("Q"), 27):
                    return "q"

        index = 0
        quit_requested = False
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
                time.sleep(args.settle)
            else:
                # Keep the live view running while the operator positions the
                # hand-guided arm.  Enter/Space captures; q/Esc finishes.
                while True:
                    preview = camera.read_fresh(raw=True)
                    preview_detection = detect_board(preview, target_board)
                    preview_degrees, preview_valid = read_joint_status()
                    shown = preview.copy()
                    cv2.putText(
                        shown,
                        f"{args.camera}  ChArUco corners: {preview_detection.count}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 255, 0) if preview_detection.usable() else (0, 180, 255),
                        2, cv2.LINE_AA,
                    )
                    cv2.putText(
                        shown,
                        (
                            "ENTER/SPACE: capture    Q/ESC: finish"
                            if not args.leader_teleop
                            else "LEADER: hold pose    ENTER/SPACE: capture    Q/ESC: finish"
                        ),
                        (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA,
                    )
                    draw_joint_status(shown, preview_degrees, preview_valid)
                    window.show(shown)
                    key = window.poll_key()
                    if key in (10, 13, 32):
                        break
                    if key in (ord("q"), ord("Q"), 27):
                        quit_requested = True
                        break

                if quit_requested:
                    break

                # Latch the exact hand-guided position before capture.  Write
                # the goal while torque is still off so enabling torque cannot
                # recall a stale goal and jump unexpectedly.
                if not args.leader_teleop:
                    current = interface.robot.get_observation()
                    hold = {
                        joint: float(current[joint]) for joint in SO101_JOINT_ORDER
                    }
                    interface.robot.send_action(hold)
                    interface.robot.bus.enable_torque()
                    interface.robot.send_action(hold)
                if args.leader_teleop:
                    time.sleep(args.settle)
                else:
                    time.sleep(args.settle)

            if args.leader_teleop:
                with robot_state_lock:
                    observation = (
                        None if latest_observation is None else dict(latest_observation)
                    )
                if robot_error is not None:
                    raise ConnectionError(f"leader teleop stopped: {robot_error}")
                if observation is None:
                    raise RuntimeError("leader teleop has not produced a follower observation yet")
            else:
                observation = interface.robot.get_observation()
            raw_values = torch.tensor(
                [float(observation[joint]) for joint in SO101_JOINT_ORDER],
                dtype=torch.float32,
            )
            radians = interface.get_mapped_actions_vectorized(raw_values)
            degrees = np.degrees(radians.numpy())
            valid = np.logical_and(
                degrees >= joint_limits_deg[:, 0], degrees <= joint_limits_deg[:, 1]
            )
            if not np.all(valid):
                bad = ", ".join(
                    f"{joint_names[i]}={degrees[i]:.1f}deg"
                    for i in range(len(joint_names)) if not valid[i]
                )
                print(f"  NOT saved: outside Isaac joint limits ({bad})")
                if manual_mode and not args.leader_teleop:
                    interface.robot.bus.disable_torque()
                    print("Torque OFF. Move to a pose within the displayed limits.")
                    continue
            frame = camera.read_fresh(raw=True)
            detection = detect_board(frame, target_board)
            if manual_mode and not detection.usable():
                print(
                    f"  NOT saved: only {detection.count} ChArUco corners detected. "
                    "Reposition so the fixed board is visible."
                )
                answer = wait_for_preview_key(
                    "ENTER/SPACE: release torque and retry    Q/ESC: finish"
                )
                if not args.leader_teleop:
                    interface.robot.bus.disable_torque()
                if answer == "q":
                    quit_requested = True
                    break
                if not args.leader_teleop:
                    print("Torque OFF. Move the arm to a visible pose.")
                continue

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
            print(f"  captured {image_path.name}  corners={detection.count}  joints(rad)="
                  f"{np.round(radians.numpy(), 4).tolist()}")
            index += 1

            if manual_mode:
                answer = wait_for_preview_key(
                    (
                        "ENTER/SPACE: release torque and move next    Q/ESC: finish"
                        if not args.leader_teleop
                        else "LEADER: move to next pose    ENTER/SPACE: capture    Q/ESC: finish"
                    )
                )
                if not args.leader_teleop:
                    interface.robot.bus.disable_torque()
                if answer == "q":
                    break
                if not args.leader_teleop:
                    print("Torque OFF. Move the arm to the next pose.")

    window.close()
    cleanup_connections()
    atexit.unregister(cleanup_connections)
    record_path = capture_dir / "records.json"
    record_path.write_text(
        json.dumps({"camera": args.camera, "records": records}, indent=2),
        encoding="utf-8",
    )
    print(f"\n[info] wrote {len(records)} records to {record_path}")
    if len(records) < 8:
        print("[warn] fewer than 8 poses; the solve will be poorly conditioned")
    return 0


def _record_joints(record, mapping, device):
    """Isaac joints for one capture record, (1, 6) on `device`.

    Recomputed from the follower's raw readings with the mapping in force now
    (so101.real.joint_mapping.follower_mapping) rather than taken from the
    radians stored at capture time, so adopting a fitted joint mapping
    re-solves the cameras against the kinematics the policy will actually
    run with. Records without raw_values fall back to the stored radians.
    """
    import torch

    if "raw_values" in record:
        raw = torch.tensor(record["raw_values"], dtype=torch.float32)
        joints = mapping.to_sim(raw)
    else:
        joints = torch.tensor(record["joint_positions_rad"], dtype=torch.float32)
    return joints.to(device).unsqueeze(0)


def _target_pose_in_camera(frame, calibration, board):
    """``T_cam_target`` for one image, or ``None`` if the target is not usable.

    Identical for both cameras now: wrist looks for the table board, front
    for the gripper board, but both are ChArUco boards solved the same way.
    A single flat board resists the head-on planar-pose-ambiguity far better
    than a lone 4-corner marker would (many more, more spread-out
    correspondences for solvePnP to work with), and hand-eye capture already
    spans a wide range of angles across poses -- see generate_handeye_poses.py
    -- which is what actually keeps any one near-head-on view from mattering.
    """
    import cv2
    import numpy as np

    from so101.charuco import board_object_points, detect_board
    from so101.handeye import pose_matrix

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

    # Hand-eye solving only needs the robot articulation for forward
    # kinematics.  Use the non-visual task so Isaac does not spawn camera
    # sensors while AppLauncher deliberately runs with cameras disabled.
    cfg = make_env_cfg("so101-StackCube-v0", num_envs=1, device=args.device)
    env = gym.make("so101-StackCube-v0", cfg=cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    device = robot.data.joint_pos.device
    env_origin = env.scene.env_origins[0].cpu().numpy()
    gripper_index = list(robot.data.body_names).index("gripper")
    # The joint readings are the follower's (web and leader-driven captures
    # alike), so they go through the follower mapping.
    from so101.real.joint_mapping import follower_mapping

    mapping = follower_mapping(args.robot_id)
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

        joints = _record_joints(record, mapping, device)
        robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        robot.set_joint_position_target(joints)
        robot.write_data_to_sim()
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

    from so101.real.cameras import calibration_name

    if not args.front_image:
        print("[fail] --via-board needs --front-image with at least one frame")
        return 1

    # Intrinsics are per-resolution, so pick the record matching the frames
    # rather than assuming they were shot at the task resolution.
    stem = args.front_calibration
    if stem is None:
        probe = cv2.imread(str(args.front_image[0]))
        if probe is None:
            print(f"[fail] could not read {args.front_image[0]}")
            return 1
        stem = calibration_name("front", probe.shape[1], probe.shape[0])
        print(f"[info] frames are {probe.shape[1]}x{probe.shape[0]}; using "
              f"intrinsics {stem!r}")
    try:
        front = load_calibration(stem)
    except Exception as error:  # noqa: BLE001 - surfaced verbatim
        print(f"[fail] intrinsics {stem!r} are required first: {error}")
        return 1

    square_mm = args.square_mm
    board = charuco_board(square_mm) if square_mm else charuco_board()

    if args.known_board_center_env is not None:
        # PnP uses the ChArUco board's outer upper-left corner as its frame
        # origin.  The user supplies the geometric centre of the printed
        # rectangle, so convert that centre to the board-frame origin.
        from so101.charuco import load_board_spec

        spec = load_board_spec()
        centre = np.asarray(args.known_board_center_env, dtype=np.float64)
        yaw = np.deg2rad(float(args.known_board_yaw_deg))
        cy, sy = np.cos(yaw), np.sin(yaw)
        # OpenCV's board frame has +x across the board, +y down the printed
        # face, and +z away from that face.  A board lying face-up on the
        # tabletop therefore maps to Rx(pi) before applying its in-plane yaw.
        board_rotation = np.array(
            [[cy, sy, 0.0], [sy, -cy, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )
        centre_offset = np.array(
            [spec.width_mm / 2000.0, spec.height_mm / 2000.0, 0.0],
            dtype=np.float64,
        )
        board_origin = centre - board_rotation @ centre_offset
        env_board = pose_matrix(board_rotation, board_origin)
        board_scatter_mm = 0.0
        board_spread_deg = 0.0
        used = 0
        print("[info] using known table-board centre; wrist extrinsic bypassed")
        print(f"[info] board centre env: {np.round(centre, 6).tolist()} m")
        print(f"[info] board origin env: {np.round(board_origin, 6).tolist()} m")
    else:
        env_board, board_scatter_mm, board_spread_deg, used = _board_pose_in_env(args, board)

    print("\n" + "=" * 68)
    print(
        "KNOWN BOARD: front camera from fixed table-board position"
        if args.known_board_center_env is not None
        else "SHARED BOARD: front camera from the wrist camera's view of the table"
    )
    print("=" * 68)
    print(f"  wrist frames that localised the board : {used}")
    print(f"  board position scatter                : {board_scatter_mm:.2f} mm")
    print(f"  board orientation spread              : {board_spread_deg:.2f} deg")
    if board_scatter_mm > 10.0:
        print("  WARNING: the board's measured pose disagrees between wrist frames")
        print("  by more than 10 mm. That is the wrist hand-eye result showing")
        print("  through -- fix it before trusting anything derived from it.")

    solutions = []
    reprojection_squared_errors = []
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
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, front.camera_matrix, front.distortion
        )
        reprojection_squared_errors.extend(
            np.sum(
                (projected.reshape(-1, 2) - image_points) ** 2,
                axis=1,
            ).tolist()
        )
        print(f"  {path.name}: {detection.count} corners")

    if not solutions:
        print("[fail] no front frame localised the board")
        return 1

    best, scatter_mm, spread_deg = consensus(solutions)

    print(f"\n  front frames used         : {len(solutions)}")
    print(f"  position across frames    : {scatter_mm:.2f} mm scatter")
    print(f"  orientation across frames : {spread_deg:.2f} deg spread")
    print(f"  camera position (env)     : {np.round(best[:3, 3], 4).tolist()} m")
    if args.known_board_center_env is not None:
        print(
            "\n  Note: this pose uses the measured board position directly and "
            "does not depend on the wrist extrinsic."
        )
    else:
        print(
            "\n  Note: this pose inherits the wrist hand-eye error on top of two PnP "
            "solves,\n  so expect it to be roughly twice as uncertain as a direct "
            "eye-to-hand result.\n  The alignment gate is what decides whether that is "
            "good enough."
        )

    # The pose goes on the record the task actually runs with.  A camera's
    # position is a physical fact, so one measured from higher-resolution
    # frames applies unchanged -- only the intrinsics are resolution-specific.
    try:
        target = load_calibration("front")
    except Exception as error:  # noqa: BLE001 - surfaced verbatim
        print(f"[fail] the task-resolution front intrinsics are needed too: {error}")
        return 1
    target.extrinsic = CameraExtrinsic(
        parent="env",
        pos=tuple(float(v) for v in best[:3, 3]),
        quat_wxyz=matrix_to_quat_wxyz(best[:3, :3]),
    )
    target.handeye_rmse_px = float(np.sqrt(np.mean(reprojection_squared_errors)))
    if args.known_board_center_env is not None:
        target.notes = (
            f"known fixed table board centre env="
            f"{np.round(args.known_board_center_env, 6).tolist()} m, "
            f"yaw={args.known_board_yaw_deg:.1f} deg, {len(solutions)} front frames "
            f"from {stem}, pose scatter {scatter_mm:.2f} mm, "
            f"orientation spread {spread_deg:.2f} deg"
        )
    output = calibration_path("front")
    target.save(output)
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
    from so101.charuco import charuco_board, gripper_board
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

    # wrist watches the table board (fixed on the table); front watches the
    # gripper board (taped to the gripper) -- disjoint marker id ranges, so
    # whichever one happens to also be in frame is simply ignored.
    if args.camera == "wrist":
        board = charuco_board(args.square_mm) if args.square_mm else charuco_board()
    else:
        board = gripper_board(args.square_mm) if args.square_mm else gripper_board()

    # Hand-eye solving only needs the robot articulation for forward
    # kinematics.  Use the non-visual task so Isaac does not spawn camera
    # sensors while AppLauncher deliberately runs with cameras disabled.
    cfg = make_env_cfg("so101-StackCube-v0", num_envs=1, device=args.device)
    env = gym.make("so101-StackCube-v0", cfg=cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    device = robot.data.joint_pos.device
    env_origin = env.scene.env_origins[0].cpu().numpy()
    gripper_index = list(robot.data.body_names).index("gripper")
    # The joint readings are the follower's (web and leader-driven captures
    # alike), so they go through the follower mapping.
    from so101.real.joint_mapping import follower_mapping

    mapping = follower_mapping(args.robot_id)

    gripper_poses: list[np.ndarray] = []
    target_poses: list[np.ndarray] = []
    observations: list[tuple[np.ndarray, np.ndarray]] = []
    skipped = 0

    for record in records:
        frame = cv2.imread(str(capture_dir / record["image"]))
        if frame is None:
            skipped += 1
            continue
        result, count = _target_pose_in_camera(frame, calibration, board)
        if result is None:
            print(f"  skipping {record['image']}: target not usable ({count} found)")
            skipped += 1
            continue
        target_pose, object_points, image_points = result

        joints = _record_joints(record, mapping, device)
        robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        robot.set_joint_position_target(joints)
        robot.write_data_to_sim()
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

    # A returned transform is not necessarily a valid calibration.  Badly
    # paired FK/image samples still produce plausible quaternions, so preserve
    # the last known pose unless independent quality signals agree.
    failures = []
    if spread_mm > 10.0:
        failures.append(f"inter-solver position spread {spread_mm:.2f} mm > 10 mm")
    if spread_deg > 5.0:
        failures.append(f"inter-solver rotation spread {spread_deg:.2f} deg > 5 deg")
    if chosen.rig_scatter_mm > 10.0:
        failures.append(
            f"rig position scatter {chosen.rig_scatter_mm:.2f} mm > 10 mm"
        )
    if rmse > 5.0:
        failures.append(f"reprojection RMSE {rmse:.2f} px > 5 px")
    if failures and not getattr(args, "force", False):
        print("\n[fail] refusing to overwrite the calibration:")
        for failure in failures:
            print(f"  - {failure}")
        print("  Recapture after checking joint readback, target rigidity, and pose diversity.")
        return 1
    if failures:
        print("\n[warn] --force: writing despite:")
        for failure in failures:
            print(f"  - {failure}")

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
    parser = build_parser()
    # Real capture has no Isaac dependency and runs in the lightweight
    # hardware environment.  Only add Isaac's launcher arguments for solve;
    # importing AppLauncher unconditionally made `--capture` fail before it
    # could even open the robot in that environment.
    if "--solve" in sys.argv:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.capture == args.solve:
        print("[fail] choose exactly one of --capture or --solve")
        return 1
    return capture(args) if args.capture else solve(args)


if __name__ == "__main__":
    raise SystemExit(main())
