"""Move the real SO-101 smoothly to a pose and stream both cameras."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", nargs=6, type=float,
                        default=[-0.07, -0.47, 0.17, 1.44, 0.0, -0.013],
                        help="Isaac joint radians; default: wrist views table board")
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--robot-id", default="my_follower")
    parser.add_argument("--move-seconds", type=float, default=5.0)
    parser.add_argument("--hold-seconds", type=float, default=120.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "real_camera_stream",
        help="directory for latest_front.png/latest_wrist.png when GUI is unavailable",
    )
    parser.add_argument("--yes", action="store_true", help="skip the move confirmation")
    args = parser.parse_args()

    from so101.real.constants import SO101_JOINT_ORDER
    from so101.real.cameras import open_camera
    from so101.real.interface import LeRobotSO101Interface
    from so101.real.preview import PreviewWindow, preview_available

    target = torch.tensor(args.pose, dtype=torch.float32)
    if not args.yes:
        answer = input(
            f"Move follower to radians {np.round(args.pose, 4).tolist()}? "
            "Type MOVE to continue: "
        ).strip()
        if answer != "MOVE":
            print("Aborted; no command sent.")
            return 1

    interface = LeRobotSO101Interface(
        device="cpu", port=args.port, id=args.robot_id,
        cameras={}, fps=30, kind="follower",
    )
    cameras = {}
    windows = {}
    gui = preview_available()
    args.out.mkdir(parents=True, exist_ok=True)
    if not gui:
        print(f"[info] OpenCV has no GUI; saving live frames under {args.out}")
    try:
        interface.init_device()
        interface.connect()
        observation = interface.robot.get_observation()
        current_raw = torch.tensor(
            [float(observation[joint]) for joint in SO101_JOINT_ORDER],
            dtype=torch.float32,
        )
        current = interface.get_mapped_actions_vectorized(current_raw)

        cameras = {
            name: open_camera(name, rectify=True)
            for name in ("front", "wrist")
        }
        if gui:
            for name in cameras:
                windows[name] = PreviewWindow(name, size=(640, 480))
        print("[info] cameras opened; moving with interpolated commands")
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            alpha = min(elapsed / max(args.move_seconds, 0.1), 1.0)
            # Smooth cosine easing avoids a sudden first/last command.
            eased = 0.5 - 0.5 * np.cos(np.pi * alpha)
            pose = current + (target - current) * float(eased)
            raw = interface.get_raw_actions_from_radians(pose)
            action = {
                joint: float(value)
                for joint, value in zip(SO101_JOINT_ORDER, raw.tolist())
            }
            interface.robot.send_action(action)

            for name, camera in cameras.items():
                frame = camera.read()
                if gui:
                    cv2.putText(
                        frame, f"{name} | target pose | q=quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 0), 2, cv2.LINE_AA,
                    )
                    windows[name].show(frame)
                cv2.imwrite(str(args.out / f"latest_{name}.png"), frame)
            quit_requested = any(
                window.poll_key() in (ord("q"), 27) for window in windows.values()
            )
            if quit_requested or elapsed >= args.move_seconds + args.hold_seconds:
                break
            time.sleep(1.0 / 30.0)
    finally:
        for camera in cameras.values():
            camera.close()
        try:
            interface.robot.disconnect()
        except Exception:
            pass
        for window in windows.values():
            window.close()
    print("[info] stopped and disconnected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
