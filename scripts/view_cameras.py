"""Look at what the real cameras see, with detection and rectification overlays.

Replaces ``stream_d435i.py``, which targeted a RealSense that is not part of
this rig and imported a package the project never declared.

The installed OpenCV is the headless wheel, so there is no preview window and
the default is to write annotated frames to disk.  That also happens to be the
mode that works over a remote session, which is when these images are most
needed.  ``--show`` opens a window if a desktop build of ``opencv-python`` is
installed instead.

Examples:

    python scripts/view_cameras.py --snapshot
    python scripts/view_cameras.py --camera front --watch 20 --interval 1.0
    python scripts/view_cameras.py --show --camera wrist
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from so101.charuco import (  # noqa: E402
    GRIPPER_TAG_IDS,
    charuco_board,
    detect_board,
    detect_gripper_tags,
)
from so101.real.cameras import DEFAULT_SPECS, Camera, open_camera  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DEFAULT_OUT = REPO_ROOT / "outputs" / "camera_views"


def gui_available() -> bool:
    try:
        cv2.namedWindow("__probe__")
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False


def annotate(
    frame: np.ndarray,
    camera: Camera,
    fps: float | None,
    detect: bool,
) -> np.ndarray:
    canvas = frame.copy()
    lines = [
        f"{camera.spec.name}  {camera.spec.device}  "
        f"{camera.spec.width}x{camera.spec.height} {camera.spec.fourcc}",
        f"rectified: {'YES' if camera.is_rectified else 'NO (uncalibrated)'}",
    ]
    if fps is not None:
        lines.append(f"{fps:.1f} fps")
    lock = camera.exposure_lock.as_dict()
    if lock:
        locked = ", ".join(f"{k}={v}" for k, v in lock.items() if k != "unsupported_controls")
        lines.append(f"locked: {locked}")
        if camera.exposure_lock.unsupported:
            lines.append(f"not lockable: {', '.join(camera.exposure_lock.unsupported)}")

    if detect:
        board = charuco_board()
        detection = detect_board(canvas, board)
        if detection.count:
            cv2.aruco.drawDetectedCornersCharuco(
                canvas, detection.charuco_corners, detection.charuco_ids, (0, 255, 0)
            )
        tags = detect_gripper_tags(canvas)
        for tag_id, corners in tags.items():
            pts = corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], True, (0, 128, 255), 2)
            cv2.putText(
                canvas,
                str(tag_id),
                tuple(corners[0].astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 128, 255),
                1,
                cv2.LINE_AA,
            )
        lines.append(
            f"board corners: {detection.count}   "
            f"gripper tags: {sorted(tags)} of {list(GRIPPER_TAG_IDS)}"
        )

    for index, line in enumerate(lines):
        y = 18 + index * 18
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="View the real SO-101 cameras.")
    parser.add_argument(
        "--camera",
        action="append",
        choices=sorted(DEFAULT_SPECS),
        help="camera to view; repeatable, defaults to both",
    )
    parser.add_argument("--raw", action="store_true", help="show unrectified frames")
    parser.add_argument(
        "--no-detect", action="store_true", help="skip ChArUco/ArUco overlays"
    )
    parser.add_argument("--snapshot", action="store_true", help="one frame each, then exit")
    parser.add_argument("--watch", type=int, default=0, help="save this many frames each")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--show", action="store_true", help="open a window instead")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="readout size; use the higher one when photographing the board for "
        "the extrinsic solve",
    )
    args = parser.parse_args()

    names = args.camera or sorted(DEFAULT_SPECS)
    detect = not args.no_detect

    if args.show and not gui_available():
        print(
            "[fail] this OpenCV build has no GUI (opencv-python-headless).\n"
            "       Install opencv-python for a window, or drop --show to write "
            "annotated frames to disk."
        )
        return 1

    cameras = {
        name: open_camera(
            name, rectify=not args.raw, width=args.width, height=args.height
        )
        for name in names
    }
    for name, camera in cameras.items():
        if args.raw:
            print(f"[info] {name}: showing RAW frames (rectification bypassed)")
        elif not camera.is_rectified:
            print(f"[info] {name}: no calibration yet, frames are raw")

    try:
        if args.show:
            print("[info] press q to quit")
            last = time.time()
            while True:
                for name, camera in cameras.items():
                    frame = camera.read_raw() if args.raw else camera.read()
                    now = time.time()
                    fps = 1.0 / max(now - last, 1e-6)
                    last = now
                    cv2.imshow(name, annotate(frame, camera, fps, detect))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()
            return 0

        args.out.mkdir(parents=True, exist_ok=True)
        shots = 1 if args.snapshot or not args.watch else args.watch
        for index in range(shots):
            if index:
                time.sleep(args.interval)
            for name, camera in cameras.items():
                # Timing a single read measures how long a buffered frame took
                # to hand over, not the stream rate; average a few instead.
                start = time.time()
                for _ in range(10):
                    frame = camera.read_raw() if args.raw else camera.read()
                fps = 10.0 / max(time.time() - start, 1e-6)
                suffix = "raw" if args.raw else "rect"
                path = args.out / f"{name}_{suffix}_{index:03d}.png"
                cv2.imwrite(str(path), annotate(frame, camera, fps, detect))
                print(f"wrote {path}")
        return 0
    finally:
        for camera in cameras.values():
            camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
