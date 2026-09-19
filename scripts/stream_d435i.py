"""Display an Intel RealSense D435i color stream."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    profile = pipeline.start(config)
    device = profile.get_device()
    device_name = device.get_info(rs.camera_info.name)
    serial = device.get_info(rs.camera_info.serial_number)

    print(
        f"[INFO] Streaming {device_name} ({serial}) at "
        f"{args.width}x{args.height} @ {args.fps} Hz"
    )
    print("[INFO] Press q or Esc in the viewer window to stop.")

    frame_count = 0
    measured_at = time.monotonic()
    measured_fps = 0.0
    window_name = "Intel RealSense D435i: RGB"

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            frame_count += 1
            now = time.monotonic()
            elapsed = now - measured_at
            if elapsed >= 1.0:
                measured_fps = frame_count / elapsed
                frame_count = 0
                measured_at = now

            cv2.putText(
                color_image,
                f"{args.width}x{args.height}  {measured_fps:.1f} FPS",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, color_image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
