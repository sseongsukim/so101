"""Capture portfolio evidence without changing calibration or training data.

real-pose saves rectified clean frames and measured joints for stream_sim_cameras.
export-rollout exports recorded RGB cameras at their measured wall-clock timing.
touch-guide makes a coordinate guide, not a photograph or a calibration result.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
DEFAULT_OUT = REPO_ROOT / "outputs/portfolio_capture"


def touch_guide(args):
    args.out.mkdir(parents=True, exist_ok=True)
    # Diagram only: coordinates are in the yellow base-frame convention.
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="650" viewBox="0 0 1000 650">',
             '<rect width="1000" height="650" fill="white"/>',
             '<g font-family="sans-serif" fill="#182b39">',
             '<text x="40" y="45" font-size="25">9-point touch placement guide (not to scale)</text>',
             '<text x="40" y="80" font-size="18">Origin: yellow base marker; +X into table; +Y robot left. Units: cm.</text>']
    for i, (x, y) in enumerate((x, y) for x in (18, 25, 32) for y in (-14, 0, 14)):
        px, py = 500 - y * 22, 550 - (x - 18) * 27
        parts += [f'<circle cx="{px}" cy="{py}" r="9" fill="#1677a6"/>',
                  f'<text x="{px}" y="{py - 22}" text-anchor="middle" font-size="20">{i + 1}: ({x}, {y:+})</text>']
    parts += ['</g></svg>']
    path = args.out / "touch_placement_guide.svg"
    path.write_text("\n".join(parts), encoding="utf-8")
    print(path)
    print("Mark physical distances with a ruler. Photograph the real robot and all nine points.")


def real_pose(args):
    import cv2
    import numpy as np
    import torch
    from so101.real.cameras import open_camera
    from so101.real.follower import FollowerArm, SafetyLimits
    from so101.real.preview import PreviewWindow, preview_available

    args.out.mkdir(parents=True, exist_ok=True)
    target = torch.tensor(args.pose, dtype=torch.float32)
    if not np.isfinite(args.pose).all():
        raise ValueError("pose must contain finite joint radians")
    arm = FollowerArm(args.port, args.robot_id, SafetyLimits(max_step_rad=0.06))
    cameras, windows = {}, {}

    def save(frames):
        measured = arm.read().tolist()
        for name, frame in frames.items():
            if not cv2.imwrite(str(args.out / f"real_{name}.png"), frame):
                raise RuntimeError(f"cannot write {name} frame")
        # Sequential camera reads and encoder read: approximate static alignment.
        record = {"captured_at": datetime.now().astimezone().isoformat(),
                  "joints_rad": measured, "commanded_joints_rad": target.tolist(),
                  "read_only": args.read_only, "rectified": True,
                  "capture_timing": "sequential static camera reads, followed by encoder read",
                  "joint_order": ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]}
        (args.out / "pose.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"[capture] clean frames + measured pose -> {args.out}")

    try:
        arm.connect()
        if not args.read_only:
            print(f"[move] target radians {args.pose}; easing {args.move_seconds}s")
            arm.move_to(target, seconds=args.move_seconds)
        for name in ("front", "wrist"):
            cameras[name] = open_camera(name, rectify=True)
        if args.show and preview_available():
            for name in cameras:
                windows[name] = PreviewWindow(f"REAL {name} | s=save, q=exit", size=(640, 480))
        elif args.show:
            print("[info] no display; PNG capture still works")
        time.sleep(0.5)
        # Drain buffered camera frames after arrival at the stationary pose.
        for _ in range(5):
            for camera in cameras.values():
                camera.read()
        deadline = time.monotonic() + args.seconds
        first = True
        while first or time.monotonic() < deadline:
            frames = {name: camera.read() for name, camera in cameras.items()}
            if first:
                save(frames)
                first = False
            for name, window in windows.items():
                window.show(frames[name])
            keys = [window.poll_key() for window in windows.values()]
            if ord("s") in keys or 32 in keys:
                save(frames)
            if any(key in (ord("q"), 27) for key in keys):
                break
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        print("[info] capture interrupted")
    finally:
        for camera in cameras.values():
            camera.close()
        for window in windows.values():
            window.close()
        if arm.connected:
            try:
                if not args.read_only:
                    arm.go_home()
            finally:
                arm.close()


def export_rollout(args):
    import cv2
    import numpy as np

    # Only load this project's locally recorded pickle files.
    with args.episode.open("rb") as handle:
        episode = pickle.load(handle)
    front, wrist = (np.asarray(episode[f"{name}_images"]) for name in ("front", "wrist"))
    if len(front) == 0 or len(front) != len(wrist):
        raise ValueError("episode needs equal, nonempty front/wrist frame arrays")
    for frames in (front, wrist):
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("expected recorded uint8 RGB images, (T,H,W,3)")
    stamps = np.asarray(episode.get("wall_times", []), dtype=float)
    if len(stamps) == len(front) and len(stamps) > 1:
        if not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
            raise ValueError("wall_times must be finite and strictly increasing")
        elapsed = stamps - stamps[0]
        duration = elapsed[-1] + np.median(np.diff(elapsed))
        times = np.arange(0, duration, 1 / args.fps)
        index = np.maximum(np.searchsorted(elapsed, times, side="right") - 1, 0)
        timing = "wall_times resampled by holding last captured frame"
    else:
        index = np.arange(len(front))
        timing = "assumed frame rate; original wall_times missing or insufficient"
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.episode.stem
    writers = {}
    try:
        for name, width in (("front", front.shape[2]), ("wrist", wrist.shape[2]),
                            ("front_wrist", front.shape[2] + wrist.shape[2])):
            height = front.shape[1]
            if wrist.shape[1] != height:
                raise ValueError("camera image heights differ")
            path = args.out / f"{stem}_{name}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
            writers[name] = writer
            if not writer.isOpened():
                raise RuntimeError(f"MP4 encoder unavailable: {path}")
        for i in index:
            bgr_front, bgr_wrist = front[i][..., ::-1], wrist[i][..., ::-1]
            writers["front"].write(np.ascontiguousarray(bgr_front))
            writers["wrist"].write(np.ascontiguousarray(bgr_wrist))
            writers["front_wrist"].write(np.concatenate([bgr_front, bgr_wrist], axis=1))
        for name, frames in (("front", front), ("wrist", wrist)):
            for label, i in (("start", 0), ("end", len(frames) - 1)):
                cv2.imwrite(str(args.out / f"{stem}_{name}_{label}.png"), frames[i][..., ::-1])
    finally:
        for writer in writers.values():
            writer.release()
    metadata = {"episode": str(args.episode.resolve()), "checkpoint": episode.get("checkpoint"),
                "success": bool(np.asarray(episode.get("successes", [])).any()),
                "interrupted": bool(episode.get("interrupted", False)),
                "timing": timing, "fps": args.fps, "frames": len(index),
                "stored_resolution": [front.shape[2], front.shape[1]],
                "source": episode.get("source"), "dry_run": episode.get("dry_run")}
    metadata["deployment_args"] = episode.get("deployment_args")
    (args.out / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"[export] front, wrist, paired MP4 + start/end PNG -> {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    guide = sub.add_parser("touch-guide")
    guide.add_argument("--out", type=Path, default=DEFAULT_OUT / "slide6_touch")
    guide.set_defaults(run=touch_guide)
    real = sub.add_parser("real-pose")
    real.add_argument("--pose", nargs=6, type=float, default=[-0.07, -0.47, 0.17, 1.44, 0.0, -0.013])
    real.add_argument("--read-only", action="store_true", help="read current pose without sending motion commands")
    real.add_argument("--move-seconds", type=float, default=5)
    real.add_argument("--seconds", type=float, default=120)
    real.add_argument("--show", action="store_true")
    real.add_argument("--port", default="/dev/so101-follower")
    real.add_argument("--robot-id", default="my_follower")
    real.add_argument("--out", type=Path, default=DEFAULT_OUT / "slide6_alignment/real")
    real.set_defaults(run=real_pose)
    export = sub.add_parser("export-rollout")
    export.add_argument("--episode", type=Path, required=True)
    export.add_argument("--fps", type=float, default=30)
    export.add_argument("--out", type=Path, default=DEFAULT_OUT / "slide8/real_video")
    export.set_defaults(run=export_rollout)
    args = parser.parse_args()
    if getattr(args, "fps", 30) <= 0 or getattr(args, "seconds", 0) < 0 or getattr(args, "move_seconds", 1) <= 0:
        parser.error("fps/move-seconds must be positive; seconds must be nonnegative")
    args.run(args)


if __name__ == "__main__":
    main()
