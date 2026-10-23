#!/usr/bin/env python3
"""Audit one collector episode's camera-ID masks, frame counters and projection."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def _quat_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def _world_quat_to_ros_rotation(q_world: np.ndarray) -> np.ndarray:
    """Match IsaacLab world(+X,+Z-up) -> ROS optical(+Z,-Y-up) conversion."""
    roll, pitch = np.pi / 2, -np.pi / 2
    rx = np.asarray([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)],
                     [0, np.sin(roll), np.cos(roll)]], dtype=np.float64)
    ry = np.asarray([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0],
                     [-np.sin(pitch), 0, np.cos(pitch)]], dtype=np.float64)
    return _quat_matrix_wxyz(q_world) @ rx @ ry @ np.diag([1., -1., -1.])


def audit_episode(episode: dict) -> dict:
    required = ("cube_instance_masks_packed", "camera_sensor_frame_counters",
                "camera_poses_robot_relative_m_wxyz", "camera_intrinsics_source_px",
                "cube_positions_robot_relative_m")
    missing = [key for key in required if key not in episode]
    if missing:
        raise ValueError(f"episode missing audit fields {missing}")
    packed = np.asarray(episode["cube_instance_masks_packed"])
    shapes = episode["cube_instance_mask_source_shape_hw"]
    if packed.ndim != 5 or packed.shape[1:3] != (2, 2) or packed.dtype != np.uint8:
        raise ValueError("packed cube masks must be uint8 [T,camera=2,object=2,H,W/8]")
    height, width = shapes[0]
    if shapes[1] != shapes[0] or width % 8 or packed.shape != (len(packed), 2, 2, height, width // 8):
        raise ValueError("packed mask shape disagrees with declared source image shape")
    masks = np.unpackbits(packed, axis=-1, bitorder="big")[..., :width].astype(bool)
    poses = np.asarray(episode["camera_poses_robot_relative_m_wxyz"], dtype=np.float64)
    intrinsics = np.asarray(episode["camera_intrinsics_source_px"], dtype=np.float64)
    positions = np.asarray(episode["cube_positions_robot_relative_m"], dtype=np.float64)
    counters = np.asarray(episode["camera_sensor_frame_counters"])
    n = len(packed)
    if poses.shape != (n, 2, 7) or intrinsics.shape != (n, 2, 3, 3) or positions.shape != (n, 2, 3):
        raise ValueError("pose/K/XYZ dimensions do not align with the mask frame count")
    if counters.shape != (n, 2) or not np.issubdtype(counters.dtype, np.integer):
        raise ValueError("camera frame counters must be integer [T,2]")
    synced = bool(np.array_equal(counters[:, 0], counters[:, 1]))
    contiguous = bool(n < 2 or np.all(np.diff(counters, axis=0) == 1))
    per_view = []
    maximum_roundtrip = 0.0
    for camera_index, camera_name in enumerate(("front", "wrist")):
        for object_index, object_name in enumerate(("held_cube", "fixed_cube")):
            visible = 0
            center_in_bbox = 0
            center_pixel_in_mask = 0
            distances = []
            for tick in range(n):
                mask = masks[tick, camera_index, object_index]
                ys, xs = np.nonzero(mask)
                if not len(xs):
                    continue
                visible += 1
                rotation = _world_quat_to_ros_rotation(poses[tick, camera_index, 3:])
                camera_xyz = rotation.T @ (positions[tick, object_index] - poses[tick, camera_index, :3])
                if camera_xyz[2] <= 0:
                    distances.append(float("inf"))
                    continue
                matrix = intrinsics[tick, camera_index]
                u = matrix[0, 0] * camera_xyz[0] / camera_xyz[2] + matrix[0, 2]
                v = matrix[1, 1] * camera_xyz[1] / camera_xyz[2] + matrix[1, 2]
                projected_back = rotation @ camera_xyz + poses[tick, camera_index, :3]
                maximum_roundtrip = max(maximum_roundtrip,
                                        float(np.linalg.norm(projected_back - positions[tick, object_index])))
                in_bbox = xs.min() <= u <= xs.max() and ys.min() <= v <= ys.max()
                center_in_bbox += int(in_bbox)
                iu, iv = int(np.floor(u + .5)), int(np.floor(v + .5))
                if 0 <= iu < width and 0 <= iv < height:
                    center_pixel_in_mask += int(mask[iv, iu])
                dx, dy = max(xs.min() - u, 0, u - xs.max()), max(ys.min() - v, 0, v - ys.max())
                distances.append(float(np.hypot(dx, dy)))
            per_view.append({
                "camera": camera_name, "object": object_name,
                "visible_frames": visible,
                "projected_center_inside_mask_bbox_fraction": center_in_bbox / visible if visible else None,
                "projected_center_pixel_in_mask_fraction": center_pixel_in_mask / visible if visible else None,
                "projected_center_to_mask_bbox_distance_px_p95": float(np.quantile(distances, .95)) if distances else None,
            })
    mapping = episode.get("cube_instance_id_to_usd_path_by_camera", {})
    expected_fragments = ("/HeldAsset", "/FixedAsset")
    path_map_checks = {
        camera: {fragment: any(fragment in str(path) for path in paths.values())
                 for fragment in expected_fragments}
        for camera, paths in mapping.items()
    }
    return {
        "transitions": n, "source_mask_shape": [height, width],
        "packed_mask_bytes": int(packed.nbytes),
        "camera_frame_counters_synchronized": synced,
        "camera_frame_counters_contiguous": contiguous,
        "camera_frame_counter_first_last": [counters[0].tolist(), counters[-1].tolist()] if n else [],
        "id_to_usd_path_checks": path_map_checks,
        "world_to_ros_xyz_roundtrip_max_error_m": maximum_roundtrip,
        "per_camera_object_projection": per_view,
        "projection_claim": "algebra and cross-modal consistency diagnostics; not camera-calibration proof",
    }


def make_overlay(episode: dict, output: Path, frames: list[int]) -> None:
    from PIL import Image

    rgb_views = (np.asarray(episode["front_images"]), np.asarray(episode["wrist_images"]))
    packed = np.asarray(episode["cube_instance_masks_packed"])
    masks = np.unpackbits(packed, axis=-1, bitorder="big")[..., :packed.shape[-1] * 8].astype(bool)
    colors = ((0, 255, 255), (255, 80, 200))
    height, width = rgb_views[0].shape[1:3]
    rows = []
    for tick in frames:
        if not 0 <= tick < len(packed):
            raise ValueError(f"overlay tick {tick} outside [0,{len(packed)})")
        pair = []
        for camera_index in range(2):
            image = Image.fromarray(rgb_views[camera_index][tick]).convert("RGB")
            pixels = np.asarray(image).copy()
            for object_index, color in enumerate(colors):
                source_mask = Image.fromarray(masks[tick, camera_index, object_index].astype(np.uint8) * 255)
                mask = np.asarray(source_mask.resize((width, height), Image.Resampling.NEAREST)) > 0
                pixels[mask] = (.45 * pixels[mask] + .55 * np.asarray(color)).astype(np.uint8)
            pair.append(Image.fromarray(pixels))
        rows.append(pair)
    canvas = Image.new("RGB", (width * 2, height * len(rows)))
    for row, pair in enumerate(rows):
        canvas.paste(pair[0], (0, row * height)); canvas.paste(pair[1], (width, row * height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overlay-out", type=Path)
    parser.add_argument("--overlay-ticks", type=int, nargs="+", default=[0, 60, 150, 299])
    args = parser.parse_args()
    with args.trajectory.open("rb") as stream:
        episode = pickle.load(stream)
    report = audit_episode(episode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    if args.overlay_out:
        make_overlay(episode, args.overlay_out, args.overlay_ticks)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
