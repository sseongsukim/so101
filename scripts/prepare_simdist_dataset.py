"""Pack trajectory pickle files into the aligned SO101 SimDist contract.

All image/state/action entries represent a real simulator tick. `next_*` fields
are captured after that action; the converter never fabricates a terminal
transition by duplicating the final observation.
"""
from __future__ import annotations

import argparse
import pickle
import json
import hashlib
from pathlib import Path

import numpy as np


def pack_trajectories(paths: list[Path], gamma: float = 0.99, task_horizon_ticks: int | None = None) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in (
        "observations", "next_observations", "actions", "rewards", "values",
        "expert_policy_flag", "terminated", "truncated", "front_images",
        "wrist_images", "next_front_images", "next_wrist_images", "ticks", "tick_seconds",
        "cube_positions_robot_relative_m", "next_cube_positions_robot_relative_m",
        "next_grasp_points_robot_relative_m", "table_height_robot_relative_m",
        "camera_poses_robot_relative_m_wxyz", "camera_intrinsics_source_px",
        "cube_instance_masks_packed", "camera_sensor_frame_counters",
        "value_valid", "reward_valid", "trajectory_quality", "phase_labels", "reward_semantics",
        "remaining_ticks", "next_remaining_ticks", "task_horizon_ticks", "objective_terminal",
    )}
    starts, lengths, elapsed = [], [], []
    episode_metadata = []
    object_label_mode: bool | None = None
    geometry_label_mode: bool | None = None
    geometry_reward_config_json: str | None = None
    camera_label_mode: bool | None = None
    camera_contract_json: str | None = None
    instance_mask_mode: bool | None = None
    instance_mask_contract_json: str | None = None
    for path in sorted(paths):
        with path.open("rb") as f:
            episode = pickle.load(f)
        required = ("observations", "next_observations", "actions", "rewards",
                    "front_images", "wrist_images", "next_front_images", "next_wrist_images")
        missing = [key for key in required if key not in episode]
        if missing:
            raise ValueError(f"{path}: missing aligned transition fields {missing}; regenerate diverse data")
        recorded_n = len(episode["actions"])
        if recorded_n == 0:
            continue
        for key in required:
            if len(episode[key]) != recorded_n:
                raise ValueError(f"{path}: {key} does not match recorded action length")
        horizon = episode.get("task_horizon_ticks", task_horizon_ticks)
        if horizon is None:
            raise ValueError(f"{path}: task_horizon_ticks missing; declare the task horizon explicitly")
        if not isinstance(horizon, (int, np.integer)) or int(horizon) < 1:
            raise ValueError(f"{path}: task_horizon_ticks must be a positive integer")
        horizon = int(horizon)
        original_ticks = np.asarray(episode.get("tick", np.arange(recorded_n)))
        if original_ticks.shape != (recorded_n,) or not np.issubdtype(original_ticks.dtype, np.integer):
            raise ValueError(f"{path}: tick must be an integer vector matching actions")
        if (original_ticks < 0).any() or (recorded_n > 1 and np.any(np.diff(original_ticks) != 1)):
            raise ValueError(f"{path}: negative or non-contiguous ticks")
        # A post-success tail may exceed the declared finite task horizon.
        # Retain actual transitions only inside the objective, without changing
        # simulator terminated/truncated flags into synthetic env flags.
        n = int(np.count_nonzero(original_ticks < horizon))
        if n == 0:
            raise ValueError(f"{path}: recording starts outside task horizon")
        transition_keys = set(required) | {"tick", "values", "expert_policy_flag", "terminated", "truncated",
                                           "phase_labels", "teacher_observations", "terminals", "successes"}
        object_label_keys = ("cube_positions_robot_relative_m", "next_cube_positions_robot_relative_m")
        present_object_labels = [key in episode for key in object_label_keys]
        if any(present_object_labels) and not all(present_object_labels):
            raise ValueError(f"{path}: current and successor cube labels must be provided together")
        if all(present_object_labels):
            if episode.get("cube_position_label_semantics") != "simulator_xyz_robot_root_relative_training_only":
                raise ValueError(f"{path}: cube labels must be declared simulator training-only robot-relative XYZ")
            if episode.get("cube_position_order") != ["held_cube", "fixed_cube"]:
                raise ValueError(f"{path}: cube label order must be [held_cube, fixed_cube]")
            transition_keys.update(object_label_keys)
        geometry_keys = (
            "next_grasp_points_robot_relative_m", "table_height_robot_relative_m",
        )
        present_geometry = [key in episode for key in geometry_keys]
        if any(present_geometry) and not all(present_geometry):
            raise ValueError(f"{path}: grasp-point and table-height labels must be provided together")
        has_geometry = all(present_geometry)
        if has_geometry:
            if not all(present_object_labels):
                raise ValueError(f"{path}: geometry reward labels require cube pose labels")
            if episode.get("grasp_point_order") != ["midpoint", "fixed_fingertip", "moving_fingertip"]:
                raise ValueError(f"{path}: unexpected grasp-point label order")
            config = episode.get("geometry_reward_config")
            if not isinstance(config, dict):
                raise ValueError(f"{path}: missing geometry reward constants")
            config_json = json.dumps(config, sort_keys=True)
            if geometry_reward_config_json is not None and geometry_reward_config_json != config_json:
                raise ValueError(f"{path}: geometry reward constants differ across episodes")
            geometry_reward_config_json = config_json
            transition_keys.update(geometry_keys)
        camera_keys = ("camera_poses_robot_relative_m_wxyz", "camera_intrinsics_source_px")
        present_camera = [key in episode for key in camera_keys]
        if any(present_camera) and not all(present_camera):
            raise ValueError(f"{path}: camera poses and intrinsics must be provided together")
        has_camera = all(present_camera)
        if has_camera:
            if episode.get("camera_order") != ["front", "wrist"]:
                raise ValueError(f"{path}: camera order must be [front, wrist]")
            if episode.get("camera_pose_semantics") != (
                    "robot_root_relative_translation_world_axes_quaternion_wxyz_forward_x_up_z"):
                raise ValueError(f"{path}: unexpected camera pose semantics")
            if episode.get("camera_intrinsics_semantics") != "pinhole_3x3_source_sensor_pixel_coordinates":
                raise ValueError(f"{path}: unexpected camera intrinsics semantics")
            source_shapes = episode.get("camera_source_image_shape_hw")
            if (not isinstance(source_shapes, list) or len(source_shapes) != 2
                    or any(len(shape) != 2 or min(shape) < 1 for shape in source_shapes)):
                raise ValueError(f"{path}: camera source image shapes must be two positive [H,W] pairs")
            contract_json = json.dumps({
                "camera_order": episode["camera_order"],
                "camera_pose_semantics": episode["camera_pose_semantics"],
                "camera_intrinsics_semantics": episode["camera_intrinsics_semantics"],
                "camera_source_image_shape_hw": source_shapes,
            }, sort_keys=True)
            if camera_contract_json is not None and camera_contract_json != contract_json:
                raise ValueError(f"{path}: camera geometry contract differs across episodes")
            camera_contract_json = contract_json
            transition_keys.update(camera_keys)
        mask_key = "cube_instance_masks_packed"
        has_instance_masks = mask_key in episode
        mask_fields = ("cube_instance_mask_camera_order", "cube_instance_mask_object_order",
                       "cube_instance_mask_semantics", "cube_instance_mask_storage",
                       "cube_instance_mask_source_shape_hw", "camera_sensor_frame_counter_semantics")
        if has_instance_masks:
            if not all(field in episode for field in mask_fields):
                raise ValueError(f"{path}: packed cube masks require their complete camera/object/path contract")
            if episode["cube_instance_mask_camera_order"] != ["front", "wrist"]:
                raise ValueError(f"{path}: unexpected cube mask camera order")
            if episode["cube_instance_mask_object_order"] != ["held_cube", "fixed_cube"]:
                raise ValueError(f"{path}: unexpected cube mask object order")
            if episode["cube_instance_mask_semantics"] != (
                    "uncolorized_instance_id_segmentation_fast_resolved_by_usd_prim_path"):
                raise ValueError(f"{path}: unexpected cube mask semantics")
            if episode["cube_instance_mask_storage"] != "np.packbits_axis_last_bitorder_big":
                raise ValueError(f"{path}: unsupported cube mask bit packing")
            shapes = episode["cube_instance_mask_source_shape_hw"]
            if not isinstance(shapes, list) or len(shapes) != 2 or any(
                    len(shape) != 2 or min(shape) < 1 or shape[1] % 8 for shape in shapes):
                raise ValueError(f"{path}: cube mask source shapes need two [H,W%8==0] pairs")
            packed = np.asarray(episode[mask_key])
            expected_shape = (recorded_n, 2, 2, shapes[0][0], shapes[0][1] // 8)
            if shapes[0] != shapes[1] or packed.shape != expected_shape or packed.dtype != np.uint8:
                raise ValueError(f"{path}: cube masks must be uint8 {expected_shape}, got {packed.shape}/{packed.dtype}")
            frame_counters = np.asarray(episode.get("camera_sensor_frame_counters"))
            if (episode["camera_sensor_frame_counter_semantics"] !=
                    "same_current_render_used_for_rgb_and_instance_masks"):
                raise ValueError(f"{path}: unexpected camera frame counter semantics")
            if frame_counters.shape != (recorded_n, 2) or not np.issubdtype(frame_counters.dtype, np.integer):
                raise ValueError(f"{path}: camera sensor frame counters must be integer (N,2)")
            if not np.array_equal(frame_counters[:, 0], frame_counters[:, 1]):
                raise ValueError(f"{path}: front/wrist annotation frames are not synchronized")
            if recorded_n > 1 and not np.all(np.diff(frame_counters, axis=0) == 1):
                raise ValueError(f"{path}: camera annotation frame counters are not contiguous")
            path_map = episode.get("cube_instance_id_to_usd_path_by_camera")
            if not isinstance(path_map, dict) or set(path_map) != {"front", "wrist"} or any(
                    not isinstance(camera_map, dict) for camera_map in path_map.values()):
                raise ValueError(f"{path}: cube mask ID mapping must be a per-camera object")
            contract_json = json.dumps({field: episode[field] for field in mask_fields}, sort_keys=True)
            if instance_mask_contract_json is not None and contract_json != instance_mask_contract_json:
                raise ValueError(f"{path}: cube mask contract differs across episodes")
            instance_mask_contract_json = contract_json
            transition_keys.add(mask_key)
            transition_keys.add("camera_sensor_frame_counters")
        if instance_mask_mode is not None and instance_mask_mode != has_instance_masks:
            raise ValueError("cannot mix trajectories with and without cube instance masks in one archive")
        instance_mask_mode = has_instance_masks
        if camera_label_mode is not None and camera_label_mode != has_camera:
            raise ValueError("cannot mix trajectories with and without camera geometry labels in one archive")
        camera_label_mode = has_camera
        if geometry_label_mode is not None and geometry_label_mode != has_geometry:
            raise ValueError("cannot mix trajectories with and without geometry reward labels")
        geometry_label_mode = has_geometry
        has_object_labels = all(present_object_labels)
        if object_label_mode is not None and object_label_mode != has_object_labels:
            raise ValueError("cannot mix trajectories with and without cube pose labels in one archive")
        object_label_mode = has_object_labels
        episode = {k: (v[:n] if k in transition_keys and
                       (isinstance(v, list) or isinstance(v, np.ndarray) and v.ndim > 0)
                       and len(v) == recorded_n else v)
                   for k, v in episode.items()}
        for key in required:
            if len(episode[key]) != n:
                raise ValueError(f"{path}: {key} has {len(episode[key])} rows, expected {n}")
        if np.asarray(episode["actions"]).shape != (n, 6):
            raise ValueError(f"{path}: expected six applied joint targets, got {np.shape(episode['actions'])}")
        if np.asarray(episode["rewards"]).shape != (n,):
            raise ValueError(f"{path}: rewards must contain one scalar per applied tick")
        if np.asarray(episode["observations"]).shape != (n, 6) or np.asarray(episode["next_observations"]).shape != (n, 6):
            raise ValueError(f"{path}: measured current/next joint observations must both be (N, 6)")
        for key in ("front_images", "wrist_images", "next_front_images", "next_wrist_images"):
            image = np.asarray(episode[key])
            image_shape = image.shape
            if len(image_shape) != 4 or image_shape[-1] != 3:
                raise ValueError(f"{path}: {key} must be NHWC uint8 RGB; got {image_shape}")
            if image.dtype != np.uint8:
                raise ValueError(f"{path}: {key} must be uint8 RGB")
        for key in ("observations", "next_observations", "actions", "rewards"):
            if not np.isfinite(np.asarray(episode[key])).all():
                raise ValueError(f"{path}: {key} contains non-finite entries")
        if all(present_object_labels):
            for key in object_label_keys:
                labels = np.asarray(episode[key])
                if labels.shape != (n, 2, 3) or not np.isfinite(labels).all():
                    raise ValueError(f"{path}: {key} must be finite (N, 2, 3) robot-relative XYZ")
            if n > 1 and not np.allclose(
                np.asarray(episode["next_cube_positions_robot_relative_m"])[:-1],
                np.asarray(episode["cube_positions_robot_relative_m"])[1:],
                atol=2e-6, rtol=0,
            ):
                raise ValueError(f"{path}: cube successor labels do not align with next current labels")
        if has_geometry:
            grasp = np.asarray(episode["next_grasp_points_robot_relative_m"])
            table = np.asarray(episode["table_height_robot_relative_m"])
            if grasp.shape != (n, 3, 3) or table.shape != (n,):
                raise ValueError(f"{path}: grasp points must be (N,3,3) and table height (N,)")
            if not np.isfinite(grasp).all() or not np.isfinite(table).all():
                raise ValueError(f"{path}: geometry reward labels must be finite")
        if has_camera:
            poses = np.asarray(episode["camera_poses_robot_relative_m_wxyz"], dtype=np.float32)
            intrinsics = np.asarray(episode["camera_intrinsics_source_px"], dtype=np.float32)
            if poses.shape != (n, 2, 7) or intrinsics.shape != (n, 2, 3, 3):
                raise ValueError(f"{path}: camera poses/intrinsics must have shapes (N,2,7)/(N,2,3,3)")
            if not np.isfinite(poses).all() or not np.isfinite(intrinsics).all():
                raise ValueError(f"{path}: camera geometry must be finite")
            quat_norm = np.linalg.norm(poses[..., 3:], axis=-1)
            if not np.allclose(quat_norm, 1.0, atol=2e-3, rtol=0):
                raise ValueError(f"{path}: camera world quaternions must be unit (w,x,y,z)")
            if (np.any(intrinsics[..., 0, 0] <= 0) or np.any(intrinsics[..., 1, 1] <= 0)
                    or not np.allclose(intrinsics[..., 2, :], [0, 0, 1], atol=1e-5, rtol=0)):
                raise ValueError(f"{path}: invalid pinhole camera intrinsics")
        if np.asarray(episode["front_images"]).shape != np.asarray(episode["wrist_images"]).shape:
            raise ValueError(f"{path}: front/wrist image shapes differ")
        if episode.get("action_semantics") != "applied_absolute_joint_target_radians":
            raise ValueError(f"{path}: actions are not declared absolute radians joint targets")
        ticks = np.asarray(episode.get("tick", np.arange(n)), dtype=np.int64)
        fps = float(episode.get("fps", 30))
        if fps <= 0:
            raise ValueError(f"{path}: fps must be positive, got {fps}")
        if n > 1 and np.any(np.diff(ticks) != 1):
            raise ValueError(f"{path}: non-contiguous ticks; windows would cross a time gap")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be within [0, 1], got {gamma}")
        starts.append(sum(lengths)); lengths.append(n); elapsed.extend((ticks / fps).tolist())
        arrays["ticks"].append(ticks)
        arrays["tick_seconds"].append(np.full(n, 1.0 / fps, dtype=np.float32))
        defaults = {
            "values": np.zeros(n, np.float32),
            "expert_policy_flag": np.full(
                n, episode.get("trajectory_quality") not in ("diverse", "perturbed"), np.bool_
            ),
            "terminated": np.zeros(n, np.bool_), "truncated": np.zeros(n, np.bool_),
            "ticks": ticks,
        }
        phases = np.asarray(episode.get("phase_labels", ["unknown"] * n), dtype=np.str_)
        if phases.shape != (n,):
            raise ValueError(f"{path}: phase_labels must have shape ({n},)")
        arrays["trajectory_quality"].append(
            np.repeat(np.asarray([str(episode.get("trajectory_quality", "unknown"))], dtype=np.str_), n)
        )
        arrays["phase_labels"].append(phases)
        reward_semantics = str(episode.get(
            "reward_semantics",
            "dense_environment_reward" if episode.get("trajectory_quality") == "diverse" else "sparse_success_indicator",
        ))
        if reward_semantics not in ("dense_environment_reward", "sparse_success_indicator"):
            raise ValueError(f"{path}: unsupported reward_semantics {reward_semantics!r}")
        arrays["reward_semantics"].append(
            np.repeat(np.asarray([reward_semantics], dtype=np.str_), n)
        )
        arrays["reward_valid"].append(np.full(n, reward_semantics == "dense_environment_reward", dtype=np.bool_))
        for key, target in (("observations", "observations"), ("next_observations", "next_observations"),
                            ("actions", "actions"), ("rewards", "rewards"),
                            ("front_images", "front_images"), ("wrist_images", "wrist_images"),
                            ("next_front_images", "next_front_images"), ("next_wrist_images", "next_wrist_images")):
            arrays[target].append(np.asarray(episode[key]))
        if has_instance_masks:
            arrays[mask_key].append(np.asarray(episode[mask_key])[:n])
            arrays["camera_sensor_frame_counters"].append(np.asarray(episode["camera_sensor_frame_counters"])[:n])
        if all(present_object_labels):
            for key in object_label_keys:
                arrays[key].append(np.asarray(episode[key], dtype=np.float32))
        if has_geometry:
            for key in geometry_keys:
                arrays[key].append(np.asarray(episode[key], dtype=np.float32))
        if has_camera:
            for key in camera_keys:
                arrays[key].append(np.asarray(episode[key], dtype=np.float32))
        for key in ("values", "expert_policy_flag", "terminated", "truncated"):
            v = episode.get(key, defaults[key])
            if np.asarray(v).shape != (n,):
                raise ValueError(f"{path}: {key} must contain one scalar per applied tick")
            if key == "expert_policy_flag" and not np.isin(v, [0,1]).all():
                raise ValueError(f"{path}: expert_policy_flag must be boolean")
            arrays[key].append(np.asarray(v, dtype=np.float32 if key in ("values", "expert_policy_flag") else np.bool_))
        terminated = np.asarray(episode.get("terminated", defaults["terminated"]), dtype=np.bool_)
        truncated = np.asarray(episode.get("truncated", defaults["truncated"]), dtype=np.bool_)
        if terminated.shape != (n,) or truncated.shape != (n,):
            raise ValueError(f"{path}: terminated/truncated must each have shape ({n},)")
        if np.any(terminated & truncated):
            raise ValueError(f"{path}: a transition cannot be both terminated and truncated")
        if np.any((terminated | truncated)[:-1]):
            raise ValueError(f"{path}: transitions continue after an episode boundary")
        remaining = horizon - ticks
        successor_remaining = remaining - 1
        at_horizon = bool(successor_remaining[-1] == 0)
        has_true_terminal_boundary = bool(terminated[-1] and not truncated[-1])
        completed_objective = at_horizon or has_true_terminal_boundary
        valid = np.full(n, completed_objective and reward_semantics == "dense_environment_reward", dtype=np.bool_)
        objective_terminal = terminated.copy()
        objective_terminal |= successor_remaining == 0
        arrays["value_valid"].append(valid)
        arrays["remaining_ticks"].append(remaining.astype(np.int64))
        arrays["next_remaining_ticks"].append(successor_remaining.astype(np.int64))
        arrays["task_horizon_ticks"].append(np.full(n, horizon, dtype=np.int64))
        arrays["objective_terminal"].append(objective_terminal)
        generation_path = path.parent / "generation.json"
        generation = json.loads(generation_path.read_text()) if generation_path.is_file() else {}
        fingerprint = hashlib.sha256()
        for key in ("observations", "next_observations", "actions", "tick"):
            value = ticks if key == "tick" else np.asarray(episode[key])
            fingerprint.update(key.encode())
            fingerprint.update(str(value.shape).encode())
            fingerprint.update(value.astype(np.float32 if key != "tick" else np.int64).tobytes())
        episode_metadata.append({"seed": int(episode.get("seed", -1)), "env_index": int(episode.get("env_index", -1)),
                                 "quality": str(episode.get("trajectory_quality", "unknown")),
                                 "noise_std": float(episode.get("perturbation_std", generation.get("perturbation_std", -1))),
                                 "cube_instance_id_to_usd_path_by_camera": episode.get(
                                     "cube_instance_id_to_usd_path_by_camera", {}),
                                 "source": str(path.resolve()), "fingerprint": fingerprint.hexdigest()})
    if not lengths:
        raise ValueError("no trajectories found")
    out = {k: np.concatenate(v, axis=0) for k, v in arrays.items() if v}
    # Finite-horizon MC: successor continuation is exactly zero at success
    # task termination or the declared task horizon. An earlier recording
    # cutoff remains censored, including all its prefix values.
    returns = np.zeros(len(out["rewards"]), dtype=np.float32)
    for start, length in zip(starts, lengths):
        carry = 0.0
        for i in range(start + length - 1, start - 1, -1):
            carry = float(out["rewards"][i]) + gamma * carry
            returns[i] = carry
        stop = start + length
        for i in range(start, stop):
            # V(s_{i+1}); terminal successor is known to be zero. Truncation
            # values are marked invalid above rather than mislabelled as zero.
            if out["objective_terminal"][i]:
                out["values"][i] = 0.0
            elif i + 1 < stop:
                out["values"][i] = returns[i + 1]
            else:
                out["values"][i] = 0.0
    out["episode_starts"] = np.asarray(starts, dtype=np.int64)
    out["episode_lengths"] = np.asarray(lengths, dtype=np.int64)
    out["timestamps_seconds"] = np.asarray(elapsed, dtype=np.float32)
    out["gamma"] = np.asarray(gamma, dtype=np.float32)
    out["action_semantics"] = np.asarray("applied_absolute_joint_target_radians")
    out["value_objective"] = np.asarray("finite_horizon_dense_mc_v2")
    if object_label_mode:
        out["cube_position_label_semantics"] = np.asarray(
            "simulator_xyz_robot_root_relative_training_only"
        )
        out["cube_position_order"] = np.asarray(["held_cube", "fixed_cube"])
    if geometry_label_mode:
        out["grasp_point_order"] = np.asarray(
            ["midpoint", "fixed_fingertip", "moving_fingertip"]
        )
        out["geometry_reward_config_json"] = np.asarray(geometry_reward_config_json)
    if camera_label_mode:
        out["camera_geometry_contract_json"] = np.asarray(camera_contract_json)
    if instance_mask_mode:
        out["cube_instance_mask_contract_json"] = np.asarray(instance_mask_contract_json)
        out["episode_cube_instance_id_to_usd_path_json"] = np.asarray([
            json.dumps(row["cube_instance_id_to_usd_path_by_camera"], sort_keys=True)
            for row in episode_metadata
        ])
    for field, key in (("seed", "episode_seeds"), ("env_index", "episode_env_indices"),
                       ("quality", "episode_quality"), ("noise_std", "episode_noise_std"),
                       ("source", "episode_sources"), ("fingerprint", "episode_fingerprints")):
        out[key] = np.asarray([row[field] for row in episode_metadata])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, nargs="+", required=True,
                   help="one or more directories containing trajectory_*.pkl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--task-horizon-ticks", type=int, default=None,
                   help="explicit horizon fallback for old recordings missing horizon metadata")
    a = p.parse_args()
    paths = [path for directory in a.input for path in directory.glob("trajectory_*.pkl")]
    data = pack_trajectories(paths, a.gamma, a.task_horizon_ticks)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.out.with_suffix(a.out.suffix + ".tmp")
    with temporary.open("wb") as f:
        np.savez_compressed(f, **data)
    temporary.replace(a.out)
    quality, quality_counts = np.unique(data["trajectory_quality"], return_counts=True)
    phases, phase_counts = np.unique(data["phase_labels"], return_counts=True)
    print(f"saved {a.out}: {len(data['episode_starts'])} episodes, {len(data['actions'])} transitions; "
          f"expert_actions={int(data['expert_policy_flag'].sum())}; "
          f"quality={dict(zip(quality.tolist(), quality_counts.tolist()))}; "
          f"phases={dict(zip(phases.tolist(), phase_counts.tolist()))}")


if __name__ == "__main__":
    main()
