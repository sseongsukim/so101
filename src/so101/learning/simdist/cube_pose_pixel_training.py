"""Bounded pixel pose trainer. RGB/masks stream; early maps exist only per batch."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from .cube_pose_pixel import (MASK_SEMANTICS, PIXEL_VERSION, POSE_SEMANTICS, RGB_HW,
                             FrozenEarlyDPEncoder, PixelCubePose, front_centre_targets,
                             pixel_pose_loss, unpack_resize_masks)
from .cube_pose_region import (CubeTaskRegion, current_successor_regions, fit_region_class_weights,
                              fit_region_normalization, masked_xyz_terms, neutral_pose_targets,
                              region_pose_metrics)
from .instance_masks import cube_instance_ids_by_path


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _record(episode, source, image_source, domain):
    """Validate immutable archive contracts and retain compact targets only."""
    seed = int(episode["seed"])
    if 1900 <= seed <= 1903:
        raise ValueError("H2 seeds 1900–1903 are reserved")
    expected = {
        "cube_position_order": ["held_cube", "fixed_cube"],
        "cube_position_label_semantics": "simulator_xyz_robot_root_relative_training_only",
        "camera_order": ["front", "wrist"], "camera_pose_semantics": POSE_SEMANTICS,
        "camera_intrinsics_semantics": "pinhole_3x3_source_sensor_pixel_coordinates",
        "cube_instance_mask_camera_order": ["front", "wrist"],
        "cube_instance_mask_object_order": ["held_cube", "fixed_cube"],
        "cube_instance_mask_semantics": MASK_SEMANTICS,
        "cube_instance_mask_storage": "np.packbits_axis_last_bitorder_big",
        "camera_sensor_frame_counter_semantics": "same_current_render_used_for_rgb_and_instance_masks",
    }
    for key, value in expected.items():
        if episode.get(key) != value:
            raise ValueError(f"{source}: unexpected or missing {key}")
    shapes = episode["cube_instance_mask_source_shape_hw"]
    if (len(shapes) != 2 or shapes[0] != shapes[1] or len(shapes[0]) != 2
            or min(shapes[0]) < 1 or episode.get("camera_source_image_shape_hw") != shapes):
        raise ValueError("mask/camera source shape contracts disagree")
    # Exact full-frame source -> RGB resize; shape ratio alone cannot prove absence
    # of historical cropping, so the source contract remains part of provenance.
    if shapes[0][0]*RGB_HW[1] != shapes[0][1]*RGB_HW[0]:
        raise ValueError("source sensor aspect ratio does not match full RGB")
    path_maps = episode.get("cube_instance_id_to_usd_path_by_camera")
    if not isinstance(path_maps, dict) or set(path_maps) != {"front", "wrist"}:
        raise ValueError("path-resolved mask provenance requires both camera maps")
    for mapping in path_maps.values():
        cube_instance_ids_by_path(mapping, int(episode["env_index"]))
    q = np.asarray(episode["observations"], np.float32)
    n = len(q)
    ticks = np.asarray(episode["tick"])
    horizon = int(episode["task_horizon_ticks"])
    if (q.shape != (n, 6) or not np.isfinite(q).all() or horizon < 1
            or ticks.shape != (n,) or not np.array_equal(ticks, np.arange(n))):
        raise ValueError("measured q/task clock is malformed")
    keep = min(n, horizon)
    if keep < 1:
        raise ValueError("no observed task-horizon frames")
    for key in ("front_images", "wrist_images"):
        images = np.asarray(episode[key])
        if images.dtype != np.uint8 or images.shape != (n, *RGB_HW, 3):
            raise ValueError("pixel mode requires full 240x320 uint8 RGB archives")
    frames = np.asarray(episode["camera_sensor_frame_counters"])
    if (frames.shape != (n, 2) or not np.issubdtype(frames.dtype, np.integer)
            or not np.array_equal(frames[:, 0], frames[:, 1])
            or (n > 1 and not np.all(np.diff(frames, axis=0) == 1))):
        raise ValueError("RGB/mask camera frame counters are not synchronized/contiguous")
    cubes = np.asarray(episode["cube_positions_robot_relative_m"], np.float32)[:keep]
    successor = np.asarray(episode["next_cube_positions_robot_relative_m"], np.float32)[:keep]
    if cubes.shape != (keep, 2, 3) or successor.shape != cubes.shape:
        raise ValueError("current/successor cube label shape mismatch")
    inside, next_inside = current_successor_regions(cubes, successor, domain)
    if "table_height_robot_relative_m" not in episode:
        raise ValueError("pixel mode requires recorded table height cross-check")
    domain.validate_height(np.asarray(episode["table_height_robot_relative_m"])[:keep])
    auxiliary = front_centre_targets(cubes,
        np.asarray(episode["camera_poses_robot_relative_m_wxyz"])[:keep],
        np.asarray(episode["camera_intrinsics_source_px"])[:keep],
        np.asarray(episode["cube_instance_masks_packed"])[:keep], tuple(shapes[0]), inside)
    return {"seed": seed, "source": str(source), "image_source": image_source, "length": keep,
            "source_hw": tuple(shapes[0]), "q": q[:keep].copy(), "target": cubes.copy(),
            "inside": inside, "next_inside": next_inside, **auxiliary}


def load_pixel_records(data_dirs, archives, domain):
    records, provenance = [], []
    for directory in data_dirs or []:
        for path in sorted(Path(directory).glob("trajectory_*.pkl")):
            with path.open("rb") as stream:
                episode = pickle.load(stream)
            records.append(_record(episode, path, path, domain))
            provenance.append({"path": str(path), "sha256": _sha256(path)})
            del episode
    for path in archives or []:
        # NPZ current RGB/masks stay in host RAM; no dataset-wide feature maps.
        with np.load(path, allow_pickle=False) as z:
            meta = {**json.loads(str(z["camera_geometry_contract_json"])),
                    **json.loads(str(z["cube_instance_mask_contract_json"])),
                    "cube_position_order": z["cube_position_order"].tolist(),
                    "cube_position_label_semantics": str(z["cube_position_label_semantics"])}
            keys = ("front_images", "wrist_images", "observations", "cube_positions_robot_relative_m",
                    "next_cube_positions_robot_relative_m", "table_height_robot_relative_m",
                    "camera_poses_robot_relative_m_wxyz", "camera_intrinsics_source_px",
                    "cube_instance_masks_packed", "camera_sensor_frame_counters", "ticks", "task_horizon_ticks")
            arrays = {key: z[key] for key in keys}
            starts, lengths = z["episode_starts"], z["episode_lengths"]
            seeds, envs = z["episode_seeds"], z["episode_env_indices"]
            maps = z["episode_cube_instance_id_to_usd_path_json"]
            total = len(arrays["observations"])
            if (not all(len(x) == len(starts) for x in (lengths, seeds, envs, maps))
                    or not np.array_equal(starts, np.r_[0, np.cumsum(lengths)[:-1]])
                    or np.any(lengths < 1) or lengths.sum() != total
                    or any(len(x) != total for x in arrays.values())):
                raise ValueError("aligned archive episode inventory is malformed")
            for i, (start, length) in enumerate(zip(starts, lengths)):
                start, stop = int(start), int(start+length)
                data = {key: value[start:stop] for key, value in arrays.items()}
                horizons = data.pop("task_horizon_ticks")
                if not np.all(horizons == horizons[0]):
                    raise ValueError("episode task horizon is not constant")
                episode = {**meta, **data, "tick": data["ticks"], "seed": int(seeds[i]),
                           "env_index": int(envs[i]), "task_horizon_ticks": int(horizons[0]),
                           "cube_instance_id_to_usd_path_by_camera": json.loads(str(maps[i]))}
                records.append(_record(episode, f"{path}#episode={i}", data, domain))
            provenance.append({"path": str(path), "sha256": _sha256(path)})
    if not records:
        raise ValueError("provide annotated --data-dir or --data-archive")
    return records, provenance


class PixelBatchStream:
    """At most one decoded raw pickle retained; NPZ RGB/masks are host-resident.

    Trainer shuffles episodes and within-episode indices to avoid reloading a raw
    archive for every sample. No model features or uncompressed dataset masks cached.
    """
    def __init__(self, device):
        self.device = device
        self.last_path, self.last_episode = None, None
        self.raw_episode_loads, self.batches, self.peak_batch_frames = 0, 0, 0

    def get(self, row, ids):
        source = row["image_source"]
        if isinstance(source, Path):
            if self.last_path != source:
                # Release the previous episode before decoding the next one.
                self.last_episode = None
                with source.open("rb") as stream:
                    self.last_episode = pickle.load(stream)
                self.last_path = source
                self.raw_episode_loads += 1
            source = self.last_episode
        self.batches += 1
        self.peak_batch_frames = max(self.peak_batch_frames, len(ids))
        images = [torch.as_tensor(np.asarray(source[key])[ids], device=self.device).permute(0, 3, 1, 2).contiguous()
                  for key in ("front_images", "wrist_images")]
        masks = unpack_resize_masks(np.asarray(source["cube_instance_masks_packed"])[ids], row["source_hw"]).to(self.device)
        return (*images, masks)

    def statistics(self):
        return {"feature_cache_bytes": 0, "early_maps": "GPU/CPU_minibatch_only_no_full_prefill",
                "raw_episode_loads": self.raw_episode_loads, "batches": self.batches,
                "peak_batch_frames": self.peak_batch_frames,
                "archive_storage": "current_NPZ_RGB_packed_masks_host_RAM_or_one_raw_pickle"}


def run_pixel_training(args, dp, checkpoint):
    """Opt-in seam called by existing CLI after strict DP checkpoint load."""
    if not args.region_conditioned or args.vision_mode != "both":
        raise ValueError("pixel mode requires --region-conditioned and --vision-mode both")
    if any(not np.isfinite(w) or w < 0 for w in (args.pixel_centre_weight, args.pixel_mask_weight)):
        raise ValueError("pixel auxiliary weights must be finite nonnegative")
    device = torch.device(args.device)
    domain = CubeTaskRegion(args.region_table_height_m, args.region_height_source)
    records, provenance = load_pixel_records(args.data_dir, args.data_archive, domain)
    seeds = sorted({r["seed"] for r in records})
    groups = [args.fit_seeds, args.calibration_seeds, args.selection_seeds]
    if not any(groups) and len(seeds) == 4:
        groups = [seeds[:2], seeds[2:3], seeds[3:]]
    if not all(groups):
        raise ValueError("pixel mode requires three explicit disjoint split groups, or four default seeds")
    groups = [sorted(set(x)) for x in groups]
    sets = [set(x) for x in groups]
    if set.union(*sets) != set(seeds) or any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("pixel split groups must be disjoint and exactly cover dataset reset seeds")
    splits = [[r for r in records if r["seed"] in group] for group in groups]
    fit_q = np.concatenate([r["q"] for r in splits[0]])
    fit_xyz = np.concatenate([r["target"] for r in splits[0]])
    fit_inside = np.concatenate([r["inside"] for r in splits[0]])
    stats, norm_contract = fit_region_normalization(fit_q, fit_xyz, fit_inside)
    weights, weight_contract = fit_region_class_weights(fit_inside)
    if np.any(np.concatenate([r["inside"] for r in splits[1]]).sum(0) == 0):
        raise ValueError("calibration needs valid GT-inside XYZ for both objects")
    model = PixelCubePose(FrozenEarlyDPEncoder(dp), stats).to(device)
    stream = PixelBatchStream(device)
    class_weights = torch.as_tensor(weights, device=device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.learning_rate, weight_decay=1e-4)
    # Pixel maps are recomputed; independently cap memory with the extraction batch flag.
    batch_size = min(args.train_batch_size, args.feature_batch_size)

    def batch(row, ids):
        front, wrist, masks = stream.get(row, ids)
        q = torch.as_tensor(row["q"][ids], device=device)
        inside = torch.as_tensor(row["inside"][ids], device=device)
        target = torch.as_tensor(row["target"][ids], device=device)
        target = neutral_pose_targets(target, inside, model.normalization.pose_mean)
        target = model.normalization.normalize_targets(target)
        return model(front, wrist, q), target, inside, masks

    def evaluate(rows):
        parts, targets, masks, logits, seed_frames, episodes = [], [], [], [], [], []
        sums, counts = torch.zeros(2, device=device), torch.zeros(2, device=device)
        centre_errors, intersection, union = [], np.zeros((2, 2), np.int64), np.zeros((2, 2), np.int64)
        with torch.no_grad():
            for row in rows:
                for start in range(0, row["length"], batch_size):
                    ids = np.arange(start, min(start+batch_size, row["length"]))
                    pred, target, inside, mask = batch(row, ids)
                    s, c = masked_xyz_terms(pred.normalized_xyz, target, inside)
                    sums += s; counts += c
                    parts.append(model.normalization.denormalize_poses(pred.normalized_xyz.reshape(-1, 6)).cpu().numpy())
                    targets.append(row["target"][ids]); masks.append(row["inside"][ids])
                    logits.append(pred.region_logits.cpu().numpy())
                    seed_frames.extend([row["seed"]]*len(ids)); episodes.extend([row["source"]]*len(ids))
                    eligible = row["centre_eligible"][ids]
                    error = (pred.centres[:, 0].cpu().numpy()-row["centre_targets"][ids])*np.asarray(RGB_HW[::-1])
                    centre_errors.extend(np.linalg.norm(error[eligible], axis=-1).tolist())
                    binary = pred.mask_logits >= 0
                    intersection += (binary & mask).sum((0, 3, 4)).cpu().numpy()
                    union += (binary | mask).sum((0, 3, 4)).cpu().numpy()
        report = region_pose_metrics(np.concatenate(parts), np.concatenate(targets), np.concatenate(masks),
                                    np.concatenate(logits), seeds=np.asarray(seed_frames), episode_ids=np.asarray(episodes))
        report["pixel_auxiliary"] = {"front_eligible_object_frames": len(centre_errors),
            "front_centre_l2_rgb_px_p95": float(np.quantile(centre_errors, .95)) if centre_errors else None,
            "mask_iou_per_camera_object": [[float(intersection[i,j]/union[i,j]) if union[i,j] else None
                                           for j in range(2)] for i in range(2)],
            "mask_iou_denominator": "all_frames_per_camera_object_no_visibility_filter"}
        report["round_pose_gates"] = {}
        seed_array = np.asarray(seed_frames)
        for seed in sorted(set(seed_frames)):
            selected = seed_array == seed
            per_round = region_pose_metrics(np.concatenate(parts)[selected], np.concatenate(targets)[selected],
                                           np.concatenate(masks)[selected])
            report["round_pose_gates"][str(seed)] = per_round["pose_gate"]
        return float((.5*(sums/counts.clamp_min(1)).sum()).item()), report

    best_loss, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(args.epochs):
        model.train()
        for index in np.random.permutation(len(splits[0])):
            row = splits[0][index]
            order = np.random.permutation(row["length"])
            for start in range(0, len(order), batch_size):
                ids = order[start:start+batch_size]
                pred, target, inside, masks = batch(row, ids)
                terms = pixel_pose_loss(pred, target, inside, class_weights, masks,
                    torch.as_tensor(row["centre_targets"][ids], device=device),
                    torch.as_tensor(row["centre_eligible"][ids], device=device),
                    centre_weight=args.pixel_centre_weight, mask_weight=args.pixel_mask_weight)
                if not torch.isfinite(terms["loss"]):
                    raise ValueError("nonfinite pixel training loss")
                optimizer.zero_grad(set_to_none=True)
                terms["loss"].backward(); optimizer.step()
        model.eval()
        cal_loss, _ = evaluate(splits[1])
        if cal_loss < best_loss:
            best_loss, best_epoch = cal_loss, epoch+1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no finite calibration-selected pixel checkpoint")
    model.load_state_dict(best_state, strict=True); model.eval()
    reports = {name: evaluate(rows)[1] for name, rows in zip(("fit", "calibration", "selection"), splits)}
    contract = {"pixel_version": PIXEL_VERSION, "representation": "pixel", "region_conditioned": True,
        "model_inputs": ["current_front_rgb_uint8_240x320", "current_wrist_rgb_uint8_240x320", "current_measured_q_radians"],
        "preprocessing": "full_source_RGB_no_DP_crop_ImageNet_norm", "camera_order": ["front", "wrist"],
        "cube_position_order": ["held_cube", "fixed_cube"], "encoder_frozen": True,
        "fpn_channels": 32, "rgb_shape_hw": list(RGB_HW), "centre_auxiliary_cameras": ["front"],
        "centre_eligibility": "GT_inside_positive_optical_Z_in_frustum_source_visible_pixels>=8",
        "centre_coordinate_contract": "(source_u+.5)/source_W,(source_v+.5)/source_H",
        "mask_resize": "big_endian_unpack_then_nearest_exact_to_RGB_grid",
        "auxiliary_loss_weights": {"centre": args.pixel_centre_weight, "mask": args.pixel_mask_weight},
        "mask_class_weighting": "batch_per_camera_object_observed_foreground_background_equal_no_resampling",
        "region_domain": domain.metadata(), "region_normalization": norm_contract,
        "region_class_weighting": weight_contract, "epoch_selection": "calibration_masked_XYZ_only",
        "fit_seeds": groups[0], "calibration_seeds": groups[1], "selection_seeds": groups[2],
        "data_provenance": provenance, "dp_checkpoint_sha256": _sha256(args.dp_checkpoint),
        "dp_config": checkpoint["config"], "dp_camera_indices": list(model.encoder.dp_camera_indices),
        "best_epoch": best_epoch, "calibration_normalized_mse": best_loss,
        "seed": args.seed, "effective_batch_size": batch_size,
        "device_parameter_storage": "only_checkpoint_conv1_through_layer3_plus_FPN_pose_head_no_source_DP_actor_or_layer4",
        "frozen_encoder_parameter_bytes": sum(p.numel()*p.element_size() for p in model.encoder.parameters()),
        "termination_value_semantics": "unchanged_no_absorbing_outside_or_new_zero_value"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**contract, "model": best_state, "normalization": {k: torch.as_tensor(v) for k,v in stats.items()}}, args.out)
    selection = reports["selection"]
    gate = selection["pose_gate"]["passed"] and all(g["passed"] for g in selection["round_pose_gates"].values())
    report = {**contract, **reports, "checkpoint": str(args.out), "episodes": len(records),
              "transitions": sum(r["length"] for r in records), "stream": stream.statistics(),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "pose_gate": {"passed": bool(gate), "thresholds_mm": {"held_height_p95": 2.5,"relative_xy_axis_p95": 9.},
                            "claim_scope": "development_only_no_planner_or_real_robustness_claim"},
              "limitations": ["wrist projection auxiliary disabled", "GT visibility masks targets only",
                              "outside eligibility remains data-dependent", "no temporal/score/planner validation"]}
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"checkpoint": str(args.out), "pose_gate": report["pose_gate"], "stream": report["stream"]}, indent=2))
    return report
