"""Versioned GT object domain, conditional pose losses and honest denominators.

The domain is neither tabletop support nor visibility nor termination. These
functions receive labels for supervision/evaluation only, never deployed inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

OBJECTS = ("held_cube", "fixed_cube")
REGION_VERSION = "so101_cube_task_region_v1_20261008"
REGION_LOSS_WEIGHT = 0.1


@dataclass(frozen=True)
class CubeTaskRegion:
    table_height_m: float = 0.0300814467
    height_source: str = "declared_so101.scenes.tabletop.ROBOT_BASE_BOTTOM_Z_minus_robot_root_z"

    def __post_init__(self):
        if not np.isfinite(self.table_height_m) or not self.height_source.strip():
            raise ValueError("region requires a finite table height and explicit height source")

    def metadata(self) -> dict:
        return {
            "version": REGION_VERSION,
            "bounds_source": "Sol_high_approved_conservative_model_target_domain_20261008",
            "frame": "simulator_world_xyz_minus_robot_root_world_translation_no_rotation",
            "units": "metres", "object_order": list(OBJECTS), "inclusive": True,
            "lower_xyz_m": [-1.0, -1.0, self.table_height_m - 0.10],
            "upper_xyz_m": [1.0, 1.0, self.table_height_m + 1.00],
            "table_height_m": self.table_height_m, "table_height_source": self.height_source,
            "definition": "per_object_model_domain_not_support_visibility_success_or_terminal",
            "outside_is_absorbing": False,
        }

    def labels(self, positions: np.ndarray) -> np.ndarray:
        positions = _positions(positions)
        bounds = self.metadata()
        return ((positions >= np.asarray(bounds["lower_xyz_m"])) &
                (positions <= np.asarray(bounds["upper_xyz_m"]))).all(-1)

    def validate_height(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all() or not np.allclose(values, self.table_height_m, atol=1e-6, rtol=0):
            raise ValueError("recorded table height disagrees with declared static region height")


def _positions(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 2 and value.shape[1] == 6:
        value = value.reshape(-1, 2, 3)
    if value.ndim != 3 or value.shape[1:] != (2, 3) or not np.isfinite(value).all():
        raise ValueError("cube positions must be finite [N,2,3] or [N,6]; nonfinite is corrupt, not outside")
    return value


def current_successor_regions(current: np.ndarray, successor: np.ndarray,
                              domain: CubeTaskRegion) -> tuple[np.ndarray, np.ndarray]:
    current, successor = _positions(current), _positions(successor)
    if successor.shape != current.shape:
        raise ValueError("current/successor shape mismatch")
    if len(current) > 1 and not np.allclose(successor[:-1], current[1:], atol=2e-6, rtol=0):
        raise ValueError("cube successor discontinuity")
    inside, next_inside = domain.labels(current), domain.labels(successor)
    if len(current) > 1 and not np.array_equal(next_inside[:-1], inside[1:]):
        raise ValueError("region boundary labels disagree across successor/current rows")
    return inside, next_inside


def fit_region_normalization(q: np.ndarray, positions: np.ndarray, inside: np.ndarray) -> tuple[dict, dict]:
    positions = _positions(positions)
    q, inside = np.asarray(q, dtype=np.float64), np.asarray(inside)
    if q.shape != (len(positions), 6) or not len(q) or not np.isfinite(q).all():
        raise ValueError("normalization needs finite nonempty fit q [N,6]")
    if inside.shape != (len(q), 2) or inside.dtype != np.bool_:
        raise ValueError("normalization needs boolean per-object GT inside labels")
    counts = inside.sum(0)
    if np.any(counts == 0):
        raise ValueError("every fit object must have at least one GT-inside pose")
    means = np.stack([positions[inside[:, j], j].mean(0) for j in range(2)])
    stds = np.stack([positions[inside[:, j], j].std(0) for j in range(2)])
    stats = {"q_mean": q.mean(0), "q_std": np.maximum(q.std(0), 1e-3),
             "pose_mean": means.reshape(6), "pose_std": np.maximum(stds, 1e-3).reshape(6)}
    return {k: v.astype(np.float32) for k, v in stats.items()}, {
        "q_fit_rows": len(q), "pose_fit_rows_per_object": counts.tolist(),
        "pose_fit_mask": "GT_inside_per_object_only", "q_fit_mask": "all_fit_rows",
        "std_floor_m": 1e-3,
    }


def neutral_pose_targets(positions: torch.Tensor, inside: torch.Tensor, pose_mean: torch.Tensor) -> torch.Tensor:
    """Replace invalid XYZ by fit means BEFORE normalization/arithmetic."""
    positions = positions.reshape(-1, 2, 3)
    if inside.shape != positions.shape[:2] or inside.dtype != torch.bool:
        raise ValueError("pose targets need matching boolean GT mask")
    return torch.where(inside[..., None], positions, pose_mean.reshape(1, 2, 3)).reshape(-1, 6)


def fit_region_class_weights(inside: np.ndarray) -> tuple[np.ndarray, dict]:
    inside = np.asarray(inside)
    if inside.ndim != 2 or inside.shape[1] != 2 or inside.dtype != np.bool_ or not len(inside):
        raise ValueError("class weights require nonempty boolean fit labels [N,2]")
    counts = np.stack(((~inside).sum(0), inside.sum(0)), -1)
    weights = np.zeros((2, 2), dtype=np.float32)
    for j in range(2):
        present = counts[j] > 0
        weights[j, present] = (len(inside) / (2 * counts[j, present])) if present.all() else 1.0
    return weights, {"class_order": ["outside", "inside"], "fit_counts_per_object_class": counts.tolist(),
                     "weights_per_object_class": weights.tolist(),
                     "both_classes_observed_per_object": (counts > 0).all(-1).tolist(),
                     "source": "fit_only_inverse_frequency_no_oversampling"}


def masked_xyz_terms(prediction: torch.Tensor, target: torch.Tensor,
                     inside: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    prediction, target = prediction.reshape(-1, 2, 3), target.reshape(-1, 2, 3)
    if prediction.shape != target.shape or inside.shape != target.shape[:2] or inside.dtype != torch.bool:
        raise ValueError("XYZ loss needs matching pose and boolean per-object masks")
    mask = inside[..., None].expand_as(target)
    if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all():
        raise ValueError("valid XYZ loss inputs must be finite")
    safe_pred = torch.where(mask, prediction, 0.0)
    safe_target = torch.where(mask, target, 0.0)
    return (safe_pred - safe_target).square().sum((0, 2)), mask.sum((0, 2))


def masked_xyz_mse(prediction: torch.Tensor, target: torch.Tensor, inside: torch.Tensor) -> torch.Tensor:
    sums, counts = masked_xyz_terms(prediction, target, inside)
    # Empty object/batch contributes differentiable zero; object weights stay .5.
    return .5 * (sums / counts.clamp_min(1)).sum()


def weighted_region_bce(logits: torch.Tensor, inside: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if logits.shape != inside.shape or logits.ndim != 2 or logits.shape[1] != 2 or inside.dtype != torch.bool:
        raise ValueError("region BCE requires logits and boolean labels [N,2]")
    if weights.shape != (2, 2) or not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("region BCE needs finite nonnegative fit weights [2,2]")
    if not torch.isfinite(logits).all() or not len(logits):
        raise ValueError("region logits must be finite and nonempty")
    frame_weights = torch.stack([weights[j][inside[:, j].long()] for j in range(2)], -1)
    terms = nn.functional.binary_cross_entropy_with_logits(logits, inside.to(logits.dtype), reduction="none")
    return (terms * frame_weights).mean()


def conditional_pose_loss(prediction: torch.Tensor, target: torch.Tensor, logits: torch.Tensor,
                          inside: torch.Tensor, weights: torch.Tensor) -> dict:
    xyz = masked_xyz_mse(prediction, target, inside)
    region = weighted_region_bce(logits, inside, weights)
    return {"loss": xyz + REGION_LOSS_WEIGHT * region, "xyz": xyz, "region": region}


def region_pose_metrics(prediction: np.ndarray, target: np.ndarray, inside: np.ndarray,
                        logits: np.ndarray | None = None, *, seeds: np.ndarray | None = None,
                        episode_ids: np.ndarray | None = None) -> dict:
    """GT-only pose denominators; model abstention never removes pose errors."""
    prediction, target = _positions(prediction), _positions(target)
    inside = np.asarray(inside)
    n = len(target)
    if prediction.shape != target.shape or inside.shape != (n, 2) or inside.dtype != np.bool_:
        raise ValueError("metrics need matching finite poses and boolean GT labels")
    if seeds is not None and np.asarray(seeds).shape != (n,):
        raise ValueError("seed labels must match frame count")
    if episode_ids is not None and np.asarray(episode_ids).shape != (n,):
        raise ValueError("episode ids must match frame count")
    held, both = inside[:, 0], inside.all(-1)
    z = np.abs(prediction[held, 0, 2] - target[held, 0, 2]) * 1000
    xy = np.abs((prediction[both, 0, :2] - prediction[both, 1, :2]) -
                (target[both, 0, :2] - target[both, 1, :2])) * 1000
    z_metrics = np.quantile(z, [.5, .95]).tolist() if len(z) else None
    xy_metrics = np.quantile(xy, .95, axis=0).tolist() if len(xy) else None
    rmse = [(np.sqrt(((prediction[inside[:, j], j] - target[inside[:, j], j]) ** 2).mean(0)) * 1000).tolist()
            if inside[:, j].any() else None for j in range(2)]
    report = {
        "samples": n, "denominators": {"all_region_frames_per_object": n,
        "inside_frames_per_object": inside.sum(0).tolist(), "outside_frames_per_object": (~inside).sum(0).tolist(),
        "held_height_frames": int(held.sum()), "relative_xy_both_inside_frames": int(both.sum())},
        "held_height_abs_mm_p50_p95": z_metrics, "relative_xy_abs_mm_p95_per_axis": xy_metrics,
        "xyz_rmse_mm_per_object_axis": rmse,
        "pose_gate": {"held_height_p95_mm": 2.5, "relative_xy_axis_p95_mm": 9.0,
                      "passed": bool(z_metrics is not None and xy_metrics is not None and
                                     z_metrics[1] < 2.5 and all(v < 9 for v in xy_metrics)),
                      "status": "evaluated" if len(z) and len(xy) else "inconclusive",
                      "denominator": "GT_inside_only_never_predicted_mask"},
    }
    if logits is None:
        report["region_classifier"] = None
        return report
    logits = np.asarray(logits)
    if logits.shape != (n, 2) or not np.isfinite(logits).all():
        raise ValueError("metrics require finite region logits [N,2]")
    predicted_inside = logits >= 0  # sigmoid>=.5 without overflow
    classifiers = []
    for j, name in enumerate(OBJECTS):
        truth, pred = inside[:, j], predicted_inside[:, j]
        tp, fn = int((truth & pred).sum()), int((truth & ~pred).sum())
        fp, tn = int((~truth & pred).sum()), int((~truth & ~pred).sum())
        ni, no = tp + fn, fp + tn
        n_out_ep = None if episode_ids is None else len(np.unique(np.asarray(episode_ids)[~truth]))
        n_out_seed = None if seeds is None else len(np.unique(np.asarray(seeds)[~truth]))
        enough = n_out_ep is not None and n_out_seed is not None and n_out_ep >= 3 and n_out_seed >= 2
        coverage = tp / ni if ni else None
        recall = tn / no if no else None
        classifiers.append({"object": name, "tp_inside": tp, "fn_inside": fn, "fp_inside": fp, "tn_outside": tn,
            "inside_fnr": fn / ni if ni else None, "inside_coverage": coverage,
            "outside_recall": recall, "accuracy": (tp + tn) / n if n else None,
            "outside_episode_count": n_out_ep, "outside_reset_round_count": n_out_seed,
            "inside_coverage_gate_passed": coverage >= .99 if coverage is not None else None,
            "outside_recall_gate_passed": recall >= .95 if enough and recall is not None else None,
            "outside_gate_status": "evaluated" if enough else "inconclusive",
            "decision_threshold": .5})
    report["region_classifier"] = classifiers
    return report
