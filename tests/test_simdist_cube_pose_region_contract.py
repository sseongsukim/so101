from __future__ import annotations

import hashlib
import json
import pickle
import sys

import numpy as np
import pytest
import torch

from so101.learning.simdist.cube_pose import CubePoseHead, CubeSpatialPoseHead
from so101.learning.simdist.cube_pose_region import (
    CubeTaskRegion, REGION_VERSION, conditional_pose_loss, current_successor_regions,
    fit_region_class_weights, fit_region_normalization, masked_xyz_mse,
    neutral_pose_targets, region_pose_metrics, weighted_region_bce,
)


def test_domain_is_inclusive_and_retains_lift_outside_table_support():
    domain = CubeTaskRegion()
    metadata = domain.metadata()
    points = np.array([[metadata["lower_xyz_m"], metadata["upper_xyz_m"]],
                       [[-.5, .9, .4], [.3, -.3, .08]]], dtype=np.float64)
    assert domain.labels(points).all()
    assert metadata["version"] == REGION_VERSION
    assert metadata["outside_is_absorbing"] is False
    assert metadata["table_height_source"] and metadata["bounds_source"]
    points[0, 0, 2] -= 1e-8
    assert domain.labels(points).tolist() == [[False, True], [True, True]]
    domain.validate_height(np.full(3, domain.table_height_m, dtype=np.float32))
    with pytest.raises(ValueError, match="disagrees"):
        domain.validate_height(np.array([0.0]))
    with pytest.raises(ValueError, match="height source"):
        CubeTaskRegion(height_source="")


def test_actual_successor_label_including_final_boundary_is_not_absorbing():
    current = np.zeros((3, 2, 3))
    current[1, 0, 2] = -241.94
    successor = np.concatenate((current[1:], current[-1:])).copy()
    successor[-1, 1, 0] = 2
    inside, next_inside = current_successor_regions(current, successor, CubeTaskRegion())
    assert inside.tolist() == [[True, True], [False, True], [True, True]]
    assert next_inside.tolist() == [[False, True], [True, True], [True, False]]
    np.testing.assert_array_equal(next_inside[:-1], inside[1:])
    with pytest.raises(ValueError, match="discontinuity"):
        current_successor_regions(current, successor + .01, CubeTaskRegion())
    current[1, 0, 2] = np.nan
    with pytest.raises(ValueError, match="corrupt"):
        CubeTaskRegion().labels(current)


def test_fit_norm_excludes_only_each_objects_outside_xyz_but_keeps_all_q():
    positions = np.zeros((4, 2, 3), np.float64)
    positions[:, 0, 2] = [.03, .04, .05, -241.94]
    positions[:, 1, 0] = [.1, .2, .3, .4]
    original = positions.copy()
    q = np.repeat(np.array([0, 1, 2, 10.])[:, None], 6, 1)
    inside = CubeTaskRegion().labels(positions)
    stats, contract = fit_region_normalization(q, positions, inside)
    np.testing.assert_allclose(stats["q_mean"], q.mean(0))
    np.testing.assert_allclose(stats["pose_mean"].reshape(2, 3)[0, 2], .04)
    np.testing.assert_allclose(stats["pose_mean"].reshape(2, 3)[1, 0], .25)
    assert contract["pose_fit_rows_per_object"] == [3, 4]
    assert contract["q_fit_rows"] == 4 and (stats["pose_std"] >= .001).all()
    np.testing.assert_array_equal(positions, original)
    with pytest.raises(ValueError, match="every fit object"):
        fit_region_normalization(q, positions, np.zeros((4, 2), bool))


def test_neutral_targets_and_empty_batch_masks_have_finite_zero_gradients():
    mask = torch.tensor([[False, True], [False, True]])
    targets = torch.full((2, 6), float("nan"))
    targets[:, 3:] = 0
    means = torch.arange(6, dtype=torch.float32)
    safe = neutral_pose_targets(targets, mask, means)
    assert torch.isfinite(safe).all()
    assert torch.equal(safe[:, :3], means[:3].expand(2, -1))
    predictions = torch.ones(2, 6, requires_grad=True)
    loss = masked_xyz_mse(predictions, targets, mask)
    assert loss.item() == .5  # fixed object MSE 1, held empty keeps fixed weight .5
    loss.backward()
    assert predictions.grad[:, :3].count_nonzero() == 0
    assert predictions.grad[:, 3:].count_nonzero() == 6
    prediction = torch.ones(2, 6, requires_grad=True)
    empty_loss = masked_xyz_mse(prediction, targets, torch.zeros(2, 2, dtype=torch.bool))
    empty_loss.backward()
    assert empty_loss.item() == 0 and prediction.grad.count_nonzero() == 0


def test_class_weights_are_fit_only_per_object_and_missing_class_is_untested():
    inside = np.array([[True, True], [True, True], [True, True], [False, True]])
    weights, metadata = fit_region_class_weights(inside)
    np.testing.assert_allclose(weights, [[2, 2/3], [0, 1]])
    assert metadata["fit_counts_per_object_class"] == [[1, 3], [0, 4]]
    assert metadata["both_classes_observed_per_object"] == [True, False]
    gt = torch.as_tensor(inside)
    logits = torch.zeros(4, 2, requires_grad=True)
    bce = weighted_region_bce(logits, gt, torch.as_tensor(weights))
    torch.testing.assert_close(bce, torch.tensor(np.log(2), dtype=torch.float32))
    result = conditional_pose_loss(torch.zeros(4, 6), torch.zeros(4, 6), logits, gt, torch.as_tensor(weights))
    torch.testing.assert_close(result["loss"], .1 * bce)
    result["loss"].backward()
    assert torch.isfinite(logits.grad).all()


def test_gt_denominators_cannot_be_gamed_by_predicted_abstention():
    target = np.zeros((4, 2, 3))
    target[3, 0, 2] = -72
    inside = CubeTaskRegion().labels(target)
    prediction = target.copy()
    prediction[0, 0, 2] = .20
    logits = np.full((4, 2), -10.)  # claim every object outside, including erroneous pose
    report = region_pose_metrics(prediction, target, inside, logits,
                                 seeds=np.full(4, 27200), episode_ids=np.array([0, 0, 1, 1]))
    assert report["denominators"]["inside_frames_per_object"] == [3, 4]
    assert report["denominators"]["relative_xy_both_inside_frames"] == 3
    assert report["held_height_abs_mm_p50_p95"][1] > 100
    assert not report["pose_gate"]["passed"]
    assert report["region_classifier"][0]["inside_fnr"] == 1


def test_317_outside_frames_two_episodes_one_round_do_not_pass_recall_gate():
    target = np.zeros((400, 2, 3))
    target[:317, 0, 2] = -2
    inside = CubeTaskRegion().labels(target)
    logits = np.where(inside, 10., -10.)
    episodes = np.array(["env2"] * 159 + ["env3"] * 158 + ["normal"] * 83)
    report = region_pose_metrics(target, target, inside, logits,
                                 seeds=np.full(400, 27200), episode_ids=episodes)
    held, fixed = report["region_classifier"]
    assert held["outside_recall"] == 1
    assert held["outside_episode_count"] == 2 and held["outside_reset_round_count"] == 1
    assert held["outside_gate_status"] == "inconclusive" and held["outside_recall_gate_passed"] is None
    assert fixed["outside_recall"] is None and fixed["outside_recall_gate_passed"] is None


def test_empty_metric_denominators_are_na_and_eligible_outside_counts_are_explicit():
    target = np.zeros((4, 2, 3))
    target[..., 2] = -2
    inside = CubeTaskRegion().labels(target)
    report = region_pose_metrics(target, target, inside, np.full((4, 2), -10.),
                                 seeds=np.array([1, 1, 2, 2]), episode_ids=np.arange(4))
    assert report["held_height_abs_mm_p50_p95"] is None
    assert report["relative_xy_abs_mm_p95_per_axis"] is None
    assert report["pose_gate"]["status"] == "inconclusive"
    assert report["region_classifier"][0]["inside_fnr"] is None
    assert report["region_classifier"][0]["outside_recall_gate_passed"]
    empty = region_pose_metrics(np.empty((0, 2, 3)), np.empty((0, 2, 3)), np.empty((0, 2), bool))
    assert empty["samples"] == 0 and not empty["pose_gate"]["passed"]


@pytest.mark.parametrize("spatial", [False, True])
def test_optional_two_output_head_keeps_legacy_pose_initialization_and_keys(spatial):
    torch.manual_seed(44)
    legacy = CubeSpatialPoseHead(4) if spatial else CubePoseHead()
    torch.manual_seed(44)
    model = CubeSpatialPoseHead(4, region_conditioned=True) if spatial else CubePoseHead(region_conditioned=True)
    cameras = [torch.randn(2, 4, 3, 5) for _ in range(2)] if spatial else [torch.randn(2, 128) for _ in range(2)]
    q = torch.randn(2, 6)
    prediction = model(*cameras, q)
    assert prediction.normalized_xyz.shape == (2, 2, 3) and prediction.region_logits.shape == (2, 2)
    assert torch.equal(prediction.normalized_xyz.reshape(2, 6), legacy(*cameras, q))
    assert prediction.region_logits.count_nonzero() == 0
    assert not any(key.startswith("region_head") for key in legacy.state_dict())
    with pytest.raises(TypeError):
        model(*cameras, q, GT_region=torch.ones(2, 2))


@pytest.mark.parametrize("representation,archive_source", [("pooled", False), ("spatial", False),
                                                           ("spatial-mean", False), ("spatial", True)])
def test_cpu_region_trainer_preserves_all_outcomes_and_metadata(tmp_path, monkeypatch, representation, archive_source):
    import train_simdist_cube_pose as trainer
    from test_simdist_cube_pose_spatial_contract import StubDP
    from so101.learning.dp.config import DPConfig

    torch.set_num_threads(1)
    config = DPConfig(pretrained_backbone=False, image_shape=(32, 32), crop_size=32, front_crop_margin=0)
    dp_path = tmp_path / "dp.pt"
    torch.save({"config": config.to_dict(), "model": StubDP(config).state_dict()}, dp_path)
    paths, fit_q, fit_pos, fit_mask, episodes = [], [], [], [], []
    for seed in range(501, 505):
        pose = np.full((3, 2, 3), .10, np.float32)
        pose[:, 0, 2] += np.arange(3, dtype=np.float32) * .002
        if seed == 501:
            pose[1, 0, 2] = -241.94
        q = np.full((3, 6), seed * .01, np.float32)
        q += np.arange(3, dtype=np.float32)[:, None] * .001
        image = np.full((3, 32, 32, 3), seed % 255, np.uint8)
        episode = {"seed": seed, "task_horizon_ticks": 2, "tick": np.arange(3),
                   "actions": np.zeros((3, 6)), "observations": q,
                   "front_images": image, "wrist_images": image + 1,
                   "cube_positions_robot_relative_m": pose,
                   "next_cube_positions_robot_relative_m": np.concatenate((pose[1:], pose[-1:])),
                   "table_height_robot_relative_m": np.full(3, .0300814467, np.float32),
                   "cube_position_order": ["held_cube", "fixed_cube"],
                   "cube_position_label_semantics": "simulator_xyz_robot_root_relative_training_only",
                   "terminated": np.array([False, False, True]), "truncated": np.array([False, True, False]),
                   "rewards": np.arange(3, dtype=np.float32)}
        path = tmp_path / f"trajectory_{seed}.pkl"
        with path.open("wb") as stream:
            pickle.dump(episode, stream)
        paths.append((path, hashlib.sha256(path.read_bytes()).hexdigest()))
        episodes.append(episode)
        if seed <= 502:
            fit_q.append(q[:2]); fit_pos.append(pose[:2]); fit_mask.append(CubeTaskRegion().labels(pose[:2]))
    output = tmp_path / "region.pt"
    source_args = ["--data-dir", str(tmp_path)]
    if archive_source:
        archive = tmp_path / "aligned.npz"
        arrays = {key: np.concatenate([np.asarray(ep[key])[:2] for ep in episodes]) for key in (
            "observations", "front_images", "wrist_images", "cube_positions_robot_relative_m",
            "next_cube_positions_robot_relative_m", "table_height_robot_relative_m", "terminated")}
        np.savez_compressed(archive, **arrays, episode_starts=np.arange(4) * 2, episode_lengths=np.full(4, 2),
            episode_seeds=np.arange(501, 505), phase_labels=np.full(8, "pre_success"),
            trajectory_quality=np.full(8, "perturbed"), cube_position_order=np.array(["held_cube", "fixed_cube"]),
            cube_position_label_semantics=np.asarray("simulator_xyz_robot_root_relative_training_only"))
        source_args = ["--data-archive", str(archive)]
    monkeypatch.setattr(trainer, "DiffusionPolicy", StubDP)
    monkeypatch.setattr(sys, "argv", ["train_simdist_cube_pose.py", *source_args,
                        "--dp-checkpoint", str(dp_path), "--out", str(output), "--device", "cpu",
                        "--epochs", "1", "--train-batch-size", "1", "--feature-batch-size", "1",
                        "--representation", representation, "--region-conditioned", "--spatial-cache-mib", ".5"])
    trainer.main()
    saved = torch.load(output, weights_only=True)
    report = json.loads(output.with_suffix(".json").read_text())
    expected, _ = fit_region_normalization(np.concatenate(fit_q), np.concatenate(fit_pos), np.concatenate(fit_mask))
    np.testing.assert_allclose(saved["normalization"]["pose_mean"], expected["pose_mean"])
    np.testing.assert_allclose(saved["normalization"]["q_mean"], expected["q_mean"])
    assert saved["region_conditioned"] and saved["region_domain"]["version"] == REGION_VERSION
    assert saved["region_class_weighting"]["fit_counts_per_object_class"] == [[1, 3], [0, 4]]
    assert saved["region_normalization"]["q_fit_rows"] == 4
    assert saved["recorded_height_cross_checked_episodes"] == 4
    assert saved["termination_value_semantics"].startswith("unchanged")
    assert report["transitions"] == 8 and report["fit"]["samples"] == 4
    assert report["fit"]["denominators"]["inside_frames_per_object"] == [3, 4]
    assert report["selection"]["denominators"]["held_height_frames"] == 2
    assert report["pose_gate"]["thresholds_mm"] == {"held_height_p95": 2.5, "relative_xy_axis_p95": 9.0}
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in paths)
    head = (CubePoseHead(region_conditioned=True) if representation == "pooled" else
            CubeSpatialPoseHead(512, attention=representation == "spatial", region_conditioned=True))
    head.load_state_dict(saved["model"], strict=True)
