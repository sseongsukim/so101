from __future__ import annotations

import json
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from so101.learning.simdist.cube_pose import CubeSpatialPoseHead
from so101.learning.simdist.cube_pose_features import SpatialFeatureStream
from so101.learning.simdist.cube_pose_spatial import (
    CubePoseNormalization, FrozenDPSpatialEncoder, attention_coordinates,
)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    torch.set_num_threads(1)


def test_soft_coordinates_use_camera_map_cell_centers_u_right_v_down():
    weights = torch.zeros(2, 2, 3, 7)
    weights[0, :, 0, 0] = 1
    weights[1, :, -1, -1] = 1
    positions = attention_coordinates(weights)
    torch.testing.assert_close(positions[0], torch.tensor([[-6/7, -2/3]]).expand(2, -1))
    torch.testing.assert_close(positions[1], torch.tensor([[6/7, 2/3]]).expand(2, -1))
    uniform_positions = attention_coordinates(torch.full((1, 2, 3, 7), 1/21))
    torch.testing.assert_close(uniform_positions, torch.zeros(1, 2, 2), atol=1e-7, rtol=0)


def test_fit_normalization_round_trip_preserves_held_fixed_robot_xyz_metres():
    stats = {
        "q_mean": torch.arange(6) * .1, "q_std": torch.full((6,), .2),
        "pose_mean": torch.tensor([.1, -.2, .03, -.4, .5, .02]),
        "pose_std": torch.tensor([.1, .2, .001, .3, .2, .001]),
    }
    normalizer = CubePoseNormalization(stats)
    poses = stats["pose_mean"][None] + stats["pose_std"][None] * torch.arange(6)[None]
    normalized = normalizer.normalize_targets(poses)
    torch.testing.assert_close(normalized, torch.arange(6, dtype=torch.float32)[None])
    torch.testing.assert_close(normalizer.denormalize_poses(normalized), poses)
    torch.testing.assert_close(normalizer.normalize_joints(stats["q_mean"][None]), torch.zeros(1, 6))
    with pytest.raises(ValueError, match="positive"):
        CubePoseNormalization({**stats, "pose_std": torch.zeros(6)})


@pytest.mark.parametrize("front_first", [False, True])
def test_frozen_source_keeps_exact_named_camera_transforms_weights_and_pooling_parity(front_first):
    from so101.learning.dp.vision import FrontCameraTransform, ResnetEncoder, WristCameraTransform

    dp = nn.Module()
    dp.config = SimpleNamespace(
        vision_backbone="resnet18", image_keys=("front", "wrist") if front_first else ("wrist", "front"),
        front_key="front", image_shape=(240, 320), crop_size=224,
    )
    dp.encoders = nn.ModuleList([ResnetEncoder(pretrained=False) for _ in range(2)])
    front_transform = FrontCameraTransform((240, 320), 224, 20)
    wrist_transform = WristCameraTransform(224)
    dp.transforms = nn.ModuleList([front_transform, wrist_transform] if front_first
                                 else [wrist_transform, front_transform])
    saved_weight = dp.encoders[0].model.conv1.weight.detach().clone()
    saved_keys = list(dp.encoders[0].state_dict())
    source = FrozenDPSpatialEncoder(dp).train()
    front, wrist = [torch.randint(0, 256, (1, 3, 240, 320), dtype=torch.uint8) for _ in range(2)]
    maps = source(front, wrist)
    assert maps.shape == (1, 2, 512, 7, 7)
    assert not maps.requires_grad and not source.training
    assert all(not module.training for module in source.modules())
    assert all(not parameter.requires_grad for parameter in source.parameters())
    assert source.transforms[0] is front_transform and source.transforms[1] is wrist_transform
    for output_index, (image, dp_index) in enumerate(zip((front, wrist), source.dp_camera_indices)):
        assert source.encoders[output_index] is dp.encoders[dp_index]
        with torch.no_grad():
            expected = dp.encoders[dp_index](dp.transforms[dp_index](image))
        assert torch.equal(maps[:, output_index].mean((-2, -1)), expected)
    assert saved_keys == list(dp.encoders[0].state_dict())
    assert torch.equal(saved_weight, dp.encoders[0].model.conv1.weight)
    # A cached frozen map is an ordinary tensor usable in head backpropagation.
    CubeSpatialPoseHead()(maps[:, 0], maps[:, 1], torch.zeros(1, 6)).sum().backward()
    assert all(parameter.grad is None for parameter in source.parameters())
    with pytest.raises(ValueError, match="uint8"):
        source(front.float(), wrist.float())
    with pytest.raises(TypeError):
        source(front, wrist, cube_positions=torch.zeros(1, 2, 3))


class TinyMapEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, front, wrist):
        self.calls += 1
        values = torch.stack((front[:, 0, 0, 0], wrist[:, 0, 0, 0]), 1).float()
        return values[:, :, None, None, None].expand(-1, -1, 4, 2, 2).contiguous()


def make_stream_sources(tmp_path, lengths=(3, 2)):
    sources = []
    for episode, length in enumerate(lengths):
        path = tmp_path / f"trajectory_{episode}.pkl"
        front = np.zeros((length + 1, 4, 5, 3), dtype=np.uint8)
        for frame in range(length + 1):
            front[frame] = 10 * episode + frame
        with path.open("wb") as stream:
            pickle.dump({"front_images": front, "wrist_images": front + 50,
                         "cube_positions_robot_relative_m": "never an input",
                         "phase_labels": "never an input", "tick": "never an input"}, stream)
        sources.append((path, length))
    return sources


def test_stream_bounds_cache_preserves_order_and_rejects_tail_or_privileged_indices(tmp_path):
    encoder = TinyMapEncoder()
    # One full two-frame chunk = 256 bytes; the second episode has a shorter tail.
    stream = SpatialFeatureStream(encoder, make_stream_sources(tmp_path), torch.device("cpu"),
                                  feature_batch_size=2, cache_mib=256/1024**2)
    ids = np.array([4, 0, 2, 4, 3], dtype=np.int64)
    maps = stream.get(ids)
    assert maps.shape == (5, 2, 4, 2, 2)
    torch.testing.assert_close(maps[:, 0, 0, 0, 0], torch.tensor([11., 0., 2., 11., 10.]))
    torch.testing.assert_close(maps[:, 1, 0, 0, 0], maps[:, 0, 0, 0, 0] + 50)
    assert stream.cached_bytes <= 256 and stream.peak_cached_bytes <= 256
    assert not stream.prefill_if_fits()
    with pytest.raises(ValueError, match="valid integer"):
        stream.get(np.array([5]))
    with pytest.raises(ValueError, match="valid integer"):
        stream.get(np.array([1.5]))
    assert sorted(path.name for path in tmp_path.iterdir()) == ["trajectory_0.pkl", "trajectory_1.pkl"]


def test_prefill_fits_exact_byte_budget_and_amortizes_extraction_across_epochs(tmp_path):
    encoder = TinyMapEncoder()
    # Five retained frames, ignoring two stored post-horizon tails: 5*128 bytes.
    stream = SpatialFeatureStream(encoder, make_stream_sources(tmp_path), torch.device("cpu"),
                                  feature_batch_size=2, cache_mib=640/1024**2)
    assert stream.prefill_if_fits()
    assert stream.statistics()["full_dataset_bytes"] == 640
    assert stream.cached_bytes == 640 and stream.peak_cached_bytes == 640
    calls = encoder.calls
    for ids in ([4, 0, 2, 1, 3], [1, 3, 0, 4, 2]):
        stream.get(np.array(ids))
    assert encoder.calls == calls == 3
    assert stream.episode_loads == 2
    assert stream.reencoded_chunks == 0
    assert stream.cache_hits > 0


def test_zero_cache_reencodes_without_retaining_maps(tmp_path):
    stream = SpatialFeatureStream(TinyMapEncoder(), make_stream_sources(tmp_path), torch.device("cpu"),
                                  feature_batch_size=2, cache_mib=0)
    first = stream.get(np.array([0]))
    second = stream.get(np.array([0]))
    assert torch.equal(first, second)
    assert stream.cached_bytes == 0 and not stream.cache
    assert stream.reencoded_chunks == 1 and stream.extractions == 2


def test_memory_backed_npz_episode_sources_keep_streaming_cache_contract():
    front = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    wrist = np.ones_like(front)
    stream = SpatialFeatureStream(TinyMapEncoder(), [((front, wrist), 3)], torch.device("cpu"),
                                  feature_batch_size=2, cache_mib=640/1024**2)
    assert stream.prefill_if_fits()
    first = stream.get(np.array([2, 0]))
    second = stream.get(np.array([1, 2]))
    assert first.shape == second.shape == (2, 2, 4, 2, 2)
    assert stream.episode_loads == 0
    assert stream.reencoded_chunks == 0
    assert stream.statistics()["full_dataset_fits"]


def test_no_privileged_state_argument_in_pose_head():
    head = CubeSpatialPoseHead(channels=4)
    with pytest.raises(TypeError):
        head(torch.zeros(1, 4, 2, 2), torch.zeros(1, 4, 2, 2), torch.zeros(1, 6),
             cube_positions_robot_relative_m=torch.zeros(1, 2, 3))


class StubResnet(nn.Module):
    """Small camera stub for trainer wiring; real ResNet parity is tested above."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward_spatial(self, image):
        image = nn.functional.adaptive_avg_pool2d(image.float().mean(1, keepdim=True) / 255, 2)
        return image.expand(-1, 512, -1, -1) * self.scale

    def forward(self, image):
        return self.forward_spatial(image).mean((-2, -1))


class StubDP(nn.Module):
    def __init__(self, config, stats=None):
        super().__init__()
        self.config = config
        self.encoders = nn.ModuleList([StubResnet(), StubResnet()])
        self.transforms = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.projections = nn.ModuleList([nn.Linear(512, 128), nn.Linear(512, 128)])
        self.layernorms = nn.ModuleList([nn.LayerNorm(128), nn.LayerNorm(128)])


@pytest.mark.parametrize("representation,vision", [("pooled", "both"), ("pooled", "none"),
                                                   ("spatial", "both"), ("spatial-mean", "both")])
def test_cpu_trainer_modes_keep_fit_only_normalization_horizon_and_checkpoint_contract(
        tmp_path, monkeypatch, capsys, representation, vision):
    import train_simdist_cube_pose as trainer
    from so101.learning.dp.config import DPConfig

    config = DPConfig(pretrained_backbone=False, image_shape=(32, 32), crop_size=32, front_crop_margin=0)
    dp_checkpoint = tmp_path / "dp.pt"
    torch.save({"config": config.to_dict(), "model": StubDP(config).state_dict(), "step": 7}, dp_checkpoint)
    data = tmp_path / "data"
    data.mkdir()
    fit_q, fit_pose = [], []
    for seed in range(101, 105):
        q = np.full((3, 6), seed * .01, dtype=np.float32) + np.arange(3)[:, None] * .001
        current = np.full((3, 2, 3), seed * .001, dtype=np.float32)
        current[:, 0, 2] += np.arange(3) * .002
        successor = np.concatenate((current[1:], current[-1:]))
        image = np.full((3, 32, 32, 3), seed, dtype=np.uint8)
        episode = {
            "seed": seed, "task_horizon_ticks": 2, "tick": np.arange(3),
            "actions": np.zeros((3, 6)), "observations": q,
            "front_images": image, "wrist_images": image + 1,
            "cube_positions_robot_relative_m": current,
            "next_cube_positions_robot_relative_m": successor,
            "cube_position_order": ["held_cube", "fixed_cube"],
            "cube_position_label_semantics": "simulator_xyz_robot_root_relative_training_only",
        }
        with (data / f"trajectory_{seed}.pkl").open("wb") as stream:
            pickle.dump(episode, stream)
        if seed <= 102:
            fit_q.append(q[:2])
            fit_pose.append(current[:2].reshape(2, 6))
    out = tmp_path / "head.pt"
    monkeypatch.setattr(trainer, "DiffusionPolicy", StubDP)
    monkeypatch.setattr(sys, "argv", ["train_simdist_cube_pose.py", "--data-dir", str(data),
                                     "--dp-checkpoint", str(dp_checkpoint), "--out", str(out),
                                     "--device", "cpu", "--epochs", "1", "--train-batch-size", "2",
                                     "--feature-batch-size", "1", "--representation", representation,
                                     "--vision-mode", vision, "--spatial-cache-mib", ".5"])
    trainer.main()
    saved = torch.load(out, weights_only=True)
    report = json.loads(out.with_suffix(".json").read_text())
    assert report["transitions"] == 8
    assert saved["model_inputs"] == ["current_front_rgb", "current_wrist_rgb", "current_measured_q"]
    assert "robot_root_relative_metres" in saved["pose_output_contract"]
    np.testing.assert_allclose(saved["normalization"]["q_mean"], np.concatenate(fit_q).mean(0))
    np.testing.assert_allclose(saved["normalization"]["pose_mean"], np.concatenate(fit_pose).mean(0))
    assert saved["fit_seeds"] == [101, 102] and saved["selection_seeds"] == [104]
    assert saved["representation"] == representation and saved["vision_mode"] == vision
    if representation != "pooled":
        assert saved["feature_shape"] == [512, 2, 2]
        assert saved["spatial_feature_contract"]["nominal_input_stride"] == 32
        assert report["spatial_cache_statistics"]["extractions"] == 8
        assert report["spatial_cache_statistics"]["reencoded_chunks"] == 0
    assert len(list(data.iterdir())) == 4
    capsys.readouterr()


def test_cpu_trainer_accepts_aligned_npz_shards_without_pickle_images(tmp_path, monkeypatch):
    import train_simdist_cube_pose as trainer
    from so101.learning.dp.config import DPConfig

    config = DPConfig(pretrained_backbone=False, image_shape=(32, 32), crop_size=32, front_crop_margin=0)
    dp_checkpoint = tmp_path / "dp.pt"
    torch.save({"config": config.to_dict(), "model": StubDP(config).state_dict(), "step": 7}, dp_checkpoint)
    lengths = np.array([3, 3, 3, 3], dtype=np.int64)
    seeds = np.array([101, 102, 103, 104], dtype=np.int64)
    starts = np.arange(4, dtype=np.int64) * 3
    q = np.arange(72, dtype=np.float32).reshape(12, 6) * .001
    pose = np.arange(72, dtype=np.float32).reshape(12, 2, 3) * .0001
    successor = np.concatenate([pose[1:], pose[-1:]], axis=0)
    images = np.arange(12, dtype=np.uint8)[:, None, None, None] * np.ones((12, 32, 32, 3), np.uint8)
    archive = tmp_path / "aligned.npz"
    np.savez_compressed(
        archive, episode_starts=starts, episode_lengths=lengths, episode_seeds=seeds,
        observations=q, cube_positions_robot_relative_m=pose,
        next_cube_positions_robot_relative_m=successor,
        front_images=images, wrist_images=images + 1,
        phase_labels=np.full(12, "pre_success"), trajectory_quality=np.full(12, "perturbed"),
        terminated=np.zeros(12, dtype=bool),
        cube_position_label_semantics=np.asarray("simulator_xyz_robot_root_relative_training_only"),
        cube_position_order=np.asarray(["held_cube", "fixed_cube"]),
    )
    out = tmp_path / "archive_head.pt"
    monkeypatch.setattr(trainer, "DiffusionPolicy", StubDP)
    monkeypatch.setattr(sys, "argv", [
        "train_simdist_cube_pose.py", "--data-archive", str(archive),
        "--dp-checkpoint", str(dp_checkpoint), "--out", str(out), "--device", "cpu",
        "--epochs", "1", "--train-batch-size", "2", "--feature-batch-size", "2",
        "--representation", "spatial", "--spatial-cache-mib", ".5",
    ])
    trainer.main()
    report = json.loads(out.with_suffix(".json").read_text())
    assert report["transitions"] == 12
    assert report["reset_rounds"] == {"fit": [101, 102], "calibration": [103], "selection": [104]}
    assert report["spatial_cache_statistics"]["full_dataset_fits"]
