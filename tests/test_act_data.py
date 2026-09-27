"""Teleop pickle -> ACT sample adapter, and a CPU smoke run of train_act.py."""

import json
import pickle

import numpy as np
import pytest
import torch

from so101.learning.act import ACTION, OBS_STATE, ACTPolicy, load_stats
from so101.learning.act.data import (
    CAMERA_SOURCES,
    TeleopACTDataset,
    compute_stats,
    image_to_tensor,
    load_episodes,
    to_uint8_rgb,
)

H, W = 24, 32


def _write_trajectory(directory, index, length, *, images=True, image_shape=(H, W, 3)):
    """Same keys/dtypes as scripts/teleop_task.py TrajectoryBuffer.save()."""
    rng = np.random.default_rng(index)
    data = {
        "observations": rng.normal(size=(length, 6)).astype(np.float32),
        # action[t] = index*1000 + t makes chunk contents checkable.
        "actions": (index * 1000 + np.arange(length, dtype=np.float32))[:, None].repeat(6, 1),
        "rewards": np.zeros(length, np.float32),
        "terminals": np.eye(1, length, length - 1, dtype=bool)[0],
        "successes": np.eye(1, length, length - 1, dtype=bool)[0],
        "next_observations": rng.normal(size=(length, 6)).astype(np.float32),
    }
    if images:
        data["front_images"] = rng.integers(0, 256, (length, *image_shape), dtype=np.uint8)
        data["wrist_images"] = rng.integers(0, 256, (length, *image_shape), dtype=np.uint8)
    path = directory / f"trajectory_{index:06d}.pkl"
    with path.open("wb") as f:
        pickle.dump(data, f)
    return data


def test_samples_align_state_images_and_action_chunk(tmp_path):
    data = _write_trajectory(tmp_path, 0, 5)
    ds = TeleopACTDataset(load_episodes(tmp_path), chunk_size=3)
    assert len(ds) == 5

    s = ds[1]
    torch.testing.assert_close(s[OBS_STATE], torch.from_numpy(data["observations"][1]))
    torch.testing.assert_close(s[ACTION][:, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert not s["action_is_pad"].any()
    for key, src in CAMERA_SOURCES.items():
        assert s[key].shape == (3, H, W) and s[key].dtype == torch.float32
        torch.testing.assert_close(s[key], torch.from_numpy(data[src][1]).permute(2, 0, 1).float() / 255)


def test_chunk_past_episode_end_repeats_last_action_and_is_padded(tmp_path):
    _write_trajectory(tmp_path, 0, 5)
    ds = TeleopACTDataset(load_episodes(tmp_path), chunk_size=4)
    s = ds[3]
    torch.testing.assert_close(s[ACTION][:, 0], torch.tensor([3.0, 4.0, 4.0, 4.0]))
    assert s["action_is_pad"].tolist() == [False, False, True, True]


def test_chunks_never_cross_into_the_next_episode(tmp_path):
    _write_trajectory(tmp_path, 0, 3)
    _write_trajectory(tmp_path, 1, 4)
    ds = TeleopACTDataset(load_episodes(tmp_path), chunk_size=5)
    assert len(ds) == 7
    last_of_first = ds[2]
    assert last_of_first[ACTION][:, 0].tolist() == [2.0] * 5
    assert last_of_first["action_is_pad"].tolist() == [False, True, True, True, True]
    first_of_second = ds[3]
    assert first_of_second[ACTION][:, 0].tolist() == [1000.0, 1001.0, 1002.0, 1003.0, 1003.0]


def test_min_length_skips_stray_saves(tmp_path, capsys):
    _write_trajectory(tmp_path, 0, 2)
    _write_trajectory(tmp_path, 1, 6)
    episodes = load_episodes(tmp_path, min_length=3)
    assert [ep.path.name for ep in episodes] == ["trajectory_000001.pkl"]
    assert "skip trajectory_000000.pkl" in capsys.readouterr().out


def test_state_only_trajectories_are_rejected(tmp_path):
    _write_trajectory(tmp_path, 0, 4, images=False)
    with pytest.raises(ValueError, match="--visual"):
        load_episodes(tmp_path)


def test_mismatched_image_sizes_are_rejected(tmp_path):
    _write_trajectory(tmp_path, 0, 4)
    _write_trajectory(tmp_path, 1, 4, image_shape=(H * 2, W * 2, 3))
    with pytest.raises(ValueError, match="differs"):
        load_episodes(tmp_path)


def test_empty_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_episodes(tmp_path)


def test_compute_stats_matches_numpy(tmp_path):
    d0 = _write_trajectory(tmp_path, 0, 4)
    d1 = _write_trajectory(tmp_path, 1, 3)
    stats = compute_stats(load_episodes(tmp_path))
    state = np.concatenate([d0["observations"], d1["observations"]]).astype(np.float64)
    np.testing.assert_allclose(stats[OBS_STATE]["mean"], state.mean(0), rtol=1e-6)
    np.testing.assert_allclose(stats[OBS_STATE]["std"], state.std(0), rtol=1e-6)
    for key, src in CAMERA_SOURCES.items():
        px = np.concatenate([d0[src], d1[src]]).reshape(-1, 3) / 255.0
        np.testing.assert_allclose(stats[key]["mean"], px.mean(0), rtol=1e-6)
        np.testing.assert_allclose(stats[key]["std"], px.std(0), rtol=1e-5)


def test_near_constant_dims_are_left_unscaled(tmp_path):
    _write_trajectory(tmp_path, 0, 4)  # every action column moves by 1 per step
    episodes = load_episodes(tmp_path)
    episodes[0].state[:, 2] = 0.5  # a joint that never moved
    stats = compute_stats(episodes)
    assert stats[OBS_STATE]["std"][2] == 1.0
    assert stats[OBS_STATE]["std"][0] != 1.0
    assert all(s > 1.0 for s in stats[ACTION]["std"])


def test_to_uint8_rgb_resizes_float_and_uint8_frames_alike():
    frame = torch.rand(48, 64, 3)
    from_float = to_uint8_rgb(frame, (24, 32))
    from_uint8 = to_uint8_rgb(frame.mul(255).round().to(torch.uint8), (24, 32))
    assert from_float.shape == (24, 32, 3) and from_float.dtype == np.uint8
    np.testing.assert_array_equal(from_float, from_uint8)
    assert image_to_tensor(from_float).shape == (3, 24, 32)


def test_train_act_smoke(tmp_path):
    from train_act import train

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_trajectory(data_dir, 0, 6)
    _write_trajectory(data_dir, 1, 5)
    out = tmp_path / "out"

    ckpt = train(
        data_dir, out, steps=2, batch_size=4, chunk_size=4, device="cpu",
        log_freq=1, pretrained_backbone=False,
    )

    policy = ACTPolicy.from_checkpoint(ckpt)
    assert policy.config.image_keys == tuple(CAMERA_SOURCES)
    assert set(load_stats(out / "stats.json")) == {OBS_STATE, ACTION, *CAMERA_SOURCES}
    info = json.loads((out / "train_info.json").read_text())
    assert info["image_shape"] == {key: [H, W, 3] for key in CAMERA_SOURCES}
    batch = {OBS_STATE: torch.zeros(1, 6), **{k: torch.rand(1, 3, H, W) for k in CAMERA_SOURCES}}
    assert policy.select_action(batch).shape == (1, 6)


def test_slow_recording_is_reported(tmp_path, capsys):
    import pickle as pk
    _write_trajectory(tmp_path, 0, 4)
    path = tmp_path / "trajectory_000000.pkl"
    data = pk.loads(path.read_bytes())
    data["wall_times"] = np.arange(4) * (1 / 15)  # 15 Hz loop
    path.write_bytes(pk.dumps(data))
    load_episodes(tmp_path)
    assert "2.00x faster" in capsys.readouterr().out


def test_init_from_reuses_source_stats(tmp_path):
    from train_act import train

    first = tmp_path / "first"
    first.mkdir()
    _write_trajectory(first, 0, 6)
    ckpt = train(first, tmp_path / "a", steps=1, batch_size=2, chunk_size=3, device="cpu",
                 pretrained_backbone=False)
    second = tmp_path / "second"
    second.mkdir()
    _write_trajectory(second, 7, 6)
    train(second, tmp_path / "b", steps=2, batch_size=2, chunk_size=3, device="cpu",
          pretrained_backbone=False, init_from=ckpt)
    assert load_stats(tmp_path / "b" / "stats.json") == load_stats(tmp_path / "a" / "stats.json")
