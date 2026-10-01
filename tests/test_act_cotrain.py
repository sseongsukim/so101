"""ACT on memmap caches: samples match the in-RAM dataset; the co-training script runs."""

import subprocess
import sys

import numpy as np
import torch

from so101.learning.act import ACTION, OBS_STATE, ACTPolicy
from so101.learning.act.cache_data import ACTCacheDataset, PhotometricAugment, compute_stats
from so101.learning.act.data import CAMERA_SOURCES, TeleopACTDataset, load_episodes
from so101.learning.dp.data import DataSource, prepare_sources
from test_act_data import H, W, _write_trajectory


def _sources(tmp_path):
    sim, real = tmp_path / "sim", tmp_path / "real"
    sim.mkdir(); real.mkdir()
    for i in range(3):
        _write_trajectory(sim, i, 6 + i)
    _write_trajectory(real, 7, 9)
    return prepare_sources([DataSource("sim", sim), DataSource("real", real)], cache_root=tmp_path / "cache")


def test_cache_samples_match_in_memory_dataset(tmp_path):
    sources = _sources(tmp_path)
    cached = ACTCacheDataset(sources, chunk_size=4)
    ref = TeleopACTDataset(load_episodes(tmp_path / "sim") + load_episodes(tmp_path / "real"), chunk_size=4)
    assert len(cached) == len(ref) == 6 + 7 + 8 + 9
    for i in (0, 5, 6, 20, len(ref) - 1):
        a, b = cached[i], ref[i]
        torch.testing.assert_close(a[OBS_STATE], b[OBS_STATE])
        torch.testing.assert_close(a[ACTION], b[ACTION])
        assert torch.equal(a["action_is_pad"], b["action_is_pad"])
        for key in CAMERA_SOURCES:
            assert a[key].dtype == torch.uint8
            torch.testing.assert_close(a[key].float() / 255, b[key])


def test_stats_and_augment(tmp_path):
    dataset = ACTCacheDataset(_sources(tmp_path), chunk_size=4)
    stats = compute_stats(dataset, image_stride=1)
    assert set(stats) == {OBS_STATE, ACTION, *CAMERA_SOURCES}
    x = torch.randint(0, 256, (4, 3, H, W), dtype=torch.uint8)
    y = PhotometricAugment()(x)
    assert y.dtype == torch.uint8 and y.shape == x.shape


def test_train_act_cotrain_smoke(tmp_path):
    _sources(tmp_path)
    out = tmp_path / "out"
    cmd = [sys.executable, "scripts/train_act_cotrain.py", "--data", f"sim={tmp_path / 'sim'}",
           "--data", f"real={tmp_path / 'real'}:2", "--out", str(out), "--steps", "3", "--batch-size", "4",
           "--chunk-size", "4", "--num-workers", "0", "--device", "cpu", "--log-freq", "1", "--save-freq", "0",
           "--cache-root", str(tmp_path / "cache")]
    for extra in ([], ["--no-augment"]):
        subprocess.run(cmd + extra + ["--out", str(out / ("plain" if extra else "aug"))], check=True,
                       capture_output=True, env={"PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin"} | dict(__import__("os").environ))
    for name in ("aug", "plain"):
        policy = ACTPolicy.from_checkpoint(out / name / "act_so101.pt")
        assert policy.config.image_keys == tuple(CAMERA_SOURCES)
        info = __import__("json").loads((out / name / "train_info.json").read_text())
        assert info["image_shape"] == {key: [H, W, 3] for key in CAMERA_SOURCES}
        assert (out / name / "stats.json").is_file()
