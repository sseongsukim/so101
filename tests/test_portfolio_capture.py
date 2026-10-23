"""Video exports must preserve recorded delays rather than accelerating a rollout."""

import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def capture_module():
    path = Path(__file__).resolve().parents[1] / "scripts/portfolio_capture.py"
    spec = importlib.util.spec_from_file_location("portfolio_capture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_preserves_long_real_frame_gap(capture_module, tmp_path):
    frames = np.zeros((3, 48, 64, 3), dtype=np.uint8)
    frames[0, :, :, 0] = 240  # RGB red, checks VideoWriter's BGR conversion.
    frames[1, :, :, 1] = 240
    frames[2, :, :, 2] = 240
    episode = tmp_path / "trajectory_000000.pkl"
    with episode.open("wb") as handle:
        pickle.dump({"front_images": frames, "wrist_images": frames,
                     "wall_times": np.array([100., 100.1, 100.5]),
                     "successes": np.array([False, False, True]), "checkpoint": "example.pt"}, handle)
    out = tmp_path / "video"
    capture_module.export_rollout(argparse.Namespace(episode=episode, out=out, fps=10))
    video = cv2.VideoCapture(str(out / "trajectory_000000_front.mp4"))
    try:
        assert video.isOpened()
        assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
        ok, first = video.read()
        assert ok and first[:, :, 2].mean() > 200 and first[:, :, 0].mean() < 20
    finally:
        video.release()
    metadata = json.loads((out / "trajectory_000000_metadata.json").read_text())
    assert metadata["success"] is True
    assert metadata["checkpoint"] == "example.pt"
    assert "wall_times" in metadata["timing"]


def test_export_rejects_nonmonotonic_timing(capture_module, tmp_path):
    episode = tmp_path / "episode.pkl"
    with episode.open("wb") as handle:
        frames = np.zeros((2, 48, 64, 3), dtype=np.uint8)
        pickle.dump({"front_images": frames, "wrist_images": frames, "wall_times": [5., 4.]}, handle)
    with pytest.raises(ValueError, match="strictly increasing"):
        capture_module.export_rollout(argparse.Namespace(episode=episode, out=tmp_path / "out", fps=30))
