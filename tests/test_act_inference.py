"""ACTRunner: checkpoint + stats + train_info -> sim-radian joint targets."""

import pytest
import torch

from so101.learning.act import ACTION, OBS_STATE
from so101.learning.act.data import CAMERA_SOURCES, image_to_tensor, to_uint8_rgb
from so101.learning.act.inference import ACTRunner
from test_act_data import H, W, _write_trajectory


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    from train_act import train

    root = tmp_path_factory.mktemp("run")
    data = root / "data"
    data.mkdir()
    _write_trajectory(data, 0, 8)
    return train(data, root / "out", steps=1, batch_size=4, chunk_size=4, device="cpu",
                 pretrained_backbone=False)


def _obs(batch=2, source=(48, 64)):
    state = torch.randn(batch, 6)
    images = {key: torch.rand(batch, *source, 3) for key in CAMERA_SOURCES}
    return state, images


def test_act_matches_manual_preprocess_and_unnormalize(checkpoint):
    runner = ACTRunner(checkpoint, device="cpu")
    state, images = _obs()

    batch = {OBS_STATE: state}
    for key in CAMERA_SOURCES:
        # Full-resolution frames are resized to the recorded size, exactly as teleop stored them.
        batch[key] = torch.stack([image_to_tensor(to_uint8_rgb(f, (H, W))) for f in images[key]])
    for key in batch:
        mean, std = runner.norm[key]
        batch[key] = (batch[key] - mean) / std
    runner.policy.reset()
    expected = runner.policy.select_action(batch) * runner.norm[ACTION][1] + runner.norm[ACTION][0]

    runner.reset()
    torch.testing.assert_close(runner.act(state, images), expected)


def test_action_queue_requeries_after_n_action_steps(checkpoint):
    runner = ACTRunner(checkpoint, device="cpu", n_action_steps=2)
    state, images = _obs(batch=1)
    calls = []
    original = runner.policy.predict_action_chunk
    runner.policy.predict_action_chunk = lambda b: calls.append(1) or original(b)
    for _ in range(5):
        assert runner.act(state, images).shape == (1, 6)
    assert len(calls) == 3


def test_temporal_ensemble_forces_single_step_queries(checkpoint):
    runner = ACTRunner(checkpoint, device="cpu", temporal_ensemble_coeff=0.01)
    assert runner.policy.config.n_action_steps == 1
    state, images = _obs(batch=1)
    for _ in range(3):
        assert runner.act(state, images).shape == (1, 6)


def test_n_action_steps_above_chunk_size_is_rejected(checkpoint):
    with pytest.raises(ValueError):
        ACTRunner(checkpoint, device="cpu", n_action_steps=5)
