"""checkpoint.py: on-disk format + MEAN_STD normalization helpers.

Runs on CPU with tiny inputs and no pretrained weights (offline-safe).
"""

import torch

from so101.learning.act.checkpoint import (
    build_normalizer,
    load_checkpoint,
    load_stats,
    normalize_batch,
    save_checkpoint,
    save_stats,
)
from so101.learning.act.config import ACTION, OBS_STATE, ACTConfig
from so101.learning.act.policy import ACTPolicy

STATE, ACT_DIM = 6, 6
CAMS = ("observation.images.wrist", "observation.images.top")


def _cfg(**kw):
    base = dict(
        action_dim=ACT_DIM,
        robot_state_dim=STATE,
        image_keys=CAMS,
        chunk_size=4,
        n_action_steps=4,
        pretrained_backbone_weights=None,
    )
    base.update(kw)
    return ACTConfig(**base)


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    cfg = _cfg()
    policy = ACTPolicy(cfg)
    ckpt = tmp_path / "ckpt.pt"

    save_checkpoint(ckpt, policy, cfg.__dict__, step=123)
    saved = load_checkpoint(ckpt)

    assert set(saved.keys()) == {"model", "config", "step"}
    assert saved["step"] == 123
    assert saved["config"]["chunk_size"] == 4
    assert set(saved["model"].keys()) == set(policy.state_dict().keys())


def test_save_and_load_stats_roundtrip(tmp_path):
    stats = {OBS_STATE: {"mean": [0.0] * STATE, "std": [1.0] * STATE}}
    path = tmp_path / "stats.json"

    save_stats(path, stats)
    assert load_stats(path) == stats


def test_build_normalizer_and_normalize_batch_apply_mean_std():
    stats = {
        OBS_STATE: {"mean": [1.0] * STATE, "std": [2.0] * STATE},
        ACTION: {"mean": [0.0] * ACT_DIM, "std": [1.0] * ACT_DIM},
        CAMS[0]: {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    }
    norm = build_normalizer(stats, (OBS_STATE, ACTION), (CAMS[0],), device=torch.device("cpu"))

    batch = {
        OBS_STATE: torch.full((2, STATE), 3.0),
        CAMS[0]: torch.full((2, 3, 4, 4), 1.0),
    }
    out = normalize_batch(batch, norm)

    torch.testing.assert_close(out[OBS_STATE], torch.full((2, STATE), 1.0))  # (3-1)/2
    torch.testing.assert_close(out[CAMS[0]], torch.full((2, 3, 4, 4), 1.0))  # (1-0.5)/0.5


def test_normalize_batch_skips_normalizer_keys_absent_from_batch():
    norm = build_normalizer(
        {OBS_STATE: {"mean": [0.0] * STATE, "std": [1.0] * STATE},
         ACTION: {"mean": [0.0] * ACT_DIM, "std": [1.0] * ACT_DIM}},
        (OBS_STATE, ACTION), (), device=torch.device("cpu"),
    )
    batch = {OBS_STATE: torch.zeros(1, STATE)}  # no ACTION key, as at inference time
    out = normalize_batch(batch, norm)
    assert set(out.keys()) == {OBS_STATE}


def test_act_policy_from_checkpoint_reproduces_the_saved_policy(tmp_path):
    cfg = _cfg()
    policy = ACTPolicy(cfg).eval()
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, policy, cfg.__dict__, step=1)

    reloaded = ACTPolicy.from_checkpoint(ckpt)

    assert reloaded.config == cfg
    batch = {OBS_STATE: torch.randn(1, STATE)}
    for key in cfg.image_keys:
        batch[key] = torch.rand(1, 3, 64, 64)

    expected = policy.select_action(batch)
    actual = reloaded.select_action(batch)
    torch.testing.assert_close(actual, expected)
