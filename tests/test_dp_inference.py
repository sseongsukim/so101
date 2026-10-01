"""train_dp.py smoke runs and DPRunner end to end (CPU)."""

import json

import numpy as np
import pytest
import torch

from so101.learning.act.data import CAMERA_SOURCES, to_uint8_rgb
from so101.learning.dp import ACTION, OBS_STATE
from so101.learning.dp.inference import DPRunner
from so101.learning.dp.policy import load_checkpoint
from test_act_data import H, W, _write_trajectory
from test_dp_data import _limit_threads  # noqa: F401  (autouse fixture)

TINY = {
    "image_shape": [H, W], "crop_size": 16, "front_crop_margin": 2,
    "pred_horizon": 8, "action_horizon": 4, "inference_steps": 4,
    "down_dims": [16, 32], "diffusion_step_embed_dim": 16, "projection_dim": 8,
    "warmup_steps": 1, "encoder_warmup_steps": 1,
}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from train_dp import train

    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    root = tmp_path_factory.mktemp("dp")
    sim, real = root / "sim", root / "real"
    sim.mkdir()
    real.mkdir()
    for i in range(3):
        _write_trajectory(sim, i, 12)
    _write_trajectory(real, 10, 10)
    ckpt = train(
        [f"sim={sim}", f"real={real}:3"], root / "out", steps=3, batch_size=4, num_workers=0, device="cpu",
        log_freq=1, save_freq=2, pretrained_backbone=False, cache_root=root / "cache", val_fraction=0.34,
        val_freq=3, overrides=dict(TINY), ema=True,
    )
    torch.set_num_threads(previous)
    return {"root": root, "ckpt": ckpt, "sim": sim, "real": real}


def test_train_writes_checkpoints_stats_and_info(run):
    out = run["ckpt"].parent
    assert run["ckpt"].name == "dp_so101.pt"
    for name in ("step_0000002.pt", "resume.pt", "stats.json", "train_info.json", "train_log.jsonl"):
        assert (out / name).is_file(), name
    info = json.loads((out / "train_info.json").read_text())
    assert info["camera_keys"] == ["observation.images.wrist", "observation.images.front"]
    assert info["image_shape"]["observation.images.front"] == [H, W, 3]
    names = [s["name"] for s in info["data_sources"]]
    assert names == ["sim", "real"] and info["data_sources"][1]["weight"] == 3.0
    # 3 sim episodes, 0.34 held out -> 1 val; the lone real episode stays in train.
    assert [s["episodes"] for s in info["data_sources"]] == [2, 1]
    assert info["num_frames"] == 2 * 12 + 10
    assert info["config"]["pred_horizon"] == 8 and info["config"]["ema_use"]
    log = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [1, 2, 3] and "val_loss" in log[-1]
    assert all(np.isfinite(r["loss"]) for r in log)
    ckpt = load_checkpoint(run["ckpt"])
    assert ckpt["step"] == 3 and "ema_model" in ckpt and "optimizers" not in ckpt
    assert ckpt["stats"] == json.loads((out / "stats.json").read_text())
    assert set(load_checkpoint(out / "resume.pt")["optimizers"]) == {"actor", "encoder"}


def test_init_from_resume_continues_step_and_reset_step_restarts(run):
    from train_dp import train

    out = run["ckpt"].parent
    resumed = train([f"sim={run['sim']}"], run["root"] / "resumed", steps=4, batch_size=4, num_workers=0,
                    device="cpu", save_freq=0, cache_root=run["root"] / "cache", init_from=out / "resume.pt")
    log = [json.loads(line) for line in (resumed.parent / "train_log.jsonl").read_text().splitlines()]
    assert load_checkpoint(resumed)["step"] == 4 and [r["step"] for r in log] == [4]
    # Normalization is inherited from the checkpoint, not refit on the new data.
    assert json.loads((resumed.parent / "stats.json").read_text()) == load_checkpoint(out / "resume.pt")["stats"]

    fresh = train([f"sim={run['sim']}"], run["root"] / "ft", steps=2, batch_size=4, num_workers=0, device="cpu",
                  save_freq=0, cache_root=run["root"] / "cache", init_from=run["ckpt"], reset_step=True, log_freq=1)
    assert load_checkpoint(fresh)["step"] == 2


def _obs(batch=2, source=(48, 64)):
    return torch.randn(batch, 6), {key: torch.rand(batch, *source, 3) for key in CAMERA_SOURCES}


def test_runner_matches_manual_pipeline(run):
    runner = DPRunner(run["ckpt"], device="cpu")
    assert runner.image_sizes == {k: (H, W) for k in runner.image_keys}
    state, images = _obs()
    manual = {
        key: torch.from_numpy(np.stack([to_uint8_rgb(f, (H, W)) for f in images[key]])).permute(0, 3, 1, 2)
        for key in CAMERA_SOURCES
    }
    torch.manual_seed(0)
    runner.policy.reset()
    expected = runner.policy.select_action(state, manual)
    torch.manual_seed(0)
    runner.reset()
    got = runner.act(state, images)
    assert got.shape == (2, 6)
    # Actions are ~1e4 in these synthetic pickles; memory layout alone moves float32 conv results slightly.
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-3)
    # Within the action range seen in training (clip_sample in normalized space).
    lo, hi = runner.policy.normalizer.action_min, runner.policy.normalizer.action_max
    assert torch.all(got >= lo - 1e-4) and torch.all(got <= hi + 1e-4)


def test_runner_requeries_every_action_horizon_and_accepts_uint8(run):
    runner = DPRunner(run["ckpt"], device="cpu", action_horizon=2, inference_steps=2, warmstart_timestep=None)
    assert runner.config.inference_steps == 2 and runner.config.warmstart_timestep is None
    calls = []
    original = runner.policy.predict_action_chunk
    runner.policy.predict_action_chunk = lambda s, im: calls.append(1) or original(s, im)
    state, images = _obs(batch=1)
    images = {k: (v * 255).to(torch.uint8) for k, v in images.items()}
    for _ in range(5):
        assert runner.act(state, images).shape == (1, 6)
    assert len(calls) == 3
    runner.reset()
    runner.act(state, images)
    assert len(calls) == 4


def test_runner_uses_ema_weights_by_default(run):
    ema = DPRunner(run["ckpt"], device="cpu").policy.state_dict()
    raw = DPRunner(run["ckpt"], device="cpu", use_ema=False).policy.state_dict()
    ckpt = load_checkpoint(run["ckpt"])
    name = "model.final_conv.1.weight"
    torch.testing.assert_close(ema[name], ckpt["ema_model"][name])
    torch.testing.assert_close(raw[name], ckpt["model"][name])


def test_train_script_cli_parses(run, monkeypatch):
    import train_dp

    captured = {}
    monkeypatch.setattr(train_dp, "train", lambda *a, **kw: captured.update(args=a, kw=kw))
    monkeypatch.setattr("sys.argv", ["train_dp.py", "--data", "sim=/x", "--data", "real=/y:2", "--out", "o",
                                     "--steps", "5", "--set", "down_dims=[16,32]", "--set", "encoder_lr=3e-5"])
    train_dp.main()
    assert captured["args"] == (["sim=/x", "real=/y:2"], "o")
    assert captured["kw"]["overrides"] == {"down_dims": [16, 32], "encoder_lr": 3e-5}
    assert captured["kw"]["ema"] is None and captured["kw"]["steps"] == 5
    assert ACTION and OBS_STATE
