"""Trajectory pickles -> memmap cache -> DP samples (CPU)."""

import json
import os
import time

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from so101.learning.act.data import CAMERA_SOURCES
from so101.learning.dp.config import ACTION, OBS_STATE, DPConfig
from so101.learning.dp.data import (
    DataSource,
    DPDataset,
    build_cache,
    cache_is_current,
    compute_min_max_stats,
    create_sample_indices,
    make_sampler,
    parse_data_arg,
    prepare_sources,
    split_episodes,
)
from so101.learning.dp.policy import LinearNormalizer
from test_act_data import H, W, _write_trajectory


@pytest.fixture(autouse=True)
def _limit_threads():
    # CPU tests share the machine with a rendering process; 8 intra-op
    # threads contending for it make the small convs ~10x slower.
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


def tiny_config(**kw):
    base = dict(
        image_shape=(H, W), crop_size=16, front_crop_margin=2,
        obs_horizon=1, pred_horizon=8, action_horizon=4,
        down_dims=(16, 32), diffusion_step_embed_dim=16, projection_dim=8,
        pretrained_backbone=False, batch_size=4, warmup_steps=2, encoder_warmup_steps=2,
    )
    base.update(kw)
    return DPConfig(**base)


def _sources(tmp_path, spec, **kw):
    """spec: {name: [episode lengths]} -> prepared DataSources with caches under tmp_path."""
    out = []
    for offset, (name, lengths) in enumerate(spec.items()):
        d = tmp_path / name
        d.mkdir(exist_ok=True)
        for i, n in enumerate(lengths):
            _write_trajectory(d, offset * 10 + i, n)
        out.append(DataSource(name=name, source_dir=d, **kw.get(name, {})))
    return prepare_sources(out, cache_root=tmp_path / "cache")


def test_cache_matches_pickles(tmp_path):
    datas = [_write_trajectory(tmp_path, i, n) for i, n in enumerate((5, 7))]
    cache = build_cache(tmp_path, cache_root=tmp_path / "cache")
    manifest = json.loads((cache / "manifest.json").read_text())
    assert manifest["lengths"] == [5, 7] and manifest["num_frames"] == 12
    np.testing.assert_array_equal(np.load(cache / "episode_ends.npy"), [5, 12])
    state = np.load(cache / "state.npy", mmap_mode="r")
    np.testing.assert_array_equal(state, np.concatenate([d["observations"] for d in datas]))
    for key, src in CAMERA_SOURCES.items():
        # Grown incrementally with a patched header: must still be a valid .npy.
        images = np.load(cache / f"{src}.npy", mmap_mode="r")
        assert images.shape == (12, H, W, 3) and images.dtype == np.uint8
        np.testing.assert_array_equal(images, np.concatenate([d[src] for d in datas]))
        assert manifest["image_shape"][key] == [H, W, 3]


def test_cache_reused_then_rebuilt_when_file_list_changes(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _write_trajectory(src, 0, 5)
    cache = build_cache(src, cache_root=tmp_path / "cache")
    stamp = (cache / "manifest.json").stat().st_mtime_ns
    time.sleep(0.01)
    assert build_cache(src, cache_root=tmp_path / "cache") == cache
    assert (cache / "manifest.json").stat().st_mtime_ns == stamp  # reused

    _write_trajectory(src, 1, 6)
    assert not cache_is_current(cache, src)
    build_cache(src, cache_root=tmp_path / "cache")
    assert json.loads((cache / "manifest.json").read_text())["lengths"] == [5, 6]

    (src / "trajectory_000000.pkl").unlink()
    build_cache(src, cache_root=tmp_path / "cache")
    assert json.loads((cache / "manifest.json").read_text())["lengths"] == [6]
    assert not [p for p in cache.parent.iterdir() if ".building-" in p.name]


def test_parse_data_arg():
    s = parse_data_arg("real=/a/b:c/real_demos:2.5")
    assert (s.name, str(s.source_dir), s.weight) == ("real", "/a/b:c/real_demos", 2.5)
    s = parse_data_arg("/x/sim_demos")
    assert (s.name, str(s.source_dir), s.weight) == ("sim_demos", "/x/sim_demos", 1.0)
    with pytest.raises(ValueError):
        parse_data_arg("a=/x:0")


def test_sample_indices_mirror_paper_padding():
    # sequence 4, pad_after 1: starts 0..(6 - 4 + 1) = 0..3 for a 6-step episode.
    rows = create_sample_indices(np.array([6, 9]), sequence_length=4, pad_before=0, pad_after=1)
    assert rows[:, 2].tolist() == [0, 1, 2, 3, 6]
    assert rows[:, :2].tolist() == [[0, 6]] * 4 + [[6, 9]]


def test_samples_align_and_pad_like_the_paper(tmp_path):
    cfg = tiny_config(pred_horizon=8, action_horizon=4)
    (src,) = _sources(tmp_path, {"sim": [12]})
    data = _write_trajectory(tmp_path / "sim", 0, 12)  # same content as the cached file
    ds = DPDataset([src], cfg)
    # seq_len 8, pad_after 3 -> t = 0 .. 12 - 8 + 3 = 7
    assert len(ds) == 8
    s = ds[2]
    assert s[OBS_STATE].shape == (1, 6) and s[ACTION].shape == (8, 6)
    torch.testing.assert_close(s[OBS_STATE][0], torch.from_numpy(data["observations"][2]))
    assert s[ACTION][:, 0].tolist() == list(range(2, 10))
    for key, pkl_key in CAMERA_SOURCES.items():
        assert s[key].shape == (1, 3, H, W) and s[key].dtype == torch.uint8
        torch.testing.assert_close(s[key][0], torch.from_numpy(data[pkl_key][2]).permute(2, 0, 1))
    last = ds[7]
    assert int(last["frame"]) == 7
    # Past the end the last action repeats (unmasked, as in the paper).
    assert last[ACTION][:, 0].tolist() == [7, 8, 9, 10, 11, 11, 11, 11]


def test_obs_horizon_pads_before_with_first_frame(tmp_path):
    cfg = tiny_config(obs_horizon=2, pred_horizon=8, action_horizon=4)
    (src,) = _sources(tmp_path, {"sim": [12]})
    ds = DPDataset([src], cfg)
    first = ds[0]
    assert int(first["frame"]) == 0
    torch.testing.assert_close(first[OBS_STATE][0], first[OBS_STATE][1])
    # first_action_idx = obs_horizon - 1: the chunk starts at the current step.
    assert first[ACTION][:, 0].tolist() == list(range(0, 8))


def test_short_episodes_give_no_samples_and_chunks_stay_in_episode(tmp_path, capsys):
    cfg = tiny_config(pred_horizon=8, action_horizon=4)
    (src,) = _sources(tmp_path, {"sim": [4, 6]})
    ds = DPDataset([src], cfg)
    # 4 < 8 - 3: none; 6 -> t = 0..1
    assert len(ds) == 2 and "contribute no samples" in capsys.readouterr().out
    assert ds[1][ACTION][:, 0].tolist() == [1001, 1002, 1003, 1004, 1005, 1005, 1005, 1005]


def test_multi_source_weights_and_min_length(tmp_path):
    cfg = tiny_config()
    sim, real = _sources(tmp_path, {"sim": [10, 10, 10], "real": [10, 2]}, real={"weight": 4.0})
    ds = DPDataset([sim, real], cfg, min_length=5)
    assert ds.num_episodes == [3, 1] and ds.num_frames == [30, 10]
    assert ds.num_samples == [18, 6]  # t = 0 .. 10 - 8 + 3 per episode
    fr = ds.effective_fractions()
    assert fr["real"] == pytest.approx(24 / (18 + 24))
    weights = ds.sample_weights()
    assert set(weights[:18]) == {1.0} and set(weights[18:]) == {4.0}
    assert ds[18]["domain"] == 1 and ds[18][ACTION][0, 0] == 10000
    assert isinstance(make_sampler(ds), torch.utils.data.WeightedRandomSampler)
    uniform = DPDataset([sim], cfg)
    assert isinstance(make_sampler(uniform), torch.utils.data.RandomSampler)


def test_episode_split_and_stats(tmp_path):
    sim, real = _sources(tmp_path, {"sim": [10] * 10, "real": [10] * 3})
    train, val = split_episodes([sim, real], 0.2, seed=0)
    assert [len(v) for v in val] == [2, 1] and [len(t) for t in train] == [8, 2]
    for t, v in zip(train, val):
        assert not set(t) & set(v)
    stats = compute_min_max_stats([sim, real], train)
    actions = []
    for source, eps in zip((sim, real), train):
        a = np.load(source.cache_dir / "action.npy")
        actions += [a[e * 10 : (e + 1) * 10] for e in eps]
    actions = np.concatenate(actions)
    np.testing.assert_allclose(stats[ACTION]["min"], actions.min(0))
    np.testing.assert_allclose(stats[ACTION]["max"], actions.max(0))


def test_normalizer_round_trip_and_constant_columns(tmp_path):
    (src,) = _sources(tmp_path, {"sim": [10]})
    stats = compute_min_max_stats([src])
    norm = LinearNormalizer(6, 6)
    norm.set_stats(stats)
    x = torch.from_numpy(np.load(src.cache_dir / "action.npy"))
    n = norm.normalize(x, ACTION)
    assert n.min() >= -1 - 1e-6 and n.max() <= 1 + 1e-6
    torch.testing.assert_close(norm.unnormalize(n, ACTION), x)
    for key in stats:
        for bound in ("min", "max"):
            np.testing.assert_allclose(norm.stats()[key][bound], stats[key][bound], rtol=1e-6)

    const = tmp_path / "const"
    const.mkdir()
    import pickle

    data = _write_trajectory(const, 0, 5)
    data["actions"][:] = 3.0
    with (const / "trajectory_000000.pkl").open("wb") as f:
        pickle.dump(data, f)
    (c,) = prepare_sources([DataSource("c", const)], cache_root=tmp_path / "cache")
    s = compute_min_max_stats([c])
    assert s[ACTION]["min"] == [2.0] * 6 and s[ACTION]["max"] == [4.0] * 6


def test_dataloader_workers_read_memmaps(tmp_path):
    cfg = tiny_config()
    sim, real = _sources(tmp_path, {"sim": [10, 12], "real": [11]})
    ds = DPDataset([sim, real], cfg)
    ds._open()  # an already-open parent must not leak mmaps into workers
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
    batches = list(loader)
    assert sum(len(b[ACTION]) for b in batches) == len(ds)
    b = batches[0]
    assert b[cfg.front_key].shape == (4, 1, 3, H, W) and b[cfg.front_key].dtype == torch.uint8
    torch.testing.assert_close(b[ACTION][1], ds[1][ACTION])
    assert os.getpid() == ds._pid
