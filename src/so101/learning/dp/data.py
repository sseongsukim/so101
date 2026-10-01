"""Trajectory pickles -> memory-mapped cache -> DP training samples.

Datasets here are tens of GB (hundreds of episodes x ~150 frames x 2 cameras
x 240x320x3), so the pickles are never all held in RAM:

  1. `build_cache(source_dir)` converts one directory of `trajectory_*.pkl`
     (the `scripts/teleop_task.py --visual` / teacher-demo format read by
     `so101.learning.act.data`) into
         <cache>/manifest.json         file list (name, size, mtime) + shapes
         <cache>/state.npy             (N, 6)  float32   observations
         <cache>/action.npy            (N, 6)  float32   actions
         <cache>/episode_ends.npy      (E,)    int64     cumulative frame counts
         <cache>/front_images.npy      (N, H, W, 3) uint8
         <cache>/wrist_images.npy      (N, H, W, 3) uint8
     one pickle at a time: image frames are appended to the .npy files as they
     are read and the header is patched with the final length at the end. The
     cache is built in a temp directory and renamed into place, and is rebuilt
     whenever the directory's pickle list (names, sizes, mtimes) changes.
  2. `DPDataset` reads samples from those arrays with np.load(mmap_mode="r"),
     opened lazily per process so it is safe with DataLoader workers.

Sample semantics mirror robust-rearrangement `src/dataset/dataset.py`
(`create_sample_indices` + `sample_sequence` + `ImageDataset.__getitem__`):
sequence_length = obs_horizon + pred_horizon - 1, pad_before = obs_horizon - 1,
pad_after = action_horizon - 1 (data.yaml pad_after=true). Out-of-episode
steps repeat the first/last frame, padded action steps are *not* masked (the
paper's loss includes them), and a sample start t only goes up to
T - sequence_length + pad_after -- the last pred_horizon - action_horizon
frames of an episode are never a sample's current observation, and episodes
shorter than sequence_length - pad_after produce no samples at all.

Co-training (`--data sim=DIR --data real=DIR:WEIGHT`): the paper concatenates
all zarr datasets into one and shuffles uniformly over samples
(combine_zarr_datasets + random_split + shuffle=True), so each domain is
represented in proportion to its frame count. Here every source's samples get
a per-sample weight (default 1.0 = the paper's concatenation); a weight of w
makes that source's samples w times as likely as a weight-1 sample
(WeightedRandomSampler, with replacement).

Normalizer stats: min/max per dimension over all frames of the used episodes
of all sources combined, constant columns widened to min-1 / max+1
(LinearNormalizer.fit).

Deliberate deviations: normalization runs in the policy instead of in the
dataset; there is no sample-level random train/test split (optional
episode-level holdout instead, see `split_episodes`); `domain` is the source
index in `--data` order rather than sim=0 / real=1.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, RandomSampler, Sampler, WeightedRandomSampler

from so101.learning.act.data import CAMERA_SOURCES, TRAJECTORY_GLOB, _load_episode
from so101.learning.dp.config import ACTION, OBS_STATE, DPConfig

CACHE_VERSION = 1
DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[4] / "outputs" / "dp_cache"
MANIFEST = "manifest.json"
_HEADER_BYTES = 128  # fixed .npy v1.0 header size so it can be rewritten in place


# --------------------------------------------------------------------------- #
# Cache building
# --------------------------------------------------------------------------- #


class _GrowingNpy:
    """Append rows to a .npy file whose final length is unknown up front."""

    def __init__(self, path: Path, dtype: np.dtype, row_shape: tuple[int, ...]):
        self.path, self.dtype, self.row_shape = path, np.dtype(dtype), tuple(row_shape)
        self.rows = 0
        self._file = path.open("wb")
        self._write_header()

    def _write_header(self) -> None:
        shape = (self.rows, *self.row_shape)
        header = f"{{'descr': {self.dtype.str!r}, 'fortran_order': False, 'shape': {shape!r}, }}"
        pad = _HEADER_BYTES - 10 - len(header) - 1
        if pad < 0:
            raise ValueError(f"shape {shape} does not fit the fixed .npy header")
        body = (header + " " * pad + "\n").encode("latin1")
        self._file.seek(0)
        self._file.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(body)) + body)

    def append(self, rows: np.ndarray) -> None:
        rows = np.ascontiguousarray(rows, dtype=self.dtype)
        if rows.shape[1:] != self.row_shape:
            raise ValueError(f"{self.path.name}: row shape {rows.shape[1:]} != {self.row_shape}")
        self._file.seek(0, os.SEEK_END)
        self._file.write(rows.tobytes())
        self.rows += len(rows)

    def close(self) -> None:
        self._write_header()
        self._file.close()


def _source_listing(source_dir: Path) -> list[list]:
    return [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(source_dir.glob(TRAJECTORY_GLOB))]


def default_cache_dir(source_dir: str | Path, cache_root: str | Path | None = None) -> Path:
    source_dir = Path(source_dir).resolve()
    digest = hashlib.sha1(str(source_dir).encode()).hexdigest()[:10]
    return Path(cache_root or DEFAULT_CACHE_ROOT) / f"{source_dir.name}-{digest}"


def cache_is_current(cache_dir: Path, source_dir: Path) -> bool:
    manifest = cache_dir / MANIFEST
    if not manifest.is_file():
        return False
    info = json.loads(manifest.read_text())
    return info.get("version") == CACHE_VERSION and info.get("files") == _source_listing(source_dir)


def build_cache(
    source_dir: str | Path,
    cache_dir: str | Path | None = None,
    *,
    cache_root: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Convert `source_dir/trajectory_*.pkl` into the memmap cache; returns the cache dir.

    A no-op when the existing cache's manifest matches the current file list.
    """
    source_dir = Path(source_dir).resolve()
    cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir(source_dir, cache_root)
    listing = _source_listing(source_dir)
    if not listing:
        raise FileNotFoundError(f"no {TRAJECTORY_GLOB} under {source_dir}")
    if not force and cache_is_current(cache_dir, source_dir):
        return cache_dir
    if cache_dir.exists():
        print(f"cache {cache_dir} is stale (file list changed); rebuilding")

    tmp = cache_dir.with_name(f"{cache_dir.name}.building-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    states, actions, lengths, names = [], [], [], []
    writers: dict[str, _GrowingNpy] = {}
    image_shape: dict[str, list[int]] = {}
    try:
        for index, (name, _, _) in enumerate(listing):
            episode = _load_episode(source_dir / name)
            if not writers:
                for key, src in CAMERA_SOURCES.items():
                    shape = episode.images[key].shape[1:]
                    image_shape[key] = list(shape)
                    writers[key] = _GrowingNpy(tmp / f"{src}.npy", np.uint8, shape)
                ref_state, ref_action = episode.state.shape[1:], episode.action.shape[1:]
            elif episode.state.shape[1:] != ref_state or episode.action.shape[1:] != ref_action:
                raise ValueError(f"{name}: state/action shape differs from {names[0]}")
            for key, writer in writers.items():
                writer.append(episode.images[key])
            states.append(episode.state)
            actions.append(episode.action)
            lengths.append(len(episode))
            names.append(name)
            print(f"\rcache {source_dir.name}: {index + 1}/{len(listing)} episodes, {sum(lengths)} frames", end="", flush=True)
            del episode
        print()
    finally:
        for writer in writers.values():
            writer.close()
    np.save(tmp / "state.npy", np.concatenate(states).astype(np.float32))
    np.save(tmp / "action.npy", np.concatenate(actions).astype(np.float32))
    np.save(tmp / "episode_ends.npy", np.cumsum(lengths).astype(np.int64))
    manifest = {
        "version": CACHE_VERSION,
        "source": str(source_dir),
        "files": listing,
        "episodes": names,
        "lengths": lengths,
        "num_frames": int(sum(lengths)),
        "state_dim": int(states[0].shape[1]),
        "action_dim": int(actions[0].shape[1]),
        "image_shape": image_shape,
        "camera_files": {key: f"{src}.npy" for key, src in CAMERA_SOURCES.items()},
    }
    # The manifest is written last: a cache without one is never "current".
    (tmp / MANIFEST).write_text(json.dumps(manifest, indent=1))
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    tmp.rename(cache_dir)
    return cache_dir


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


@dataclass
class DataSource:
    name: str
    source_dir: Path
    weight: float = 1.0
    cache_dir: Path | None = None

    @property
    def manifest(self) -> dict:
        return json.loads((self.cache_dir / MANIFEST).read_text())


def parse_data_arg(spec: str) -> DataSource:
    """`NAME=DIR[:WEIGHT]` or `DIR[:WEIGHT]` (name defaults to the directory name)."""
    name, sep, rest = spec.partition("=")
    if not sep:
        name, rest = "", spec
    weight = 1.0
    head, colon, tail = rest.rpartition(":")
    if colon:
        try:
            weight = float(tail)
            rest = head
        except ValueError:
            pass
    if weight <= 0:
        raise ValueError(f"{spec}: weight must be > 0")
    path = Path(rest)
    return DataSource(name=name or path.resolve().name, source_dir=path, weight=weight)


def prepare_sources(sources: list[DataSource], cache_root: str | Path | None = None, force: bool = False) -> list[DataSource]:
    names = [s.name for s in sources]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate --data names {names}")
    for source in sources:
        source.cache_dir = build_cache(source.source_dir, cache_root=cache_root, force=force)
    shapes = {s.name: (s.manifest["image_shape"], s.manifest["state_dim"], s.manifest["action_dim"]) for s in sources}
    if len({json.dumps(v) for v in shapes.values()}) > 1:
        raise ValueError(f"sources disagree on image/state/action shapes: {shapes}")
    return sources


def split_episodes(sources: list[DataSource], val_fraction: float, seed: int = 0) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per source, episode indices for train and for an episode-level holdout."""
    rng = np.random.default_rng(seed)
    train, val = [], []
    for source in sources:
        n = len(source.manifest["lengths"])
        order = rng.permutation(n)
        n_val = int(round(n * val_fraction)) if val_fraction > 0 else 0
        if val_fraction > 0 and n >= 2:
            n_val = min(max(n_val, 1), n - 1)
        val.append(np.sort(order[:n_val]))
        train.append(np.sort(order[n_val:]))
    return train, val


def compute_min_max_stats(sources: list[DataSource], episodes: list[np.ndarray] | None = None) -> dict:
    """LinearNormalizer.fit over every frame of the selected episodes of all sources."""
    values = {OBS_STATE: [], ACTION: []}
    for i, source in enumerate(sources):
        ends = np.asarray(source.manifest["lengths"]).cumsum()
        starts = ends - np.asarray(source.manifest["lengths"])
        chosen = range(len(ends)) if episodes is None else episodes[i]
        state = np.load(source.cache_dir / "state.npy")
        action = np.load(source.cache_dir / "action.npy")
        for e in chosen:
            values[OBS_STATE].append(state[starts[e] : ends[e]])
            values[ACTION].append(action[starts[e] : ends[e]])
    stats = {}
    for key, chunks in values.items():
        data = np.concatenate(chunks).astype(np.float64)
        lo, hi = data.min(0), data.max(0)
        constant = hi - lo == 0
        lo[constant] -= 1
        hi[constant] += 1
        stats[key] = {"min": lo.tolist(), "max": hi.tolist()}
    return stats


def save_stats(path: str | Path, stats: dict) -> None:
    Path(path).write_text(json.dumps(stats, indent=2))


def load_stats(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def create_sample_indices(episode_ends: np.ndarray, sequence_length: int, pad_before: int = 0, pad_after: int = 0) -> np.ndarray:
    """(n, 3) [episode_start, episode_end, sequence_start] per sample, with
    sequence_start possibly < episode_start (pad_before). Same sample set as
    the paper's create_sample_indices."""
    rows = []
    start = 0
    for end in episode_ends:
        length = end - start
        for idx in range(-pad_before, length - sequence_length + pad_after + 1):
            rows.append((start, end, start + idx))
        start = end
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


class DPDataset(Dataset):
    """Samples from one or more memmap caches.

    Item: OBS_STATE (obs_horizon, state_dim) float32 raw, ACTION (pred_horizon,
    action_dim) float32 raw, each camera key (obs_horizon, 3, H, W) uint8,
    "domain" () int64 source index, "frame" () int64 current frame in episode.
    """

    def __init__(
        self,
        sources: list[DataSource],
        config: DPConfig,
        *,
        episodes: list[np.ndarray] | None = None,
        min_length: int = 1,
    ):
        self.sources = sources
        self.config = config
        self.image_keys = tuple(config.image_keys)
        self.cache_dirs = [s.cache_dir for s in sources]
        self.camera_files = [s.manifest["camera_files"] for s in sources]
        pad_after = config.action_horizon - 1 if config.pad_after else 0
        index_rows, source_rows = [], []
        self.num_frames, self.num_episodes, self.num_samples = [], [], []
        dropped = 0
        for i, source in enumerate(sources):
            lengths = np.asarray(source.manifest["lengths"])
            ends = lengths.cumsum()
            chosen = np.arange(len(lengths)) if episodes is None else np.asarray(episodes[i], dtype=np.int64)
            chosen = chosen[lengths[chosen] >= min_length]
            rows = []
            for e in chosen:
                # Per episode, then shifted to the source's global frame offset.
                r = create_sample_indices(lengths[e : e + 1], config.sequence_length, config.obs_horizon - 1, pad_after)
                dropped += len(r) == 0
                rows.append(r + (ends[e] - lengths[e]))
            rows = np.concatenate(rows) if rows else np.zeros((0, 3), np.int64)
            index_rows.append(rows)
            source_rows.append(np.full(len(rows), i, dtype=np.int64))
            self.num_frames.append(int(lengths[chosen].sum()))
            self.num_episodes.append(len(chosen))
            self.num_samples.append(len(rows))
        if dropped:
            print(
                f"warning: {dropped} episode(s) shorter than {config.sequence_length - pad_after} steps "
                "contribute no samples (paper indexing: obs_horizon + pred_horizon - 1 - pad_after)"
            )
        self.indices = np.concatenate(index_rows)
        self.source_of = np.concatenate(source_rows)
        if len(self.indices) == 0:
            raise ValueError("no training samples (episodes too short for the horizons?)")
        self._arrays: list[dict] | None = None
        self._pid: int | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def sample_weights(self) -> np.ndarray:
        weights = np.asarray([s.weight for s in self.sources], dtype=np.float64)
        return weights[self.source_of]

    def effective_fractions(self) -> dict[str, float]:
        w = np.asarray([s.weight * n for s, n in zip(self.sources, self.num_samples)], dtype=np.float64)
        w = w / w.sum()
        return {s.name: float(f) for s, f in zip(self.sources, w)}

    def _open(self) -> list[dict]:
        # Lazily per process: DataLoader workers each get their own mmaps.
        if self._arrays is None or self._pid != os.getpid():
            self._arrays = []
            for cache_dir, camera_files in zip(self.cache_dirs, self.camera_files):
                arrays = {
                    OBS_STATE: np.load(cache_dir / "state.npy", mmap_mode="r"),
                    ACTION: np.load(cache_dir / "action.npy", mmap_mode="r"),
                }
                for key in self.image_keys:
                    arrays[key] = np.load(cache_dir / camera_files[key], mmap_mode="r")
                self._arrays.append(arrays)
            self._pid = os.getpid()
        return self._arrays

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        cfg = self.config
        ep_start, ep_end, seq_start = (int(v) for v in self.indices[index])
        arrays = self._open()[self.source_of[index]]
        positions = np.clip(seq_start + np.arange(cfg.sequence_length), ep_start, ep_end - 1)
        obs_pos = positions[: cfg.obs_horizon]
        act_pos = positions[cfg.first_action_idx : cfg.first_action_idx + cfg.pred_horizon]
        sample = {
            OBS_STATE: torch.from_numpy(np.asarray(arrays[OBS_STATE][obs_pos], dtype=np.float32)),
            ACTION: torch.from_numpy(np.asarray(arrays[ACTION][act_pos], dtype=np.float32)),
            "domain": torch.tensor(int(self.source_of[index])),
            "frame": torch.tensor(int(obs_pos[-1] - ep_start)),
        }
        for key in self.image_keys:
            sample[key] = torch.from_numpy(np.ascontiguousarray(arrays[key][obs_pos])).permute(0, 3, 1, 2)
        return sample


def make_sampler(dataset: DPDataset, generator: torch.Generator | None = None) -> Sampler:
    """Uniform shuffling when every source has weight 1 (the paper's
    concatenation); otherwise per-sample weighted sampling with replacement."""
    weights = dataset.sample_weights()
    if np.allclose(weights, 1.0):
        return RandomSampler(dataset, generator=generator)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(dataset), replacement=True, generator=generator)

