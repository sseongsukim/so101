"""ACT samples from the memory-mapped trajectory caches built for the DP student.

`so101.learning.act.data` holds every episode in RAM, which is fine for a few
dozen teleop demos but not for the synthetic teacher data (600 episodes, ~23
GB of frames). The DP trainer already converts pickle directories into
memmap caches (`so101.learning.dp.data.build_cache`); this reads the same
caches, so the two students train on byte-identical frames.

A sample is exactly what `TeleopACTDataset` returns -- every frame of every
episode, the action chunk `actions[t : t + chunk_size]` padded with the last
action and flagged in `action_is_pad` -- except that images come back as
uint8 CHW so the trainer can augment them on the GPU before scaling to
[0, 1]. Co-training follows the DP data (per-source sample weights; weight 1
everywhere = plain concatenation).
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset, RandomSampler, Sampler, WeightedRandomSampler

from so101.learning.act.config import ACTION, OBS_STATE
from so101.learning.act.data import CAMERA_SOURCES, MIN_STD
from so101.learning.dp.data import DataSource


class ACTCacheDataset(Dataset):
    def __init__(self, sources: list[DataSource], chunk_size: int, min_length: int = 1):
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.sources = sources
        self.chunk_size = chunk_size
        self.image_keys = tuple(CAMERA_SOURCES)
        self.camera_files = [s.manifest["camera_files"] for s in sources]
        rows, src = [], []
        self.num_frames, self.num_episodes = [], []
        for i, source in enumerate(sources):
            lengths = np.asarray(source.manifest["lengths"])
            ends = lengths.cumsum()
            starts = ends - lengths
            keep = np.flatnonzero(lengths >= min_length)
            for e in keep:
                t = np.arange(starts[e], ends[e])
                rows.append(np.stack([t, np.full_like(t, ends[e])], axis=1))
                src.append(np.full(len(t), i))
            self.num_frames.append(int(lengths[keep].sum()))
            self.num_episodes.append(len(keep))
        self.index = np.concatenate(rows).astype(np.int64)      # (frame, episode_end)
        self.source_of = np.concatenate(src).astype(np.int64)
        self._arrays: list[dict] | None = None
        self._pid: int | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _open(self) -> list[dict]:
        if self._arrays is None or self._pid != os.getpid():
            self._arrays = []
            for source, files in zip(self.sources, self.camera_files):
                arrays = {
                    OBS_STATE: np.load(source.cache_dir / "state.npy", mmap_mode="r"),
                    ACTION: np.load(source.cache_dir / "action.npy", mmap_mode="r"),
                }
                for key in self.image_keys:
                    arrays[key] = np.load(source.cache_dir / files[key], mmap_mode="r")
                self._arrays.append(arrays)
            self._pid = os.getpid()
        return self._arrays

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        frame, episode_end = (int(v) for v in self.index[index])
        arrays = self._open()[self.source_of[index]]
        chunk = frame + np.arange(self.chunk_size)
        is_pad = chunk >= episode_end
        chunk = np.minimum(chunk, episode_end - 1)
        sample = {
            OBS_STATE: torch.from_numpy(np.array(arrays[OBS_STATE][frame], dtype=np.float32)),
            ACTION: torch.from_numpy(np.array(arrays[ACTION][chunk], dtype=np.float32)),
            "action_is_pad": torch.from_numpy(is_pad),
            "domain": torch.tensor(int(self.source_of[index])),
        }
        for key in self.image_keys:
            sample[key] = torch.from_numpy(np.array(arrays[key][frame])).permute(2, 0, 1)  # uint8 CHW (copied out of the memmap)
        return sample

    def effective_fractions(self) -> dict[str, float]:
        counts = np.bincount(self.source_of, minlength=len(self.sources)).astype(np.float64)
        w = counts * np.asarray([s.weight for s in self.sources])
        w /= w.sum()
        return {s.name: float(f) for s, f in zip(self.sources, w)}


def make_sampler(dataset: ACTCacheDataset, generator: torch.Generator | None = None) -> Sampler:
    weights = np.asarray([s.weight for s in dataset.sources])[dataset.source_of]
    if np.allclose(weights, 1.0):
        return RandomSampler(dataset, generator=generator)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(dataset),
                                 replacement=True, generator=generator)


def compute_stats(dataset: ACTCacheDataset, image_stride: int = 10) -> dict:
    """MEAN_STD stats in `checkpoint.save_stats` schema over all sources.

    State/action: every frame, std below MIN_STD left unscaled (as
    `data.compute_stats`). Images: per channel in [0, 1] over every
    `image_stride`-th frame, which is plenty for three numbers per camera.
    """
    arrays = dataset._open()
    frames = dataset.index[:, 0]
    stats = {}
    for key in (OBS_STATE, ACTION):
        values = np.concatenate([
            np.asarray(a[key])[frames[dataset.source_of == i]] for i, a in enumerate(arrays)
        ]).astype(np.float64)
        std = values.std(0)
        flat = np.flatnonzero(std < MIN_STD)
        if flat.size:
            print(f"warning: {key} dims {flat.tolist()} have std < {MIN_STD}; left unscaled")
        stats[key] = {"mean": values.mean(0).tolist(), "std": np.where(std < MIN_STD, 1.0, std).tolist()}
    for key in dataset.image_keys:
        total, total_sq, count = np.zeros(3), np.zeros(3), 0
        for i, a in enumerate(arrays):
            picked = frames[dataset.source_of == i][::image_stride]
            for start in range(0, len(picked), 256):
                pixels = np.asarray(a[key][picked[start : start + 256]]).reshape(-1, 3).astype(np.float64) / 255.0
                total += pixels.sum(0)
                total_sq += np.square(pixels).sum(0)
                count += len(pixels)
        mean = total / count
        stats[key] = {"mean": mean.tolist(), "std": np.sqrt(np.maximum(total_sq / count - mean**2, 0.0)).tolist()}
    return stats


class PhotometricAugment(torch.nn.Module):
    """Training-time image augmentation for the ACT student (not in ACT or
    so101-ros; added for sim-to-real). Same ops as the DP student's camera
    transforms minus the crops, which would move the action-relevant image
    geometry the ACT backbone sees at full frame: ColorJitter(brightness,
    contrast, saturation 0.3), GaussianBlur(5, 0.01-2), per-image Gaussian
    noise (0-`noise_std` gray levels). Hue jitter defaults to 0.05 instead of
    the paper's 0.3: here the two cubes are also told apart by color (olive
    small, red large), and a +-0.3 hue rotation turns red into yellow-green.
    Input/output uint8 (B, 3, H, W)."""

    def __init__(self, noise_std: float = 4.0, hue: float = 0.05):
        super().__init__()
        from torchvision import transforms

        from so101.learning.dp.vision import RandomGaussianNoise

        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=hue),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0)),
            RandomGaussianNoise(noise_std),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)
