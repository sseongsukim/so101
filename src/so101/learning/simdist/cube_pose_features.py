"""Bounded CPU feature cache for the opt-in spatial cube-pose trainer."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import pickle

import numpy as np
import torch


class SpatialFeatureStream:
    """Read one RGB episode and extract/cache small frozen-map chunks on demand.

    The cache stores CPU float32 maps only, bounded by cache_mib. A requested
    batch and the current extraction chunk are additional transient allocations.
    No maps are written to disk or packed into a contiguous dataset array.
    Source records supply only paths/lengths; labels never enter the encoder.
    Pickle decoding still needs enough CPU RAM for one source episode.
    """

    def __init__(self, encoder, sources: list[tuple[Path | tuple[np.ndarray, np.ndarray], int]], device: torch.device,
                 feature_batch_size: int = 16, cache_mib: float = 2048) -> None:
        if not sources or any(length < 1 for _, length in sources):
            raise ValueError("spatial stream requires nonempty episode sources")
        if feature_batch_size < 1 or not np.isfinite(cache_mib) or cache_mib < 0:
            raise ValueError("feature_batch_size must be positive and cache_mib finite/nonnegative")
        self.encoder = encoder.eval()
        self.sources = sources
        self.device = device
        self.feature_batch_size = feature_batch_size
        self.cache_limit_bytes = int(cache_mib * 1024 * 1024)
        self.offsets = np.concatenate(([0], np.cumsum([length for _, length in sources])))
        self.cache: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()
        self.cached_bytes = 0
        self.peak_cached_bytes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.extractions = 0
        self.reencoded_chunks = 0
        self.encoded_frames = 0
        self.episode_loads = 0
        self._extracted_keys: set[tuple[int, int]] = set()
        self.feature_shape: tuple[int, ...] | None = None
        self._image_episode: int | None = None
        self._images: tuple[np.ndarray, np.ndarray] | None = None

    def _load_images(self, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
        if self._image_episode != episode_index:
            # Release the previous episode before decoding the next one.
            self._images = None
            self._image_episode = None
            path, length = self.sources[episode_index]
            if isinstance(path, tuple):
                # Aligned NPZ readers may keep current RGB arrays in memory;
                # episode slices are views, so this adds no second image copy.
                front, wrist = (np.asarray(images)[:length] for images in path)
            else:
                with path.open("rb") as stream:
                    episode = pickle.load(stream)
                self.episode_loads += 1
                front, wrist = (np.asarray(episode[key])[:length] for key in ("front_images", "wrist_images"))
            if len(front) != length or len(wrist) != length:
                raise ValueError(f"{path}: RGB length does not cover the retained horizon")
            self._images = (front, wrist)
            self._image_episode = episode_index
        return self._images

    def _chunk(self, episode_index: int, chunk_index: int) -> torch.Tensor:
        key = (episode_index, chunk_index)
        if key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.cache_misses += 1
        front, wrist = self._load_images(episode_index)
        start = chunk_index * self.feature_batch_size
        stop = min(start + self.feature_batch_size, self.sources[episode_index][1])
        rgb = [torch.as_tensor(image[start:stop], device=self.device).permute(0, 3, 1, 2).contiguous()
               for image in (front, wrist)]
        maps = self.encoder(*rgb).detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(maps).all():
            raise ValueError("spatial feature chunk contains non-finite values")
        shape = tuple(maps.shape[1:])
        if self.feature_shape is not None and shape != self.feature_shape:
            raise ValueError("spatial feature shape changed between source chunks")
        self.feature_shape = shape
        self.extractions += 1
        self.encoded_frames += stop - start
        self.reencoded_chunks += int(key in self._extracted_keys)
        self._extracted_keys.add(key)
        size = maps.numel() * maps.element_size()
        if size <= self.cache_limit_bytes:
            while self.cache and self.cached_bytes + size > self.cache_limit_bytes:
                _, evicted = self.cache.popitem(last=False)
                self.cached_bytes -= evicted.numel() * evicted.element_size()
            self.cache[key] = maps
            self.cached_bytes += size
            self.peak_cached_bytes = max(self.peak_cached_bytes, self.cached_bytes)
        return maps

    def statistics(self) -> dict:
        expected_bytes = (None if self.feature_shape is None else
                          int(self.offsets[-1] * np.prod(self.feature_shape) * 4))
        return {
            "limit_bytes": self.cache_limit_bytes, "cached_bytes": self.cached_bytes,
            "peak_cached_bytes": self.peak_cached_bytes,
            "full_dataset_bytes": expected_bytes,
            "full_dataset_fits": expected_bytes is not None and expected_bytes <= self.cache_limit_bytes,
            "cache_hits": self.cache_hits, "cache_misses": self.cache_misses,
            "extractions": self.extractions, "reencoded_chunks": self.reencoded_chunks,
            "encoded_frames": self.encoded_frames, "episode_loads": self.episode_loads,
        }

    def prefill_if_fits(self) -> bool:
        """Amortize backbone inference across epochs without exceeding the cap.

        Sequential episode/chunk order also limits pickle decoding to one load
        per episode when the cache can hold the entire dataset. A small cache
        continues to stream on demand instead of evicting during a full prefill.
        """
        if self.feature_shape is None:
            self.get(np.array([0], dtype=np.int64))
        if not self.statistics()["full_dataset_fits"]:
            return False
        for episode, (_, length) in enumerate(self.sources):
            for chunk in range((length + self.feature_batch_size - 1) // self.feature_batch_size):
                self._chunk(episode, chunk)
        return True

    def get(self, sample_ids: np.ndarray) -> torch.Tensor:
        """Return CPU maps (N, 2, C, H, W) in the exact requested sample order."""
        sample_ids = np.asarray(sample_ids)
        if (sample_ids.ndim != 1 or not np.issubdtype(sample_ids.dtype, np.integer)
                or not len(sample_ids) or np.any(sample_ids < 0) or np.any(sample_ids >= self.offsets[-1])):
            raise ValueError("sample_ids must be a nonempty vector of valid integer indices")
        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        episodes = np.searchsorted(self.offsets[1:], sample_ids, side="right")
        for output_index, (sample, episode) in enumerate(zip(sample_ids, episodes)):
            local = int(sample - self.offsets[episode])
            chunk, within = divmod(local, self.feature_batch_size)
            groups.setdefault((int(episode), chunk), []).append((output_index, within))
        output = None
        for (episode, chunk), indices in groups.items():
            maps = self._chunk(episode, chunk)
            if output is None:
                output = torch.empty((len(sample_ids), *maps.shape[1:]), dtype=torch.float32)
            destinations, within = zip(*indices)
            output[list(destinations)] = maps[list(within)]
        return output
