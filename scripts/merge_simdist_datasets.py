#!/usr/bin/env python3
"""Merge aligned SimDist NPZ archives while preserving episode provenance."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


EPISODE_FIELDS = {
    "episode_starts", "episode_lengths", "episode_seeds", "episode_env_indices",
    "episode_quality", "episode_noise_std", "episode_sources", "episode_fingerprints",
}


def merge_archives(paths: list[Path]) -> dict[str, np.ndarray]:
    if len(paths) < 2:
        raise ValueError("provide at least two aligned archives")
    archives = [np.load(path, allow_pickle=False) for path in paths]
    try:
        keys = set(archives[0].files)
        if any(set(archive.files) != keys for archive in archives[1:]):
            raise ValueError("archives must have identical fields")
        transition_counts = [len(archive["actions"]) for archive in archives]
        episode_counts = [len(archive["episode_starts"]) for archive in archives]
        for archive, n in zip(archives, transition_counts):
            if any(len(archive[key]) != n for key in ("observations", "next_observations", "actions",
                                                       "rewards", "values", "front_images", "wrist_images",
                                                       "next_front_images", "next_wrist_images")):
                raise ValueError("transition fields have inconsistent lengths")
        seed_sets = [set(map(int, archive["episode_seeds"])) for archive in archives]
        if any(seed_sets[i] & seed_sets[j] for i in range(len(seed_sets))
               for j in range(i + 1, len(seed_sets))):
            raise ValueError("reset seeds overlap across input archives")
        result = {}
        for key in archives[0].files:
            arrays = [archive[key] for archive in archives]
            if key == "episode_starts":
                result[key] = np.concatenate([a + sum(transition_counts[:i]) for i, a in enumerate(arrays)])
            elif key in EPISODE_FIELDS:
                result[key] = np.concatenate(arrays)
            elif arrays[0].ndim and arrays[0].shape[0] == transition_counts[0]:
                if any(a.ndim != arrays[0].ndim or a.shape[1:] != arrays[0].shape[1:] for a in arrays[1:]):
                    raise ValueError(f"transition shape mismatch for {key}")
                result[key] = np.concatenate(arrays)
            else:
                if any(not np.array_equal(arrays[0], a) for a in arrays[1:]):
                    raise ValueError(f"dataset-level metadata mismatch for {key}")
                result[key] = arrays[0].copy()
        starts, lengths = result["episode_starts"], result["episode_lengths"]
        if len(starts) != sum(episode_counts) or int(starts[-1] + lengths[-1]) != sum(transition_counts):
            raise ValueError("merged episode extents do not cover transition arrays")
        if np.any(starts[1:] != starts[:-1] + lengths[:-1]):
            raise ValueError("episode extents are not contiguous")
        return result
    finally:
        for archive in archives:
            archive.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    merged = merge_archives(args.inputs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary.open("wb") as f:
        np.savez_compressed(f, **merged)
    temporary.replace(args.out)
    print(f"wrote {args.out}: {len(merged['episode_starts'])} episodes, {len(merged['actions'])} transitions")


if __name__ == "__main__":
    main()
