import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.merge_simdist_datasets import merge_archives


def archive(seed, transitions):
    n = transitions
    return {
        "observations": np.zeros((n, 6), np.float32),
        "next_observations": np.ones((n, 6), np.float32),
        "actions": np.full((n, 6), seed, np.float32),
        "rewards": np.zeros(n, np.float32), "values": np.zeros(n, np.float32),
        "front_images": np.zeros((n, 2, 2, 3), np.uint8),
        "wrist_images": np.zeros((n, 2, 2, 3), np.uint8),
        "next_front_images": np.zeros((n, 2, 2, 3), np.uint8),
        "next_wrist_images": np.zeros((n, 2, 2, 3), np.uint8),
        "episode_starts": np.array([0]), "episode_lengths": np.array([n]),
        "episode_seeds": np.array([seed]), "episode_env_indices": np.array([0]),
        "episode_quality": np.array(["mixed"]), "episode_noise_std": np.array([0.04]),
        "episode_sources": np.array([f"seed-{seed}"]), "episode_fingerprints": np.array([str(seed)]),
        "gamma": np.array(0.99, np.float32), "episode_label": np.full(n, seed),
    }


class SimdistMergeTests(unittest.TestCase):
    def test_offsets_episode_starts_and_keeps_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for seed in (10, 11):
                path = Path(directory) / f"{seed}.npz"
                np.savez_compressed(path, **archive(seed, 4))
                paths.append(path)
            merged = merge_archives(paths)
        np.testing.assert_array_equal(merged["episode_starts"], [0, 4])
        np.testing.assert_array_equal(merged["episode_lengths"], [4, 4])
        np.testing.assert_array_equal(merged["episode_seeds"], [10, 11])
        np.testing.assert_array_equal(merged["episode_label"], [10]*4 + [11]*4)

    def test_rejects_overlapping_reset_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i in range(2):
                path = Path(directory) / f"{i}.npz"
                np.savez_compressed(path, **archive(10, 4))
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "reset seeds overlap"):
                merge_archives(paths)


if __name__ == "__main__":
    unittest.main()
