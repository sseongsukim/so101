import numpy as np
import pytest
import torch

from so101.learning.simdist.dynamics_audit import (
    aggregate_by_cluster,
    bootstrap_ratio,
    summarize_groups,
    time_aware_persistence,
    training_seed_inventory,
    validate_episode_keys,
    validate_fingerprint_split,
)


def test_bootstrap_uses_paired_episode_macro_ratio():
    learned = np.array([[2., 2.], [2., 2.], [2., 2.], [2., 2.]])
    persistence = np.ones_like(learned)
    result = bootstrap_ratio(learned, persistence, repetitions=200, seed=3)
    assert [row["ratio_of_means"] for row in result] == [2., 2.]
    assert all(row["ratio_ci95_percentile"] == [2., 2.] for row in result)
    assert all(row["verdict"] == "fail" for row in result)


def test_heterogeneous_bootstrap_is_paired_and_reproducible():
    learned = np.array([[1., 2.], [10., 4.], [3., 8.], [7., 1.]])
    persistence = np.array([[1., 5.], [5., 2.], [1., 4.], [6., 1.]])
    seed, repetitions = 17, 500
    first = bootstrap_ratio(learned, persistence, repetitions=repetitions, seed=seed)
    second = bootstrap_ratio(learned, persistence, repetitions=repetitions, seed=seed)
    assert first == second
    assert first[0]["ratio_of_means"] == pytest.approx(learned[:, 0].mean() / persistence[:, 0].mean())
    assert first[0]["ratio_of_means"] != pytest.approx(np.mean(learned[:, 0] / persistence[:, 0]))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(learned), size=(repetitions, len(learned)))
    expected = np.percentile(learned[indices, 0].mean(1) / persistence[indices, 0].mean(1), [2.5, 97.5])
    assert first[0]["ratio_ci95_percentile"] == pytest.approx(expected.tolist())
    independent_rng = np.random.default_rng(seed)
    li = independent_rng.integers(0, len(learned), size=(repetitions, len(learned)))
    pi = independent_rng.integers(0, len(learned), size=(repetitions, len(learned)))
    independent = np.percentile(learned[li, 0].mean(1) / persistence[pi, 0].mean(1), [2.5, 97.5])
    assert not np.allclose(first[0]["ratio_ci95_percentile"], independent)


def test_bootstrap_marks_zero_persistence_inconclusive():
    result = bootstrap_ratio(np.ones((4, 1)), np.zeros((4, 1)), repetitions=100)
    assert result[0]["verdict"] == "inconclusive_zero_persistence_error"


def test_group_counts_are_episodes_not_windows():
    episodes = [dict(quality="expert", noise_std=0.0, terminal_class="success", windows=100 + i,
                     learned_mse=[.5], persistence_mse=[1.]) for i in range(4)]
    result = summarize_groups(episodes, min_episodes=4, min_group_episodes=4,
                              repetitions=100)
    assert result["all"]["episodes"] == 4
    assert result["all"]["windows"] == sum(x["windows"] for x in episodes)
    assert result["all"]["verdict"] == "pass"
    assert result["noise_std:0.0"]["episodes"] == 4


def test_bootstrap_rejects_undersized_or_invalid_samples():
    with pytest.raises(ValueError):
        bootstrap_ratio(np.ones((1, 1)), np.ones((1, 1)))
    with pytest.raises(ValueError):
        bootstrap_ratio(np.array([[np.nan], [1.]]), np.ones((2, 1)))


def test_training_seed_inventory_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        training_seed_inventory([])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no trajectory"):
        training_seed_inventory([empty])
    with (empty / "trajectory_0.pkl").open("wb") as stream:
        import pickle
        pickle.dump({"seed": 10, "actions": np.zeros((2, 6)), "tick": np.array([0, 1]),
                     "observations": np.zeros((2, 6)), "next_observations": np.zeros((2, 6))}, stream)
    seeds, count, fingerprints = training_seed_inventory([empty])
    assert (seeds, count) == ([10], 1)
    assert len(fingerprints) == 1


def test_training_seed_inventory_rejects_missing_seed(tmp_path):
    import pickle
    source = tmp_path / "train"
    source.mkdir()
    with (source / "trajectory_0.pkl").open("wb") as stream:
        pickle.dump({"observations": []}, stream)
    with pytest.raises(ValueError, match="invalid seed"):
        training_seed_inventory([source])


def test_training_inventory_requires_every_declared_directory(tmp_path):
    valid = tmp_path / "valid"
    valid.mkdir()
    with (valid / "trajectory_0.pkl").open("wb") as stream:
        import pickle
        pickle.dump({"seed": 7, "actions": np.zeros((1, 6)), "observations": np.zeros((1, 6)),
                     "next_observations": np.zeros((1, 6))}, stream)
    with pytest.raises(ValueError, match="does not exist"):
        training_seed_inventory([valid, tmp_path / "missing"])


def test_episode_seed_and_fingerprint_overlap_are_rejected():
    with pytest.raises(ValueError, match="seed overlap"):
        validate_episode_keys([10], [10], [0], 1)
    with pytest.raises(ValueError, match="unique"):
        validate_episode_keys([10], [20, 20], [0, 0], 2)
    with pytest.raises(ValueError, match="fingerprint overlap"):
        validate_fingerprint_split(["train"], ["train"], ["train"])
    with pytest.raises(ValueError, match="do not match"):
        validate_fingerprint_split(["wrong"], ["train"], ["holdout"])


def test_round_cluster_sensitivity_averages_episode_means():
    rows = [
        {"seed": 1, "windows": 10, "learned_mse": [2.], "persistence_mse": [1.]},
        {"seed": 1, "windows": 100, "learned_mse": [4.], "persistence_mse": [2.]},
        {"seed": 2, "windows": 7, "learned_mse": [8.], "persistence_mse": [4.]},
        {"seed": 2, "windows": 9, "learned_mse": [10.], "persistence_mse": [5.]},
    ]
    clusters = aggregate_by_cluster(rows, "seed")
    assert clusters[0]["learned_mse"] == [3.]
    assert clusters[0]["persistence_mse"] == [1.5]
    assert clusters[0]["windows"] == 110


def test_time_aware_persistence_changes_clock_not_current_content():
    class Scaler:
        @staticmethod
        def scale(value):
            return value

    class Encoder:
        def __init__(self):
            self.content_input = None

        def content(self, joints, front, wrist):
            self.content_input = joints.clone()
            return joints

        @staticmethod
        def latent_from_content(content, remaining):
            return content + remaining

    class Model:
        def __init__(self):
            self.scaler, self.encoder = Scaler(), Encoder()

    model = Model()
    joints = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    x = {"proprio_obs_hist": joints,
         "front_hist": torch.zeros(1, 2, 2, 2, 3),
         "wrist_hist": torch.zeros(1, 2, 2, 2, 3)}
    remaining = torch.tensor([[[3.], [2.], [1.]]])
    prediction = time_aware_persistence(model, x, remaining)
    assert torch.equal(model.encoder.content_input, joints[:, -1])
    assert torch.equal(prediction[0, :, 0], joints[0, -1, 0] + remaining[0, :, 0])


def test_episode_forecast_mse_weights_each_window_even_with_partial_batch():
    from so101.learning.simdist.dynamics_audit import episode_forecast_mse

    class Scaler:
        @staticmethod
        def scale(value):
            return value

    class Encoder:
        @staticmethod
        def content(joints, front, wrist):
            return torch.zeros((len(joints), 1))

        @staticmethod
        def latent_from_content(content, remaining):
            return torch.zeros((len(content), remaining.shape[1], 1))

    class Model:
        scaler, encoder = Scaler(), Encoder()

        @staticmethod
        def eval():
            pass

        @staticmethod
        def inference(x):
            return {"latents": x["prediction"]}

        @staticmethod
        def encode_latent(joints, front, wrist, remaining):
            return torch.zeros((len(joints), len(joints[0]), 1))

    def batch(predictions):
        b = len(predictions)
        x = {"prediction": torch.tensor(predictions, dtype=torch.float32).reshape(b, 1, 1),
             "proprio_obs_hist": torch.zeros((b, 1, 6)),
             "front_hist": torch.zeros((b, 1, 2, 2, 3)),
             "wrist_hist": torch.zeros((b, 1, 2, 2, 3))}
        y = {"proprio_obs": torch.zeros((b, 1, 6)), "front": torch.zeros((b, 1, 2, 2, 3)),
             "wrist": torch.zeros((b, 1, 2, 2, 3)), "next_remaining_ticks": torch.ones((b, 1, 1))}
        return {"model_in": x, "labels": y}

    result = episode_forecast_mse(Model(), [batch([1., 3.]), batch([2.])])
    assert result["windows"] == 3
    assert result["learned_mse"] == [pytest.approx(14. / 3.)]
    with pytest.raises(ValueError, match="no full forecast"):
        episode_forecast_mse(Model(), [])
