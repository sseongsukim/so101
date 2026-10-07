"""Operator keys must work without Enter and preserve every rated/aborted attempt."""

import argparse
import importlib.util
import json
import os
import pickle
import pty
import termios
from pathlib import Path

import numpy as np
import pytest
import torch

from so101.real.evaluation_controls import EvaluationLog, SingleKeyInput
from so101.real.recording import EpisodeRecorder


def test_terminal_keys_need_no_enter_and_restore_terminal():
    master, slave = pty.openpty()
    original = termios.tcgetattr(slave)
    keys = None
    try:
        with os.fdopen(os.dup(slave), "r") as stream:
            keys = SingleKeyInput(stream=stream)
            assert not termios.tcgetattr(slave)[3] & termios.ICANON
            for char in "rsynq":
                os.write(master, char.encode())
                assert keys.get() == char
            assert keys.get() is None
            keys.stop()
            assert termios.tcgetattr(slave) == original
    finally:
        if keys is not None:
            keys.stop()
        os.close(master)
        os.close(slave)


@pytest.fixture
def deployment(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/deploy_policy_real.py"
    spec = importlib.util.spec_from_file_location("deploy_policy_real_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(module, "move_to_start", lambda arm, keys: arm.starts.append("start"))
    return module


class FakeKeys:
    def __init__(self, values):
        self.values = iter(values)

    def get(self):
        return next(self.values, None)


class FakeCommand:
    def reset(self, state):
        pass

    def __call__(self, state):
        return state


class FakeArm:
    def __init__(self):
        self.starts = []
        self.command = FakeCommand()
        self.last_raw_read = torch.zeros(6)
        self.last_raw_sent = torch.zeros(6)
        self.sent = []

    def read(self):
        return torch.zeros(6)

    def send(self, target):
        self.sent.append(target)
        return target


class FakeRunner:
    def reset(self):
        pass

    def act(self, state, images):
        return torch.ones(1, 6)


class FakeCamera:
    def latest(self):
        return np.zeros((32, 32, 3), dtype=np.uint8)


def evaluation_args():
    return argparse.Namespace(episodes=2, max_seconds=10., fps=30, dry_run=False)


def test_r_prepares_s_starts_each_episode_and_y_n_log_results(deployment, tmp_path):
    arm = FakeArm()
    # Per episode: r then s; pre-inference poll; post-inference label.
    keys = FakeKeys([None, "r", "s", None, "y", "r", "s", None, "n"])
    recorder = EpisodeRecorder(tmp_path, image_size=(16, 16))
    log = EvaluationLog(tmp_path, tmp_path / "model.pt", {"inference_steps": 4})
    deployment.run_evaluation(evaluation_args(), FakeRunner(), arm,
                              {name: FakeCamera() for name in ("front", "wrist")}, recorder, keys, log)
    assert arm.starts == ["start", "start"]
    summary = json.loads(log.summary_path.read_text())
    assert (summary["successes"], summary["failures"], summary["success_rate"]) == (1, 1, 0.5)
    assert summary["rated_episodes"] == 2
    paths = sorted(tmp_path.glob("trajectory_*.pkl"))
    assert len(paths) == 2
    with paths[0].open("rb") as handle:
        first = pickle.load(handle)
    assert first["evaluation_label"] == "success"
    assert first["successes"].any()


def test_q_during_rollout_records_interruption_not_failure(deployment, tmp_path):
    arm = FakeArm()
    keys = FakeKeys(["r", "s", None, None, "q"])
    recorder = EpisodeRecorder(tmp_path, image_size=(16, 16))
    log = EvaluationLog(tmp_path, tmp_path / "model.pt", {})
    with pytest.raises(deployment.EvaluationQuit):
        deployment.run_evaluation(evaluation_args(), FakeRunner(), arm,
                                  {name: FakeCamera() for name in ("front", "wrist")}, recorder, keys, log)
    summary = json.loads(log.summary_path.read_text())
    assert summary["attempts"] == summary["interrupted"] == 1
    assert summary["rated_episodes"] == summary["failures"] == 0
    assert summary["success_rate"] is None
    assert summary["results"][0]["termination_reason"] == "q"
    with next(tmp_path.glob("trajectory_*.pkl")).open("rb") as handle:
        assert pickle.load(handle)["interrupted"] is True


def test_q_during_inference_prevents_policy_target_send(deployment, tmp_path):
    arm = FakeArm()
    log = EvaluationLog(tmp_path, tmp_path / "model.pt", {})
    with pytest.raises(deployment.EvaluationQuit):
        deployment.run_evaluation(evaluation_args(), FakeRunner(), arm,
                                  {name: FakeCamera() for name in ("front", "wrist")},
                                  EpisodeRecorder(tmp_path), FakeKeys(["r", "s", None, "q"]), log)
    assert all(torch.equal(target, torch.zeros(6)) for target in arm.sent)
    assert log.summary()["interrupted"] == 1
    assert log.summary()["results"][0]["frames"] == 0


def test_q_after_r_before_s_does_not_start_policy_or_record_attempt(deployment, tmp_path):
    arm = FakeArm()
    log = EvaluationLog(tmp_path, tmp_path / "model.pt", {})
    with pytest.raises(deployment.EvaluationQuit):
        deployment.run_evaluation(evaluation_args(), FakeRunner(), arm,
                                  {name: FakeCamera() for name in ("front", "wrist")},
                                  EpisodeRecorder(tmp_path), FakeKeys(["r", "q"]), log)
    assert arm.starts == ["start"]
    assert arm.sent == []
    assert log.summary()["attempts"] == 0


def test_r_during_running_logs_interruption_then_waits_for_s(deployment, tmp_path):
    arm = FakeArm()
    log = EvaluationLog(tmp_path, tmp_path / "model.pt", {})
    args = evaluation_args()
    args.episodes = 1
    deployment.run_evaluation(args, FakeRunner(), arm,
                              {name: FakeCamera() for name in ("front", "wrist")},
                              EpisodeRecorder(tmp_path),
                              FakeKeys(["r", "s", None, None, "r", "s", None, "y"]), log)
    assert arm.starts == ["start", "start"]
    assert log.summary()["interrupted"] == log.summary()["successes"] == 1
    assert log.summary()["results"][0]["termination_reason"] == "reset"
