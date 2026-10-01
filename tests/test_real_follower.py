"""Real-rig helpers without hardware: safety clamp, follower I/O, recording format."""

import numpy as np
import torch

from so101.learning.act import ACTION, OBS_STATE
from so101.learning.act.data import TeleopACTDataset, load_episodes
from so101.real.constants import SO101_JOINT_ORDER
from so101.real.follower import JOINT_HIGH, JOINT_LOW, START_POSE, FollowerArm, RateLimitedCommand, SafetyLimits
from so101.real.joint_mapping import JointMapping
from so101.real.recording import EpisodeRecorder


class FakeRobot:
    """Stands in for LeRobot's SO101Follower: stores the last action as its pose."""

    def __init__(self, values):
        self.values = dict(zip(SO101_JOINT_ORDER, values))
        self.sent = []

    def get_observation(self):
        return dict(self.values)

    def send_action(self, action):
        self.sent.append(action)
        self.values = dict(action)


def _arm(mapping, start_raw, limits=None):
    arm = FollowerArm.__new__(FollowerArm)
    from so101.real.interface import LeRobotSO101Interface

    arm.interface = LeRobotSO101Interface("cpu", "/dev/null", "my_follower", {}, 30, kind="follower",
                                          joint_mapping=mapping)
    arm.interface.robot = FakeRobot(start_raw)
    arm.command = RateLimitedCommand(limits or SafetyLimits())
    arm.connected = True
    arm.command.reset(arm.read())
    return arm


def test_rate_limit_and_joint_limits():
    command = RateLimitedCommand(SafetyLimits(max_step_rad=0.1, max_gripper_step_rad=0.3, limit_margin_rad=0.0))
    command.reset(torch.zeros(6))
    out = command(torch.full((6,), 5.0))
    torch.testing.assert_close(out, torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.3]))  # gripper has its own limit
    for _ in range(200):
        out = command(torch.full((6,), 50.0))
    torch.testing.assert_close(out, JOINT_HIGH)
    command.reset(JOINT_LOW.clone())
    torch.testing.assert_close(command(JOINT_LOW - 1.0), JOINT_LOW)


def test_follower_send_read_round_trip_through_the_mapping():
    for mapping in (None, JointMapping.physical({
        j.split(".")[0]: {"range_min": 800 + 10 * i, "range_max": 3200 - 10 * i}
        for i, j in enumerate(SO101_JOINT_ORDER)})):
        arm = _arm(mapping, [0.0, 0.0, 0.0, 0.0, 0.0, 50.0], SafetyLimits(max_step_rad=10.0))
        target = START_POSE.clone()
        sent = arm.send(target)
        torch.testing.assert_close(sent, target)
        torch.testing.assert_close(arm.read(), target, atol=1e-4, rtol=0)


def test_move_to_is_eased_and_rate_limited():
    arm = _arm(None, [0.0, 0.0, 0.0, 0.0, 0.0, 50.0])
    import so101.real.follower as follower

    sleeps = []
    original = follower.time.sleep
    follower.time.sleep = sleeps.append
    try:
        arm.move_to(START_POSE, seconds=1.0)
    finally:
        follower.time.sleep = original
    torch.testing.assert_close(arm.read(), START_POSE, atol=2e-3, rtol=0)
    sent = [arm.interface.get_mapped_actions_vectorized(torch.tensor([a[j] for j in SO101_JOINT_ORDER]))
            for a in arm.interface.robot.sent]
    steps = torch.stack([b - a for a, b in zip(sent, sent[1:])]).abs().max()
    assert steps <= SafetyLimits().max_step_rad + 1e-5


def test_recorded_episode_loads_into_the_training_pipeline(tmp_path):
    recorder = EpisodeRecorder(tmp_path, source="real_teleop")
    rng = np.random.default_rng(0)
    for t in range(5):
        recorder.append(torch.full((6,), float(t)), torch.full((6,), t + 0.5),
                        rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
                        rng.integers(0, 256, (480, 640, 3), dtype=np.uint8))
    path = recorder.save(success=True)
    assert path.name == "trajectory_000000.pkl" and len(recorder) == 0
    episodes = load_episodes(tmp_path)
    sample = TeleopACTDataset(episodes, chunk_size=2)[1]
    assert sample[OBS_STATE].tolist() == [1.0] * 6
    assert sample[ACTION][:, 0].tolist() == [1.5, 2.5]
    assert episodes[0].images["observation.images.front"].shape == (5, 240, 320, 3)


def test_raw_values_are_recorded_and_remap_recovers_new_mapping(tmp_path):
    import pickle
    import subprocess
    import sys

    arm = _arm(None, [0.0, 0.0, 0.0, 0.0, 0.0, 50.0], SafetyLimits(max_step_rad=10.0))
    recorder = EpisodeRecorder(tmp_path / "src")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for t in range(3):
        state = arm.read()
        sent = arm.send(START_POSE + 0.01 * t)
        recorder.append(state, sent, frame, frame, arm.last_raw_read, arm.last_raw_sent)
    path = recorder.save()
    data = pickle.loads(path.read_bytes())
    assert data["raw_observations"].shape == (3, 6) and data["raw_actions"].shape == (3, 6)
    # Linear mapping in force (no follower.yaml passed): remap reproduces the stored radians.
    missing = tmp_path / "none.yaml"
    out = tmp_path / "out"
    subprocess.run([sys.executable, "scripts/remap_real_episodes.py", "--src", str(tmp_path / "src"),
                    "--out", str(out), "--mapping", str(missing)], check=True, capture_output=True)
    remapped = pickle.loads((out / path.name).read_bytes())
    np.testing.assert_allclose(remapped["actions"], data["actions"], atol=1e-4)
    np.testing.assert_allclose(remapped["observations"], data["observations"], atol=1e-4)


def test_trim_idle_removes_leading_pause_and_trailing_stillness(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    step = np.deg2rad(2.0)                      # well above the motion thresholds
    targets = ([0.0] * 10                       # idle before starting
               + [step * k for k in range(1, 11)]
               + [step * 10] * 8                # mid pause: 8 still ticks
               + [step * (10 + k) for k in range(1, 6)]
               + [step * 15] * 2                # short stop: kept
               + [step * (15 + k) for k in range(1, 4)]
               + [step * 18] * 30)              # idle at the end
    for i, q in enumerate(targets):
        recorder.append(torch.full((6,), q), torch.full((6,), q), frame, frame)
    cut_start, cut_mid, cut_end = recorder.trim_idle(threshold_deg=1.0, keep_after_motion=15,
                                                     pause_step_deg=0.3, min_pause_steps=5)
    act = np.stack(recorder.act)[:, 0]
    assert cut_start == 9 and act[0] == 0.0 and np.isclose(act[1], step)                  # one still frame before motion
    assert cut_mid == 7                                                          # 8-tick pause -> 1 frame
    assert np.isclose(act, step * 10).sum() == 2                                 # last motion frame + pause frame
    assert np.isclose(act, step * 15).sum() == 3                                 # short stop untouched
    assert cut_end == 16 and np.isclose(act, step * 18).sum() == 15              # 15 kept after the last motion
    assert len(recorder.front) == len(recorder.act) == len(recorder.stamps) == len(targets) - 9 - 7 - 16
