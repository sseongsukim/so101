"""JointMapping: linear == today's interface math; physical == LeRobot DEGREES + offset."""

import numpy as np
import pytest
import torch

from so101.real.conversions import raw_degrees_to_sim_radians, sim_radians_to_raw_degrees
from so101.real.joint_mapping import DEG_PER_TICK, JOINTS, USD_MID, JointMapping

# The follower calibration this rig used when the mapping question came up.
CALIBRATION = {
    "shoulder_pan": {"drive_mode": 0, "homing_offset": -1435, "range_min": 811, "range_max": 3277},
    "shoulder_lift": {"drive_mode": 0, "homing_offset": -1020, "range_min": 663, "range_max": 3191},
    "elbow_flex": {"drive_mode": 0, "homing_offset": 1742, "range_min": 1010, "range_max": 3164},
    "wrist_flex": {"drive_mode": 0, "homing_offset": -357, "range_min": 661, "range_max": 3115},
    "wrist_roll": {"drive_mode": 0, "homing_offset": 1988, "range_min": 0, "range_max": 4095},
    "gripper": {"drive_mode": 0, "homing_offset": -1933, "range_min": 2029, "range_max": 3529},
}
VALUES = torch.tensor([[-80.0, -27.1, 15.7, 87.1, 0.2, 8.4], [36.4, 87.4, -93.6, -54.3, -87.0, 97.4]])


def _ticks(values: np.ndarray) -> np.ndarray:
    """Invert LeRobot's RANGE_M100_100 / RANGE_0_100 normalization."""
    ticks = []
    for i, j in enumerate(JOINTS):
        lo, hi = CALIBRATION[j]["range_min"], CALIBRATION[j]["range_max"]
        frac = values[..., i] / 100.0 if j == "gripper" else (values[..., i] + 100.0) / 200.0
        ticks.append(lo + frac * (hi - lo))
    return np.stack(ticks, -1)


def test_linear_matches_current_interface_math():
    mapping = JointMapping.linear()
    for row in VALUES:
        torch.testing.assert_close(mapping.to_sim(row), raw_degrees_to_sim_radians(row))
        # float32 round trip through radians
        torch.testing.assert_close(mapping.to_lerobot(mapping.to_sim(row)), row, atol=1e-4, rtol=0)
        torch.testing.assert_close(mapping.to_lerobot(mapping.to_sim(row)), sim_radians_to_raw_degrees(mapping.to_sim(row)), atol=1e-4, rtol=0)


def test_physical_is_lerobot_degrees_plus_offset():
    offsets = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    mapping = JointMapping.physical(CALIBRATION, offsets)
    values = VALUES.numpy().astype(np.float64)
    ticks = _ticks(values)
    mid = np.array([(CALIBRATION[j]["range_min"] + CALIBRATION[j]["range_max"]) / 2 for j in JOINTS])
    lerobot_degrees = (ticks - mid) * DEG_PER_TICK  # MotorsBus._normalize, DEGREES mode
    np.testing.assert_allclose(mapping.to_sim(values), np.deg2rad(lerobot_degrees + offsets), atol=1e-9)
    np.testing.assert_allclose(mapping.to_lerobot(mapping.to_sim(values)), values, atol=1e-9)


def test_nominal_physical_agrees_with_linear_at_range_midpoint_only():
    physical, linear = JointMapping.physical(CALIBRATION), JointMapping.linear()
    mid = np.array([0, 0, 0, 0, 0, 50.0])
    np.testing.assert_allclose(physical.to_sim(mid), linear.to_sim(mid))
    np.testing.assert_allclose(physical.to_sim(mid), np.deg2rad(USD_MID))
    # shoulder_lift: 222.2 deg physical sweep vs 200 deg USD range -> 11% apart at the end.
    end = np.array([0, 100.0, 0, 0, 0, 50.0])
    ratio = np.rad2deg(physical.to_sim(end) - physical.to_sim(mid))[1] / np.rad2deg(linear.to_sim(end) - linear.to_sim(mid))[1]
    assert ratio == pytest.approx(222.2 / 200.0, rel=1e-3)


def test_torch_and_numpy_agree():
    mapping = JointMapping.physical(CALIBRATION)
    np.testing.assert_allclose(mapping.to_sim(VALUES).numpy(), mapping.to_sim(VALUES.numpy().astype(np.float64)), rtol=1e-6)


def test_save_load_round_trip_and_stale_calibration_is_refused(tmp_path):
    mapping = JointMapping.physical(CALIBRATION, np.arange(6.0))
    path = tmp_path / "follower.yaml"
    mapping.save(path, note="test")
    np.testing.assert_array_equal(JointMapping.load(path, CALIBRATION).offset_deg, np.arange(6.0))
    recalibrated = {j: dict(v) for j, v in CALIBRATION.items()}
    recalibrated["elbow_flex"]["range_max"] += 10
    with pytest.raises(ValueError, match="re-run"):
        JointMapping.load(path, recalibrated)
