from __future__ import annotations

import numpy as np
import pytest

from so101.learning.simdist.geometry import stack_cube_dense_reward


CONFIG = {
    "small_cube_size_m": .025, "large_cube_size_m": .04,
    "distance_gain": 20., "lift_clearance_m": .02,
    "success_xy_threshold_m": .018, "success_height_threshold_m": .005,
    "gripper_away_threshold_m": .02, "reward_distance_weight": .1,
    "reward_lift_weight": 1.5, "reward_align_weight": 2., "reward_success": 16.,
}


def test_stack_cube_dense_reward_emits_exact_success_reward_and_components():
    cubes = np.asarray([[[.25, 0., .0525], [.25, 0., .02]]], np.float32)
    grasp = np.asarray([[[.5, .5, .5], [.5, .5, .5], [.5, .5, .5]]], np.float32)
    reward, metrics = stack_cube_dense_reward(cubes, grasp, np.asarray([0.]), CONFIG)
    np.testing.assert_array_equal(reward, [16.])
    assert metrics["success"].tolist() == [True]


def test_stack_cube_dense_reward_rejects_bad_geometry_shapes():
    with pytest.raises(ValueError, match="cube_positions"):
        stack_cube_dense_reward(np.zeros((2, 6)), np.zeros((2, 3, 3)), np.zeros(2), CONFIG)
