from __future__ import annotations

import pickle

import numpy as np
import pytest

from scripts.prepare_simdist_dataset import pack_trajectories


def _episode(path, *, ticks=(0, 1, 2), semantics="applied_absolute_joint_target_radians"):
    n = len(ticks)
    episode = {
        "observations": np.arange(n * 6, dtype=np.float32).reshape(n, 6),
        "next_observations": np.arange(n * 6, dtype=np.float32).reshape(n, 6) + 1,
        "actions": np.full((n, 6), 0.25, np.float32),
        "rewards": np.asarray([1.0, 2.0, 4.0][:n], np.float32),
        "front_images": np.zeros((n, 4, 5, 3), np.uint8),
        "wrist_images": np.ones((n, 4, 5, 3), np.uint8),
        "next_front_images": np.full((n, 4, 5, 3), 2, np.uint8),
        "next_wrist_images": np.full((n, 4, 5, 3), 3, np.uint8),
        "terminated": np.zeros(n, bool),
        "truncated": np.asarray([False] * (n - 1) + [True]),
        "tick": np.asarray(ticks),
        "fps": 30,
        "task_horizon_ticks": 300,
        "action_semantics": semantics,
        "reward_semantics": "dense_environment_reward",
        "trajectory_quality": "diverse",
    }
    path.write_bytes(pickle.dumps(episode))
    return episode


def test_pack_preserves_real_next_observation_and_mc_next_state_value(tmp_path):
    path = tmp_path / "trajectory_000000.pkl"
    episode = _episode(path)
    packed = pack_trajectories([path], gamma=0.5)
    np.testing.assert_array_equal(packed["next_observations"], episode["next_observations"])
    np.testing.assert_array_equal(packed["next_front_images"], episode["next_front_images"])
    assert packed["trajectory_quality"].tolist() == ["diverse"] * 3
    assert packed["phase_labels"].tolist() == ["unknown"] * 3
    assert not packed["expert_policy_flag"].any()
    assert packed["action_semantics"].item() == "applied_absolute_joint_target_radians"
    # V(s') for the first action is r_1 + gamma*r_2; the last next state
    # transition has no bootstrapped value.
    np.testing.assert_allclose(packed["values"], [4.0, 4.0, 0.0])
    assert packed["truncated"][-1]
    assert not packed["terminated"].any()
    assert not packed["value_valid"].any()
    assert packed["reward_valid"].all()
    np.testing.assert_allclose(packed["tick_seconds"], 1.0 / 30.0)


def test_pack_preserves_explicit_training_only_cube_successors(tmp_path):
    path = tmp_path / "object_labels.pkl"
    episode = _episode(path)
    current = np.arange(18, dtype=np.float32).reshape(3, 2, 3) / 100
    successor = np.concatenate([current[1:], (current[-1:] + 0.01)], axis=0)
    episode.update({
        "cube_positions_robot_relative_m": current,
        "next_cube_positions_robot_relative_m": successor,
        "cube_position_label_semantics": "simulator_xyz_robot_root_relative_training_only",
        "cube_position_order": ["held_cube", "fixed_cube"],
    })
    path.write_bytes(pickle.dumps(episode))
    packed = pack_trajectories([path], gamma=0.5)
    np.testing.assert_array_equal(packed["cube_positions_robot_relative_m"], current)
    np.testing.assert_array_equal(packed["next_cube_positions_robot_relative_m"], successor)
    assert packed["cube_position_label_semantics"].item() == "simulator_xyz_robot_root_relative_training_only"
    assert packed["cube_position_order"].tolist() == ["held_cube", "fixed_cube"]


def test_pack_rejects_partial_or_mixed_cube_label_contract(tmp_path):
    partial = tmp_path / "partial.pkl"
    episode = _episode(partial)
    episode["cube_positions_robot_relative_m"] = np.zeros((3, 2, 3), np.float32)
    episode["cube_position_label_semantics"] = "simulator_xyz_robot_root_relative_training_only"
    episode["cube_position_order"] = ["held_cube", "fixed_cube"]
    partial.write_bytes(pickle.dumps(episode))
    with pytest.raises(ValueError, match="must be provided together"):
        pack_trajectories([partial])

    first, second = tmp_path / "labeled.pkl", tmp_path / "unlabeled.pkl"
    labeled = _episode(first)
    labeled["cube_positions_robot_relative_m"] = np.zeros((3, 2, 3), np.float32)
    labeled["next_cube_positions_robot_relative_m"] = np.zeros((3, 2, 3), np.float32)
    labeled["next_cube_positions_robot_relative_m"][-1] = 1.0
    labeled["cube_position_label_semantics"] = "simulator_xyz_robot_root_relative_training_only"
    labeled["cube_position_order"] = ["held_cube", "fixed_cube"]
    first.write_bytes(pickle.dumps(labeled))
    _episode(second)
    with pytest.raises(ValueError, match="cannot mix trajectories"):
        pack_trajectories([first, second])


def test_pack_rejects_discontinuous_cube_successor_labels(tmp_path):
    path = tmp_path / "discontinuous_labels.pkl"
    episode = _episode(path)
    current = np.zeros((3, 2, 3), np.float32)
    successor = np.zeros_like(current)
    successor[0, 0, 0] = 0.1
    episode.update({
        "cube_positions_robot_relative_m": current,
        "next_cube_positions_robot_relative_m": successor,
        "cube_position_label_semantics": "simulator_xyz_robot_root_relative_training_only",
        "cube_position_order": ["held_cube", "fixed_cube"],
    })
    path.write_bytes(pickle.dumps(episode))
    with pytest.raises(ValueError, match="do not align"):
        pack_trajectories([path])


def test_pack_preserves_optional_geometry_reward_inputs(tmp_path):
    path = tmp_path / "geometry_labels.pkl"
    episode = _episode(path)
    current = np.zeros((3, 2, 3), np.float32)
    successor = current.copy()
    episode.update({
        "cube_positions_robot_relative_m": current,
        "next_cube_positions_robot_relative_m": successor,
        "cube_position_label_semantics": "simulator_xyz_robot_root_relative_training_only",
        "cube_position_order": ["held_cube", "fixed_cube"],
        "next_grasp_points_robot_relative_m": np.ones((3, 3, 3), np.float32),
        "grasp_point_order": ["midpoint", "fixed_fingertip", "moving_fingertip"],
        "table_height_robot_relative_m": np.zeros(3, np.float32),
        "geometry_reward_config": {"distance_gain": 20.0},
    })
    path.write_bytes(pickle.dumps(episode))
    packed = pack_trajectories([path], gamma=.5)
    np.testing.assert_array_equal(packed["next_grasp_points_robot_relative_m"], episode["next_grasp_points_robot_relative_m"])
    np.testing.assert_array_equal(packed["table_height_robot_relative_m"], np.zeros(3))
    assert packed["grasp_point_order"].tolist() == ["midpoint", "fixed_fingertip", "moving_fingertip"]
    assert '"distance_gain": 20.0' in packed["geometry_reward_config_json"].item()


def test_pack_preserves_current_camera_pose_and_intrinsics_contract(tmp_path):
    path = tmp_path / "camera_geometry.pkl"
    episode = _episode(path)
    poses = np.zeros((3, 2, 7), np.float32)
    poses[..., 3] = 1.0
    intrinsics = np.broadcast_to(np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]], np.float32),
                                 (3, 2, 3, 3)).copy()
    episode.update({
        "camera_poses_robot_relative_m_wxyz": poses,
        "camera_intrinsics_source_px": intrinsics,
        "camera_order": ["front", "wrist"],
        "camera_pose_semantics": "robot_root_relative_translation_world_axes_quaternion_wxyz_forward_x_up_z",
        "camera_intrinsics_semantics": "pinhole_3x3_source_sensor_pixel_coordinates",
        "camera_source_image_shape_hw": [[480, 640], [480, 640]],
    })
    path.write_bytes(pickle.dumps(episode))
    packed = pack_trajectories([path])
    np.testing.assert_array_equal(packed["camera_poses_robot_relative_m_wxyz"], poses)
    np.testing.assert_array_equal(packed["camera_intrinsics_source_px"], intrinsics)
    contract = packed["camera_geometry_contract_json"].item()
    assert '"camera_order": ["front", "wrist"]' in contract
    assert "source_sensor_pixel_coordinates" in contract


def test_pack_rejects_invalid_camera_geometry_quaternion(tmp_path):
    path = tmp_path / "bad_camera_geometry.pkl"
    episode = _episode(path)
    poses = np.zeros((3, 2, 7), np.float32)
    intrinsics = np.broadcast_to(np.eye(3, dtype=np.float32), (3, 2, 3, 3)).copy()
    episode.update({
        "camera_poses_robot_relative_m_wxyz": poses,
        "camera_intrinsics_source_px": intrinsics,
        "camera_order": ["front", "wrist"],
        "camera_pose_semantics": "robot_root_relative_translation_world_axes_quaternion_wxyz_forward_x_up_z",
        "camera_intrinsics_semantics": "pinhole_3x3_source_sensor_pixel_coordinates",
        "camera_source_image_shape_hw": [[480, 640], [480, 640]],
    })
    path.write_bytes(pickle.dumps(episode))
    with pytest.raises(ValueError, match="unit"):
        pack_trajectories([path])


def test_pack_preserves_path_resolved_bitpacked_cube_instance_masks(tmp_path):
    path = tmp_path / "cube_masks.pkl"
    episode = _episode(path)
    masks = np.zeros((3, 2, 2, 4, 1), np.uint8)
    masks[:, 0, 0, 1, 0] = 0b10000000
    masks[:, 1, 1, 2, 0] = 0b00010000
    episode.update({
        "cube_instance_masks_packed": masks,
        "cube_instance_mask_camera_order": ["front", "wrist"],
        "cube_instance_mask_object_order": ["held_cube", "fixed_cube"],
        "cube_instance_mask_semantics": "uncolorized_instance_id_segmentation_fast_resolved_by_usd_prim_path",
        "cube_instance_mask_storage": "np.packbits_axis_last_bitorder_big",
        "cube_instance_mask_source_shape_hw": [[4, 8], [4, 8]],
        "camera_sensor_frame_counters": np.asarray([[10, 10], [11, 11], [12, 12]], np.int64),
        "camera_sensor_frame_counter_semantics": "same_current_render_used_for_rgb_and_instance_masks",
        "cube_instance_id_to_usd_path_by_camera": {
            "front": {"4": "/World/envs/env_0/HeldAsset", "7": "/World/envs/env_0/FixedAsset"},
            "wrist": {"2": "/World/envs/env_0/HeldAsset", "9": "/World/envs/env_0/FixedAsset"},
        },
    })
    path.write_bytes(pickle.dumps(episode))
    packed = pack_trajectories([path])
    np.testing.assert_array_equal(packed["cube_instance_masks_packed"], masks)
    np.testing.assert_array_equal(packed["camera_sensor_frame_counters"], [[10, 10], [11, 11], [12, 12]])
    assert "instance_id_segmentation_fast" in packed["cube_instance_mask_contract_json"].item()
    maps = packed["episode_cube_instance_id_to_usd_path_json"].tolist()
    assert '"/World/envs/env_0/HeldAsset"' in maps[0]


def test_instance_id_masks_resolve_paths_and_pack_bits_without_id_assumptions():
    from so101.learning.simdist.instance_masks import resolve_cube_instance_masks

    ids = np.asarray([[11, 11, 22, 0, 0, 0, 0, 0]], np.int32)
    packed, used_paths = resolve_cube_instance_masks(ids, {
        "11": "/World/envs/env_0/HeldAsset/Visuals/mesh",
        "22": "/World/envs/env_0/FixedAsset/Visuals/mesh",
        "5": "/World/envs/env_1/HeldAsset",
    }, 0)
    assert packed.shape == (2, 1, 1)
    expected = np.asarray([[[1, 1, 0, 0, 0, 0, 0, 0]],
                           [[0, 0, 1, 0, 0, 0, 0, 0]]], np.uint8)
    np.testing.assert_array_equal(np.unpackbits(packed, axis=-1, bitorder="big"), expected)
    assert used_paths["11"].endswith("HeldAsset/Visuals/mesh")


def test_true_termination_has_valid_zero_successor_value(tmp_path):
    path = tmp_path / "terminated.pkl"
    episode = _episode(path)
    episode["truncated"][-1] = False
    episode["terminated"][-1] = True
    path.write_bytes(pickle.dumps(episode))
    packed = pack_trajectories([path], gamma=0.5)
    np.testing.assert_allclose(packed["values"], [4.0, 4.0, 0.0])
    assert packed["value_valid"].all()


def test_declared_task_timeout_is_valid_finite_horizon_mc(tmp_path):
    path = tmp_path / "timeout.pkl"
    episode = _episode(path)
    episode["task_horizon_ticks"] = 3
    path.write_bytes(pickle.dumps(episode))
    packed = pack_trajectories([path], gamma=0.5)
    assert packed["truncated"].tolist() == [False, False, True]
    assert packed["objective_terminal"].tolist() == [False, False, True]
    assert packed["value_valid"].all()
    np.testing.assert_array_equal(packed["remaining_ticks"], [3, 2, 1])
    np.testing.assert_array_equal(packed["next_remaining_ticks"], [2, 1, 0])
    np.testing.assert_allclose(packed["values"], [4., 4., 0.])


def test_post_success_tail_beyond_horizon_cannot_change_labels(tmp_path):
    path = tmp_path / "long_tail.pkl"
    episode = _episode(path)
    episode["task_horizon_ticks"] = 2
    episode["rewards"][-1] = 1000
    path.write_bytes(pickle.dumps(episode))
    packed = pack_trajectories([path], gamma=0.5)
    assert len(packed["actions"]) == 2
    assert not packed["terminated"].any()
    assert not packed["truncated"].any()
    assert packed["objective_terminal"].tolist() == [False, True]
    assert packed["value_valid"].all()
    np.testing.assert_allclose(packed["values"], [2., 0.])


def test_unknown_task_horizon_requires_explicit_declaration(tmp_path):
    path = tmp_path / "old.pkl"
    episode = _episode(path)
    del episode["task_horizon_ticks"]
    path.write_bytes(pickle.dumps(episode))
    with pytest.raises(ValueError, match="task_horizon_ticks missing"):
        pack_trajectories([path])
    packed = pack_trajectories([path], task_horizon_ticks=3)
    assert packed["value_valid"].all()


def test_pack_rejects_time_gaps_and_undeclared_action_units(tmp_path):
    gap = tmp_path / "gap.pkl"
    _episode(gap, ticks=(0, 2, 3))
    with pytest.raises(ValueError, match="non-contiguous ticks"):
        pack_trajectories([gap])
    wrong = tmp_path / "wrong.pkl"
    _episode(wrong, semantics="normalized_actions")
    with pytest.raises(ValueError, match="absolute radians"):
        pack_trajectories([wrong])


def test_unmarked_episode_has_censored_value_targets(tmp_path):
    path = tmp_path / "unmarked.pkl"
    episode = _episode(path)
    episode["truncated"][:] = False
    path.write_bytes(pickle.dumps(episode))
    packed = pack_trajectories([path])
    assert not packed["value_valid"].any()


def test_pack_multiple_source_episodes_keeps_boundaries_and_reward_masks(tmp_path):
    expert_path = tmp_path / "expert.pkl"
    perturbed_path = tmp_path / "perturbed.pkl"
    expert = _episode(expert_path)
    expert["trajectory_quality"] = "expert_teacher"
    expert["expert_policy_flag"] = np.ones(3, bool)
    expert["reward_semantics"] = "sparse_success_indicator"
    expert["terminated"][-1] = True
    expert["truncated"][-1] = False
    expert_path.write_bytes(pickle.dumps(expert))
    perturbed = _episode(perturbed_path)
    perturbed["trajectory_quality"] = "perturbed"
    perturbed["expert_policy_flag"] = np.zeros(3, bool)
    perturbed_path.write_bytes(pickle.dumps(perturbed))

    packed = pack_trajectories([expert_path, perturbed_path], gamma=0.5)
    np.testing.assert_array_equal(packed["episode_starts"], [0, 3])
    np.testing.assert_array_equal(packed["episode_lengths"], [3, 3])
    np.testing.assert_array_equal(packed["expert_policy_flag"], [1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(packed["reward_valid"], [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(packed["value_valid"], [0, 0, 0, 0, 0, 0])
