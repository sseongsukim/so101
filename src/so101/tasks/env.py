"""SO-101 Direct environment for StackCube.

These environments intentionally use the SO-101 scene and joint-position
control. They do not inherit from Isaac Lab's Franka Factory environment.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils import math as math_utils

from so101.configs.base import SO101TaskEnvCfg, SO101VisualTaskEnvCfg

from .assets import LARGE_CUBE_SIZE, SMALL_CUBE_SIZE

# Distal grasp points measured from the composed SO-101 collider geometry.
# The fixed finger is part of the gripper rigid body; the moving finger is the
# jaw rigid body. Their midpoint is the task-space grasp EEF used by StackCube.
FIXED_FINGERTIP_OFFSET = (0.0, 0.0, -0.090)
MOVING_FINGERTIP_OFFSET = (0.0, -0.070, 0.0189)


class SO101TaskEnv(DirectRLEnv):
    """Joint-position SO-101 environment with task-centric state observations."""

    cfg: SO101TaskEnvCfg

    def __init__(self, cfg: SO101TaskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.fixed_asset = self.scene["fixed_asset"]
        self.held_asset = self.scene["held_asset"]
        ee_body_ids, ee_body_names = self.robot.find_bodies("gripper")
        jaw_body_ids, jaw_body_names = self.robot.find_bodies("jaw")
        if len(ee_body_ids) != 1 or len(jaw_body_ids) != 1:
            raise RuntimeError(
                "Could not resolve the SO-101 gripper bodies. "
                f"gripper={ee_body_names}, jaw={jaw_body_names}"
            )
        self._ee_body_idx = ee_body_ids[0]
        self._jaw_body_idx = jaw_body_ids[0]
        self._fixed_fingertip_offset = torch.tensor(
            FIXED_FINGERTIP_OFFSET, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self._moving_fingertip_offset = torch.tensor(
            MOVING_FINGERTIP_OFFSET, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self._joint_targets = self.robot.data.default_joint_pos.clone()
        self._successes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # DirectRLEnv calls _get_dones() immediately before _get_rewards().
        # Cache StackCube metrics only across those two hooks to avoid computing
        # the same fingertip transforms, norms, and tanh terms twice per step.
        self._step_stack_cube_metrics: dict[str, torch.Tensor] | None = None
        self._held_initial_root_z = (
            self.held_asset.data.default_root_state[:, 2] + self.scene.env_origins[:, 2]
        ).clone()

    def _setup_scene(self) -> None:
        """All scene entities are declaratively spawned by the selected config."""

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Absolute joint-position control, matching the SO-101 workshop:
        # each action value is the target joint angle in radians.
        self.actions = actions.clone()
        self._joint_targets = self.actions

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._joint_targets)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        cube_a_pos = self.held_asset.data.root_pos_w
        cube_a_quat = self.held_asset.data.root_quat_w
        cube_b_pos = self.fixed_asset.data.root_pos_w
        cube_b_quat = self.fixed_asset.data.root_quat_w
        ee_pos, _, _ = self._get_grasp_points()
        ee_quat = self.robot.data.body_quat_w[:, self._ee_body_idx]
        state = torch.cat(
            (
                cube_a_quat,
                cube_a_pos,
                cube_b_quat,
                cube_b_pos,
                cube_b_pos - cube_a_pos,
                ee_pos,
                ee_quat,
                self.robot.data.joint_pos,
                self.robot.data.joint_vel,
            ),
            dim=-1,
        )
        return {"state": state}

    def _get_grasp_points(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return grasp midpoint, fixed fingertip, and moving fingertip in world frame."""
        fixed_body_pos = self.robot.data.body_pos_w[:, self._ee_body_idx]
        fixed_body_quat = self.robot.data.body_quat_w[:, self._ee_body_idx]
        moving_body_pos = self.robot.data.body_pos_w[:, self._jaw_body_idx]
        moving_body_quat = self.robot.data.body_quat_w[:, self._jaw_body_idx]
        fixed_offset = self._fixed_fingertip_offset.expand(self.num_envs, -1)
        moving_offset = self._moving_fingertip_offset.expand(self.num_envs, -1)
        fixed_fingertip = fixed_body_pos + math_utils.quat_apply(
            fixed_body_quat, fixed_offset
        )
        moving_fingertip = moving_body_pos + math_utils.quat_apply(
            moving_body_quat, moving_offset
        )
        grasp_midpoint = 0.5 * (fixed_fingertip + moving_fingertip)
        return grasp_midpoint, fixed_fingertip, moving_fingertip

    def _get_stack_cube_metrics(self) -> dict[str, torch.Tensor]:
        """Compute the geometric terms used by the IsaacGym Franka stack task."""
        cube_a_pos = self.held_asset.data.root_pos_w
        cube_b_pos = self.fixed_asset.data.root_pos_w
        ee_pos, fixed_finger_pos, moving_finger_pos = self._get_grasp_points()

        d = torch.linalg.vector_norm(cube_a_pos - ee_pos, dim=-1)
        d_lf = torch.linalg.vector_norm(cube_a_pos - fixed_finger_pos, dim=-1)
        d_rf = torch.linalg.vector_norm(cube_a_pos - moving_finger_pos, dim=-1)
        dist_reward = 1.0 - torch.tanh(
            self.cfg.stack_distance_gain * (d + d_lf + d_rf) / 3.0
        )

        cube_a_size = SMALL_CUBE_SIZE
        cube_b_size = LARGE_CUBE_SIZE
        table_height = self._held_initial_root_z - cube_a_size / 2.0
        cube_a_height = cube_a_pos[:, 2] - table_height
        cube_a_lifted = (
            cube_a_height - cube_a_size
        ) > self.cfg.stack_lift_clearance

        cube_a_to_cube_b = cube_b_pos - cube_a_pos
        target_offset = torch.zeros_like(cube_a_to_cube_b)
        target_offset[:, 2] = (cube_a_size + cube_b_size) / 2.0
        d_ab = torch.linalg.vector_norm(cube_a_to_cube_b + target_offset, dim=-1)
        align_reward = (
            1.0 - torch.tanh(self.cfg.stack_distance_gain * d_ab)
        ) * cube_a_lifted.float()
        dist_reward = torch.maximum(dist_reward, align_reward)

        target_height = cube_b_size + cube_a_size / 2.0
        # The support surface is square, so check each horizontal axis rather
        # than using a circular XY-distance threshold.
        cube_a_align_cube_b = torch.all(
            torch.abs(cube_a_to_cube_b[:, :2]) < self.cfg.success_xy_threshold,
            dim=-1,
        )
        cube_a_on_cube_b = (
            torch.abs(cube_a_height - target_height)
            < self.cfg.success_height_threshold
        )
        gripper_away_from_cube_a = d > self.cfg.stack_gripper_away_threshold
        stack_success = torch.logical_and(
            torch.logical_and(cube_a_align_cube_b, cube_a_on_cube_b),
            gripper_away_from_cube_a,
        )
        return {
            "dist_reward": dist_reward,
            "lift_reward": cube_a_lifted.float(),
            "align_reward": align_reward,
            "stack_success": stack_success,
            "eef_cube_distance": d,
            "cube_target_distance": d_ab,
        }

    def _get_rewards(self) -> torch.Tensor:
        metrics = self._step_stack_cube_metrics
        if metrics is None:
            metrics = self._get_stack_cube_metrics()
        self._step_stack_cube_metrics = None
        shaped_reward = (
            0.1 * metrics["dist_reward"]
            + 1.5 * metrics["lift_reward"]
            + 2.0 * metrics["align_reward"]
        )
        reward = torch.where(
            metrics["stack_success"],
            16.0 * metrics["stack_success"].float(),
            shaped_reward,
        )
        self.extras["reward_distance"] = metrics["dist_reward"]
        self.extras["reward_lift"] = metrics["lift_reward"]
        self.extras["reward_align"] = metrics["align_reward"]
        self.extras["reward_success"] = metrics["stack_success"].float()
        self.extras["reach_distance"] = metrics["eef_cube_distance"]
        self.extras["held_target_distance"] = metrics["cube_target_distance"]
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._step_stack_cube_metrics = self._get_stack_cube_metrics()
        self._successes = self._step_stack_cube_metrics["stack_success"]
        self.extras["success"] = self._successes
        self.extras["successes"] = self._successes.float().mean()
        terminated = (
            self._successes
            if self.cfg.terminate_on_success
            else torch.zeros_like(self._successes)
        )
        time_out = (
            self.episode_length_buf >= self.max_episode_length - 1
            if self.cfg.truncate_on_timeout
            else torch.zeros_like(self._successes)
        )
        # Prefer task termination when success occurs on the final allowed
        # step. This keeps terminal and time-limit transitions unambiguous for
        # both on-policy rollouts and off-policy replay buffers.
        time_out = torch.logical_and(time_out, torch.logical_not(terminated))
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        held_initial_root_z = None
        num_reset_envs = len(env_ids)
        x_min, x_max = self.cfg.asset_spawn_x_range
        y_abs_min, y_abs_max = self.cfg.asset_spawn_y_abs_range
        if x_min >= x_max or y_abs_min <= 0.0 or y_abs_min >= y_abs_max:
            raise ValueError(
                "Asset spawn ranges require x_min < x_max and "
                "0 < y_abs_min < y_abs_max."
            )

        fixed_xy = torch.empty((num_reset_envs, 2), device=self.device)
        held_xy = torch.empty_like(fixed_xy)
        fixed_xy[:, 0].uniform_(x_min, x_max)
        held_xy[:, 0].uniform_(x_min, x_max)

        # Randomly put the small cube on one side of the robot and force the
        # large cube onto the opposite side. Negative Y is left, positive Y
        # is right; the center strip is never used.
        held_on_left = torch.rand(num_reset_envs, device=self.device) < 0.5
        held_y_magnitude = torch.empty(num_reset_envs, device=self.device).uniform_(
            y_abs_min, y_abs_max
        )
        fixed_y_magnitude = torch.empty(num_reset_envs, device=self.device).uniform_(
            y_abs_min, y_abs_max
        )
        held_xy[:, 1] = torch.where(
            held_on_left, -held_y_magnitude, held_y_magnitude
        )
        fixed_xy[:, 1] = torch.where(
            held_on_left, fixed_y_magnitude, -fixed_y_magnitude
        )

        for asset in (self.fixed_asset, self.held_asset):
            root_state = asset.data.default_root_state[env_ids].clone()
            sampled_xy = held_xy if asset is self.held_asset else fixed_xy
            root_state[:, :2] = sampled_xy

            # Keep each cube flat on the table and randomize only its yaw.
            yaw = torch.empty(num_reset_envs, device=self.device).uniform_(
                -torch.pi, torch.pi
            )
            root_state[:, 3] = torch.cos(yaw * 0.5)
            root_state[:, 4:6] = 0.0
            root_state[:, 6] = torch.sin(yaw * 0.5)
            root_state[:, :3] += self.scene.env_origins[env_ids]
            asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
            asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
            asset.reset(env_ids)
            if asset is self.held_asset:
                held_initial_root_z = root_state[:, 2]

        self._joint_targets[env_ids] = joint_pos
        self.actions[env_ids] = joint_pos
        self._successes[env_ids] = False
        self._step_stack_cube_metrics = None
        if held_initial_root_z is None:
            raise RuntimeError("Held asset reset state was not initialized.")
        self._held_initial_root_z[env_ids] = held_initial_root_z


class SO101TaskVisualEnv(SO101TaskEnv):
    """StackCube variant exposing normalized wrist and external RGB images."""

    cfg: SO101VisualTaskEnvCfg

    def __init__(self, cfg: SO101VisualTaskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.wrist_camera = self.scene["wrist_camera"]
        self.external_camera = self.scene["external_camera"]

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Keep visual observations deployable on the real robot: only encoder
        # positions and camera images are observable outside simulation.
        state = self.robot.data.joint_pos
        wrist_rgb = self.wrist_camera.data.output["rgb"].to(dtype=torch.float32) / 255.0
        external_rgb = self.external_camera.data.output["rgb"].to(dtype=torch.float32) / 255.0
        return {
            "state": state,
            "wrist_image": wrist_rgb,
            "front_image": external_rgb,
        }
