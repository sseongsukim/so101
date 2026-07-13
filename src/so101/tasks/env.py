"""Minimal SO-101 Direct environments for scaled contact-task assets.

These environments intentionally use the SO-101 scene and joint-position
control. They do not inherit from Isaac Lab's Franka Factory environment.
"""

from __future__ import annotations

import math
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
        self.jaw_contact_sensor = self.scene["jaw_contact"]
        ee_body_ids, ee_body_names = self.robot.find_bodies("gripper")
        if len(ee_body_ids) != 1:
            raise RuntimeError(
                "Expected exactly one SO-101 end-effector body named 'gripper', "
                f"found {ee_body_names}."
            )
        self._ee_body_idx = ee_body_ids[0]
        jaw_body_ids, jaw_body_names = self.robot.find_bodies("jaw")
        if len(jaw_body_ids) != 1:
            raise RuntimeError(
                "Expected exactly one SO-101 moving gripper body named 'jaw', "
                f"found {jaw_body_names}."
            )
        self._jaw_body_idx = jaw_body_ids[0]
        self._fixed_fingertip_offset = torch.tensor(
            FIXED_FINGERTIP_OFFSET, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self._moving_fingertip_offset = torch.tensor(
            MOVING_FINGERTIP_OFFSET, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self._held_base_offset = torch.tensor(
            cfg.held_base_offset, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self._target_offset = torch.tensor(
            cfg.target_offset, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self._joint_targets = self.robot.data.default_joint_pos.clone()
        self._successes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._has_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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

    @staticmethod
    def _canonicalize_quat(quat: torch.Tensor) -> torch.Tensor:
        """Use a unique quaternion sign so q and -q do not alternate in observations."""
        return quat * torch.where(quat[:, :1] < 0.0, -1.0, 1.0)

    @staticmethod
    def _relative_quat(target_quat: torch.Tensor, current_quat: torch.Tensor) -> torch.Tensor:
        relative = math_utils.quat_mul(math_utils.quat_conjugate(target_quat), current_quat)
        return SO101TaskEnv._canonicalize_quat(relative)

    @staticmethod
    def _apply_local_offset(
        root_pos: torch.Tensor, root_quat: torch.Tensor, local_offset: torch.Tensor
    ) -> torch.Tensor:
        offset = local_offset.expand(root_pos.shape[0], -1)
        return root_pos + math_utils.quat_apply(root_quat, offset)

    def _get_task_frames(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return held assembly frame and its target frame in world coordinates."""
        held_root_pos = self.held_asset.data.root_pos_w
        held_root_quat = self.held_asset.data.root_quat_w
        target_root_pos = self.fixed_asset.data.root_pos_w
        target_root_quat = self.fixed_asset.data.root_quat_w

        held_pos = self._apply_local_offset(held_root_pos, held_root_quat, self._held_base_offset)
        target_pos = self._apply_local_offset(target_root_pos, target_root_quat, self._target_offset)
        return held_pos, held_root_quat, target_pos, target_root_quat

    def _get_state_dict(self) -> dict[str, torch.Tensor]:
        """Build the shared 38-D state used by both the teacher policy and critic."""
        held_pos, held_quat, target_pos, target_quat = self._get_task_frames()

        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat = self.robot.data.body_quat_w[:, self._ee_body_idx]

        return {
            "joint_pos": self.robot.data.joint_pos,
            "joint_vel": self.robot.data.joint_vel,
            # Position errors intentionally stay in the global frame, matching
            # Isaac Factory's position-difference observations.
            "ee_pos_rel_target": ee_pos - target_pos,
            "ee_quat_rel_target": self._relative_quat(target_quat, ee_quat),
            "ee_linvel": self.robot.data.body_lin_vel_w[:, self._ee_body_idx],
            "ee_angvel": self.robot.data.body_ang_vel_w[:, self._ee_body_idx],
            "held_pos_rel_target": held_pos - target_pos,
            "held_quat_rel_target": self._relative_quat(target_quat, held_quat),
            # At observation time this is the absolute joint target applied by
            # the preceding environment step. The last entry is the Jaw target.
            "previous_action": self.actions,
        }

    def _get_observations(self) -> dict[str, torch.Tensor]:
        if self.cfg.task_name == "stack_cube":
            cube_a_pos = self.held_asset.data.root_pos_w
            cube_a_quat = self.held_asset.data.root_quat_w
            cube_b_pos = self.fixed_asset.data.root_pos_w
            ee_pos, _, _ = self._get_grasp_points()
            ee_quat = self.robot.data.body_quat_w[:, self._ee_body_idx]
            state = torch.cat(
                (
                    cube_a_quat,
                    cube_a_pos,
                    cube_b_pos - cube_a_pos,
                    ee_pos,
                    ee_quat,
                    self.robot.data.joint_pos,
                    self.robot.data.joint_vel,
                ),
                dim=-1,
            )
            return {"state": state}

        state_dict = self._get_state_dict()
        state = torch.cat(tuple(state_dict.values()), dim=-1)
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

    def _get_successes(self) -> torch.Tensor:
        """Evaluate Isaac Factory-style task success without ending the episode."""
        if self.cfg.task_name == "stack_cube":
            return self._get_stack_cube_metrics()["stack_success"]

        held_pos, _, target_pos, _ = self._get_task_frames()
        xy_dist = torch.linalg.vector_norm(held_pos[:, :2] - target_pos[:, :2], dim=-1)
        z_disp = held_pos[:, 2] - target_pos[:, 2]
        if self.cfg.check_success_height_absolute:
            z_disp = torch.abs(z_disp)
        successes = torch.logical_and(
            xy_dist < self.cfg.success_xy_threshold,
            z_disp < self.cfg.success_height_threshold,
        )

        if self.cfg.check_success_rotation:
            ee_quat = self.robot.data.body_quat_w[:, self._ee_body_idx]
            _, _, ee_yaw = math_utils.euler_xyz_from_quat(ee_quat, wrap_to_2pi=True)
            # Match Factory's nut-thread yaw convention: angles above 235 deg
            # are represented as negative angles before checking progress.
            ee_yaw = torch.where(
                ee_yaw > math.radians(235.0),
                ee_yaw - 2 * torch.pi,
                ee_yaw,
            )
            successes = torch.logical_and(successes, ee_yaw < self.cfg.ee_success_yaw)

        return successes

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
        cube_a_align_cube_b = (
            torch.linalg.vector_norm(cube_a_to_cube_b[:, :2], dim=-1)
            < self.cfg.success_xy_threshold
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
        if self.cfg.task_name == "stack_cube":
            metrics = self._step_stack_cube_metrics
            if metrics is None:
                # Keep direct/manual calls to this hook correct outside the
                # standard DirectRLEnv.step() call order.
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

        held_pos, _, target_pos, _ = self._get_task_frames()
        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx]

        # Phase 0 contains no orientation objective. It only guides the end
        # effector to the held asset and rewards physical jaw contact/lifting.
        reach_distance = torch.linalg.vector_norm(ee_pos - held_pos, dim=-1)
        reach_reward = torch.exp(
            -0.5 * (reach_distance / self.cfg.reach_reward_std).square()
        )

        contact_forces = self.jaw_contact_sensor.data.force_matrix_w
        if contact_forces is None:
            raise RuntimeError(
                "jaw_contact requires filter_prim_paths_expr so filtered contact forces are available."
            )
        jaw_contact_force = torch.linalg.vector_norm(contact_forces, dim=-1).sum(dim=(1, 2))
        jaw_has_contact = jaw_contact_force > self.cfg.contact_force_threshold

        lift_height = self.held_asset.data.root_pos_w[:, 2] - self._held_initial_root_z
        lift_progress = torch.clamp(lift_height / self.cfg.lift_height_threshold, 0.0, 1.0)
        valid_lift = torch.logical_and(
            jaw_has_contact,
            lift_height >= self.cfg.lift_height_threshold,
        )
        # Once the robot has physically contacted and lifted the asset, keep
        # the target-reaching phase active for the rest of that episode.
        self._has_lifted = torch.logical_or(self._has_lifted, valid_lift)

        pre_lift_reward = (
            self.cfg.reach_reward_weight * reach_reward
            + self.cfg.contact_reward_weight * jaw_has_contact.float()
            + self.cfg.lift_progress_reward_weight * lift_progress * jaw_has_contact.float()
        )

        # Phase 1 starts above the complete Phase-0 reward range. The original
        # task-frame Gaussian then guides the already-lifted asset to its target.
        held_target_dist = torch.linalg.vector_norm(held_pos - target_pos, dim=-1)

        coarse = torch.exp(
            -0.5 * (held_target_dist / self.cfg.distance_reward_coarse_std).square()
        )
        fine = torch.exp(
            -0.5 * (held_target_dist / self.cfg.distance_reward_fine_std).square()
        )
        target_reward = (
            self.cfg.distance_reward_coarse_weight * coarse
            + self.cfg.distance_reward_fine_weight * fine
        )
        post_lift_reward = self.cfg.lifted_bonus + target_reward
        reward = torch.where(self._has_lifted, post_lift_reward, pre_lift_reward)

        # Phase 2 is the task-specific Isaac Factory geometric success. Gate
        # its bonus on a previously valid lift so pushing cannot earn it.
        reward_success = torch.logical_and(self._has_lifted, self._get_successes())
        reward = reward + self.cfg.success_bonus * reward_success.float()

        self.extras["reward_phase"] = (
            self._has_lifted.long() + reward_success.long()
        )
        self.extras["jaw_contact"] = jaw_has_contact
        self.extras["jaw_contact_force"] = jaw_contact_force
        self.extras["lift_height"] = lift_height
        self.extras["has_lifted"] = self._has_lifted
        self.extras["reach_distance"] = reach_distance
        self.extras["reward_reach"] = reach_reward.mean()
        self.extras["held_target_distance"] = held_target_dist
        self.extras["reward_distance_coarse"] = coarse.mean()
        self.extras["reward_distance_fine"] = fine.mean()
        self.extras["reward_success"] = reward_success.float().mean()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.task_name == "stack_cube":
            self._step_stack_cube_metrics = self._get_stack_cube_metrics()
            self._successes = self._step_stack_cube_metrics["stack_success"]
        else:
            self._successes = self._get_successes()
        self.extras["success"] = self._successes
        self.extras["successes"] = self._successes.float().mean()
        terminated = (
            self._successes
            if self.cfg.task_name == "stack_cube"
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
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
        fixed_xy = None
        held_xy = None
        if self.cfg.randomize_asset_poses:
            x_min, x_max = self.cfg.asset_spawn_x_range
            y_min, y_max = self.cfg.asset_spawn_y_range
            fixed_xy = torch.empty((num_reset_envs, 2), device=self.device)
            fixed_xy[:, 0].uniform_(x_min, x_max)
            fixed_xy[:, 1].uniform_(y_min, y_max)

            held_xy = torch.empty_like(fixed_xy)
            held_xy[:, 0].uniform_(x_min, x_max)
            held_xy[:, 1].uniform_(y_min, y_max)
            for _ in range(self.cfg.asset_spawn_max_attempts):
                too_close = (
                    torch.linalg.vector_norm(held_xy - fixed_xy, dim=-1)
                    < self.cfg.asset_spawn_min_separation
                )
                if not torch.any(too_close):
                    break
                num_resamples = int(too_close.sum().item())
                held_xy[too_close, 0] = torch.empty(
                    num_resamples, device=self.device
                ).uniform_(x_min, x_max)
                held_xy[too_close, 1] = torch.empty(
                    num_resamples, device=self.device
                ).uniform_(y_min, y_max)
            else:
                raise RuntimeError(
                    "Could not sample separated tabletop asset poses within "
                    f"{self.cfg.asset_spawn_max_attempts} attempts."
                )
        for asset in (self.fixed_asset, self.held_asset):
            root_state = asset.data.default_root_state[env_ids].clone()
            if self.cfg.randomize_asset_poses:
                sampled_xy = held_xy if asset is self.held_asset else fixed_xy
                if sampled_xy is None:
                    raise RuntimeError("Randomized asset poses were not initialized.")
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
        self._has_lifted[env_ids] = False
        self._step_stack_cube_metrics = None
        if held_initial_root_z is None:
            raise RuntimeError("Held asset reset state was not initialized.")
        self._held_initial_root_z[env_ids] = held_initial_root_z


class SO101TaskVisualEnv(SO101TaskEnv):
    """Visual variant exposing task state and two normalized RGB images."""

    cfg: SO101VisualTaskEnvCfg

    def __init__(self, cfg: SO101VisualTaskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.wrist_camera = self.scene["wrist_camera"]
        self.external_camera = self.scene["external_camera"]

    def _get_observations(self) -> dict[str, torch.Tensor]:
        state = SO101TaskEnv._get_observations(self)["state"]
        wrist_rgb = self.wrist_camera.data.output["rgb"].to(dtype=torch.float32) / 255.0
        external_rgb = self.external_camera.data.output["rgb"].to(dtype=torch.float32) / 255.0
        return {
            "state": state,
            "wrist_image": wrist_rgb,
            "front_image": external_rgb,
        }
