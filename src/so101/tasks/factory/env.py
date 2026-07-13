"""Minimal SO-101 Direct environments for scaled contact-task assets.

These environments intentionally use the SO-101 scene and joint-position
control. They do not inherit from Isaac Lab's Franka Factory environment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from .assets import FACTORY_ASSET_SCALE
from .scenes import (
    SO101GearMeshSceneCfg,
    SO101NutThreadSceneCfg,
    SO101PegInsertSceneCfg,
    SO101VisualGearMeshSceneCfg,
    SO101VisualNutThreadSceneCfg,
    SO101VisualPegInsertSceneCfg,
)


@configclass
class SO101FactoryEnvCfg(DirectRLEnvCfg):
    """Common configuration for the three SO-101 Factory-inspired tasks."""

    decimation = 2
    episode_length_s = 10.0
    action_space = 6
    observation_space = 38
    state_space = 38

    # Task-frame definitions. Offsets are expressed in the corresponding USD
    # asset's local frame and transformed into world coordinates at runtime.
    task_name: str = "peg_insert"
    held_base_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Isaac Factory success criteria, expressed in meters after asset scaling.
    success_xy_threshold: float = 0.0025
    success_height_threshold: float = 0.025 * 0.04
    check_success_rotation: bool = False
    ee_success_yaw: float = 0.0

    # Three reward phases: approach/grasp, move the lifted asset, and success.
    reach_reward_std: float = 0.05
    reach_reward_weight: float = 0.5
    contact_reward_weight: float = 0.25
    lift_progress_reward_weight: float = 0.25
    contact_force_threshold: float = 0.1
    lift_height_threshold: float = 0.01
    lifted_bonus: float = 1.0
    success_bonus: float = 2.0

    # Multi-scale held-to-target reward, enabled only after a valid lift.
    distance_reward_coarse_std: float = 0.05
    distance_reward_fine_std: float = 0.005
    distance_reward_coarse_weight: float = 0.5
    distance_reward_fine_weight: float = 0.5

    scene = SO101PegInsertSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
    sim = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=64,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.004,
        ),
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
    )

    def __post_init__(self) -> None:
        self.viewer.eye = (0.82, -0.68, 0.50)
        self.viewer.lookat = (0.27, 0.0, 0.10)


@configclass
class SO101PegInsertEnvCfg(SO101FactoryEnvCfg):
    scene = SO101PegInsertSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
    episode_length_s = 10.0


@configclass
class SO101GearMeshEnvCfg(SO101FactoryEnvCfg):
    scene = SO101GearMeshSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
    episode_length_s = 20.0
    task_name = "gear_mesh"
    held_base_offset = (0.02025 * FACTORY_ASSET_SCALE, 0.0, 0.0)
    target_offset = held_base_offset
    success_xy_threshold = 0.0025 * FACTORY_ASSET_SCALE
    success_height_threshold = 0.020 * FACTORY_ASSET_SCALE * 0.05


@configclass
class SO101NutThreadEnvCfg(SO101FactoryEnvCfg):
    scene = SO101NutThreadSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
    episode_length_s = 30.0
    task_name = "nut_thread"
    held_base_offset = (0.0, 0.0, 0.010)
    target_offset = (0.0, 0.0, 0.010 + 0.025 - 0.002 * 1.5)
    success_height_threshold = 0.002 * 0.375
    check_success_rotation = True


@configclass
class SO101VisualFactoryEnvCfg(SO101FactoryEnvCfg):
    """Visual-policy space: proprioception plus both RGB camera streams."""

    observation_space = {
        "proprio": 12,
        "rgb_wrist": [480, 640, 3],
        "rgb_external": [480, 640, 3],
    }
    state_space = 0


@configclass
class SO101VisualPegInsertEnvCfg(SO101VisualFactoryEnvCfg):
    scene = SO101VisualPegInsertSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
    episode_length_s = 10.0


@configclass
class SO101VisualGearMeshEnvCfg(SO101VisualFactoryEnvCfg):
    scene = SO101VisualGearMeshSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
    episode_length_s = 20.0
    task_name = "gear_mesh"
    held_base_offset = (0.02025 * FACTORY_ASSET_SCALE, 0.0, 0.0)
    target_offset = held_base_offset
    success_xy_threshold = 0.0025 * FACTORY_ASSET_SCALE
    success_height_threshold = 0.020 * FACTORY_ASSET_SCALE * 0.05


@configclass
class SO101VisualNutThreadEnvCfg(SO101VisualFactoryEnvCfg):
    scene = SO101VisualNutThreadSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
    episode_length_s = 30.0
    task_name = "nut_thread"
    held_base_offset = (0.0, 0.0, 0.010)
    target_offset = (0.0, 0.0, 0.010 + 0.025 - 0.002 * 1.5)
    success_height_threshold = 0.002 * 0.375
    check_success_rotation = True


class SO101FactoryEnv(DirectRLEnv):
    """Joint-position SO-101 environment with task-centric state observations."""

    cfg: SO101FactoryEnvCfg

    def __init__(self, cfg: SO101FactoryEnvCfg, render_mode: str | None = None, **kwargs):
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
        self._held_base_offset = torch.tensor(
            cfg.held_base_offset, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self._target_offset = torch.tensor(
            cfg.target_offset, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self._joint_targets = self.robot.data.default_joint_pos.clone()
        self._successes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._has_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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
        return SO101FactoryEnv._canonicalize_quat(relative)

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
        state_dict = self._get_state_dict()
        state = torch.cat(tuple(state_dict.values()), dim=-1)
        return {"policy": state, "critic": state}

    def _get_successes(self) -> torch.Tensor:
        """Evaluate Isaac Factory-style task success without ending the episode."""
        held_pos, _, target_pos, _ = self._get_task_frames()
        xy_dist = torch.linalg.vector_norm(held_pos[:, :2] - target_pos[:, :2], dim=-1)
        z_disp = held_pos[:, 2] - target_pos[:, 2]
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

    def _get_rewards(self) -> torch.Tensor:
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
        self._successes = self._get_successes()
        self.extras["success"] = self._successes
        self.extras["successes"] = self._successes.float().mean()
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        held_initial_root_z = None
        for asset in (self.fixed_asset, self.held_asset):
            root_state = asset.data.default_root_state[env_ids].clone()
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
        if held_initial_root_z is None:
            raise RuntimeError("Held asset reset state was not initialized.")
        self._held_initial_root_z[env_ids] = held_initial_root_z


class SO101FactoryVisualEnv(SO101FactoryEnv):
    """Visual variant exposing proprioception and two normalized RGB images."""

    cfg: SO101VisualFactoryEnvCfg

    def __init__(self, cfg: SO101VisualFactoryEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.wrist_camera = self.scene["wrist_camera"]
        self.external_camera = self.scene["external_camera"]

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        proprio = torch.cat((self.robot.data.joint_pos, self.robot.data.joint_vel), dim=-1)
        wrist_rgb = self.wrist_camera.data.output["rgb"].to(dtype=torch.float32) / 255.0
        external_rgb = self.external_camera.data.output["rgb"].to(dtype=torch.float32) / 255.0
        return {
            "policy": {
                "proprio": proprio,
                "rgb_wrist": wrist_rgb,
                "rgb_external": external_rgb,
            }
        }
