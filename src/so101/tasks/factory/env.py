"""Minimal SO-101 Direct environments for scaled contact-task assets.

These environments intentionally use the SO-101 scene and joint-position
control. They do not inherit from Isaac Lab's Franka Factory environment.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

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
    observation_space = 21
    state_space = 0

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

    # Normalized actions are offsets from the robot's default joint pose.
    joint_action_scale = (0.35, 0.35, 0.35, 0.35, 0.50, 0.15)

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


@configclass
class SO101NutThreadEnvCfg(SO101FactoryEnvCfg):
    scene = SO101NutThreadSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
    episode_length_s = 30.0


@configclass
class SO101VisualFactoryEnvCfg(SO101FactoryEnvCfg):
    """Visual-policy space: proprioception plus both RGB camera streams."""

    observation_space = {
        "proprio": 12,
        "rgb_wrist": [480, 640, 3],
        "rgb_external": [480, 640, 3],
    }


@configclass
class SO101VisualPegInsertEnvCfg(SO101VisualFactoryEnvCfg):
    scene = SO101VisualPegInsertSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
    episode_length_s = 10.0


@configclass
class SO101VisualGearMeshEnvCfg(SO101VisualFactoryEnvCfg):
    scene = SO101VisualGearMeshSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
    episode_length_s = 20.0


@configclass
class SO101VisualNutThreadEnvCfg(SO101VisualFactoryEnvCfg):
    scene = SO101VisualNutThreadSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
    episode_length_s = 30.0


class SO101FactoryEnv(DirectRLEnv):
    """Joint-position SO-101 environment with task assets and a proximity reward.

    The initial implementation provides a valid Gym/Isaac Lab stepping surface,
    scaled contact assets, observations, reset behavior, and a basic proximity
    reward. Task-specific grasp initialization and insertion/threading rewards
    can be layered on this environment without bringing in Franka code.
    """

    cfg: SO101FactoryEnvCfg

    def __init__(self, cfg: SO101FactoryEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.fixed_asset = self.scene["fixed_asset"]
        self.held_asset = self.scene["held_asset"]
        self._joint_action_scale = torch.tensor(cfg.joint_action_scale, device=self.device).unsqueeze(0)
        self._joint_targets = self.robot.data.default_joint_pos.clone()

    def _setup_scene(self) -> None:
        """All scene entities are declaratively spawned by the selected config."""

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)
        self._joint_targets = self.robot.data.default_joint_pos + self.actions * self._joint_action_scale

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._joint_targets)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        env_origins = self.scene.env_origins
        held_pos = self.held_asset.data.root_pos_w - env_origins
        fixed_pos = self.fixed_asset.data.root_pos_w - env_origins
        policy_obs = torch.cat(
            (
                self.robot.data.joint_pos,
                self.robot.data.joint_vel,
                held_pos,
                fixed_pos,
                held_pos - fixed_pos,
            ),
            dim=-1,
        )
        return {"policy": policy_obs}

    def _get_rewards(self) -> torch.Tensor:
        held_pos = self.held_asset.data.root_pos_w
        fixed_pos = self.fixed_asset.data.root_pos_w
        return -torch.linalg.vector_norm(held_pos - fixed_pos, dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        for asset in (self.fixed_asset, self.held_asset):
            root_state = asset.data.default_root_state[env_ids].clone()
            root_state[:, :3] += self.scene.env_origins[env_ids]
            asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
            asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
            asset.reset(env_ids)

        self._joint_targets[env_ids] = joint_pos


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
