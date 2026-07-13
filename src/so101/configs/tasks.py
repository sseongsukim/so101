"""Task-specific SO-101 environment configurations."""

from isaaclab.utils import configclass

from so101.tasks.assets import FACTORY_ASSET_SCALE
from so101.tasks.scenes import (
    SO101GearMeshSceneCfg,
    SO101NutThreadSceneCfg,
    SO101PegInsertSceneCfg,
    SO101StackCubeSceneCfg,
    SO101VisualGearMeshSceneCfg,
    SO101VisualNutThreadSceneCfg,
    SO101VisualPegInsertSceneCfg,
    SO101VisualStackCubeSceneCfg,
)

from .base import SO101TaskEnvCfg, SO101VisualTaskEnvCfg

STACK_CUBE_MAX_EPISODE_STEPS = 350
STACK_CUBE_CONTROL_HZ = 30.0
STACK_CUBE_EPISODE_LENGTH_S = STACK_CUBE_MAX_EPISODE_STEPS / STACK_CUBE_CONTROL_HZ


@configclass
class SO101PegInsertEnvCfg(SO101TaskEnvCfg):
    scene = SO101PegInsertSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
    episode_length_s = 10.0


@configclass
class SO101GearMeshEnvCfg(SO101TaskEnvCfg):
    scene = SO101GearMeshSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
    episode_length_s = 20.0
    task_name = "gear_mesh"
    held_base_offset = (0.02025 * FACTORY_ASSET_SCALE, 0.0, 0.0)
    target_offset = held_base_offset
    success_xy_threshold = 0.0025 * FACTORY_ASSET_SCALE
    success_height_threshold = 0.020 * FACTORY_ASSET_SCALE * 0.05


@configclass
class SO101NutThreadEnvCfg(SO101TaskEnvCfg):
    scene = SO101NutThreadSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
    episode_length_s = 30.0
    task_name = "nut_thread"
    held_base_offset = (0.0, 0.0, 0.010)
    target_offset = (0.0, 0.0, 0.010 + 0.025 - 0.002 * 1.5)
    success_height_threshold = 0.002 * 0.375
    check_success_rotation = True


@configclass
class SO101StackCubeEnvCfg(SO101TaskEnvCfg):
    scene = SO101StackCubeSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
    observation_space = {"state": 29}
    state_space = 0
    episode_length_s = STACK_CUBE_EPISODE_LENGTH_S
    task_name = "stack_cube"
    target_offset = (0.0, 0.0, 0.0325)
    success_xy_threshold = 0.01
    success_height_threshold = 0.005
    check_success_height_absolute = True
    randomize_asset_poses = True
    asset_spawn_x_range = (0.22, 0.30)
    asset_spawn_y_range = (-0.10, 0.10)
    asset_spawn_min_separation = 0.06
    stack_distance_gain = 20.0
    stack_lift_clearance = 0.02
    stack_gripper_away_threshold = 0.02


@configclass
class SO101VisualPegInsertEnvCfg(SO101VisualTaskEnvCfg):
    scene = SO101VisualPegInsertSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
    episode_length_s = 10.0


@configclass
class SO101VisualGearMeshEnvCfg(SO101VisualTaskEnvCfg):
    scene = SO101VisualGearMeshSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
    episode_length_s = 20.0
    task_name = "gear_mesh"
    held_base_offset = (0.02025 * FACTORY_ASSET_SCALE, 0.0, 0.0)
    target_offset = held_base_offset
    success_xy_threshold = 0.0025 * FACTORY_ASSET_SCALE
    success_height_threshold = 0.020 * FACTORY_ASSET_SCALE * 0.05


@configclass
class SO101VisualNutThreadEnvCfg(SO101VisualTaskEnvCfg):
    scene = SO101VisualNutThreadSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
    episode_length_s = 30.0
    task_name = "nut_thread"
    held_base_offset = (0.0, 0.0, 0.010)
    target_offset = (0.0, 0.0, 0.010 + 0.025 - 0.002 * 1.5)
    success_height_threshold = 0.002 * 0.375
    check_success_rotation = True


@configclass
class SO101VisualStackCubeEnvCfg(SO101VisualTaskEnvCfg):
    scene = SO101VisualStackCubeSceneCfg(
        num_envs=1,
        env_spacing=1.0,
        clone_in_fabric=False,
    )
    observation_space = {
        "state": 29,
        "wrist_image": [480, 640, 3],
        "front_image": [480, 640, 3],
    }
    episode_length_s = STACK_CUBE_EPISODE_LENGTH_S
    task_name = "stack_cube"
    target_offset = (0.0, 0.0, 0.0325)
    success_xy_threshold = 0.01
    success_height_threshold = 0.005
    check_success_height_absolute = True
    randomize_asset_poses = True
    asset_spawn_x_range = (0.22, 0.30)
    asset_spawn_y_range = (-0.10, 0.10)
    asset_spawn_min_separation = 0.06
    stack_distance_gain = 20.0
    stack_lift_clearance = 0.02
    stack_gripper_away_threshold = 0.02
