"""Task-specific SO-101 environment configurations."""

from isaaclab.utils import configclass

from so101.tasks.scenes import SO101StackCubeSceneCfg, SO101VisualStackCubeSceneCfg

from .base import SO101TaskEnvCfg

STACK_CUBE_MAX_EPISODE_STEPS = 1000
STACK_CUBE_CONTROL_HZ = 30.0
STACK_CUBE_EPISODE_LENGTH_S = STACK_CUBE_MAX_EPISODE_STEPS / STACK_CUBE_CONTROL_HZ


@configclass
class SO101StackCubeEnvCfg(SO101TaskEnvCfg):
    scene = SO101StackCubeSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
    observation_space = {"state": 36}
    state_space = 0
    episode_length_s = STACK_CUBE_EPISODE_LENGTH_S
    task_name = "stack_cube"
    success_xy_threshold = 0.018
    success_height_threshold = 0.005
    asset_spawn_x_range = (0.20, 0.30)
    asset_spawn_y_abs_range = (0.055, 0.15)
    stack_distance_gain = 20.0
    stack_lift_clearance = 0.02
    stack_gripper_away_threshold = 0.02


@configclass
class SO101VisualStackCubeEnvCfg(SO101StackCubeEnvCfg):
    """StackCube with wrist and external camera observations."""

    scene = SO101VisualStackCubeSceneCfg(
        num_envs=1, env_spacing=1.0, clone_in_fabric=False
    )
    observation_space = {
        "state": 6,
        "wrist_image": [480, 640, 3],
        "front_image": [480, 640, 3],
    }
