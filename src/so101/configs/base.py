"""Shared Isaac Lab configuration for SO-101 task environments."""

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from so101.tasks.scenes import SO101PegInsertSceneCfg


@configclass
class SO101TaskEnvCfg(DirectRLEnvCfg):
    """Common configuration for state-based SO-101 manipulation tasks."""

    # 120 Hz physics and decimation=4 produce 30 Hz policy/control steps.
    decimation = 4
    episode_length_s = 10.0
    action_space = 6
    observation_space = {"state": 38}
    state_space = 0

    viewer = ViewerCfg(
        eye=(0.82, -0.68, 0.50),
        lookat=(0.27, 0.0, 0.10),
        resolution=(1280, 720),
        origin_type="env",
        env_index=0,
    )

    task_name: str = "peg_insert"
    held_base_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    success_xy_threshold: float = 0.0025
    success_height_threshold: float = 0.025 * 0.04
    check_success_rotation: bool = False
    check_success_height_absolute: bool = False
    ee_success_yaw: float = 0.0

    randomize_asset_poses: bool = False
    asset_spawn_x_range: tuple[float, float] = (0.0, 0.0)
    asset_spawn_y_range: tuple[float, float] = (0.0, 0.0)
    asset_spawn_min_separation: float = 0.0
    asset_spawn_max_attempts: int = 100

    stack_distance_gain: float = 10.0
    stack_lift_clearance: float = 0.04
    stack_gripper_away_threshold: float = 0.04

    reach_reward_std: float = 0.05
    reach_reward_weight: float = 0.5
    contact_reward_weight: float = 0.25
    lift_progress_reward_weight: float = 0.25
    contact_force_threshold: float = 0.1
    lift_height_threshold: float = 0.01
    lifted_bonus: float = 1.0
    success_bonus: float = 2.0

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
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=0.8,
        ),
    )


@configclass
class SO101VisualTaskEnvCfg(SO101TaskEnvCfg):
    """Shared state-plus-images observation configuration."""

    observation_space = {
        "state": 38,
        "wrist_image": [480, 640, 3],
        "front_image": [480, 640, 3],
    }
    state_space = 0
