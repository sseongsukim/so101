"""Shared Isaac Lab configuration for SO-101 task environments."""

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from so101.tasks.scenes import SO101StackCubeSceneCfg


@configclass
class SO101TaskEnvCfg(DirectRLEnvCfg):
    """Configuration for the state-based SO-101 StackCube environment."""

    # 120 Hz physics and decimation=4 produce 30 Hz policy/control steps.
    decimation = 4
    episode_length_s = 10.0
    action_space = 6
    observation_space = {"state": 36}
    state_space = 0

    viewer = ViewerCfg(
        eye=(0.0, 0.0, 0.65),
        lookat=(0.30, 0.0, 0.0),
        resolution=(1280, 720),
        origin_type="env",
        env_index=0,
    )

    task_name: str = "stack_cube"
    success_xy_threshold: float = 0.01
    success_height_threshold: float = 0.005

    asset_spawn_x_range: tuple[float, float] = (0.0, 0.0)
    asset_spawn_y_abs_range: tuple[float, float] = (0.0, 0.0)

    stack_distance_gain: float = 10.0
    stack_lift_clearance: float = 0.04
    stack_gripper_away_threshold: float = 0.04

    scene = SO101StackCubeSceneCfg(num_envs=1, env_spacing=1.0, clone_in_fabric=False)
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
    """State observation plus wrist and external RGB images."""

    observation_space = {
        "state": 6,
        "wrist_image": [480, 640, 3],
        "front_image": [480, 640, 3],
    }
