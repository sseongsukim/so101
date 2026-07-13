"""Scaled Isaac Factory asset configurations used by the SO-101 tasks."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

FACTORY_ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"
FACTORY_ASSET_SCALE = 0.75

PEG_USD = f"{FACTORY_ASSET_DIR}/factory_peg_8mm.usd"
HOLE_USD = f"{FACTORY_ASSET_DIR}/factory_hole_8mm.usd"
GEAR_BASE_USD = f"{FACTORY_ASSET_DIR}/factory_gear_base.usd"
GEAR_SMALL_USD = f"{FACTORY_ASSET_DIR}/factory_gear_small.usd"
GEAR_MEDIUM_USD = f"{FACTORY_ASSET_DIR}/factory_gear_medium.usd"
GEAR_LARGE_USD = f"{FACTORY_ASSET_DIR}/factory_gear_large.usd"
NUT_USD = f"{FACTORY_ASSET_DIR}/factory_nut_m16.usd"
BOLT_USD = f"{FACTORY_ASSET_DIR}/factory_bolt_m16.usd"

SMALL_CUBE_SIZE = 0.025
LARGE_CUBE_SIZE = 0.04


def cube_asset_cfg(
    prim_path: str,
    *,
    size: float,
    mass: float,
    fixed: bool,
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    """Build an exact, axis-aligned cube from an Isaac Lab primitive."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=(size, size, size),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=fixed,
                kinematic_enabled=fixed,
                max_depenetration_velocity=2.0,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.5,
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def factory_asset_cfg(
    prim_path: str,
    usd_path: str,
    *,
    mass: float,
    fixed: bool,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    scale: float = FACTORY_ASSET_SCALE,
) -> ArticulationCfg:
    """Build a scaled Factory articulation without importing its Franka environment.

    Mass is scaled by ``scale**3`` to preserve the original material density.
    Fixed fixtures have a fixed root. Held assets remain movable and are always
    affected by gravity so they must be physically supported or grasped.
    """
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(scale, scale, scale),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=2.0,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=1,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=fixed,
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass * scale**3),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=pos, rot=rot, joint_pos={}, joint_vel={}),
        actuators={},
    )
