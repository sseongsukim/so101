"""Primitive asset configurations used by the SO-101 StackCube task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

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
