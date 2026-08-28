"""A minimal SO-101 scene with a fixed 50 cm square tabletop."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from so101.assets import SO101_CFG, SO101_PARALLEL_CFG

TABLETOP_LENGTH = 0.50
TABLETOP_WIDTH = 0.50
TABLETOP_THICKNESS = 0.04
# The SO-101 USD root is at z=0, while the lowest point of its base mesh is
# 30.081 mm above that origin.  Put the tabletop surface at the mesh bottom so
# the robot is visually and physically supported without moving its root pose.
ROBOT_BASE_BOTTOM_Z = 0.0300814467

# Camera poses are deliberately kept as module-level constants so they can be
# calibrated against the real setup without changing the scene structure.
# Match the local physical SO-101: its wrist camera is the right-side mirror of
# the workshop asset. The proper camera frame therefore uses the mirrored
# local-X rotation; the USD visual/collider also mirror their local Y geometry.
WRIST_CAMERA_OFFSET_POS = (-0.005, -0.060, -0.062)
WRIST_CAMERA_OFFSET_ROT = (0.9238795, 0.3826834, 0.0, 0.0)
EXTERNAL_CAMERA_POS = (0.78, -0.62, 0.48)
EXTERNAL_CAMERA_ROT = (0.7979214, 0.4994660, 0.1790317, 0.2860119)


def camera_cfg(
    prim_path: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    *,
    width: int = 640,
    height: int = 480,
):
    """Create an RGB-D camera; ``pos`` and ``rot`` use the OpenGL convention."""
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            projection_type="pinhole",
            focal_length=13.5,
            focus_distance=0.25,
        ),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="opengl"),
    )


@configclass
class SO101TabletopSceneCfg(InteractiveSceneCfg):
    """Robot at the origin, mounted on the midpoint of a tabletop edge.

    The tabletop's top surface touches the bottom of the robot base mesh at
    z=0.030081 m. It extends forward along +X, so its bounds are x=[0.0, 0.5]
    and y=[-0.25, 0.25]. Omitting rigid-body properties makes it a static
    collider: it cannot fall or move.
    """

    tabletop = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tabletop",
        spawn=sim_utils.CuboidCfg(
            size=(TABLETOP_LENGTH, TABLETOP_WIDTH, TABLETOP_THICKNESS),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                # Slightly reflective black-painted tabletop. It remains
                # distinct from the robot's true-black printed parts.
                diffuse_color=(0.012, 0.012, 0.012),
                roughness=0.42,
                metallic=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                TABLETOP_LENGTH / 2.0,
                0.0,
                ROBOT_BASE_BOTTOM_Z - TABLETOP_THICKNESS / 2.0,
            ),
        ),
    )

    robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 0.96, 0.90),
            intensity=2500.0,
            angle=25.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9239, 0.2209, -0.2209, 0.2209)),
    )

    fill_light = AssetBaseCfg(
        prim_path="/World/FillLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.82, 1.0),
            intensity=700.0,
        ),
    )


@configclass
class SO101VisualTabletopSceneCfg(SO101TabletopSceneCfg):
    """Tabletop scene variant that adds wrist and external RGB-D cameras."""

    wrist_camera = camera_cfg(
        "{ENV_REGEX_NS}/Robot/gripper/gripper_cam",
        WRIST_CAMERA_OFFSET_POS,
        WRIST_CAMERA_OFFSET_ROT,
    )

    external_camera = camera_cfg(
        "{ENV_REGEX_NS}/ExternalCamera",
        EXTERNAL_CAMERA_POS,
        EXTERNAL_CAMERA_ROT,
    )


@configclass
class SO101ParallelTabletopSceneCfg(SO101TabletopSceneCfg):
    """Tabletop scene using the symmetric parallel-gripper SO-101 asset."""

    robot = SO101_PARALLEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
