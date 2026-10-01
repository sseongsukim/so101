"""SO-101 tabletop scene for the StackCube task."""

from __future__ import annotations

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from so101.assets import SO101_CAMERA_CONTACT_GRASP_CFG, SO101_CONTACT_GRASP_CFG
from so101.scenes.tabletop import (
    EXTERNAL_CAMERA_POS,
    EXTERNAL_CAMERA_ROT,
    ROBOT_BASE_BOTTOM_Z,
    ROBOT_ROOT_POS,
    SO101TabletopSceneCfg,
    WRIST_CAMERA_OFFSET_POS,
    WRIST_CAMERA_OFFSET_ROT,
    calibrated_camera_cfg,
)

from .assets import (
    LARGE_CUBE_SIZE,
    SMALL_CUBE_SIZE,
    cube_asset_cfg,
)

TASK_Y = 0.0
NEAR_TASK_X = 0.28
STAGING_X = 0.22
STAGING_Y = -0.09

# Defined outside Isaac-dependent modules so real-robot code can start from
# the same pose the simulated episodes (and so the policies) start from.
from so101.real.constants import STACK_CUBE_DEFAULT_JOINT_POS  # noqa: E402

# Material colors fitted with scripts/match_cube_colors.py so the front camera
# renders the cubes like the real ones (2026-09-29: small cube olive green,
# real pixels ~(71, 86, 46); large cube red, ~(167, 43, 61)). Rendering
# randomization varies appearance around these.
SMALL_CUBE_COLOR = (0.064, 0.111, 0.011)
LARGE_CUBE_COLOR = (0.501, 0.011, 0.037)

def jaw_contact_cfg(held_body_name: str | None) -> ContactSensorCfg:
    """Create a jaw sensor filtered to the held cube."""
    return ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        # PhysX filtered-contact views require the exact rigid-body prim here.
        # The configured HeldAsset prim is only the articulation container.
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/HeldAsset"
            if held_body_name is None
            else f"{{ENV_REGEX_NS}}/HeldAsset/{held_body_name}"
        ],
    )


@configclass
class SO101TaskSceneCfg(SO101TabletopSceneCfg):
    """Tabletop scene with filtered contact sensing on the moving jaw."""

    robot = SO101_CONTACT_GRASP_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=SO101_CONTACT_GRASP_CFG.init_state.replace(
            joint_pos=STACK_CUBE_DEFAULT_JOINT_POS,
            pos=ROBOT_ROOT_POS,
        ),
    )


@configclass
class SO101StackCubeSceneCfg(SO101TaskSceneCfg):
    """SO-101 and exact 2.5 cm and 4 cm cubes for the stacking task."""

    jaw_contact = jaw_contact_cfg(None)

    fixed_asset = cube_asset_cfg(
        "{ENV_REGEX_NS}/FixedAsset",
        size=LARGE_CUBE_SIZE,
        mass=0.064,
        fixed=False,
        pos=(
            ROBOT_ROOT_POS[0] + NEAR_TASK_X,
            ROBOT_ROOT_POS[1] + TASK_Y,
            ROBOT_BASE_BOTTOM_Z + LARGE_CUBE_SIZE / 2.0,
        ),
        color=LARGE_CUBE_COLOR,
    )
    held_asset = cube_asset_cfg(
        "{ENV_REGEX_NS}/HeldAsset",
        size=SMALL_CUBE_SIZE,
        mass=0.015625,
        fixed=False,
        pos=(
            ROBOT_ROOT_POS[0] + STAGING_X,
            ROBOT_ROOT_POS[1] + STAGING_Y,
            ROBOT_BASE_BOTTOM_Z + SMALL_CUBE_SIZE / 2.0,
        ),
        color=SMALL_CUBE_COLOR,
    )
@configclass
class SO101VisualStackCubeSceneCfg(SO101StackCubeSceneCfg):
    """StackCube with camera hardware plus wrist and external RGB-D sensors."""

    robot = SO101_CAMERA_CONTACT_GRASP_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=SO101_CAMERA_CONTACT_GRASP_CFG.init_state.replace(
            joint_pos=STACK_CUBE_DEFAULT_JOINT_POS,
            pos=ROBOT_ROOT_POS,
        ),
    )
    wrist_camera = calibrated_camera_cfg(
        "wrist",
        "{ENV_REGEX_NS}/Robot/gripper/gripper_cam",
        WRIST_CAMERA_OFFSET_POS,
        WRIST_CAMERA_OFFSET_ROT,
    )
    external_camera = calibrated_camera_cfg(
        "front",
        "{ENV_REGEX_NS}/ExternalCamera",
        EXTERNAL_CAMERA_POS,
        EXTERNAL_CAMERA_ROT,
    )
