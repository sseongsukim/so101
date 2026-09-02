"""SO-101 tabletop scene for the StackCube task."""

from __future__ import annotations

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from so101.assets import SO101_CAMERA_CONTACT_GRASP_CFG, SO101_CONTACT_GRASP_CFG
from so101.scenes.tabletop import (
    EXTERNAL_CAMERA_POS,
    EXTERNAL_CAMERA_ROT,
    ROBOT_BASE_BOTTOM_Z,
    SO101TabletopSceneCfg,
    WRIST_CAMERA_OFFSET_POS,
    WRIST_CAMERA_OFFSET_ROT,
    camera_cfg,
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

STACK_CUBE_DEFAULT_JOINT_POS = {
    "Rotation": -0.0685,
    "Pitch": -1.3674,
    "Elbow": 1.3919,
    "Wrist_Pitch": 1.0408,
    "Wrist_Roll": -0.0211,
    "Jaw": 0.0808,
}

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
            joint_pos=STACK_CUBE_DEFAULT_JOINT_POS
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
        pos=(NEAR_TASK_X, TASK_Y, ROBOT_BASE_BOTTOM_Z + LARGE_CUBE_SIZE / 2.0),
        color=(0.15, 0.35, 0.85),
    )
    held_asset = cube_asset_cfg(
        "{ENV_REGEX_NS}/HeldAsset",
        size=SMALL_CUBE_SIZE,
        mass=0.015625,
        fixed=False,
        pos=(STAGING_X, STAGING_Y, ROBOT_BASE_BOTTOM_Z + SMALL_CUBE_SIZE / 2.0),
        color=(0.90, 0.25, 0.12),
    )
@configclass
class SO101VisualStackCubeSceneCfg(SO101StackCubeSceneCfg):
    """StackCube with camera hardware plus wrist and external RGB-D sensors."""

    robot = SO101_CAMERA_CONTACT_GRASP_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=SO101_CAMERA_CONTACT_GRASP_CFG.init_state.replace(
            joint_pos=STACK_CUBE_DEFAULT_JOINT_POS
        ),
    )
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
