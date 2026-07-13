"""SO-101 tabletop scenes populated with scaled Factory task assets."""

from __future__ import annotations

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from so101.assets import SO101_CONTACT_GRASP_CFG
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
    BOLT_USD,
    FACTORY_ASSET_SCALE,
    GEAR_BASE_USD,
    GEAR_LARGE_USD,
    GEAR_MEDIUM_USD,
    GEAR_SMALL_USD,
    HOLE_USD,
    LARGE_CUBE_SIZE,
    NUT_USD,
    PEG_USD,
    SMALL_CUBE_SIZE,
    cube_asset_cfg,
    factory_asset_cfg,
)

TASK_X = 0.32
TASK_Y = 0.0
NEAR_TASK_X = 0.28
STAGING_X = 0.22
STAGING_Y = -0.09

# StackCube-specific third-person view. It looks from the front-right side at
# the center of the randomized cube workspace (x=0.26, y=0.0) and slightly
# above the tabletop (z=0.06) so both loose cubes and a completed stack remain
# visible. The quaternion uses CameraCfg's OpenGL convention.
STACK_CUBE_EXTERNAL_CAMERA_POS = (0.62, -0.50, 0.42)
STACK_CUBE_EXTERNAL_CAMERA_ROT = (
    0.8254465,
    0.4737144,
    0.1527951,
    0.2662452,
)
STACK_CUBE_RENDER_CAMERA_POS = (0.68, -0.58, 0.46)
STACK_CUBE_RENDER_CAMERA_ROT = (
    0.8204539,
    0.4814836,
    0.1560243,
    0.2658672,
)

FACTORY_DEFAULT_JOINT_POS = {
    "Rotation": -0.0685,
    "Pitch": -1.3674,
    "Elbow": 1.3919,
    "Wrist_Pitch": 1.0408,
    "Wrist_Roll": -0.0211,
    "Jaw": 0.0808,
}


def jaw_contact_cfg(held_body_name: str | None) -> ContactSensorCfg:
    """Create a jaw sensor filtered to one Factory articulation rigid body."""
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
            joint_pos=FACTORY_DEFAULT_JOINT_POS
        ),
    )


@configclass
class SO101PegInsertSceneCfg(SO101TaskSceneCfg):
    """SO-101, table, and the original-scale Factory 8 mm peg and hole."""

    jaw_contact = jaw_contact_cfg("forge_round_peg_8mm")

    fixed_asset = factory_asset_cfg(
        "{ENV_REGEX_NS}/FixedAsset",
        HOLE_USD,
        mass=0.05,
        fixed=True,
        pos=(TASK_X, TASK_Y, ROBOT_BASE_BOTTOM_Z),
        scale=1.0,
    )
    held_asset = factory_asset_cfg(
        "{ENV_REGEX_NS}/HeldAsset",
        PEG_USD,
        mass=0.019,
        fixed=False,
        # The peg USD origin is at its bottom face (local z=0), not its center.
        pos=(STAGING_X, STAGING_Y, ROBOT_BASE_BOTTOM_Z),
        scale=1.0,
    )


@configclass
class SO101GearMeshSceneCfg(SO101TaskSceneCfg):
    """SO-101, table, and the complete 75%-scale Factory gear set."""

    jaw_contact = jaw_contact_cfg("factory_gear_medium")

    fixed_asset = factory_asset_cfg(
        "{ENV_REGEX_NS}/FixedAsset",
        GEAR_BASE_USD,
        mass=0.05,
        fixed=True,
        pos=(NEAR_TASK_X, TASK_Y, ROBOT_BASE_BOTTOM_Z),
    )
    held_asset = factory_asset_cfg(
        "{ENV_REGEX_NS}/HeldAsset",
        GEAR_MEDIUM_USD,
        mass=0.012,
        fixed=False,
        # Medium-gear USD has a +5 mm local-Z bottom offset. Compensate for
        # that offset after scaling so the gear rests on the tabletop.
        pos=(STAGING_X, STAGING_Y, ROBOT_BASE_BOTTOM_Z - 0.005 * FACTORY_ASSET_SCALE),
    )
    small_gear = factory_asset_cfg(
        "{ENV_REGEX_NS}/SmallGearAsset",
        GEAR_SMALL_USD,
        mass=0.019,
        fixed=True,
        pos=(NEAR_TASK_X, TASK_Y, ROBOT_BASE_BOTTOM_Z),
    )
    large_gear = factory_asset_cfg(
        "{ENV_REGEX_NS}/LargeGearAsset",
        GEAR_LARGE_USD,
        mass=0.019,
        fixed=True,
        pos=(NEAR_TASK_X, TASK_Y, ROBOT_BASE_BOTTOM_Z),
    )


@configclass
class SO101NutThreadSceneCfg(SO101TaskSceneCfg):
    """SO-101, table, and the original-scale Factory M16 nut and bolt."""

    jaw_contact = jaw_contact_cfg("factory_nut_loose")

    fixed_asset = factory_asset_cfg(
        "{ENV_REGEX_NS}/FixedAsset",
        BOLT_USD,
        mass=0.05,
        fixed=True,
        pos=(NEAR_TASK_X, TASK_Y, ROBOT_BASE_BOTTOM_Z),
        scale=1.0,
    )
    held_asset = factory_asset_cfg(
        "{ENV_REGEX_NS}/HeldAsset",
        NUT_USD,
        mass=0.03,
        fixed=False,
        # Nut USD geometry starts at local z=+10 mm, so its root must sit
        # 10 mm below the tabletop for the geometry bottom to touch it.
        pos=(STAGING_X, STAGING_Y, ROBOT_BASE_BOTTOM_Z - 0.010),
        scale=1.0,
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


class _VisualCamerasMixin:
    """Camera fields shared by visual task scenes."""

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
class SO101VisualPegInsertSceneCfg(_VisualCamerasMixin, SO101PegInsertSceneCfg):
    """Peg-insert scene with wrist and external cameras."""


@configclass
class SO101VisualGearMeshSceneCfg(_VisualCamerasMixin, SO101GearMeshSceneCfg):
    """Gear-mesh scene with wrist and external cameras."""


@configclass
class SO101VisualNutThreadSceneCfg(_VisualCamerasMixin, SO101NutThreadSceneCfg):
    """Nut-thread scene with wrist and external cameras."""


@configclass
class SO101VisualStackCubeSceneCfg(_VisualCamerasMixin, SO101StackCubeSceneCfg):
    """Stack-cube scene with wrist and external cameras."""

    external_camera = camera_cfg(
        "{ENV_REGEX_NS}/ExternalCamera",
        STACK_CUBE_EXTERNAL_CAMERA_POS,
        STACK_CUBE_EXTERNAL_CAMERA_ROT,
    )


@configclass
class SO101RenderStackCubeSceneCfg(SO101StackCubeSceneCfg):
    """StackCube scene with one task-evaluation camera for image capture."""

    render_camera = camera_cfg(
        "{ENV_REGEX_NS}/RenderCamera",
        STACK_CUBE_RENDER_CAMERA_POS,
        STACK_CUBE_RENDER_CAMERA_ROT,
        width=1280,
        height=720,
    )
