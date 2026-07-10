"""SO-101 tabletop scenes populated with scaled Factory task assets."""

from __future__ import annotations

from isaaclab.utils import configclass

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
    NUT_USD,
    PEG_USD,
    factory_asset_cfg,
)

TASK_X = 0.32
TASK_Y = 0.0
STAGING_X = 0.22
STAGING_Y = -0.09


@configclass
class SO101PegInsertSceneCfg(SO101TabletopSceneCfg):
    """SO-101, table, and the original-scale Factory 8 mm peg and hole."""

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
class SO101GearMeshSceneCfg(SO101TabletopSceneCfg):
    """SO-101, table, and the complete 75%-scale Factory gear set."""

    fixed_asset = factory_asset_cfg(
        "{ENV_REGEX_NS}/FixedAsset",
        GEAR_BASE_USD,
        mass=0.05,
        fixed=True,
        pos=(TASK_X, TASK_Y, ROBOT_BASE_BOTTOM_Z),
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
        pos=(TASK_X, TASK_Y, ROBOT_BASE_BOTTOM_Z),
    )
    large_gear = factory_asset_cfg(
        "{ENV_REGEX_NS}/LargeGearAsset",
        GEAR_LARGE_USD,
        mass=0.019,
        fixed=True,
        pos=(TASK_X, TASK_Y, ROBOT_BASE_BOTTOM_Z),
    )


@configclass
class SO101NutThreadSceneCfg(SO101TabletopSceneCfg):
    """SO-101, table, and the original-scale Factory M16 nut and bolt."""

    fixed_asset = factory_asset_cfg(
        "{ENV_REGEX_NS}/FixedAsset",
        BOLT_USD,
        mass=0.05,
        fixed=True,
        pos=(TASK_X, TASK_Y, ROBOT_BASE_BOTTOM_Z),
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
