"""Isaac Lab asset configurations for SO-101."""

from .so101 import SO101_CFG, SO101_CONTACT_GRASP_CFG, SO101_NO_CAMERA_CFG
from .materials import DEFAULT_ROBOT_COLOR, ROBOT_COLORS, set_so101_robot_color

__all__ = [
    "DEFAULT_ROBOT_COLOR",
    "ROBOT_COLORS",
    "SO101_CFG",
    "SO101_CONTACT_GRASP_CFG",
    "SO101_NO_CAMERA_CFG",
    "set_so101_robot_color",
]
