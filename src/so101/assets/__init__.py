"""Isaac Lab asset configurations for SO-101."""

from .parallel_gripper import (
    PARALLEL_GRIPPER_CLOSED,
    PARALLEL_GRIPPER_OPEN,
    logical_to_parallel_joint_pos,
    parallel_to_logical_joint_pos,
)
from .so101 import (
    SO101_CFG,
    SO101_CONTACT_GRASP_CFG,
    SO101_NO_CAMERA_CFG,
    SO101_PARALLEL_CFG,
    SO101_PARALLEL_CONTACT_CFG,
)
from .materials import DEFAULT_ROBOT_COLOR, ROBOT_COLORS, set_so101_robot_color

__all__ = [
    "DEFAULT_ROBOT_COLOR",
    "ROBOT_COLORS",
    "SO101_CFG",
    "SO101_CONTACT_GRASP_CFG",
    "SO101_NO_CAMERA_CFG",
    "SO101_PARALLEL_CFG",
    "SO101_PARALLEL_CONTACT_CFG",
    "PARALLEL_GRIPPER_CLOSED",
    "PARALLEL_GRIPPER_OPEN",
    "logical_to_parallel_joint_pos",
    "parallel_to_logical_joint_pos",
    "set_so101_robot_color",
]
