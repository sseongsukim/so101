"""Logical SO-101 commands for the parallel-gripper articulation.

The project keeps the calibrated six-value SO-101 command space for the new
asset. These helpers preserve the first five values and only replace the Jaw
angle with two symmetric clamp displacements.
"""

from __future__ import annotations

import math

import torch

# Do not invent zero offsets: preserve the first five command values exactly.
PARALLEL_ARM_OFFSETS = (0.0, 0.0, 0.0, 0.0, 0.0)

# Keep the existing logical Jaw range and use the imported right-clamp limit.
LOGICAL_JAW_LOWER = math.radians(-10.0)
LOGICAL_JAW_UPPER = math.radians(100.0)
PARALLEL_GRIPPER_CLOSED = 0.0
PARALLEL_GRIPPER_OPEN = 0.037


def logical_to_parallel_joint_pos(joint_pos: torch.Tensor) -> torch.Tensor:
    """Convert calibrated 6-D SO-101 positions to parallel-USD positions.

    The input has six entries and the output has seven: five arm joints,
    ``left_clamp``, and ``right_clamp`` (the USD articulation order). Clamp
    targets are in metres and use
    opposite signs so the tips move symmetrically.
    """
    if joint_pos.shape[-1] != 6:
        raise ValueError(f"Expected 6 SO-101 joint values, got shape {joint_pos.shape}.")

    result = joint_pos.new_empty((*joint_pos.shape[:-1], 7))
    result[..., :5] = joint_pos[..., :5]
    offsets = joint_pos.new_tensor(PARALLEL_ARM_OFFSETS) * torch.pi / 180.0
    result[..., :5] += offsets
    opening = (joint_pos[..., 5] - LOGICAL_JAW_LOWER) / (
        LOGICAL_JAW_UPPER - LOGICAL_JAW_LOWER
    )
    right_clamp = opening.clamp(0.0, 1.0) * PARALLEL_GRIPPER_OPEN
    result[..., 5] = -right_clamp
    result[..., 6] = right_clamp
    return result


def parallel_to_logical_joint_pos(joint_pos: torch.Tensor) -> torch.Tensor:
    """Convert parallel-USD positions back to the calibrated 6-D space."""
    if joint_pos.shape[-1] != 7:
        raise ValueError(f"Expected 7 parallel joint values, got shape {joint_pos.shape}.")

    result = joint_pos[..., :6].clone()
    offsets = joint_pos.new_tensor(PARALLEL_ARM_OFFSETS) * torch.pi / 180.0
    result[..., :5] -= offsets
    opening = (
        (joint_pos[..., 6] - PARALLEL_GRIPPER_CLOSED)
        / (PARALLEL_GRIPPER_OPEN - PARALLEL_GRIPPER_CLOSED)
    )
    result[..., 5] = LOGICAL_JAW_LOWER + opening.clamp(0.0, 1.0) * (
        LOGICAL_JAW_UPPER - LOGICAL_JAW_LOWER
    )
    return result
