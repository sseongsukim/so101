"""Conversions between LeRobot SO-101 action space and Isaac Lab radians."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .constants import SO101_JOINT_ORDER, SO101_USD_MAPPING


def joint_limits_tensor(device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    joint_names = [joint.split(".")[0] for joint in SO101_JOINT_ORDER]
    joint_mins = torch.tensor(
        [SO101_USD_MAPPING[name]["joint_min"] for name in joint_names],
        dtype=torch.float32,
        device=device,
    )
    joint_maxs = torch.tensor(
        [SO101_USD_MAPPING[name]["joint_max"] for name in joint_names],
        dtype=torch.float32,
        device=device,
    )
    return joint_mins, joint_maxs


def real_action_to_tensor(
    real_action: Mapping[str, float],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [real_action[joint] for joint in SO101_JOINT_ORDER],
        dtype=torch.float32,
        device=device,
    )


def raw_degrees_to_sim_radians(raw_values: torch.Tensor) -> torch.Tensor:
    """Map LeRobot SO-101 degree/percent commands to USD joint radians."""
    joint_mins, joint_maxs = joint_limits_tensor(raw_values.device)
    normalized = torch.zeros_like(raw_values)
    normalized[:-1] = (raw_values[:-1] + 100.0) / 200.0
    normalized[-1] = raw_values[-1] / 100.0
    mapped_deg = joint_mins + normalized * (joint_maxs - joint_mins)
    return mapped_deg * torch.pi / 180.0


def sim_radians_to_raw_degrees(sim_values: torch.Tensor) -> torch.Tensor:
    """Map USD joint radians back to LeRobot SO-101 degree/percent values."""
    joint_mins, joint_maxs = joint_limits_tensor(sim_values.device)
    mapped_deg = sim_values * 180.0 / torch.pi
    normalized = (mapped_deg - joint_mins) / (joint_maxs - joint_mins)
    raw_degrees = torch.zeros_like(normalized)
    raw_degrees[:-1] = normalized[:-1] * 200.0 - 100.0
    raw_degrees[-1] = normalized[-1] * 100.0
    return raw_degrees
