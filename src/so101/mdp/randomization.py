# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SO-101 domain-randomization/event helpers."""

from __future__ import annotations

import torch

from so101.assets.materials import ROBOT_COLORS, set_so101_robot_color


def set_robot_color(
    env,
    env_ids: torch.Tensor | None,
    color: str | tuple[float, float, float] = "black",
    robot_name: str = "robot",
) -> None:
    """Set a spawned SO-101 robot color from an Isaac Lab event term."""
    del env_ids
    robot = env.scene[robot_name]
    set_so101_robot_color(robot.cfg.prim_path, color)


def randomize_robot_color(
    env,
    env_ids: torch.Tensor | None,
    color_names: list[str] | None = None,
    robot_name: str = "robot",
) -> None:
    """Randomly set SO-101 color from the workshop palette on reset."""
    del env_ids
    if color_names is None:
        color_names = list(ROBOT_COLORS.keys())
    idx = torch.randint(0, len(color_names), (1,), device="cpu").item()
    set_robot_color(env, None, color_names[idx], robot_name=robot_name)
