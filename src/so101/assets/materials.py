# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SO-101 visual material helpers.

The workshop randomizes the SO-101 printed body color by changing the
``material_a_3d_printed`` shader diffuse color.  These helpers keep that
behavior reusable without copying the workshop task/environment.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from pxr import Sdf

import isaaclab.sim as sim_utils

LOGGER = logging.getLogger(__name__)

SO101_PRINTED_MATERIAL_SHADER_PATH = "Looks/material_a_3d_printed/Shader"

# Palette based on the Sim-to-Real SO-101 workshop domain-randomization colors.
ROBOT_COLORS: dict[str, tuple[float, float, float]] = {
    "orange": (0.876, 0.317, 0.132),
    "teal": (0.0, 0.8, 0.502),
    "white": (0.95, 0.95, 0.95),
    # Use true black for the local robot. The previous 0.08 value was a dark
    # gray and appeared noticeably gray under the scene's strong key light.
    "black": (0.0, 0.0, 0.0),
}

DEFAULT_ROBOT_COLOR_NAME = "black"
DEFAULT_ROBOT_COLOR = ROBOT_COLORS[DEFAULT_ROBOT_COLOR_NAME]


def resolve_robot_color(color: str | Sequence[float]) -> tuple[float, float, float]:
    """Resolve a named palette color or RGB tuple to a USD color tuple."""
    if isinstance(color, str):
        try:
            return ROBOT_COLORS[color]
        except KeyError as exc:
            names = ", ".join(sorted(ROBOT_COLORS))
            raise ValueError(f"Unknown SO-101 color '{color}'. Available colors: {names}") from exc

    if len(color) != 3:
        raise ValueError(f"SO-101 color must have exactly three RGB values, got {color!r}")
    return (float(color[0]), float(color[1]), float(color[2]))


def set_so101_robot_color(
    robot_prim_path: str,
    color: str | Sequence[float] = DEFAULT_ROBOT_COLOR,
    material_shader_path: str = SO101_PRINTED_MATERIAL_SHADER_PATH,
) -> None:
    """Set the printed SO-101 body material color on a spawned robot prim.

    Args:
        robot_prim_path: Root prim path of the SO-101 articulation. Regex-style
            paths are supported through Isaac Lab's ``find_matching_prims``.
        color: Palette name from ``ROBOT_COLORS`` or RGB values in 0-1 range.
        material_shader_path: Relative shader path under the robot prim.
    """
    selected_color = resolve_robot_color(color)
    shader_prim_path = f"{robot_prim_path}/{material_shader_path}"
    material_prims = sim_utils.find_matching_prims(shader_prim_path)

    if not material_prims:
        LOGGER.warning("SO-101 material shader not found at %s", shader_prim_path)
        return

    with Sdf.ChangeBlock():
        for material_prim in material_prims:
            attr = material_prim.GetAttribute("inputs:diffuse_color_constant")
            if not attr:
                LOGGER.warning(
                    "SO-101 material shader %s has no inputs:diffuse_color_constant attribute",
                    material_prim.GetPath(),
                )
                continue
            attr.Set(selected_color)


def spawn_so101_usd_with_color(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the SO-101 USD and apply the default printed-body color."""
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    set_so101_robot_color(prim_path, DEFAULT_ROBOT_COLOR)
    return prim
