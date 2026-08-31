# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab articulation config for the SO-101 arm.

This module intentionally contains only the robot asset/configuration layer.
Task and scene configuration should live in the consuming Isaac Lab environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat

from .materials import spawn_so101_usd_with_color

ASSET_DIR = Path(__file__).resolve().parent
USD_DIR = ASSET_DIR / "usd"
SO101_USD_PATH = USD_DIR / "SO-ARM101-USD-NO-CAMERA.usd"
SO101_CAMERA_USD_PATH = USD_DIR / "SO-ARM101-USD-RIGHT-CAMERA.usd"

SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SO101_USD_PATH),
        func=spawn_so101_usd_with_color,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "Rotation": -0.2736,
            "Pitch": -0.6109,
            "Elbow": -0.0745,
            "Wrist_Pitch": 1.5148,
            "Wrist_Roll": -1.6034,
            "Jaw": -0.1465,
        },
        # Keep the robot base at the world/environment origin.  Environment
        # geometry (for example, the tabletop) is positioned relative to it.
        pos=(0.0, 0.0, 0.0),
        rot=euler_angles_to_quat(np.array([0.0, 0.0, 90.0]), degrees=True),
    ),
    actuators={
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"],
            effort_limit_sim=30,
            stiffness=55,
            damping=0.7,
        ),
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"],
            effort_limit_sim=30,
            stiffness=30,
            damping=0.8,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"],
            effort_limit_sim=30,
            stiffness=25,
            damping=0.7,
        ),
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"],
            effort_limit_sim=30,
            stiffness=12,
            damping=0.5,
        ),
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"],
            effort_limit_sim=30,
            stiffness=7,
            damping=0.5,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["Jaw"],
            effort_limit_sim=30,
            stiffness=4,
            damping=0.3,
        ),
    },
)

SO101_CONTACT_GRASP_CFG = SO101_CFG.copy()
SO101_CONTACT_GRASP_CFG.spawn.activate_contact_sensors = True

SO101_CAMERA_CFG = SO101_CFG.copy()
SO101_CAMERA_CFG.spawn.usd_path = str(SO101_CAMERA_USD_PATH)

SO101_CAMERA_CONTACT_GRASP_CFG = SO101_CAMERA_CFG.copy()
SO101_CAMERA_CONTACT_GRASP_CFG.spawn.activate_contact_sensors = True
