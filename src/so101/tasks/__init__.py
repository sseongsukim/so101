"""Gym registration for the SO-101 task environments."""

import gymnasium as gym

from so101.configs import (
    SO101GearMeshEnvCfg,
    SO101NutThreadEnvCfg,
    SO101PegInsertEnvCfg,
    SO101StackCubeEnvCfg,
    SO101VisualGearMeshEnvCfg,
    SO101VisualNutThreadEnvCfg,
    SO101VisualPegInsertEnvCfg,
    SO101VisualStackCubeEnvCfg,
)

from .env import SO101TaskEnv, SO101TaskVisualEnv

ENV_SPECS = {
    "so101-PegInsert-v0": (SO101TaskEnv, SO101PegInsertEnvCfg),
    "so101-GearMesh-v0": (SO101TaskEnv, SO101GearMeshEnvCfg),
    "so101-NutThread-v0": (SO101TaskEnv, SO101NutThreadEnvCfg),
    "so101-StackCube-v0": (SO101TaskEnv, SO101StackCubeEnvCfg),
    "so101-visual-PegInsert-v0": (SO101TaskVisualEnv, SO101VisualPegInsertEnvCfg),
    "so101-visual-GearMesh-v0": (SO101TaskVisualEnv, SO101VisualGearMeshEnvCfg),
    "so101-visual-NutThread-v0": (SO101TaskVisualEnv, SO101VisualNutThreadEnvCfg),
    "so101-visual-StackCube-v0": (SO101TaskVisualEnv, SO101VisualStackCubeEnvCfg),
}

for env_id, (env_cls, cfg_cls) in ENV_SPECS.items():
    if env_id not in gym.registry:
        gym.register(
            id=env_id,
            entry_point=env_cls,
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": cfg_cls},
        )

__all__ = [
    "ENV_SPECS",
    "SO101GearMeshEnvCfg",
    "SO101NutThreadEnvCfg",
    "SO101PegInsertEnvCfg",
    "SO101StackCubeEnvCfg",
    "SO101TaskEnv",
    "SO101TaskVisualEnv",
    "SO101VisualGearMeshEnvCfg",
    "SO101VisualNutThreadEnvCfg",
    "SO101VisualPegInsertEnvCfg",
    "SO101VisualStackCubeEnvCfg",
]
