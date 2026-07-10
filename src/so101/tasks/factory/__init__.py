"""Gym registration for SO-101 contact-rich manipulation environments."""

import gymnasium as gym

from .env import (
    SO101FactoryEnv,
    SO101FactoryVisualEnv,
    SO101GearMeshEnvCfg,
    SO101NutThreadEnvCfg,
    SO101PegInsertEnvCfg,
    SO101VisualGearMeshEnvCfg,
    SO101VisualNutThreadEnvCfg,
    SO101VisualPegInsertEnvCfg,
)

ENV_SPECS = {
    "so101-PegInsert-v0": (SO101FactoryEnv, SO101PegInsertEnvCfg),
    "so101-GearMesh-v0": (SO101FactoryEnv, SO101GearMeshEnvCfg),
    "so101-NutThread-v0": (SO101FactoryEnv, SO101NutThreadEnvCfg),
    "so101-visual-PegInsert-v0": (SO101FactoryVisualEnv, SO101VisualPegInsertEnvCfg),
    "so101-visual-GearMesh-v0": (SO101FactoryVisualEnv, SO101VisualGearMeshEnvCfg),
    "so101-visual-NutThread-v0": (SO101FactoryVisualEnv, SO101VisualNutThreadEnvCfg),
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
    "SO101FactoryEnv",
    "SO101FactoryVisualEnv",
    "SO101GearMeshEnvCfg",
    "SO101NutThreadEnvCfg",
    "SO101PegInsertEnvCfg",
    "SO101VisualGearMeshEnvCfg",
    "SO101VisualNutThreadEnvCfg",
    "SO101VisualPegInsertEnvCfg",
]
