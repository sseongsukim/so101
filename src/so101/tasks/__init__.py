"""Gym registration for the SO-101 task environments."""

import gymnasium as gym

from so101.configs import (
    SO101StackCubeEnvCfg,
    SO101VisualStackCubeEnvCfg,
)

from .env import SO101TaskEnv, SO101TaskVisualEnv

ENV_SPECS = {
    "so101-StackCube-v0": (SO101TaskEnv, SO101StackCubeEnvCfg),
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
    "SO101StackCubeEnvCfg",
    "SO101TaskEnv",
    "SO101TaskVisualEnv",
    "SO101VisualStackCubeEnvCfg",
]
