"""Explicit task-name to configuration-class registry."""

from typing import TypeAlias

from .base import SO101TaskEnvCfg
from .tasks import (
    SO101GearMeshEnvCfg,
    SO101NutThreadEnvCfg,
    SO101PegInsertEnvCfg,
    SO101StackCubeEnvCfg,
    SO101VisualGearMeshEnvCfg,
    SO101VisualNutThreadEnvCfg,
    SO101VisualPegInsertEnvCfg,
    SO101VisualStackCubeEnvCfg,
)

EnvCfgType: TypeAlias = type[SO101TaskEnvCfg]

ENV_CFG_REGISTRY: dict[str, EnvCfgType] = {
    "so101-PegInsert-v0": SO101PegInsertEnvCfg,
    "so101-GearMesh-v0": SO101GearMeshEnvCfg,
    "so101-NutThread-v0": SO101NutThreadEnvCfg,
    "so101-StackCube-v0": SO101StackCubeEnvCfg,
    "so101-visual-PegInsert-v0": SO101VisualPegInsertEnvCfg,
    "so101-visual-GearMesh-v0": SO101VisualGearMeshEnvCfg,
    "so101-visual-NutThread-v0": SO101VisualNutThreadEnvCfg,
    "so101-visual-StackCube-v0": SO101VisualStackCubeEnvCfg,
}


def make_env_cfg(
    task_name: str,
    *,
    num_envs: int = 1,
    device: str = "cuda:0",
) -> SO101TaskEnvCfg:
    """Instantiate one task config and apply common runtime overrides."""
    try:
        cfg_cls = ENV_CFG_REGISTRY[task_name]
    except KeyError as exc:
        available = ", ".join(sorted(ENV_CFG_REGISTRY))
        raise ValueError(
            f"Unknown SO-101 task {task_name!r}. Available tasks: {available}"
        ) from exc

    cfg = cfg_cls()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    return cfg
