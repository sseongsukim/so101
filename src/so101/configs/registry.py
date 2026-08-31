"""Explicit task-name to configuration-class registry."""

from typing import TypeAlias

from .base import SO101TaskEnvCfg
from .tasks import SO101StackCubeEnvCfg, SO101VisualStackCubeEnvCfg

EnvCfgType: TypeAlias = type[SO101TaskEnvCfg]

ENV_CFG_REGISTRY: dict[str, EnvCfgType] = {
    "so101-StackCube-v0": SO101StackCubeEnvCfg,
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
