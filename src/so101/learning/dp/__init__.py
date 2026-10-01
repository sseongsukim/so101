"""Image Diffusion Policy student, ported from robust-rearrangement (ResiP,
github.com/ankile/robust-rearrangement) to plain PyTorch.

    config     DPConfig (paper Hydra config, flattened)
    unet       ConditionalUnet1D              <- src/models/unet.py
    vision     ResNet18+GroupNorm, camera aug <- src/models/vision.py, src/common/vision.py
    policy     loss, DDPM/DDIM, action queue  <- src/behavior/diffusion.py, base.py
    data       pickle -> memmap cache, dataset <- src/dataset/dataset.py
    inference  DPRunner (ACTRunner API)

Train with `scripts/train_dp.py`, evaluate with `scripts/eval_dp_sim.py`.
"""

from so101.learning.dp.config import ACTION, FRONT_KEY, OBS_STATE, WRIST_KEY, DPConfig
from so101.learning.dp.policy import DiffusionPolicy, load_checkpoint, save_checkpoint

__all__ = [
    "DPConfig",
    "DiffusionPolicy",
    "save_checkpoint",
    "load_checkpoint",
    "OBS_STATE",
    "ACTION",
    "FRONT_KEY",
    "WRIST_KEY",
]
