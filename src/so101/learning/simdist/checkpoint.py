"""Self-contained torch WM checkpoints. JAX is used only by teacher export."""
import torch

from .config import WorldModelConfig
from .model import WorldModel


def load_worldmodel(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("schema") != "so101-visual-simdist-v2":
        raise ValueError("unsupported worldmodel checkpoint contract; v2 requires explicit remaining-time conditioning")
    model = WorldModel(WorldModelConfig(**saved["config"]), saved["stats"])
    state = dict(saved["model"])
    # Experimental joint-head v1 checkpoints predate its frozen-calibration
    # buffer; their mathematically correct default is unit residual scale.
    if model.config.joint_dynamics and "joint_residual_scale" not in state:
        state["joint_residual_scale"] = torch.ones(model.config.prediction_length)
    if model.config.visual_dynamics and "visual_residual_scale" not in state:
        state["visual_residual_scale"] = torch.ones(model.config.prediction_length, 2)
    for name, value in (("reward_calibration_scale", torch.ones(model.config.prediction_length)),
                        ("reward_calibration_bias", torch.zeros(model.config.prediction_length)),
                        ("value_calibration_scale", torch.ones(model.config.prediction_length)),
                        ("value_calibration_bias", torch.zeros(model.config.prediction_length))):
        state.setdefault(name, value)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), saved
