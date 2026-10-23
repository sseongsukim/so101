"""SO101 visual SimDist configuration; adapted from adjacent PushT torch port.

See docs/simdist-sol-contract-20261007.md for deliberate SO101 differences.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorldModelConfig:
    # --- input / output structure (the only per-system part) ---
    proprio_obs_dim: int = 6
    action_dim: int = 6

    # --- windowing: H observations of history, T steps of prediction ---
    history_length: int = 3
    prediction_length: int = 8

    # SO101 cameras are RGB uint8; deterministic resize, random initialized CNN.
    image_size: int = 64
    stored_image_shape: tuple[int, int] = (240, 320)
    tick_seconds: float = 1.0 / 30.0
    gamma: float = 0.99
    task_horizon_ticks: int = 300
    value_objective: str = "finite_horizon_dense_mc_v2"
    # Experimental residual variant: predict a correction to the exact
    # time-aware persistence latent instead of an unconstrained absolute latent.
    residual_dynamics: bool = False
    # Optional ResNet18 visual features transferred from the existing DP student.
    visual_encoder_mode: str = "small_cnn"
    # Auxiliary physical prediction of measured next joint positions.
    joint_dynamics: bool = False
    # Re-encode predicted joint states with current camera features for heads.
    joint_conditioned_latents: bool = False
    # Predict camera encoder features as action-conditioned deltas around current RGB.
    visual_dynamics: bool = False

    # --- widths / regularization ---
    latent_dim: int = 64
    mlp_dropout: float = 0.2
    attention_dropout: float = 0.05

    # --- encoder ---
    enc_mlp_hidden_size_factor: int = 2
    enc_proprio_obs_layers: int = 2
    enc_action_layers: int = 0
    enc_latent_layers: int = 2

    # --- input embeddings ---
    emb_mlp_hidden_size_factor: int = 2
    emb_future_acts_layers: int = 0
    emb_reward_layers: int = 0
    emb_value_layers: int = 0

    # --- latent dynamics (causal transformer decoder over future actions) ---
    dyn_layers: int = 2
    dyn_heads: int = 8
    dyn_mlp_hidden_size_factor: int = 4

    # --- reward head (non-causal transformer encoder + MLP decoder) ---
    reward_attn_layers: int = 1
    reward_attn_heads: int = 4
    reward_attn_mlp_hidden_size_factor: int = 4
    reward_dec_layers: int = 0
    reward_dec_mlp_hidden_size_factor: int = 2

    # --- value head ---
    value_attn_layers: int = 1
    value_attn_heads: int = 4
    value_attn_mlp_hidden_size_factor: int = 4
    value_dec_layers: int = 0
    value_dec_mlp_hidden_size_factor: int = 2

    # --- policy (imitation) head, causal like the dynamics ---
    policy_attn_layers: int = 4
    policy_attn_heads: int = 8
    policy_attn_mlp_hidden_size_factor: int = 4
    policy_dec_layers: int = 0
    policy_dec_mlp_hidden_size_factor: int = 2

    # --- loss weights (config/loss/world_model.yaml) ---
    latent_dynamics_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    action_loss_weight: float = 4.0
    joint_dynamics_loss_weight: float = 1.0
    visual_dynamics_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        self.validate()

    @property
    def loss_weights(self) -> dict[str, float]:
        return {
            "latent_dynamics": self.latent_dynamics_loss_weight,
            "reward": self.reward_loss_weight,
            "value": self.value_loss_weight,
            "action": self.action_loss_weight,
            "joint_dynamics": self.joint_dynamics_loss_weight,
            "visual_dynamics": self.visual_dynamics_loss_weight,
        }

    def validate(self) -> None:
        self.stored_image_shape = tuple(self.stored_image_shape)
        if len(self.stored_image_shape) != 2 or min(self.stored_image_shape) < 1:
            raise ValueError("positive stored RGB input dimensions required")
        if self.task_horizon_ticks <= self.history_length or self.value_objective != "finite_horizon_dense_mc_v2":
            raise ValueError("finite-horizon-v2 objective and positive task horizon required")
        if self.proprio_obs_dim != 6 or self.action_dim != 6:
            raise ValueError("SO101 inputs/actions must be six ordered joint radians")
        if self.visual_encoder_mode not in ("small_cnn", "dp_resnet18"):
            raise ValueError("visual_encoder_mode must be small_cnn or dp_resnet18")
        if self.visual_encoder_mode == "dp_resnet18" and tuple(self.stored_image_shape) != (240, 320):
            raise ValueError("DP ResNet transfer expects the DP checkpoint's 240x320 stored RGB frames")
        if self.joint_conditioned_latents and not self.joint_dynamics:
            raise ValueError("joint_conditioned_latents requires joint_dynamics")
        if self.joint_conditioned_latents and self.residual_dynamics:
            raise ValueError("joint_conditioned_latents and residual_dynamics are mutually exclusive predictors")
        if self.visual_dynamics and not self.joint_conditioned_latents:
            raise ValueError("visual_dynamics requires joint_conditioned_latents")
        if self.image_size < 16 or self.tick_seconds <= 0 or not 0 < self.gamma <= 1:
            raise ValueError("invalid camera size/tick/gamma")
        if self.history_length < 1 or self.prediction_length < 1:
            raise ValueError(
                "history_length and prediction_length must be >= 1; got "
                f"{self.history_length} and {self.prediction_length}."
            )
        for name in ("dyn_heads", "reward_attn_heads", "value_attn_heads", "policy_attn_heads"):
            heads = getattr(self, name)
            if self.latent_dim % heads != 0:
                raise ValueError(
                    f"latent_dim ({self.latent_dim}) must be divisible by {name} ({heads})."
                )
