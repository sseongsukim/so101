"""ACT configuration.

Ported from LeRobot's `configuration_act.py`, but decoupled: instead of the
`PreTrainedConfig` + `input_features`/`output_features` (PolicyFeature dict)
machinery, we declare the input/output SHAPES explicitly. To reuse ACT on a new
robot you now only set `robot_state_dim`, `action_dim`, and `image_keys`.

Everything under "Architecture" / "VAE" / "Training and loss" is copied verbatim
from the original defaults (bimanual ALOHA), so behavior matches upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Batch-dict key convention (was lerobot.utils.constants). A batch fed to the
# model/policy uses these string keys. Kept identical to LeRobot so datasets
# produced for LeRobot load unchanged.
# ---------------------------------------------------------------------------
OBS_STATE = "observation.state"
OBS_ENV_STATE = "observation.environment_state"
OBS_IMAGES = "observation.images"  # internal: a list of image tensors
ACTION = "action"


@dataclass
class ACTConfig:
    """Configuration for the Action Chunking Transformer policy.

    Input/output structure (explicit — replaces the PolicyFeature dicts):
        robot_state_dim: proprioceptive state size, or None if unused.
        action_dim:      action vector size (required).
        env_state_dim:   environment-state size, or None if unused.
        image_keys:      batch keys for camera images, e.g.
                         ("observation.images.wrist", "observation.images.external").
                         Empty means no images. All cameras must share H, W.

    At least one of images / env_state must be present (see `validate()`).
    """

    # --- input / output structure (the only per-robot part) ---
    action_dim: int = 6
    robot_state_dim: int | None = 6
    env_state_dim: int | None = None
    image_keys: tuple[str, ...] = ()

    # --- chunking ---
    n_obs_steps: int = 1          # observations fed in (only 1 supported)
    chunk_size: int = 100         # actions predicted per forward pass
    n_action_steps: int = 100     # actions executed before re-querying (<= chunk_size)

    # --- vision backbone ---
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False

    # --- transformer ---
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Original ACT sets 7 but a bug means only the first decoder layer runs;
    # upstream matches that by using 1. See tonyzhaozh/act#25.
    n_decoder_layers: int = 1

    # --- CVAE ---
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # --- inference ---
    # None disables temporal ensembling. If set, n_action_steps must be 1
    # (policy is queried every step to form the ensemble). ACT paper uses 0.01.
    temporal_ensemble_coeff: float | None = None

    # --- training / loss ---
    dropout: float = 0.1
    kl_weight: float = 10.0
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        self.image_keys = tuple(self.image_keys)
        self.validate_architecture()

    # --- derived presence flags (replace *_feature properties) ---
    @property
    def has_images(self) -> bool:
        return len(self.image_keys) > 0

    @property
    def has_robot_state(self) -> bool:
        return self.robot_state_dim is not None

    @property
    def has_env_state(self) -> bool:
        return self.env_state_dim is not None

    def validate_architecture(self) -> None:
        """Architecture-only checks (upstream runs these in __post_init__)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be a ResNet variant. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling: the policy "
                "must be queried every step to form the ensemble."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})."
            )
        if self.n_obs_steps != 1:
            raise ValueError(f"Only n_obs_steps=1 is supported. Got {self.n_obs_steps}.")

    def validate_features(self) -> None:
        """Input/output presence checks. Called by the policy at build time
        (upstream calls this in ACTPolicy.__init__, not __post_init__)."""
        if not self.has_images and not self.has_env_state:
            raise ValueError("Provide at least one of `image_keys` or `env_state_dim`.")
