"""Diffusion Policy (image) student configuration.

Mirrors the Hydra config that robust-rearrangement (ResiP, Ankile et al.) trains
its image DP with, flattened into one dataclass:

    src/config/base.yaml                         horizons, lr_scheduler
    src/config/actor/diffusion.yaml              DDPM/DDIM, projection_dim, loss_fn
    src/config/actor/diffusion_model/unet.yaml   ConditionalUnet1D
    src/config/vision_encoder/resnet.yaml        resnet18 + use_groupnorm
    src/config/training.yaml                     lrs, ema, clip_grad_norm
    src/config/regularization.yaml               feature regularization defaults
    src/config/experiment/image/real_ol_cotrain.yaml
        -> pretrained encoder, obs 1 / pred 32 / action 8, batch 256,
           encoder_lr 1e-5, warmup 2000 / encoder warmup 50000,
           feature_layernorm, front_camera_dropout 0.1, weight_decay 1e-3

Deliberate deviations (see also the module docstrings):
  * robot_state is 6 joint positions (sim radians), not EE pose + velocity,
    and actions are 6-D absolute joint targets, so there is no rot_6d
    conversion and `include_proprioceptive_pos/ori` has no meaning here.
  * real_ol_cotrain.yaml overrides the encoder to r3m and the backbone to a
    transformer; this port uses resnet18 (ImageNet) + ConditionalUnet1D
    (baseline.yaml / unet.yaml), which is the chosen configuration.
  * image sizes are fields instead of the hard-coded 240x320 -> 224 so the
    CPU tests can use tiny frames; the defaults are the paper's values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

OBS_STATE = "observation.state"
ACTION = "action"
FRONT_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"


@dataclass
class DPConfig:
    # --- data contract -------------------------------------------------------
    state_dim: int = 6
    action_dim: int = 6
    # Order matters for the conditioning vector: the paper concatenates
    # [robot_state, feature(color_image1 = wrist), feature(color_image2 = front)].
    image_keys: tuple[str, ...] = (WRIST_KEY, FRONT_KEY)
    # The key that gets FrontCameraTransform and front_camera_dropout; every
    # other key gets WristCameraTransform and wrist_camera_dropout.
    front_key: str = FRONT_KEY
    image_shape: tuple[int, int] = (240, 320)  # stored (H, W) the transforms expect
    crop_size: int = 224                        # front random/center crop, wrist resize
    front_crop_margin: int = 20                 # FrontCameraTransform `margin`

    # --- horizons (base.yaml) -------------------------------------------------
    obs_horizon: int = 1
    pred_horizon: int = 32
    action_horizon: int = 8
    predict_past_actions: bool = False
    pad_after: bool = True  # data.yaml: pad_after -> action_horizon - 1 padded steps

    # --- diffusion (actor/diffusion.yaml) ------------------------------------
    num_diffusion_iters: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    inference_steps: int = 16
    ddim_eta: float = 0.0            # DiffusionPolicy.eta
    # DiffusionPolicy.warmstart_timestep: each chunk starts from the previous
    # chunk's leftover plan noised to this DDPM step. None = start from pure
    # Gaussian noise (standard DDIM).
    warmstart_timestep: int | None = 50
    loss_fn: str = "MSELoss"

    # --- ConditionalUnet1D (diffusion_model/unet.yaml) -----------------------
    down_dims: tuple[int, ...] = (256, 512, 1024)
    diffusion_step_embed_dim: int = 256
    kernel_size: int = 5
    n_groups: int = 8

    # --- vision (vision_encoder/resnet.yaml + experiment override) ----------
    vision_backbone: str = "resnet18"
    pretrained_backbone: bool = True  # IMAGENET1K_V1
    use_groupnorm: bool = True
    freeze_encoder: bool = False      # True shares one frozen encoder (paper)
    projection_dim: int | None = 128  # actor/diffusion.yaml
    augment_image: bool = True
    # Deviation (not in the paper): per-image Gaussian pixel noise with std
    # drawn from U(0, image_noise_std) in 0-255 units, added after the paper's
    # augmentations during training. The real SO-101 cameras are noisier than
    # the renderer (wrist at 320x240: ~2.4 vs ~1.5 gray levels). 0 disables.
    image_noise_std: float = 4.0

    # --- regularization (regularization.yaml + real_ol_cotrain.yaml) --------
    feature_layernorm: bool = True
    front_camera_dropout: float = 0.1
    wrist_camera_dropout: float = 0.0
    feature_dropout: float = 0.0
    feature_noise: float = 0.0
    proprioception_dropout: float = 0.0
    state_noise: float = 0.0
    weight_decay: float = 1e-3

    # --- optimization (training.yaml + real_ol_cotrain.yaml) ----------------
    actor_lr: float = 1e-4
    encoder_lr: float = 1e-5
    lr_scheduler: str = "cosine"      # diffusers get_scheduler("cosine"): warmup + half cosine to 0
    warmup_steps: int = 2000
    encoder_warmup_steps: int = 50000
    batch_size: int = 256
    clip_grad_norm: bool = False      # False -> max_norm 1001 (only for logging), as bc.py
    ema_use: bool = False
    ema_decay: float = 0.999

    def __post_init__(self) -> None:
        self.image_keys = tuple(self.image_keys)
        self.image_shape = tuple(self.image_shape)
        self.down_dims = tuple(self.down_dims)

    @property
    def cond_dim(self) -> int:
        """Global FiLM conditioning size (without the diffusion-step embedding)."""
        return self.obs_horizon * (self.state_dim + len(self.image_keys) * self.encoding_dim)

    @property
    def encoding_dim(self) -> int:
        if self.projection_dim is not None:
            return self.projection_dim
        return {"resnet18": 512, "resnet34": 512, "resnet50": 2048}[self.vision_backbone]

    @property
    def sequence_length(self) -> int:
        """dataset.ImageDataset.sequence_length."""
        return self.pred_horizon if self.predict_past_actions else self.obs_horizon + self.pred_horizon - 1

    @property
    def first_action_idx(self) -> int:
        return 0 if self.predict_past_actions else self.obs_horizon - 1

    def validate(self) -> None:
        if self.obs_horizon < 1 or self.pred_horizon < 1 or self.action_horizon < 1:
            raise ValueError("horizons must be >= 1")
        start = self.obs_horizon - 1 if self.predict_past_actions else 0
        if start + self.action_horizon > self.pred_horizon:
            raise ValueError(
                f"action_horizon {self.action_horizon} (+{start} past steps) exceeds pred_horizon {self.pred_horizon}"
            )
        factor = 2 ** (len(self.down_dims) - 1)
        if self.pred_horizon % factor:
            raise ValueError(f"pred_horizon {self.pred_horizon} must be divisible by {factor} for {len(self.down_dims)} UNet levels")
        if self.front_key not in self.image_keys:
            raise ValueError(f"front_key {self.front_key!r} not in image_keys {self.image_keys}")
        if self.prediction_type != "epsilon":
            raise ValueError("only prediction_type='epsilon' is ported")
        h, w = self.image_shape
        if self.crop_size > h or self.crop_size > w - 2 * self.front_crop_margin:
            raise ValueError(f"crop_size {self.crop_size} does not fit {self.image_shape} with margin {self.front_crop_margin}")
        if self.warmstart_timestep is not None and not 0 <= self.warmstart_timestep < self.num_diffusion_iters:
            raise ValueError(f"warmstart_timestep must be in [0, {self.num_diffusion_iters})")
        if self.freeze_encoder and not self.pretrained_backbone:
            # ResnetEncoder: "If not pretrained, then freeze must be False"
            raise ValueError("freeze_encoder requires pretrained_backbone")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DPConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown DPConfig fields {sorted(unknown)}")
        return cls(**data)


__all__ = ["DPConfig", "OBS_STATE", "ACTION", "FRONT_KEY", "WRIST_KEY"]
