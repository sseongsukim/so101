"""Image Diffusion Policy: observation encoding, DDPM loss, DDIM sampling, action queue.

Mirrors robust-rearrangement:
  * `DiffusionPolicy.compute_loss` / `_normalized_action`   <- src/behavior/diffusion.py
  * obs encoding, regularization, action queue, reset       <- src/behavior/base.py (Actor)
  * `LinearNormalizer` (min_max to [-1, 1])                 <- src/dataset/normalizer.py
  * `SwitchEMA`                                             <- src/models/ema.py
  * optimizers + LR schedules                               <- src/train/bc.py
  * `DDPMNoiseScheduler.add_noise`, `DDIMNoiseScheduler`    <- the diffusers
    DDPMScheduler/DDIMScheduler the paper instantiates with default arguments
    (timestep_spacing="leading", steps_offset=0, set_alpha_to_one=True,
    clip_sample_range=1.0). Re-implemented here so the repo does not depend on
    diffusers; tests/test_dp_policy.py checks them against diffusers when it
    is installed.

Inference, exactly as `DiffusionPolicy._normalized_action`: DDIM with
`inference_steps` (16) steps, eta 0, and a *warm start* -- instead of pure
noise, the chunk starts from the previous chunk's not-yet-executed plan
(shifted by action_horizon; zeros for the first chunk) noised to DDPM step
`warmstart_timestep` (50), and is then denoised over the full DDIM timestep
list (90, 84, ..., 0). That the noise level (50) and the first denoising step
(90) disagree is the paper code's behavior and is kept; set
`warmstart_timestep=None` for standard pure-noise DDIM.

Deliberate deviations:
  * robot_state is 6 joint positions; no rot_6d conversion, no
    include_proprioceptive_pos/ori masking, no relative-action path.
  * normalization happens inside the policy (`compute_loss` takes raw
    state/actions) instead of in the dataset; numerically the same, and the
    min/max stats travel with the checkpoint as buffers.
  * `reset()` also clears the warm-start plan. The paper keeps `prev_naction`
    across episodes, so the first chunk of an episode is warm-started from the
    last episode's leftover plan; here every episode starts like the paper's
    very first one (zeros + noise).
  * images are encoded only when a new chunk is planned; the paper encodes
    every control step and discards the features while the queue is non-empty
    (same actions, less compute).
  * VIB, confusion loss, domain loss rescaling and attention pooling are not
    ported (all off in the chosen configuration).
  * multiple cameras are handled as an ordered `image_keys` list with one
    encoder each (paper: exactly two, encoder1 = wrist, encoder2 = front);
    `freeze_encoder=True` shares one frozen encoder like the paper.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from so101.learning.dp.config import ACTION, OBS_STATE, DPConfig
from so101.learning.dp.unet import ConditionalUnet1D
from so101.learning.dp.vision import FrontCameraTransform, ResnetEncoder, WristCameraTransform

# --------------------------------------------------------------------------- #
# Noise schedulers (diffusers-equivalent subset)
# --------------------------------------------------------------------------- #


def make_betas(num_train_timesteps: int, beta_schedule: str) -> Tensor:
    if beta_schedule == "squaredcos_cap_v2":
        # diffusers betas_for_alpha_bar(alpha_transform_type="cosine", max_beta=0.999)
        def alpha_bar(t: float) -> float:
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = [
            min(1 - alpha_bar((i + 1) / num_train_timesteps) / alpha_bar(i / num_train_timesteps), 0.999)
            for i in range(num_train_timesteps)
        ]
        return torch.tensor(betas, dtype=torch.float32)
    if beta_schedule == "linear":
        return torch.linspace(1e-4, 0.02, num_train_timesteps, dtype=torch.float32)
    raise ValueError(f"unsupported beta_schedule {beta_schedule}")


class DDPMNoiseScheduler:
    """Forward process only -- all training needs (`DDPMScheduler.add_noise`)."""

    def __init__(self, num_train_timesteps: int = 100, beta_schedule: str = "squaredcos_cap_v2"):
        self.num_train_timesteps = num_train_timesteps
        self.betas = make_betas(num_train_timesteps, beta_schedule)
        self.alphas_cumprod = torch.cumprod(1.0 - self.betas, dim=0)

    def add_noise(self, original: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        acp = self.alphas_cumprod.to(device=original.device, dtype=original.dtype)[timesteps.to(original.device)]
        shape = (-1,) + (1,) * (original.ndim - 1)
        return acp.sqrt().view(shape) * original + (1 - acp).sqrt().view(shape) * noise


class DDIMNoiseScheduler(DDPMNoiseScheduler):
    """diffusers DDIMScheduler with default arguments, epsilon prediction."""

    def __init__(self, num_train_timesteps: int = 100, beta_schedule: str = "squaredcos_cap_v2", clip_sample: bool = True):
        super().__init__(num_train_timesteps, beta_schedule)
        self.clip_sample = clip_sample
        self.final_alpha_cumprod = torch.tensor(1.0)  # set_alpha_to_one=True
        self.num_inference_steps: int | None = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

    def set_timesteps(self, num_inference_steps: int) -> None:
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError("num_inference_steps exceeds num_train_timesteps")
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps  # timestep_spacing="leading"
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def _alpha_prod(self, t: int) -> Tensor:
        return self.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod

    def step(self, model_output: Tensor, timestep: int | Tensor, sample: Tensor, eta: float = 0.0, generator=None) -> Tensor:
        """x_t -> x_{t-1} (DDIM paper eq. 12). Returns prev_sample."""
        t = int(timestep)
        prev_t = t - self.num_train_timesteps // self.num_inference_steps
        alpha_t = self._alpha_prod(t).item()
        alpha_prev = self._alpha_prod(prev_t).item()
        beta_t = 1 - alpha_t
        pred_x0 = (sample - beta_t**0.5 * model_output) / alpha_t**0.5
        if self.clip_sample:
            pred_x0 = pred_x0.clamp(-1.0, 1.0)
        variance = (1 - alpha_prev) / beta_t * (1 - alpha_t / alpha_prev)
        std = eta * variance**0.5
        prev = alpha_prev**0.5 * pred_x0 + (1 - alpha_prev - std**2) ** 0.5 * model_output
        if eta > 0:
            noise = torch.randn(model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
            prev = prev + std * noise
        return prev


# --------------------------------------------------------------------------- #
# Normalizer
# --------------------------------------------------------------------------- #


class LinearNormalizer(nn.Module):
    """min_max normalization to [-1, 1] for `observation.state` and `action`.

    Stats schema (also written to stats.json):
        {"observation.state": {"min": [...], "max": [...]}, "action": {...}}
    Constant columns get min-1 / max+1 at fit time (see data.compute_min_max_stats),
    as the paper's LinearNormalizer.fit does.
    """

    KEYS = {OBS_STATE: "state", ACTION: "action"}

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        for key, dim in (("state", state_dim), ("action", action_dim)):
            self.register_buffer(f"{key}_min", -torch.ones(dim))
            self.register_buffer(f"{key}_max", torch.ones(dim))

    def set_stats(self, stats: dict) -> None:
        for key, name in self.KEYS.items():
            for bound in ("min", "max"):
                value = torch.as_tensor(stats[key][bound], dtype=torch.float32)
                getattr(self, f"{name}_{bound}").copy_(value)

    def stats(self) -> dict:
        return {
            key: {bound: getattr(self, f"{name}_{bound}").cpu().tolist() for bound in ("min", "max")}
            for key, name in self.KEYS.items()
        }

    def normalize(self, x: Tensor, key: str) -> Tensor:
        name = self.KEYS[key]
        lo, hi = getattr(self, f"{name}_min"), getattr(self, f"{name}_max")
        return 2 * (x - lo) / (hi - lo) - 1

    def unnormalize(self, x: Tensor, key: str) -> Tensor:
        name = self.KEYS[key]
        lo, hi = getattr(self, f"{name}_min"), getattr(self, f"{name}_max")
        return (x + 1) / 2 * (hi - lo) + lo


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


class DiffusionPolicy(nn.Module):
    def __init__(self, config: DPConfig, stats: dict | None = None):
        super().__init__()
        config.validate()
        self.config = config
        self.normalizer = LinearNormalizer(config.state_dim, config.action_dim)
        if stats is not None:
            self.normalizer.set_stats(stats)

        # Actor._initiate_image_encoder: one encoder per camera, or one shared
        # frozen encoder when freeze_encoder.
        def encoder() -> ResnetEncoder:
            return ResnetEncoder(
                config.vision_backbone,
                pretrained=config.pretrained_backbone,
                use_groupnorm=config.use_groupnorm,
                freeze=config.freeze_encoder,
            )

        shared = encoder() if config.freeze_encoder else None
        self.encoders = nn.ModuleList([shared or encoder() for _ in config.image_keys])
        raw_dim = self.encoders[0].encoding_dim
        self.projections = nn.ModuleList(
            [nn.Linear(raw_dim, config.projection_dim) if config.projection_dim is not None else nn.Identity() for _ in config.image_keys]
        )
        self.layernorms = nn.ModuleList(
            [nn.LayerNorm(config.encoding_dim) if config.feature_layernorm else nn.Identity() for _ in config.image_keys]
        )
        self.transforms = nn.ModuleList(
            [
                FrontCameraTransform(config.image_shape, config.crop_size, config.front_crop_margin,
                                     noise_std=config.image_noise_std)
                if key == config.front_key
                else WristCameraTransform(config.crop_size, noise_std=config.image_noise_std)
                for key in config.image_keys
            ]
        )
        self.camera_dropout = [
            config.front_camera_dropout if key == config.front_key else config.wrist_camera_dropout for key in config.image_keys
        ]

        self.model = ConditionalUnet1D(
            input_dim=config.action_dim,
            global_cond_dim=config.cond_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
        )
        self.train_noise_scheduler = DDPMNoiseScheduler(config.num_diffusion_iters, config.beta_schedule)
        self.inference_noise_scheduler = DDIMNoiseScheduler(config.num_diffusion_iters, config.beta_schedule, config.clip_sample)
        self.loss_fn = getattr(nn, config.loss_fn)(reduction="none")

        self.observations: deque = deque(maxlen=config.obs_horizon)
        self.actions: deque = deque(maxlen=config.action_horizon)
        self.prev_naction: Tensor | None = None

    # --- parameter groups (Actor.actor_parameters / encoder_parameters) ------
    def encoder_parameters(self) -> list[nn.Parameter]:
        return [p for n, p in self.named_parameters() if n.startswith("encoders.")]

    def actor_parameters(self) -> list[nn.Parameter]:
        return [p for n, p in self.named_parameters() if not n.startswith("encoders.")]

    def train(self, mode: bool = True) -> "DiffusionPolicy":
        """Actor.train/eval: augmentations only when training and augment_image."""
        super().train(mode)
        for transform in self.transforms:
            transform.train(mode and self.config.augment_image)
        return self

    # --- observation encoding (Actor._training_obs / _normalized_obs) --------
    def encode_obs(self, nstate: Tensor, images: dict[str, Tensor]) -> Tensor:
        """nstate (B, obs_horizon, state_dim) normalized; images key -> (B, obs_horizon, 3, H, W)
        uint8. Returns the flattened conditioning (B, cond_dim)."""
        cfg = self.config
        B, oh = nstate.shape[:2]
        if self.training and cfg.proprioception_dropout > 0:
            mask = torch.rand(B, oh, 1, device=nstate.device) > cfg.proprioception_dropout
            nstate = nstate * mask

        features = []
        for i, key in enumerate(cfg.image_keys):
            image = images[key]
            image = image.reshape(B * oh, *image.shape[-3:])
            image = self.transforms[i](image)
            features.append(self.projections[i](self.encoders[i](image)).reshape(B, oh, -1))

        # Actor.regularize_features (order kept: dropout, noise, wrist, front)
        if self.training:
            for i in range(len(features)):
                if cfg.feature_dropout > 0:
                    features[i] = F.dropout(features[i], p=cfg.feature_dropout, training=True)
                if cfg.feature_noise:
                    features[i] = features[i] + torch.randn_like(features[i]) * cfg.feature_noise
            for i, p in enumerate(self.camera_dropout):
                if p > 0:
                    mask = torch.rand(B, oh, 1, device=nstate.device) > p
                    features[i] = features[i] * mask
        # LayerNorm after the camera dropout, as in the paper (a dropped camera
        # therefore reaches the UNet as the LayerNorm bias, not zeros).
        features = [ln(f) for ln, f in zip(self.layernorms, features)]

        nobs = torch.cat([nstate, *features], dim=-1).flatten(start_dim=1)
        if self.training and cfg.state_noise:
            nobs = nobs + torch.randn_like(nobs) * cfg.state_noise
        return nobs

    # --- training ------------------------------------------------------------
    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """batch: OBS_STATE (B, obs_horizon, state_dim) raw, ACTION (B, pred_horizon, action_dim)
        raw, image keys (B, obs_horizon, 3, H, W) uint8. Padded chunk steps are in the
        loss (the paper does not mask them)."""
        nstate = self.normalizer.normalize(batch[OBS_STATE].float(), OBS_STATE)
        naction = self.normalizer.normalize(batch[ACTION].float(), ACTION)
        obs_cond = self.encode_obs(nstate, {k: batch[k] for k in self.config.image_keys})

        noise = torch.randn(naction.shape, device=naction.device)
        timesteps = torch.randint(0, self.train_noise_scheduler.num_train_timesteps, (naction.shape[0],), device=naction.device).long()
        noisy_action = self.train_noise_scheduler.add_noise(naction, noise, timesteps)
        noise_pred = self.model(noisy_action, timesteps, global_cond=obs_cond.float())
        loss = self.loss_fn(noise_pred, noise).mean(dim=[1, 2]).mean()
        return loss, {"bc_loss": loss.item()}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        return self.compute_loss(batch)

    # --- inference -----------------------------------------------------------
    def _normalized_action(self, nobs: Tensor) -> Tensor:
        """DiffusionPolicy._normalized_action: warm-started DDIM -> (B, pred_horizon, action_dim)."""
        cfg = self.config
        B = nobs.shape[0]
        shape = (B, cfg.pred_horizon, cfg.action_dim)
        if self.prev_naction is None or self.prev_naction.shape[0] != B or self.prev_naction.device != nobs.device:
            self.prev_naction = torch.zeros(shape, device=nobs.device)
        noise = torch.randn(shape, device=nobs.device)
        scheduler = self.inference_noise_scheduler
        scheduler.set_timesteps(cfg.inference_steps)
        if cfg.warmstart_timestep is None:
            naction = noise
        else:
            t = torch.full((B,), cfg.warmstart_timestep, device=nobs.device, dtype=torch.long)
            naction = scheduler.add_noise(self.prev_naction, noise, t)
        for k in scheduler.timesteps:
            noise_pred = self.model(sample=naction, timestep=k, global_cond=nobs)
            naction = scheduler.step(noise_pred, k, naction, eta=cfg.ddim_eta)
        # Keep the unexecuted remainder to warm-start the next chunk (the tail
        # keeps whatever was there before, as in the paper).
        self.prev_naction[:, : cfg.pred_horizon - cfg.action_horizon] = naction[:, cfg.action_horizon :]
        return naction

    @torch.no_grad()
    def predict_action_chunk(self, state: Tensor, images: dict[str, Tensor]) -> Tensor:
        """state (B, obs_horizon, state_dim) raw; images key -> (B, obs_horizon, 3, H, W) uint8.
        Returns the full (B, pred_horizon, action_dim) un-normalized chunk."""
        nstate = self.normalizer.normalize(state.float(), OBS_STATE)
        nobs = self.encode_obs(nstate, images)
        return self.normalizer.unnormalize(self._normalized_action(nobs), ACTION)

    @torch.no_grad()
    def select_action(self, state: Tensor, images: dict[str, Tensor]) -> Tensor:
        """Actor.action: state (B, state_dim), images key -> (B, 3, H, W) uint8 for the
        current step. Plans a chunk when the queue is empty and returns its next
        (B, action_dim) action; re-plans every action_horizon calls."""
        cfg = self.config
        obs = {OBS_STATE: state, **{k: images[k] for k in cfg.image_keys}}
        self.observations.append(obs)
        while len(self.observations) < cfg.obs_horizon:
            self.observations.append(obs)
        if not self.actions:
            stacked_state = torch.stack([o[OBS_STATE] for o in self.observations], dim=1)
            stacked_images = {k: torch.stack([o[k] for o in self.observations], dim=1) for k in cfg.image_keys}
            chunk = self.predict_action_chunk(stacked_state, stacked_images)
            start = cfg.obs_horizon - 1 if cfg.predict_past_actions else 0
            self.actions.extend(chunk[:, i] for i in range(start, start + cfg.action_horizon))
        return self.actions.popleft()

    def reset(self) -> None:
        self.actions.clear()
        self.observations.clear()
        self.prev_naction = None

    # --- checkpoints ---------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu", *, use_ema: bool = True) -> "DiffusionPolicy":
        ckpt = load_checkpoint(path, device="cpu")
        config = DPConfig.from_dict(ckpt["config"])
        # The checkpoint holds every weight; skip re-reading ImageNet weights
        # (a frozen, shared encoder needs pretrained=True to pass validate()).
        if not config.freeze_encoder:
            config.pretrained_backbone = False
        policy = cls(config)
        state = ckpt["ema_model"] if use_ema and ckpt.get("ema_model") is not None else ckpt["model"]
        policy.load_state_dict(state)
        return policy.to(device).eval()


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #


def cosine_with_warmup(warmup_steps: int, total_steps: int):
    """diffusers get_cosine_schedule_with_warmup (num_cycles=0.5) as a LambdaLR factor."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * 0.5 * progress)))

    return factor


def build_optimizers(policy: DiffusionPolicy, total_steps: int) -> list[tuple[str, torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]]:
    """src/train/bc.py: AdamW for the actor (UNet, projections, LayerNorms) and a
    second AdamW for the encoders, each with its own warmup + cosine schedule,
    both stepped every optimizer step."""
    cfg = policy.config
    groups = [("actor", policy.actor_parameters(), cfg.actor_lr, cfg.warmup_steps)]
    encoder_params = [p for p in policy.encoder_parameters() if p.requires_grad]
    if encoder_params:
        groups.append(("encoder", encoder_params, cfg.encoder_lr, cfg.encoder_warmup_steps))
    out = []
    for name, params, lr, warmup in groups:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
        if cfg.lr_scheduler == "cosine":
            sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_with_warmup(warmup, total_steps))
        elif cfg.lr_scheduler == "constant":
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        else:
            raise ValueError(f"unsupported lr_scheduler {cfg.lr_scheduler}")
        out.append((name, opt, sched))
    return out


class SwitchEMA:
    """src/models/ema.py: shadow = decay * shadow + (1 - decay) * param, every step."""

    def __init__(self, model: nn.Module, decay: float):
        self.model = model
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self) -> None:
        for n, p in self.model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Tensor]:
        """Model state_dict with the EMA weights substituted (buffers from the live model)."""
        state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        state.update({k: v.clone() for k, v in self.shadow.items()})
        return state

    def load_shadow(self, state: dict[str, Tensor]) -> None:
        for n in self.shadow:
            self.shadow[n].copy_(state[n])

    def apply_shadow(self) -> None:
        """Swap the EMA weights in (bc.py does this for test loss and saving)."""
        self._backup = {}
        for n, p in self.model.named_parameters():
            if n in self.shadow:
                self._backup[n] = p.data
                p.data = self.shadow[n].clone()

    def restore(self) -> None:
        for n, p in self.model.named_parameters():
            if n in self._backup:
                p.data = self._backup[n]
        self._backup = {}


def save_checkpoint(path: str | Path, policy: DiffusionPolicy, step: int, **extra) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": policy.state_dict(),
        "config": policy.config.to_dict(),
        "stats": policy.normalizer.stats(),
        "step": int(step),
        **extra,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict:
    ckpt = torch.load(Path(path), map_location=device, weights_only=True)
    ckpt["config"]["image_keys"] = tuple(ckpt["config"]["image_keys"])
    return ckpt
