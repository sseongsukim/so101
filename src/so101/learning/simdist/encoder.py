"""RGB + measured joint encoder, with interleaved sent-action history.

The CNN is trained from scratch; no downloaded weights or privileged state.
Inputs are uint8 RGB NHWC. Resizing to 64 is a small baseline budget choice.
"""
from __future__ import annotations

import torch
import hashlib
from pathlib import Path
from torch import nn
from torch.nn import functional as F

from .modules import MLP
from ..dp.vision import FrontCameraTransform, ResnetEncoder, WristCameraTransform


def load_dp_vision_checkpoint(encoder: "VisualEncoder", checkpoint: str | Path) -> dict:
    """Transfer the trained DP ResNet trunks, camera projections and norms."""
    if getattr(encoder.front, "mode", None) != "dp_resnet18" or getattr(encoder.wrist, "mode", None) != "dp_resnet18":
        raise ValueError("DP vision weights require visual_encoder_mode='dp_resnet18'")
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved.get("config", {})
    image_keys = list(config.get("image_keys", ()))
    if image_keys != ["observation.images.wrist", "observation.images.front"]:
        raise ValueError(f"unsupported DP camera ordering: {image_keys}")
    if config.get("vision_backbone") != "resnet18" or not config.get("use_groupnorm", False):
        raise ValueError("DP checkpoint must use the expected ResNet18 GroupNorm vision encoder")
    if int(config.get("projection_dim", -1)) != 128:
        raise ValueError("DP checkpoint vision projection must be 128-dimensional")
    state = saved["model"]
    for camera_name, index in (("wrist", 0), ("front", 1)):
        camera = getattr(encoder, camera_name)
        prefix = f"encoders.{index}.model."
        trunk = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if not trunk:
            raise ValueError(f"DP checkpoint is missing {camera_name} ResNet weights")
        camera.backbone.model.load_state_dict(trunk, strict=True)
        camera.projection.load_state_dict({
            key.rsplit(".", 1)[-1]: state[f"projections.{index}.{key.rsplit('.', 1)[-1]}"]
            for key in ("weight", "bias")
        }, strict=True)
        camera.feature_norm.load_state_dict({
            key.rsplit(".", 1)[-1]: state[f"layernorms.{index}.{key.rsplit('.', 1)[-1]}"]
            for key in ("weight", "bias")
        }, strict=True)
    return {"path": str(checkpoint.resolve()), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "training_step": int(saved.get("step", -1)), "camera_order": image_keys,
            "source_has_real_data": True}


class CameraEncoder(nn.Module):
    def __init__(self, dim: int, size: int, stored_shape, *, camera: str, mode: str = "small_cnn"):
        super().__init__()
        self.size = size
        self.stored_shape = tuple(stored_shape)
        self.mode = mode
        self.transferred_features_frozen = False
        if mode == "dp_resnet18":
            self.backbone = ResnetEncoder("resnet18", pretrained=False, use_groupnorm=True)
            self.transform = (FrontCameraTransform(self.stored_shape, crop_size=224, margin=20)
                              if camera == "front" else WristCameraTransform(crop_size=224))
            self.projection = nn.Linear(512, 128)
            self.feature_norm = nn.LayerNorm(128)
            self.to_latent = nn.Linear(128, dim)
            self.transform.eval()
            return
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, dim),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.mode == "dp_resnet18":
            # Match the deterministic DP inference crop; never randomize target latents.
            self.transform.eval()
            if self.transferred_features_frozen:
                self.backbone.eval()
                self.projection.eval()
                self.feature_norm.eval()
        return self

    def freeze_transferred_features(self):
        if self.mode != "dp_resnet18":
            raise ValueError("only DP ResNet camera encoders have transferred feature blocks")
        self.transferred_features_frozen = True
        for module in (self.backbone, self.projection, self.feature_norm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def forward(self, rgb):
        if rgb.ndim < 4 or rgb.shape[-1] != 3 or rgb.dtype != torch.uint8:
            raise ValueError("camera input must be uint8 (..., H, W, 3) RGB")
        if tuple(rgb.shape[-3:-1]) != self.stored_shape:
            raise ValueError(f"RGB must first use collector preprocessing to {self.stored_shape}, got {tuple(rgb.shape[-3:-1])}")
        prefix = rgb.shape[:-3]
        if self.mode == "dp_resnet18":
            x = rgb.reshape(-1, *rgb.shape[-3:]).permute(0, 3, 1, 2).contiguous()
            x = self.transform(x)
            features = self.feature_norm(self.projection(self.backbone(x)))
            return self.to_latent(features).reshape(*prefix, -1)
        x = rgb.reshape(-1, *rgb.shape[-3:]).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, (self.size, self.size), mode="bilinear", align_corners=False)
        return self.net(x).reshape(*prefix, -1)


class VisualEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        D, hidden = config.latent_dim, config.latent_dim * config.enc_mlp_hidden_size_factor
        self.front = CameraEncoder(D, config.image_size, config.stored_image_shape,
                                  camera="front", mode=config.visual_encoder_mode)
        self.wrist = CameraEncoder(D, config.image_size, config.stored_image_shape,
                                   camera="wrist", mode=config.visual_encoder_mode)
        self.joints = MLP(6, [hidden] * config.enc_proprio_obs_layers, D)
        self.time = MLP(1, [hidden], D)
        self.horizon = config.task_horizon_ticks
        self.fusion = MLP(4 * D, [hidden] * config.enc_latent_layers, D)
        self.act_proj = MLP(6, [hidden] * config.enc_action_layers, D)
        self.temporal_enc = nn.Parameter(torch.randn(config.history_length + 1, D) * 0.02)
        self.type_enc = nn.Parameter(torch.randn(3, D) * 0.02)

    def content(self, joints, front, wrist):
        """Observable image/joint features without the known episode clock."""
        return torch.cat((self.joints(joints), self.front(front), self.wrist(wrist)), -1)

    def observation_from_content(self, content, remaining_ticks):
        time = self.time(remaining_ticks.float() / self.horizon)
        return self.fusion(torch.cat((content, time), -1))

    def observation(self, joints, front, wrist, remaining_ticks):
        return self.observation_from_content(self.content(joints, front, wrist), remaining_ticks)

    def latent_from_content(self, content, remaining_ticks):
        return self.observation_from_content(content, remaining_ticks) + self.temporal_enc[-1] + self.type_enc[2]

    def encode_latent(self, joints, front, wrist, remaining_ticks):
        # The future target uses the same canonical current-observation tags.
        return self.latent_from_content(self.content(joints, front, wrist), remaining_ticks)

    def forward(self, joints, actions, front, wrist, remaining_ticks):
        obs = self.observation(joints, front, wrist, remaining_ticks)
        past = obs[:, :-1] + self.temporal_enc[:-1] + self.type_enc[0]
        acts = self.act_proj(actions) + self.temporal_enc[:-1] + self.type_enc[1]
        latent = obs[:, -1] + self.temporal_enc[-1] + self.type_enc[2]
        return torch.stack((past, acts), dim=2).flatten(1, 2), latent
