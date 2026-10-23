"""Opt-in frozen DP maps and object attention for current robot-frame cube XYZ.

Only current front/wrist RGB and measured joints are estimator inputs. Camera
map coordinates describe localization after DP crop/resize; the trainable head
learns their relationship to the robot-root-relative XYZ supervision.
"""
from __future__ import annotations

import torch
from torch import nn


class FrozenDPSpatialEncoder(nn.Module):
    """Reuse exact checkpoint camera transforms/trunks; output order front, wrist.

    Construct after loading the source DP checkpoint strictly. The source
    modules are shared, frozen, and held in eval mode even when train() is used.
    No source projection, LayerNorm, action model, or privileged state is used.
    """

    def __init__(self, dp: nn.Module) -> None:
        super().__init__()
        config = dp.config
        if config.vision_backbone != "resnet18":
            raise ValueError("spatial pose baseline requires DP resnet18")
        keys = list(config.image_keys)
        if len(keys) != 2 or keys.count(config.front_key) != 1:
            raise ValueError("spatial pose baseline requires exactly front and wrist cameras")
        front = keys.index(config.front_key)
        self.dp_camera_indices = (front, 1 - front)
        self.image_shape = tuple(config.image_shape)
        self.crop_size = config.crop_size
        self.transforms = nn.ModuleList([dp.transforms[i] for i in self.dp_camera_indices])
        self.encoders = nn.ModuleList([dp.encoders[i] for i in self.dp_camera_indices])
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenDPSpatialEncoder":
        super().train(False)
        return self

    def forward(self, front: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
        expected = (3, *self.image_shape)
        if front.ndim != 4 or tuple(front.shape[1:]) != expected or wrist.shape != front.shape:
            raise ValueError(f"front/wrist RGB must have matching (N, {expected}) shapes")
        if front.dtype != torch.uint8 or wrist.dtype != torch.uint8:
            raise ValueError("front/wrist RGB must be uint8 in [0, 255]")
        # no_grad creates ordinary detached tensors usable in a trainable head.
        with torch.no_grad():
            maps = [encoder.forward_spatial(transform(image))
                    for image, transform, encoder in zip((front, wrist), self.transforms, self.encoders)]
        return torch.stack(maps, dim=1)


def attention_coordinates(attention: torch.Tensor) -> torch.Tensor:
    """Mean map-cell center coordinates (u right, v down), [-1, 1].

    Input is a normalized spatial probability map (..., H, W). These are image
    coordinates, not metres or robot axes, and are not geometric triangulation.
    """
    height, width = attention.shape[-2:]
    u = 2 * (torch.arange(width, device=attention.device, dtype=attention.dtype) + .5) / width - 1
    v = 2 * (torch.arange(height, device=attention.device, dtype=attention.dtype) + .5) / height - 1
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    grid = torch.stack((uu, vv), dim=-1)
    return (attention[..., None] * grid).sum(dim=(-3, -2))


class CubePoseNormalization(nn.Module):
    """Fit-only q/XYZ affine transforms; XYZ is held then fixed, in metres."""

    def __init__(self, stats: dict) -> None:
        super().__init__()
        for name in ("q_mean", "q_std", "pose_mean", "pose_std"):
            value = torch.as_tensor(stats[name], dtype=torch.float32).clone()
            if value.shape != (6,) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be six finite values")
            if name.endswith("std") and (value <= 0).any():
                raise ValueError(f"{name} must be positive")
            self.register_buffer(name, value)

    def normalize_joints(self, joints: torch.Tensor) -> torch.Tensor:
        return (joints - self.q_mean) / self.q_std

    def normalize_targets(self, robot_xyz_metres: torch.Tensor) -> torch.Tensor:
        return (robot_xyz_metres - self.pose_mean) / self.pose_std

    def denormalize_poses(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * self.pose_std + self.pose_mean
