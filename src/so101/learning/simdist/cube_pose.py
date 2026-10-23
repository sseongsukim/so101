"""Small RGB + measured-joint estimator for simulator cube pose labels."""
from __future__ import annotations

import torch
from torch import nn
from typing import NamedTuple

from .cube_pose_spatial import attention_coordinates


class ConditionalCubePose(NamedTuple):
    normalized_xyz: torch.Tensor  # [..., held/fixed, XYZ]
    region_logits: torch.Tensor  # [..., held/fixed], inside is positive


def _pose_output(mlp: nn.Sequential, region_head: nn.Module | None,
                 features: torch.Tensor) -> torch.Tensor | ConditionalCubePose:
    if region_head is None:
        return mlp(features)
    hidden = features
    for layer in list(mlp.children())[:-1]:
        hidden = layer(hidden)
    xyz = mlp[-1](hidden)
    return ConditionalCubePose(xyz.reshape(*xyz.shape[:-1], 2, 3), region_head(hidden))


def _region_head(enabled: bool) -> nn.Module | None:
    if not enabled:
        return None
    head = nn.Linear(64, 2)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head


class CubePoseHead(nn.Module):
    """Map frozen front/wrist DP features and current q to two cube XYZ poses."""

    def __init__(self, *, region_conditioned: bool = False) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(128 * 2 + 6, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 6),
        )
        self.region_head = _region_head(region_conditioned)

    def forward(self, front: torch.Tensor, wrist: torch.Tensor, joints: torch.Tensor) -> torch.Tensor | ConditionalCubePose:
        if front.shape[-1] != 128 or wrist.shape != front.shape:
            raise ValueError("front and wrist features must have matching 128-D shapes")
        if joints.shape[:-1] != front.shape[:-1] or joints.shape[-1] != 6:
            raise ValueError("joint input must match feature batch dimensions and end in six values")
        return _pose_output(self.mlp, self.region_head, torch.cat((front, wrist, joints), dim=-1))


def dp_spatial_features(encoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Frozen DP ResNet layer4 maps, before global pooling/projection/LN.

    Images have already received the named DP camera's eval crop/resize.
    Apply precisely the encoder's uint8-to-ImageNet normalization. No cube
    labels, camera calibration, clock or future state enter this interface.
    """
    return encoder.forward_spatial(images)


class CubeSpatialPoseHead(nn.Module):
    """Object attention over frozen spatial DP features plus current q.

    Each named camera has two learned attention maps. Weighted visual content
    and soft image coordinates are fused to regress held/fixed XYZ. Attention
    is a latent mechanism, not a supervised/identified image keypoint: no
    pixel labels are assumed. The ``attention=False`` control uses identical
    pooled content/head capacity with fixed uniform attention, removing the
    spatial selection mechanism. Output remains fit-normalized pose, [B, 6].
    """

    def __init__(self, channels: int = 512, *, attention: bool = True,
                 region_conditioned: bool = False) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("spatial channels must be positive")
        self.channels = channels
        self.attention = attention
        self.front_attention = nn.Conv2d(channels, 2, 1)
        self.wrist_attention = nn.Conv2d(channels, 2, 1)
        # Uniform initialization: the spatial model starts exactly at its
        # matched mean-pooling control, not at randomly selected pixels.
        for head in (self.front_attention, self.wrist_attention):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            if not attention:
                head.requires_grad_(False)
        self.mlp = nn.Sequential(
            nn.Linear(4 * (channels + 2) + 6, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 6),
        )
        self.region_head = _region_head(region_conditioned)

    def _camera(self, features: torch.Tensor, head: nn.Module) -> torch.Tensor:
        features = features.float()  # cache can be fp16; head training is fp32
        batch, _, height, width = features.shape
        if self.attention:
            weights = head(features).flatten(2).softmax(-1)
        else:
            weights = features.new_full((batch, 2, height * width), 1.0 / (height * width))
        # Coordinates are cell centres in [-1,1], in the transformed image;
        # they are not robot metres or raw-image keypoint labels.
        content = torch.bmm(weights, features.flatten(2).transpose(1, 2))
        positions = attention_coordinates(weights.reshape(batch, 2, height, width))
        return torch.cat((content, positions), -1).flatten(1)

    def forward(self, front: torch.Tensor, wrist: torch.Tensor, joints: torch.Tensor) -> torch.Tensor | ConditionalCubePose:
        if (front.ndim != 4 or wrist.shape != front.shape or front.shape[1] != self.channels
                or min(front.shape[-2:]) < 1):
            raise ValueError("camera features must match [B, spatial_channels, H, W]")
        if joints.shape != (front.shape[0], 6):
            raise ValueError("joint input must have matching batch and six values")
        return _pose_output(self.mlp, self.region_head, torch.cat((
            self._camera(front, self.front_attention),
            self._camera(wrist, self.wrist_attention), joints.float(),
        ), -1))
