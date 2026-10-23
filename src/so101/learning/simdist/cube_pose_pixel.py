"""Opt-in RGB+q pixel-supervised pose estimator; annotations are loss targets only."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .cube_pose_region import conditional_pose_loss
from .cube_pose_spatial import CubePoseNormalization

PIXEL_VERSION = "so101_full_rgb_pixel_pose_v1_20261008"
RGB_HW = (240, 320)
MASK_SEMANTICS = "uncolorized_instance_id_segmentation_fast_resolved_by_usd_prim_path"
POSE_SEMANTICS = "robot_root_relative_translation_world_axes_quaternion_wxyz_forward_x_up_z"


def unpack_resize_masks(packed: np.ndarray, source_hw: tuple[int, int],
                        output_hw: tuple[int, int] = RGB_HW) -> torch.Tensor:
    """Lossless big-endian unpack then PIL-compatible nearest pixel-centre resize.

    Only a minibatch is decoded. Output is boolean [B, camera, object, H, W].
    Width padding is explicitly stripped; numeric IDs are never interpreted here.
    """
    h, w = source_hw
    packed = np.asarray(packed)
    if (h < 1 or w < 1 or min(output_hw) < 1 or packed.dtype != np.uint8
            or packed.ndim != 5 or packed.shape[1:] != (2, 2, h, (w + 7) // 8)):
        raise ValueError("packed masks need uint8 [B,2,2,H,ceil(W/8)] and valid dimensions")
    masks = np.unpackbits(packed, axis=-1, count=w, bitorder="big")
    flat = torch.from_numpy(masks).reshape(-1, 1, h, w).float()
    resized = F.interpolate(flat, size=output_hw, mode="nearest-exact")
    return resized.reshape(len(packed), 2, 2, *output_hw).bool()


def resize_pixel_centres(uv: np.ndarray, source_hw, output_hw=RGB_HW) -> np.ndarray:
    """Source and destination integer coordinates denote pixel centres (PIL resize)."""
    scale = np.asarray(output_hw[::-1], dtype=np.float64) / np.asarray(source_hw[::-1])
    return (np.asarray(uv, dtype=np.float64) + .5) * scale - .5


def front_centre_targets(cubes: np.ndarray, poses: np.ndarray, intrinsics: np.ndarray,
                         packed: np.ndarray, source_hw, inside: np.ndarray) -> dict:
    """Offline GT front centre projection. Wrist calibration does not enter this target.

    Stored WORLD camera orientation is +X forward/+Z up, wxyz. Optical axes are
    +Z forward/+X right/+Y down. UV targets use normalized pixel-centre coordinates
    (u+.5)/W, unchanged by full-frame resize. Mask centroid is not the target.
    """
    cubes, poses, intrinsics = [np.asarray(x, dtype=np.float64) for x in (cubes, poses, intrinsics)]
    n = len(cubes)
    if (cubes.shape != (n, 2, 3) or poses.shape != (n, 2, 7)
            or intrinsics.shape != (n, 2, 3, 3) or inside.shape != (n, 2)):
        raise ValueError("front projection needs aligned cube/pose/K/inside shapes")
    if not all(np.isfinite(x).all() for x in (cubes, poses, intrinsics)):
        raise ValueError("nonfinite camera or cube label")
    quat = poses[:, 0, 3:]
    if not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=1e-4, rtol=0):
        raise ValueError("camera quaternions must be unit wxyz")
    k = intrinsics[:, 0]
    if (np.any(k[:, [0, 1], [0, 1]] <= 0) or not np.allclose(k[:, 2], [0, 0, 1])
            or np.any(np.abs(k[:, 0, 1]) > 1e-8) or np.any(np.abs(k[:, 1, 0]) > 1e-8)):
        raise ValueError("front K must be positive-focal zero-skew pinhole")
    w, x, y, z = quat.T
    rotation = np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                         2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                         2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(n, 3, 3)
    world_from_optical = rotation @ np.asarray([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    camera = np.einsum("nji,noj->noi", world_from_optical, cubes - poses[:, :1, :3])
    depth = camera[..., 2]
    safe_depth = np.where(depth > 0, depth, 1.)
    uv = camera[..., :2] / safe_depth[..., None] * k[:, None, [0, 1], [0, 1]] + k[:, None, :2, 2]
    h, width = source_hw
    packed = np.asarray(packed)
    if packed.dtype != np.uint8 or packed.shape != (n, 2, 2, h, (width + 7)//8):
        raise ValueError("source packed mask shape mismatch")
    # Count source pixels before downsampling and ignore padded width bits.
    lut = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
    counts = lut[packed].sum((-2, -1)).astype(np.int64)
    if width % 8:
        padding = (1 << (8 - width % 8)) - 1
        counts -= lut[packed[..., -1] & padding].sum(-1)
    norm_uv = (uv + .5) / np.asarray([width, h])
    eligible = (np.asarray(inside, bool) & (depth > 0) & (counts[:, 0] >= 8)
                & (norm_uv >= 0).all(-1) & (norm_uv < 1).all(-1))
    # Neutralize invalid UV before loss arithmetic; preserve raw depth as diagnostic.
    return {"centre_targets": np.where(eligible[..., None], norm_uv, .5).astype(np.float32),
            "centre_eligible": eligible, "source_mask_counts": counts, "front_optical_depth_m": depth}


class EarlyCameraTrunk(nn.Module):
    """Share only used checkpoint layers; source layer4/fc/actor stay off GPU."""
    def __init__(self, encoder):
        super().__init__()
        names = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3")
        self.model = nn.ModuleDict({name: getattr(encoder.model, name) for name in names})
        self.register_buffer("mean", encoder.mean.detach().clone())
        self.register_buffer("std", encoder.std.detach().clone())


class FrozenEarlyDPEncoder(nn.Module):
    """Checkpoint convolution trunks, full RGB, ImageNet norm, minibatch-only maps."""
    channels = (64, 128, 256)

    def __init__(self, dp: nn.Module):
        super().__init__()
        if dp.config.vision_backbone != "resnet18":
            raise ValueError("pixel baseline requires checkpoint ResNet18")
        keys = list(dp.config.image_keys)
        if len(keys) != 2 or keys.count(dp.config.front_key) != 1:
            raise ValueError("pixel baseline requires exactly front and wrist")
        front = keys.index(dp.config.front_key)
        self.dp_camera_indices = (front, 1-front)
        self.encoders = nn.ModuleList([EarlyCameraTrunk(dp.encoders[i]) for i in self.dp_camera_indices])
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, front, wrist):
        maps = []
        with torch.no_grad():
            for encoder, image in zip(self.encoders, (front, wrist)):
                x = (image.float()/255 - encoder.mean)/encoder.std
                for name in ("conv1", "bn1", "relu", "maxpool"):
                    x = getattr(encoder.model, name)(x)
                levels = []
                for name in ("layer1", "layer2", "layer3"):
                    x = getattr(encoder.model, name)(x)
                    levels.append(x)
                maps.append(levels)
        return maps


class PixelPoseOutput(NamedTuple):
    normalized_xyz: torch.Tensor
    region_logits: torch.Tensor
    mask_logits: torch.Tensor  # [B,camera,object,H,W], exact RGB pixel grid
    centres: torch.Tensor      # [B,camera,object,uv], normalized pixel centres


class PixelCubePose(nn.Module):
    """Current RGB + measured radians only. No target/calibration arguments."""
    def __init__(self, encoder: nn.Module, stats: dict, width: int = 32):
        super().__init__()
        self.encoder = encoder
        self.normalization = CubePoseNormalization(stats)
        self.laterals = nn.ModuleList([nn.ModuleList([nn.Conv2d(c, width, 1) for c in encoder.channels])
                                      for _ in range(2)])
        self.refine = nn.ModuleList([nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU())
                                     for _ in range(2)])
        self.pixel_heads = nn.ModuleList([nn.Conv2d(width, 8, 1) for _ in range(2)])
        self.mlp = nn.Sequential(nn.Linear(4*(width+2)+6, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU())
        self.xyz = nn.Linear(64, 6)
        self.region = nn.Linear(64, 2)
        nn.init.zeros_(self.region.weight); nn.init.zeros_(self.region.bias)

    def forward(self, front, wrist, measured_q):
        if (front.ndim != 4 or tuple(front.shape[1:]) != (3, *RGB_HW) or wrist.shape != front.shape
                or front.dtype != torch.uint8 or wrist.dtype != torch.uint8):
            raise ValueError("pixel model inputs must be matching full 240x320 uint8 RGB")
        if measured_q.shape != (len(front), 6) or not torch.isfinite(measured_q).all():
            raise ValueError("pixel model needs finite measured q [B,6]")
        maps = self.encoder(front, wrist)
        pooled, masks, centres = [], [], []
        for levels, lateral, refine, head in zip(maps, self.laterals, self.refine, self.pixel_heads):
            x = lateral[-1](levels[-1])
            for index in (1, 0):
                x = F.interpolate(x, size=levels[index].shape[-2:], mode="bilinear", align_corners=False)
                x = x + lateral[index](levels[index])
            x = refine(F.interpolate(x, size=(120, 160), mode="bilinear", align_corners=False))
            logits = head(x)
            probability = logits[:, :2].flatten(-2).softmax(-1).reshape(len(front), 2, 120, 160)
            # A learned subpixel residual per object and dimension, bounded to half a cell.
            offsets = logits[:, 4:].reshape(len(front), 2, 2, 120, 160).tanh()*.5
            v, u = torch.meshgrid(torch.arange(120, device=x.device, dtype=x.dtype),
                                  torch.arange(160, device=x.device, dtype=x.dtype), indexing="ij")
            grid = torch.stack((u+.5, v+.5), 0)[None, None]
            uv = ((grid+offsets)*probability[:, :, None]).sum((-2, -1))
            uv = uv / x.new_tensor([160, 120])
            content = torch.einsum("bohw,bchw->boc", probability, x)
            pooled.append(torch.cat((content, uv), -1).flatten(1))
            centres.append(uv)
            # Separate mask and centre channels: occluded geometric centre need not
            # coincide with visible-mask centroid or an actual foreground pixel.
            masks.append(F.interpolate(logits[:, 2:4], size=RGB_HW, mode="bilinear", align_corners=False))
        hidden = self.mlp(torch.cat((*pooled, self.normalization.normalize_joints(measured_q)), -1))
        return PixelPoseOutput(self.xyz(hidden).reshape(-1, 2, 3), self.region(hidden),
                               torch.stack(masks, 1), torch.stack(centres, 1))


def pixel_pose_loss(pred: PixelPoseOutput, normalized_xyz, inside, region_weights,
                    masks: torch.Tensor, centre_targets: torch.Tensor,
                    centre_eligible: torch.Tensor, *, centre_weight=.1, mask_weight=.1) -> dict:
    """All GT-inside XYZ; masks both cameras; eligible GT centres front only."""
    terms = conditional_pose_loss(pred.normalized_xyz, normalized_xyz, pred.region_logits, inside, region_weights)
    if masks.shape != pred.mask_logits.shape or masks.dtype != torch.bool:
        raise ValueError("pixel mask targets must match full RGB logits as boolean")
    pixel_bce = F.binary_cross_entropy_with_logits(pred.mask_logits, masks.to(pred.mask_logits.dtype), reduction="none")
    # Class-balanced independently per camera/object in this minibatch. Missing class
    # contributes zero, observed class retains weight 1. No fabricated foreground.
    fg, bg = masks.sum((0, 3, 4)), (~masks).sum((0, 3, 4))
    fg_mean = (pixel_bce*masks).sum((0, 3, 4))/fg.clamp_min(1)
    bg_mean = (pixel_bce*~masks).sum((0, 3, 4))/bg.clamp_min(1)
    mask_loss = ((fg_mean+bg_mean)/((fg > 0).to(fg_mean.dtype)+(bg > 0)).clamp_min(1)).mean()
    if centre_targets.shape != pred.centres[:, 0].shape or centre_eligible.shape != inside.shape or centre_eligible.dtype != torch.bool:
        raise ValueError("front centre targets/boolean eligibility shape mismatch")
    eligible = centre_eligible[..., None]
    if not torch.isfinite(centre_targets[eligible.expand_as(centre_targets)]).all():
        raise ValueError("eligible front centre targets must be finite")
    difference = torch.where(eligible, pred.centres[:, 0], 0.) - torch.where(eligible, centre_targets, 0.)
    centre_loss = difference.square().sum()/(2*centre_eligible.sum()).clamp_min(1)
    return {**terms, "loss": terms["loss"] + centre_weight*centre_loss + mask_weight*mask_loss,
            "centre": centre_loss, "mask": mask_loss}
