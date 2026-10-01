"""Image encoders and camera augmentations for the DP student.

Mirrors robust-rearrangement:
  * `ResnetEncoder`, `get_resnet`, `VisionEncoder.normalize`  <- src/models/vision.py
  * `replace_submodules`                                       <- src/common/pytorch_util.py
  * `FrontCameraTransform`, `WristCameraTransform`             <- src/common/vision.py

How pretrained weights interact with `use_groupnorm` (mirrored exactly): the
torchvision ResNet is built with its ImageNet weights first, then every
BatchNorm2d is swapped for a *freshly initialized* GroupNorm with
num_groups = num_features // 16 (weight 1, bias 0). Only the conv weights
survive from ImageNet; BN affine parameters and running statistics are
discarded. `fc` is replaced by Identity, so the encoder outputs the 512-d
global-average-pooled feature.

Input contract (as in the paper): (N, 3, H, W) uint8 (or float in [0, 255]).
The augmentations run on those uint8 tensors, then the encoder divides by
255 and applies ImageNet mean/std normalization.

Augmentations are torchvision v1 transforms applied to the whole batch
tensor, so -- exactly like the paper -- one ColorJitter/GaussianBlur/
RandomCrop parameter draw is shared by every image in a batch (and per
camera). Deviation: crop/resize sizes are parameters (defaults 240x320 ->
224, margin 20) instead of constants, for tiny-image tests.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torchvision
from torch import Tensor, nn
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """Replace every submodule matching `predicate` by `func(submodule)`."""
    if predicate(root_module):
        return func(root_module)
    targets = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parent, k in targets:
        parent_module = root_module.get_submodule(".".join(parent)) if parent else root_module
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = func(parent_module[int(k)])
        else:
            setattr(parent_module, k, func(getattr(parent_module, k)))
    assert not any(predicate(m) for m in root_module.modules())
    return root_module


def get_resnet(model_name: str, weights: str | None = None) -> nn.Module:
    resnet = getattr(torchvision.models, model_name)(weights=weights)
    resnet.encoding_dim = resnet.fc.in_features
    resnet.fc = nn.Identity()
    return resnet


class ResnetEncoder(nn.Module):
    """ImageNet ResNet trunk -> (N, encoding_dim) features."""

    def __init__(self, model_name: str = "resnet18", *, pretrained: bool = True, use_groupnorm: bool = True, freeze: bool = False):
        super().__init__()
        if model_name not in ("resnet18", "resnet34", "resnet50"):
            raise ValueError(f"unsupported backbone {model_name}")
        self.model = get_resnet(model_name, weights="IMAGENET1K_V1" if pretrained else None)
        self.encoding_dim = self.model.encoding_dim
        if use_groupnorm:
            self.model = replace_submodules(
                self.model,
                predicate=lambda m: isinstance(m, nn.BatchNorm2d),
                func=lambda m: nn.GroupNorm(num_groups=m.num_features // 16, num_channels=m.num_features),
            )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.frozen = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True) -> "ResnetEncoder":
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        """x (N, 3, H, W) in [0, 255] (uint8 or float)."""
        x = x.float() / 255.0
        x = (x - self.mean) / self.std
        return self.model(x)


class _CameraTransform(nn.Module):
    """train/eval switch shared by both camera transforms (follows nn.Module.train)."""

    transform_train: nn.Module
    transform_eval: nn.Module

    def forward(self, x: Tensor) -> Tensor:
        return self.transform_train(x) if self.training else self.transform_eval(x)


def _photometric(brightness: float = 0.3, contrast: float = 0.3, saturation: float = 0.3, hue: float = 0.3) -> list:
    return [
        transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0)),
    ]


class RandomGaussianNoise(nn.Module):
    """Per-image additive Gaussian noise, std ~ U(0, max_std) in 0-255 units.
    Keeps the input dtype (uint8 stays uint8). Deviation from the paper, see
    DPConfig.image_noise_std."""

    def __init__(self, max_std: float):
        super().__init__()
        self.max_std = float(max_std)

    def forward(self, x: Tensor) -> Tensor:
        if self.max_std <= 0:
            return x
        std = torch.rand(x.shape[0], 1, 1, 1, device=x.device) * self.max_std
        noisy = x.float() + torch.randn(x.shape, device=x.device) * std
        return noisy.clamp(0, 255).round().to(x.dtype) if x.dtype == torch.uint8 else noisy.clamp(0, 255)


class FrontCameraTransform(_CameraTransform):
    """train: ColorJitter(0.3 x4) -> GaussianBlur(5, 0.01-2) -> CenterCrop(H, W - 2 margin) -> RandomCrop(crop)
    eval:  CenterCrop(crop)."""

    def __init__(self, input_size: tuple[int, int] = (240, 320), crop_size: int = 224, margin: int = 20,
                 noise_std: float = 0.0):
        super().__init__()
        self.input_size = tuple(input_size)
        crop = (crop_size, crop_size)
        self.transform_train = transforms.Compose(
            [*_photometric(), transforms.CenterCrop((input_size[0], input_size[1] - 2 * margin)), transforms.RandomCrop(crop),
             RandomGaussianNoise(noise_std)]
        )
        self.transform_eval = transforms.CenterCrop(crop)

    def forward(self, x: Tensor) -> Tensor:
        assert tuple(x.shape[-2:]) == self.input_size, f"Invalid input shape: {tuple(x.shape)}"
        return super().forward(x)


class WristCameraTransform(_CameraTransform):
    """train: ColorJitter(0.3 x4) -> GaussianBlur(5, 0.01-2) -> Resize(crop, crop)
    eval:  Resize(crop, crop). (Aspect ratio is not preserved, as in the paper.)"""

    def __init__(self, crop_size: int = 224, noise_std: float = 0.0):
        super().__init__()
        size = (crop_size, crop_size)
        self.transform_train = transforms.Compose(
            [*_photometric(), transforms.Resize(size, antialias=True), RandomGaussianNoise(noise_std)]
        )
        self.transform_eval = transforms.Resize(size, antialias=True)
