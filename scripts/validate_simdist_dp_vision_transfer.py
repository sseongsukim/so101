#!/usr/bin/env python3
"""Verify SO101 DP camera-feature transfer reproduces the source features."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from so101.learning.dp.config import DPConfig
from so101.learning.dp.policy import DiffusionPolicy
from so101.learning.simdist.config import WorldModelConfig
from so101.learning.simdist.encoder import VisualEncoder, load_dp_vision_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    dp_config = dict(saved["config"])
    dp_config["pretrained_backbone"] = False  # all trained weights load strictly below
    dp = DiffusionPolicy(DPConfig.from_dict(dp_config), saved.get("stats"))
    dp.load_state_dict(saved["model"], strict=True)
    dp = dp.to(args.device).eval()

    wm_config = WorldModelConfig(visual_encoder_mode="dp_resnet18", stored_image_shape=(240, 320))
    visual = VisualEncoder(wm_config).to(args.device).eval()
    provenance = load_dp_vision_checkpoint(visual, args.checkpoint)
    with np.load(args.data, allow_pickle=False) as archive:
        fronts = torch.as_tensor(archive["front_images"][:8], device=args.device)
        wrists = torch.as_tensor(archive["wrist_images"][:8], device=args.device)
    inputs = {"wrist": wrists.permute(0, 3, 1, 2), "front": fronts.permute(0, 3, 1, 2)}
    maximum_error = {}
    with torch.no_grad():
        for camera_name, dp_index in (("wrist", 0), ("front", 1)):
            camera = getattr(visual, camera_name)
            image = inputs[camera_name]
            source = dp.layernorms[dp_index](dp.projections[dp_index](
                dp.encoders[dp_index](dp.transforms[dp_index](image))))
            transferred = camera.feature_norm(camera.projection(
                camera.backbone(camera.transform(image))))
            maximum_error[camera_name] = float((source - transferred).abs().max())
            torch.testing.assert_close(source, transferred, atol=1e-6, rtol=1e-6)
    print({**provenance, "parity_images_per_camera": 8,
           "max_abs_feature_error": maximum_error, "result": "PASS"})


if __name__ == "__main__":
    main()
