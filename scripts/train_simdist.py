#!/usr/bin/env python3
"""Train the visual baseline on aligned NPZ episodes; no teacher/JAX dependency."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from so101.learning.simdist.config import WorldModelConfig
from so101.learning.simdist.dataset import VisualWindowDataset, split_episode_groups
from so101.learning.simdist.model import WorldModel
from so101.learning.simdist.losses import world_model_loss
from so101.learning.simdist.encoder import load_dp_vision_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-fraction", type=float, default=0.15,
                   help="reset-seed groups reserved for fixed training diagnostics")
    p.add_argument("--calibration-fraction", type=float, default=0.15,
                   help="reset-seed groups excluded from optimizer and reserved for post-training calibration")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--residual-dynamics", action="store_true",
                   help="experimental: predict corrections around time-aware persistence")
    p.add_argument("--freeze-observation-encoder", action="store_true",
                   help="diagnostic: hold visual/joint/time target representation fixed during training")
    p.add_argument("--init-dp-vision", type=Path,
                   help="initialize ResNet camera trunks/projections from a saved SO101 DP student")
    p.add_argument("--freeze-transferred-vision", action="store_true",
                   help="freeze transferred ResNet/projection/norm blocks but train their latent adapters")
    p.add_argument("--joint-dynamics", action="store_true",
                   help="add an auxiliary normalized measured-joint transition prediction head")
    p.add_argument("--joint-conditioned-latents", action="store_true",
                   help="form future latents from predicted joints, current RGB features and successor task clock")
    p.add_argument("--visual-dynamics", action="store_true",
                   help="predict front/wrist encoder-feature deltas from candidate actions")
    p.add_argument("--freeze-visual-forecaster", action="store_true",
                   help="train only the visual delta head; freeze calibrated joint forecaster and target representation")
    p.add_argument("--init-checkpoint", type=Path,
                   help="initialize matching world-model weights; visual-dynamics keys may be newly added")
    a = p.parse_args()
    if a.steps < 1 or a.batch_size < 1:
        p.error("positive steps and batch size required")
    if a.freeze_transferred_vision and a.init_dp_vision is None:
        p.error("--freeze-transferred-vision requires --init-dp-vision")
    torch.manual_seed(a.seed)
    if a.device == "cpu":
        torch.set_num_threads(1)
    with np.load(a.data, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    dt = float(np.asarray(data["tick_seconds"]).reshape(-1)[0])
    config = WorldModelConfig(history_length=a.history, prediction_length=a.horizon,
                              image_size=a.image_size, tick_seconds=dt, gamma=float(data["gamma"]),
                              task_horizon_ticks=int(np.asarray(data["task_horizon_ticks"])[0]),
                              stored_image_shape=tuple(data["front_images"].shape[1:3]),
                              residual_dynamics=a.residual_dynamics,
                              joint_dynamics=a.joint_dynamics,
                              joint_conditioned_latents=a.joint_conditioned_latents,
                              visual_dynamics=a.visual_dynamics,
                              visual_encoder_mode="dp_resnet18" if a.init_dp_vision else "small_cnn")
    train_eps, val_eps, calibration_eps = split_episode_groups(
        data, a.seed, a.validation_fraction, a.calibration_fraction)
    train = VisualWindowDataset(data, config, train_eps)
    val = VisualWindowDataset(data, config, val_eps)
    calibration_set = VisualWindowDataset(data, config, calibration_eps) if calibration_eps else None
    stats = train.stats()
    if not any(data["expert_policy_flag"][int(data["episode_starts"][e]):
                 int(data["episode_starts"][e]) + int(data["episode_lengths"][e])].any() for e in train_eps):
        raise ValueError("training split has no expert applied actions; collect fresh dense expert trajectories")
    model = WorldModel(config, stats).to(a.device)
    initialization = None
    if a.init_checkpoint is not None:
        saved_init = torch.load(a.init_checkpoint, map_location="cpu", weights_only=True)
        prior_config = dict(saved_init.get("config", {}))
        current_config = asdict(config)
        prior_config.pop("visual_dynamics", None)
        prior_config.pop("visual_dynamics_loss_weight", None)
        current_config.pop("visual_dynamics", None)
        current_config.pop("visual_dynamics_loss_weight", None)
        if prior_config != current_config:
            raise ValueError("initial checkpoint configuration differs from requested model")
        if saved_init.get("stats") != stats:
            raise ValueError("initial checkpoint normalization differs from this training split")
        missing, unexpected = model.load_state_dict(saved_init["model"], strict=False)
        expected_missing = ({"visual_residual.weight", "visual_residual.bias", "visual_residual_scale"}
                            if a.visual_dynamics else set())
        if set(missing) != expected_missing or unexpected:
            raise ValueError(f"initial checkpoint state mismatch: missing={missing}, unexpected={unexpected}")
        initialization = {"checkpoint": str(a.init_checkpoint.resolve()),
                          "sha256": __import__("hashlib").sha256(a.init_checkpoint.read_bytes()).hexdigest()}
    dp_vision_initialization = None
    if a.init_dp_vision is not None:
        dp_vision_initialization = load_dp_vision_checkpoint(model.encoder, a.init_dp_vision)
    if a.freeze_transferred_vision:
        for camera in (model.encoder.front, model.encoder.wrist):
            camera.freeze_transferred_features()
    if a.freeze_observation_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
    if a.freeze_visual_forecaster:
        if not a.visual_dynamics:
            p.error("--freeze-visual-forecaster requires --visual-dynamics")
        model.freeze_visual_forecaster()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("training configuration has no trainable parameters")
    opt = torch.optim.Adam(trainable, lr=a.lr)
    loader = DataLoader(train, batch_size=a.batch_size, shuffle=True, drop_last=False,
                        generator=torch.Generator().manual_seed(a.seed))
    def tensors(batch):
        return ({k: v.to(a.device) for k, v in batch["model_in"].items()},
                {k: v.to(a.device) for k, v in batch["labels"].items()})
    metrics = []
    step = 0
    while step < a.steps:
        model.train()
        for batch in loader:
            x, y = tensors(batch)
            loss, terms = world_model_loss(model, x, y)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step(); step += 1
            row = {"step": step, "train_loss": float(loss), **{k: float(v) for k, v in terms.items()}}
            metrics.append(row)
            print(json.dumps(row), flush=True)
            if step >= a.steps:
                break
    model.eval()
    heldout = []
    with torch.no_grad():
        for batch in DataLoader(val, batch_size=a.batch_size):
            x, y = tensors(batch)
            loss, _ = world_model_loss(model, x, y)
            heldout.append((float(loss), len(y["rewards"])))
    validation_loss = sum(v*n for v,n in heldout) / sum(n for _,n in heldout)
    a.out.mkdir(parents=True, exist_ok=True)
    seed_metadata = np.asarray(data.get("episode_seeds", np.arange(len(data["episode_starts"]))))
    manifest = {"schema": "so101-visual-simdist-v2", "source": str(a.data.resolve()),
                "seed": a.seed, "train_episodes": train_eps, "validation_episodes": val_eps,
                "calibration_episodes": calibration_eps,
                "reset_seed_partition": {"train": sorted(set(map(int, seed_metadata[train_eps]))),
                                         "validation": sorted(set(map(int, seed_metadata[val_eps]))),
                                         "calibration": sorted(set(map(int, seed_metadata[calibration_eps])))},
                "train_windows": len(train), "validation_windows": len(val),
                "calibration_windows": len(calibration_set) if calibration_set is not None else 0, "steps": step,
                "validation_loss": validation_loss, "config": asdict(config), "stats": stats,
                "freeze_observation_encoder": a.freeze_observation_encoder,
                "freeze_visual_forecaster": a.freeze_visual_forecaster,
                "freeze_transferred_vision": a.freeze_transferred_vision,
                "dp_vision_initialization": dp_vision_initialization,
                "initialization": initialization,
                "research_status": "implementation training run; real robustness unmeasured"}
    temporary = a.out / "worldmodel.pt.tmp"
    torch.save({"schema": manifest["schema"], "model": model.state_dict(), "config": asdict(config),
                "stats": stats, "step": step, "manifest": manifest}, temporary)
    temporary.replace(a.out / "worldmodel.pt")
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(a.out / "worldmodel.pt"), "validation_loss": validation_loss}))


if __name__ == "__main__":
    main()
