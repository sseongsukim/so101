"""Train the image Diffusion Policy student on trajectory pickles.

Port of robust-rearrangement `src/train/bc.py` for `DiffusionPolicy` (see
so101.learning.dp for the model/data ports): AdamW for the actor (UNet,
projections, LayerNorms) at actor_lr and a second AdamW for the two ResNet18
encoders at encoder_lr, each with linear warmup + cosine decay to 0 over
--steps (warmup 2000 / encoder warmup 50000 by default), weight decay 1e-3,
no gradient clipping, optional EMA. One or more pickle directories are
co-trained; each is first converted once into a memory-mapped cache
(outputs/dp_cache/<dir>-<hash> by default, rebuilt when its file list changes).

Output directory:
    dp_so101.pt        final checkpoint ({"model", "config", "stats", "step"[, "ema_model"]})
    step_*.pt          periodic checkpoints (--save-freq), same format
    resume.pt          latest checkpoint + optimizer/scheduler/EMA state (for --init-from)
    stats.json         min/max normalizer stats (also inside every checkpoint)
    train_info.json    data sources, frame counts, camera contract, config
    train_log.jsonl    one line per --log-freq steps

Examples:
    python scripts/train_dp.py --data sim=outputs/teacher_demos --out outputs/dp_train/run0
    python scripts/train_dp.py --data sim=outputs/teacher_demos --data real=outputs/real_demos:4 \\
        --out outputs/dp_train/cotrain0 --steps 200000
    python scripts/train_dp.py --data real=outputs/real_demos --init-from outputs/dp_train/run0/dp_so101.pt \\
        --reset-step --out outputs/dp_train/finetune0 --set warmup_steps=500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

# so101 may also be installed editable from a sibling checkout, whose .pth
# entry would otherwise shadow this repo's src (see make_calibration_targets.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, RandomSampler  # noqa: E402

from so101.learning.act.data import CAMERA_OBSERVATION_KEYS, CAMERA_SOURCES, FPS  # noqa: E402
from so101.learning.dp.config import ACTION, OBS_STATE, DPConfig  # noqa: E402
from so101.learning.dp.data import (  # noqa: E402
    DataSource,
    DPDataset,
    compute_min_max_stats,
    make_sampler,
    parse_data_arg,
    prepare_sources,
    save_stats,
    split_episodes,
)
from so101.learning.dp.policy import (  # noqa: E402
    DiffusionPolicy,
    SwitchEMA,
    build_optimizers,
    load_checkpoint,
    save_checkpoint,
)


def _parse_overrides(items: list[str] | None) -> dict:
    """--set key=value (value parsed as JSON when possible, e.g. down_dims=[64,128])."""
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, got {item!r}")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def _batches(loader: DataLoader):
    while True:
        yield from loader


def train(
    data: list[str | DataSource],
    out_dir: str | Path,
    *,
    steps: int = 100_000,
    batch_size: int | None = None,
    num_workers: int = 8,
    device: str | None = None,
    log_freq: int = 50,
    save_freq: int = 10_000,
    min_length: int = 1,
    seed: int = 0,
    pretrained_backbone: bool = True,
    init_from: str | Path | None = None,
    reset_step: bool = False,
    cache_root: str | Path | None = None,
    rebuild_cache: bool = False,
    val_fraction: float = 0.0,
    val_freq: int = 1000,
    val_batches: int = 5,
    ema: bool | None = None,
    amp: bool = False,
    overrides: dict | None = None,
) -> Path:
    """Train DP on one or more pickle directories; returns the final checkpoint path.
    Importable so tests don't need to shell out."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    sources = [parse_data_arg(d) if isinstance(d, str) else d for d in data]
    sources = prepare_sources(sources, cache_root=cache_root, force=rebuild_cache)
    manifest = sources[0].manifest

    # --- config: fresh, or the architecture of --init-from ---------------
    init_ckpt = load_checkpoint(init_from, device="cpu") if init_from is not None else None
    if init_ckpt is not None:
        config = DPConfig.from_dict(init_ckpt["config"])
        if not config.freeze_encoder:
            config.pretrained_backbone = False  # weights come from the checkpoint
    else:
        config = DPConfig(
            state_dim=manifest["state_dim"],
            action_dim=manifest["action_dim"],
            image_shape=tuple(manifest["image_shape"][DPConfig().front_key][:2]),
            pretrained_backbone=pretrained_backbone,
        )
    config = replace(config, **(overrides or {}))
    if batch_size is not None:
        config.batch_size = batch_size
    if ema is not None:
        config.ema_use = ema
    config.validate()
    front_shape = tuple(manifest["image_shape"][config.front_key][:2])
    if front_shape != config.image_shape:
        # FrontCameraTransform crops a fixed input size; the wrist view is resized from any size.
        raise ValueError(f"{config.front_key} frames are {front_shape} but config.image_shape is {config.image_shape}")
    if steps < config.encoder_warmup_steps and config.lr_scheduler == "cosine":
        print(
            f"warning: --steps {steps} < encoder_warmup_steps {config.encoder_warmup_steps}: the encoder LR "
            f"never reaches {config.encoder_lr} (the paper trains 500k steps)"
        )

    # --- data ----------------------------------------------------------------
    train_eps, val_eps = split_episodes(sources, val_fraction, seed)
    dataset = DPDataset(sources, config, episodes=train_eps, min_length=min_length)
    val_dataset = None
    if any(len(v) for v in val_eps):
        val_dataset = DPDataset([s for s, v in zip(sources, val_eps) if len(v)], config,
                                episodes=[v for v in val_eps if len(v)], min_length=min_length)
    for i, s in enumerate(sources):
        print(
            f"data {s.name}: {s.source_dir} weight {s.weight} -> {dataset.num_episodes[i]} episodes, "
            f"{dataset.num_frames[i]} frames, {dataset.num_samples[i]} samples, "
            f"batch fraction {dataset.effective_fractions()[s.name]:.2f}  (cache {s.cache_dir})"
        )

    if init_ckpt is not None:
        stats = init_ckpt["stats"]
        print(f"stats from {init_from}")
    else:
        stats = compute_min_max_stats(sources, train_eps)
    save_stats(out_dir / "stats.json", stats)

    info = {
        "data_sources": [
            {
                "name": s.name,
                "dir": str(Path(s.source_dir).resolve()),
                "weight": s.weight,
                "cache_dir": str(s.cache_dir),
                "episodes": dataset.num_episodes[i],
                "frames": dataset.num_frames[i],
                "samples": dataset.num_samples[i],
                "batch_fraction": dataset.effective_fractions()[s.name],
                "val_episodes": [s.manifest["episodes"][e] for e in val_eps[i]],
            }
            for i, s in enumerate(sources)
        ],
        "num_frames": int(sum(dataset.num_frames)),
        "num_samples": len(dataset),
        "fps": FPS,
        "state_dim": config.state_dim,
        "action_dim": config.action_dim,
        # (H, W, 3) the policy expects; live frames must be resized with
        # so101.learning.act.data.to_uint8_rgb to this size.
        "image_shape": {key: manifest["image_shape"][key] for key in config.image_keys},
        "camera_keys": list(config.image_keys),
        "camera_pickle_keys": CAMERA_SOURCES,
        "camera_observation_keys": CAMERA_OBSERVATION_KEYS,
        "steps": steps,
        "init_from": str(init_from) if init_from is not None else None,
        "amp": amp,
        "config": config.to_dict(),
    }
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2))

    def loader_for(ds: DPDataset, sampler) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            drop_last=len(ds) >= config.batch_size,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )

    generator = torch.Generator().manual_seed(seed)
    loader = loader_for(dataset, make_sampler(dataset, generator))
    val_loader = None
    if val_dataset is not None:
        val_loader = loader_for(val_dataset, RandomSampler(val_dataset, generator=torch.Generator().manual_seed(seed + 1)))

    # --- model / optimizers ----------------------------------------------
    policy = DiffusionPolicy(config, stats=stats).to(device)
    optimizers = build_optimizers(policy, total_steps=steps)
    ema_tracker = None
    start_step = 0
    if init_ckpt is not None:
        policy.load_state_dict(init_ckpt["model"])
        policy.normalizer.set_stats(stats)
        if not reset_step:
            start_step = int(init_ckpt["step"])
            if "optimizers" in init_ckpt:
                for name, opt, sched in optimizers:
                    opt.load_state_dict(init_ckpt["optimizers"][name])
                    sched.load_state_dict(init_ckpt["schedulers"][name])
        print(f"init from {init_from} (step {init_ckpt['step']}); training from step {start_step}")
    if config.ema_use:
        ema_tracker = SwitchEMA(policy, config.ema_decay)
        if init_ckpt is not None and init_ckpt.get("ema_model") is not None:
            ema_tracker.load_shadow(init_ckpt["ema_model"])
    n_actor = sum(p.numel() for p in policy.actor_parameters())
    n_enc = sum(p.numel() for p in policy.encoder_parameters())
    print(f"params: actor {n_actor / 1e6:.1f}M, encoders {n_enc / 1e6:.1f}M; cond_dim {config.cond_dim}; device {device}")

    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def save(name: str, step: int, *, resume: bool = False) -> Path:
        extra = {"ema_model": ema_tracker.state_dict()} if ema_tracker is not None else {}
        if resume:
            extra["optimizers"] = {n: o.state_dict() for n, o, _ in optimizers}
            extra["schedulers"] = {n: s.state_dict() for n, _, s in optimizers}
        path = save_checkpoint(out_dir / name, policy, step, **extra)
        print(f"saved {path}")
        return path

    @torch.no_grad()
    def validate() -> float:
        policy.eval()
        if ema_tracker is not None:
            ema_tracker.apply_shadow()
        losses = []
        for i, batch in enumerate(_batches(val_loader)):
            if i >= val_batches:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                losses.append(policy.compute_loss(batch)[0].item())
        if ema_tracker is not None:
            ema_tracker.restore()
        policy.train()
        return float(np.mean(losses))

    keep_keys = (OBS_STATE, ACTION, *config.image_keys)
    policy.train()
    step = start_step
    batches = _batches(loader)
    log_file = (out_dir / "train_log.jsonl").open("a")
    window = {"loss": 0.0, "n": 0, "data_s": 0.0, "t0": time.perf_counter()}
    max_norm = 1.0 + 1e3 * (1 - float(config.clip_grad_norm))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while step < steps:
        t_data = time.perf_counter()
        batch = next(batches)
        batch = {k: batch[k].to(device, non_blocking=True) for k in keep_keys}
        window["data_s"] += time.perf_counter() - t_data

        for _, opt, _ in optimizers:
            opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss, _ = policy.compute_loss(batch)
        scaler.scale(loss).backward()
        for _, opt, _ in optimizers:
            scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=max_norm)
        for _, opt, sched in optimizers:
            scaler.step(opt)
            sched.step()
        scaler.update()
        if ema_tracker is not None:
            ema_tracker.update()

        step += 1
        window["loss"] += loss.item()
        window["n"] += 1
        if step % log_freq == 0 or step == steps:
            elapsed = time.perf_counter() - window["t0"]
            record = {
                "step": step,
                "loss": window["loss"] / window["n"],
                "grad_norm": float(grad_norm),
                "lr": {name: sched.get_last_lr()[0] for name, _, sched in optimizers},
                "step_time_s": elapsed / window["n"],
                "data_wait_frac": window["data_s"] / max(elapsed, 1e-9),
            }
            if device.type == "cuda":
                record["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            if val_loader is not None and (step % val_freq == 0 or step == steps):
                record["val_loss"] = validate()
            print(
                f"step {step:7d}  loss {record['loss']:.4f}  "
                + (f"val {record['val_loss']:.4f}  " if "val_loss" in record else "")
                + f"|g| {record['grad_norm']:.2f}  lr {record['lr']}  "
                f"{record['step_time_s']:.3f}s/step (data {record['data_wait_frac']:.0%})"
                + (f"  mem {record['peak_mem_gb']:.1f}GB" if "peak_mem_gb" in record else ""),
                flush=True,
            )
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            window = {"loss": 0.0, "n": 0, "data_s": 0.0, "t0": time.perf_counter()}
        if save_freq and step % save_freq == 0 and step < steps:
            save(f"step_{step:07d}.pt", step)
            save("resume.pt", step, resume=True)
    log_file.close()

    save("resume.pt", step, resume=True)
    return save("dp_so101.pt", step)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--data", action="append", required=True, metavar="NAME=DIR[:WEIGHT]",
                    help="pickle directory to train on; repeat to co-train. WEIGHT (default 1) multiplies that source's per-sample probability")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=100_000, help="total optimizer steps (also the cosine schedule length)")
    ap.add_argument("--batch-size", type=int, default=None, help="default: config.batch_size (256)")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-freq", type=int, default=50)
    ap.add_argument("--save-freq", type=int, default=10_000, help="checkpoint every N steps (0 = only at end)")
    ap.add_argument("--min-length", type=int, default=1, help="skip trajectories with fewer steps (stray saves)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pretrained-backbone", action="store_true", help="random-init ResNet18 instead of ImageNet weights")
    ap.add_argument("--init-from", default=None,
                    help="checkpoint to start from: weights (+EMA); resume.pt also restores optimizer/scheduler and the step")
    ap.add_argument("--reset-step", action="store_true", help="with --init-from: start at step 0 with fresh optimizers (fine-tuning)")
    ap.add_argument("--cache-root", default=None, help="memmap cache root (default outputs/dp_cache)")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--val-fraction", type=float, default=0.0, help="hold out this fraction of each source's episodes")
    ap.add_argument("--val-freq", type=int, default=1000)
    ap.add_argument("--ema", action="store_true", default=None, help="track EMA weights (training.yaml ema, off by default)")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast + GradScaler (paper: mixed_precision false)")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="override any DPConfig field, e.g. --set encoder_lr=3e-5")
    args = ap.parse_args()

    train(
        args.data,
        args.out,
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        min_length=args.min_length,
        seed=args.seed,
        pretrained_backbone=not args.no_pretrained_backbone,
        init_from=args.init_from,
        reset_step=args.reset_step,
        cache_root=args.cache_root,
        rebuild_cache=args.rebuild_cache,
        val_fraction=args.val_fraction,
        val_freq=args.val_freq,
        ema=args.ema,
        amp=args.amp,
        overrides=_parse_overrides(args.set),
    )


if __name__ == "__main__":
    main()
