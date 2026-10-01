"""Co-train the so101-ros ACT on synthetic teacher demos + real demos (memmap caches).

Same model, loss (L1 + kl_weight * KL), optimizer (AdamW, constant LR,
separate backbone LR) and checkpoint format as scripts/train_act.py /
so101-ros act/tools/train_so101.py. What differs is only the input side:

  * data: one or more trajectory directories (`--data sim=DIR --data
    real=DIR[:WEIGHT]`), read through the DP trainer's memmap caches
    (so101.learning.act.cache_data) -- the synthetic set does not fit in RAM.
  * augmentation (default on; not in ACT): photometric jitter, blur and pixel
    noise on the GPU, see cache_data.PhotometricAugment.
  * optional fp16 autocast (--amp).

Outputs (same contract as train_act.py, so ACTRunner / eval_act_sim.py /
deploy_policy_real.py --policy act work unchanged):
    act_so101.pt, step_*.pt, stats.json, train_info.json, train_log.jsonl

Example:
    python -u scripts/train_act_cotrain.py --data sim=outputs/synthetic/resip_v2 --data real=outputs/real_demos/v1 \\
        --out outputs/act_train/cotrain_v1 --steps 60000 --batch-size 32 --amp
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from so101.learning.act import (  # noqa: E402
    ACTION,
    OBS_STATE,
    ACTConfig,
    ACTPolicy,
    build_normalizer,
    load_checkpoint,
    load_stats,
    normalize_batch,
    save_checkpoint,
    save_stats,
)
from so101.learning.act.cache_data import ACTCacheDataset, PhotometricAugment, compute_stats, make_sampler  # noqa: E402
from so101.learning.act.data import CAMERA_OBSERVATION_KEYS, CAMERA_SOURCES, FPS  # noqa: E402
from so101.learning.dp.data import parse_data_arg, prepare_sources  # noqa: E402

VECTOR_KEYS = (OBS_STATE, ACTION)


def train(args) -> Path:
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    sources = prepare_sources([parse_data_arg(spec) for spec in args.data], cache_root=args.cache_root)
    dataset = ACTCacheDataset(sources, chunk_size=args.chunk_size, min_length=args.min_length)
    fractions = dataset.effective_fractions()
    for source, frames, episodes in zip(sources, dataset.num_frames, dataset.num_episodes):
        print(f"data {source.name}: {source.source_dir} weight {source.weight} -> {episodes} episodes, "
              f"{frames} frames, batch fraction {fractions[source.name]:.2f}")

    init = load_checkpoint(args.init_from, device="cpu") if args.init_from else None
    stats_path = Path(args.init_from).parent / "stats.json" if args.init_from else None
    if stats_path is not None and stats_path.is_file():
        stats = load_stats(stats_path)
        print(f"stats from {stats_path}")
    else:
        stats = compute_stats(dataset)
    save_stats(out / "stats.json", stats)
    image_keys = dataset.image_keys
    norm = build_normalizer(stats, VECTOR_KEYS, image_keys, device)
    manifest = sources[0].manifest
    (out / "train_info.json").write_text(json.dumps({
        "sources": [{"name": s.name, "dir": str(s.source_dir.resolve()), "weight": s.weight, "cache": str(s.cache_dir),
                     "frames": f, "episodes": e} for s, f, e in zip(sources, dataset.num_frames, dataset.num_episodes)],
        "num_frames": len(dataset),
        "fps": FPS,
        "state_dim": manifest["state_dim"],
        "action_dim": manifest["action_dim"],
        "image_shape": {key: list(manifest["image_shape"][key]) for key in image_keys},
        "camera_pickle_keys": CAMERA_SOURCES,
        "camera_observation_keys": CAMERA_OBSERVATION_KEYS,
        "augment": not args.no_augment,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }, indent=2))

    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=make_sampler(dataset),
                        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True,
                        persistent_workers=args.num_workers > 0)

    config = ACTConfig(action_dim=manifest["action_dim"], robot_state_dim=manifest["state_dim"],
                       image_keys=image_keys, chunk_size=args.chunk_size, n_action_steps=args.chunk_size,
                       separate_camera_backbones=args.separate_camera_backbones)
    if args.lr is not None:
        config.optimizer_lr = args.lr
        config.optimizer_lr_backbone = args.lr
    policy = ACTPolicy(config).to(device)
    step = 0
    if init is not None:
        policy.load_state_dict(init["model"])
        step = int(init["step"])
        print(f"init from {args.init_from} at step {step}")
    policy.train()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=config.optimizer_lr,
                                  weight_decay=config.optimizer_weight_decay)
    augment = None if args.no_augment else PhotometricAugment(noise_std=args.noise_std, hue=args.hue).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    def save(name: str) -> Path:
        path = out / name
        save_checkpoint(path, policy, config.__dict__, step)
        print(f"saved {path}")
        return path

    log = (out / "train_log.jsonl").open("a")
    keep = (OBS_STATE, ACTION, "action_is_pad", *image_keys)
    running, t_last, done = [], time.perf_counter(), step >= args.steps
    while not done:
        for batch in loader:
            batch = {k: batch[k].to(device, non_blocking=True) for k in keep}
            for key in image_keys:
                images = batch[key]
                if augment is not None:
                    images = augment(images)
                batch[key] = images.float().div_(255.0)
            batch = normalize_batch(batch, norm)
            with torch.autocast(device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                loss, loss_dict = policy.forward(batch)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running.append(loss.item())
            step += 1
            if step % args.log_freq == 0:
                now = time.perf_counter()
                record = {"step": step, "loss": sum(running) / len(running), **loss_dict,
                          "step_time_s": (now - t_last) / args.log_freq,
                          "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(f"step {step:6d}  loss {record['loss']:.4f}  l1 {loss_dict['l1_loss']:.4f}  "
                      f"{record['step_time_s']:.3f} s/step")
                running, t_last = [], now
            if args.save_freq and step % args.save_freq == 0:
                save(f"step_{step:07d}.pt")
            if step >= args.steps:
                done = True
                break
    log.close()
    return save("act_so101.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", action="append", required=True, metavar="NAME=DIR[:WEIGHT]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60000, help="total steps, counted from --init-from's step")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--lr", type=float, default=None, help="override config.optimizer_lr (and backbone LR)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-freq", type=int, default=200)
    ap.add_argument("--save-freq", type=int, default=10000)
    ap.add_argument("--min-length", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true", help="plain ACT training (no image augmentation)")
    ap.add_argument("--noise-std", type=float, default=4.0)
    ap.add_argument("--hue", type=float, default=0.3, help="ColorJitter hue (0.3 = the DP student's paper setting)")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--separate-camera-backbones", action="store_true",
                    help="use an independent trainable pretrained ResNet for each camera")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
