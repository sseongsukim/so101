"""Train ACT on the RGB trajectories recorded by `scripts/teleop_task.py --visual`.

Port of so101-ros `act/tools/train_so101.py`: same model, loss, optimizer
(AdamW, constant LR, separate backbone LR) and checkpoint format; only the
data source differs (teleop pickles instead of a LeRobotDataset).

Output directory:
    act_so101.pt       final checkpoint ({"model", "config", "step"})
    step_*.pt          periodic checkpoints (--save-freq)
    stats.json         MEAN_STD stats for state/action/images -- inference
                       must normalize with this file, not recompute it
    train_info.json    data/camera contract the policy was trained against

Example:
    python scripts/teleop_task.py --visual --dataset-dir outputs/act_sim
    python scripts/train_act.py --data-dir outputs/act_sim --out outputs/act_train/run0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# so101 may also be installed editable from a sibling checkout, whose .pth
# entry would otherwise shadow this repo's src (see make_calibration_targets.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

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
from so101.learning.act.data import (  # noqa: E402
    CAMERA_OBSERVATION_KEYS,
    CAMERA_SOURCES,
    FPS,
    TeleopACTDataset,
    compute_stats,
    load_episodes,
)

VECTOR_KEYS = (OBS_STATE, ACTION)


def train(
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    steps: int = 2000,
    batch_size: int = 8,
    chunk_size: int = 100,
    lr: float | None = None,
    device: str | None = None,
    num_workers: int = 0,
    log_freq: int = 50,
    save_freq: int = 0,
    min_length: int = 1,
    seed: int = 0,
    pretrained_backbone: bool = True,
    init_from: str | Path | None = None,
) -> Path:
    """Train ACT on a directory of teleop pickles. Returns the path to the
    final checkpoint. Importable so tests don't need to shell out."""
    torch.manual_seed(seed)
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    episodes = load_episodes(data_dir, min_length=min_length)
    dataset = TeleopACTDataset(episodes, chunk_size=chunk_size)
    image_keys = dataset.image_keys
    lengths = [len(ep) for ep in episodes]
    print(
        f"train: {len(dataset)} frames, {dataset.num_episodes} episodes "
        f"(length min/mean/max {min(lengths)}/{sum(lengths) / len(lengths):.0f}/{max(lengths)}), "
        f"image_keys={image_keys}"
    )

    # Continuing from a checkpoint keeps the normalization its weights were
    # trained under instead of re-deriving it from the (possibly new) data.
    init_stats = Path(init_from).parent / "stats.json" if init_from is not None else None
    if init_stats is not None and init_stats.is_file():
        stats = load_stats(init_stats)
        print(f"stats from {init_stats}")
    else:
        if init_from is not None:
            print(f"warning: no stats.json next to {init_from}; recomputing from {data_dir}")
        stats = compute_stats(episodes)
    save_stats(out_dir / "stats.json", stats)
    norm = build_normalizer(stats, VECTOR_KEYS, image_keys, device)

    ref = episodes[0]
    (out_dir / "train_info.json").write_text(
        json.dumps(
            {
                "data_dir": str(data_dir.resolve()),
                "episodes": [ep.path.name for ep in episodes],
                "num_frames": len(dataset),
                "fps": FPS,
                "state_dim": ref.state.shape[1],
                "action_dim": ref.action.shape[1],
                # (H, W, 3) the policy expects; live frames must be resized
                # with so101.learning.act.data.to_uint8_rgb to this size.
                "image_shape": {key: list(ref.images[key].shape[1:]) for key in image_keys},
                "camera_pickle_keys": CAMERA_SOURCES,
                "camera_observation_keys": CAMERA_OBSERVATION_KEYS,
            },
            indent=2,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) >= batch_size,
        persistent_workers=num_workers > 0,
    )

    config = ACTConfig(
        action_dim=ref.action.shape[1],
        robot_state_dim=ref.state.shape[1],
        image_keys=image_keys,
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
    )
    if lr is not None:
        config.optimizer_lr = lr
    if not pretrained_backbone:
        config.pretrained_backbone_weights = None
    policy = ACTPolicy(config).to(device)
    policy.train()

    # Continue from an earlier run's weights. Only the model is restored -- the
    # checkpoint carries no optimizer state, so AdamW's moments rebuild over the
    # first ~1k steps. Harmless here because the LR is constant (no schedule to
    # resume). The step counter picks up where the checkpoint left off so
    # `save_freq` filenames stay on one continuous series.
    start_step = 0
    if init_from is not None:
        ckpt = load_checkpoint(init_from, device=device)
        policy.load_state_dict(ckpt["model"])
        start_step = int(ckpt["step"])
        print(f"init from {init_from} at step {start_step}")

    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=config.optimizer_lr,
        weight_decay=config.optimizer_weight_decay,
    )

    def save(name: str) -> Path:
        ckpt = out_dir / name
        save_checkpoint(ckpt, policy, config.__dict__, step)
        tqdm.write(f"saved {ckpt}")
        return ckpt

    keep_keys = (OBS_STATE, ACTION, "action_is_pad", *image_keys)
    step = start_step
    running = 0.0
    done = step >= steps
    pbar = tqdm(total=steps, initial=step, desc="train", dynamic_ncols=True)
    while not done:
        for batch in loader:
            batch = {k: batch[k].to(device, non_blocking=True) for k in keep_keys}
            batch = normalize_batch(batch, norm)
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running += loss.item()
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if step % log_freq == 0:
                tqdm.write(f"step {step:5d}  loss {running / log_freq:.4f}  {loss_dict}")
                running = 0.0
            if save_freq and step % save_freq == 0:
                save(f"step_{step:07d}.pt")
            if step >= steps:
                done = True
                break
    pbar.close()

    return save("act_so101.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", required=True, help="directory of trajectory_*.pkl from teleop_task.py --visual")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000, help="total steps, counted from --init-from's step when given")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--lr", type=float, default=None, help="override config.optimizer_lr")
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-freq", type=int, default=50)
    ap.add_argument("--save-freq", type=int, default=0, help="checkpoint every N steps (0 = only at end)")
    ap.add_argument("--min-length", type=int, default=1, help="skip trajectories with fewer steps (stray saves)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
        help="random-init ResNet18 instead of ImageNet weights (no download)",
    )
    ap.add_argument("--init-from", default=None, help="checkpoint to continue from (model weights + step; no optimizer state)")
    args = ap.parse_args()

    train(
        args.data_dir,
        args.out,
        steps=args.steps,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        lr=args.lr,
        device=args.device,
        num_workers=args.num_workers,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        min_length=args.min_length,
        seed=args.seed,
        pretrained_backbone=not args.no_pretrained_backbone,
        init_from=args.init_from,
    )


if __name__ == "__main__":
    main()
