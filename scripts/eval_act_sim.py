"""Roll out a `scripts/train_act.py` checkpoint in so101-visual-StackCube-v0.

Every round resets all `--num-envs` environments together and runs up to the
task's episode limit (or `--max-steps`). Automatic success/timeout resets are disabled (as in
utils/env_utils.create_env), so the ACT action queue stays aligned across the
batch; success is latched per environment on the first step it holds, the
same accounting as utils/evaluation.evaluate.

After each reset the arm holds its reset pose for `--settle-steps` steps
before the policy runs: `num_rerenders_on_reset` is 0, so the observation
returned by reset() can still show the previous episode's camera frame.

Examples:
    python scripts/eval_act_sim.py --checkpoint outputs/act_train/run0/act_so101.pt --headless
    python scripts/eval_act_sim.py --checkpoint ... --n-action-steps 20 --video-dir outputs/act_eval/videos
    python scripts/eval_act_sim.py --checkpoint ... --temporal-ensemble-coeff 0.01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
# simulation_app.close() ends the process without flushing a redirected stdout.
sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher  # noqa: E402

TASK = "so101-visual-StackCube-v0"

parser = argparse.ArgumentParser(description="Evaluate an ACT checkpoint in simulation.")
parser.add_argument("--checkpoint", type=Path, required=True, help="act_so101.pt or step_*.pt from train_act.py")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--num-rounds", type=int, default=5, help="episodes = num-envs x num-rounds")
parser.add_argument(
    "--n-action-steps",
    type=int,
    default=None,
    help="actions executed per policy query (default: the checkpoint's chunk_size)",
)
parser.add_argument(
    "--temporal-ensemble-coeff",
    type=float,
    default=None,
    help="enable ACT temporal ensembling (queries every step; paper uses 0.01)",
)
parser.add_argument("--max-steps", type=int, default=None, help="steps per round (default: the task's episode limit)")
parser.add_argument("--settle-steps", type=int, default=1, help="hold steps after reset before the policy acts")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video-dir", type=Path, default=None, help="write env 0's front|wrist view per round as mp4")
parser.add_argument("--out", type=Path, default=None, help="metrics JSON (default: <checkpoint dir>/eval_sim.json)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs < 1 or args_cli.num_rounds < 1:
    parser.error("--num-envs and --num-rounds must be >= 1")
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import so101.tasks  # noqa: E402,F401  (registers environments)
from so101.configs import make_env_cfg  # noqa: E402
from so101.learning.act.data import CAMERA_OBSERVATION_KEYS  # noqa: E402
from so101.learning.act.inference import ACTRunner  # noqa: E402


def _images(observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: observation[obs_key] for key, obs_key in CAMERA_OBSERVATION_KEYS.items()}


def _video_frame(observation: dict[str, torch.Tensor]) -> np.ndarray:
    """Env 0's front and wrist views side by side, uint8 BGR for OpenCV."""
    views = [observation[k][0].clamp(0, 1).mul(255).byte().cpu().numpy() for k in ("front_image", "wrist_image")]
    return np.ascontiguousarray(np.concatenate(views, axis=1)[..., ::-1])


class _VideoWriter:
    """Streams frames to disk: a full episode of 640x480 pairs is ~2 GB in RAM."""

    def __init__(self, path: Path, fps: float):
        self.path, self.fps, self._writer = path, fps, None

    def write(self, frame: np.ndarray) -> None:
        import cv2

        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()


def main() -> None:
    env_cfg = make_env_cfg(TASK, num_envs=args_cli.num_envs, device=args_cli.device)
    env_cfg.terminate_on_success = False
    env_cfg.truncate_on_timeout = False
    env_cfg.seed = args_cli.seed
    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    unwrapped = env.unwrapped
    max_steps = args_cli.max_steps or unwrapped.max_episode_length
    control_hz = 1.0 / unwrapped.step_dt

    runner = ACTRunner(
        args_cli.checkpoint,
        device=unwrapped.device,
        n_action_steps=args_cli.n_action_steps,
        temporal_ensemble_coeff=args_cli.temporal_ensemble_coeff,
    )
    config = runner.policy.config
    print(
        f"[INFO] {args_cli.checkpoint}: chunk_size={config.chunk_size} "
        f"n_action_steps={config.n_action_steps} temporal_ensemble={config.temporal_ensemble_coeff} "
        f"image_sizes={runner.image_sizes}"
    )

    episodes = []
    policy_times = []
    for round_index in range(args_cli.num_rounds):
        with torch.inference_mode():
            observation, _ = env.reset()
            hold = unwrapped.robot.data.default_joint_pos.clone()
            for _ in range(args_cli.settle_steps):
                observation, *_ = env.step(hold)
        runner.reset()

        success = np.zeros(args_cli.num_envs, dtype=bool)
        stacked = np.zeros(args_cli.num_envs, dtype=bool)
        success_step = np.full(args_cli.num_envs, -1)
        video = None
        if args_cli.video_dir is not None:
            video = _VideoWriter(args_cli.video_dir / f"round_{round_index:03d}.mp4", control_hz)
        for step in range(max_steps):
            if video is not None:
                video.write(_video_frame(observation))
            started = time.perf_counter()
            with torch.inference_mode():
                action = runner.act(observation["state"], _images(observation))
            policy_times.append(time.perf_counter() - started)
            with torch.inference_mode():
                observation, _, _, _, info = env.step(action)
            step_success = info["success"].cpu().numpy().astype(bool)
            success_step[step_success & ~success] = step + 1
            success |= step_success
            stacked |= info["stacked"].cpu().numpy().astype(bool)
            if success.all():
                break

        for env_index in range(args_cli.num_envs):
            episodes.append(
                {
                    "round": round_index,
                    "env": env_index,
                    "success": bool(success[env_index]),
                    "stacked": bool(stacked[env_index]),
                    "success_step": int(success_step[env_index]),
                }
            )
        print(
            f"[INFO] round {round_index + 1}/{args_cli.num_rounds}: "
            f"success {success.sum()}/{args_cli.num_envs}, stacked {stacked.sum()}/{args_cli.num_envs}",
            flush=True,
        )
        if video is not None:
            video.close()
            print(f"[INFO] wrote {video.path}")

    successes = [e for e in episodes if e["success"]]
    metrics = {
        "checkpoint": str(args_cli.checkpoint.resolve()),
        "episodes": len(episodes),
        "success_rate": len(successes) / len(episodes),
        "stacked_rate": sum(e["stacked"] for e in episodes) / len(episodes),
        "mean_success_step": float(np.mean([e["success_step"] for e in successes])) if successes else None,
        "n_action_steps": config.n_action_steps,
        "temporal_ensemble_coeff": config.temporal_ensemble_coeff,
        # Per control step, including the resize/normalize on the host; a
        # batched number, not single-robot latency, when num_envs > 1.
        "mean_policy_time_s": float(np.mean(policy_times)),
        "per_episode": episodes,
    }
    out = args_cli.out or args_cli.checkpoint.parent / "eval_sim.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(
        f"[INFO] success {metrics['success_rate']:.2%}  stacked {metrics['stacked_rate']:.2%}  "
        f"({len(episodes)} episodes) -> {out}"
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
