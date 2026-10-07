"""Roll out a `scripts/train_dp.py` Diffusion Policy checkpoint in so101-visual-StackCube-v0.

Same rollout loop as scripts/eval_act_sim.py (kept as a copy so the verified
ACT script is untouched), with `DPRunner` instead of `ACTRunner`.

Every round resets all `--num-envs` environments together and runs up to the
task's episode limit (or `--max-steps`). Automatic success/timeout resets are disabled (as in
utils/env_utils.create_env), so the DP action queue (re-planned every
action_horizon steps) stays aligned across the batch; success is latched per environment on the first step it holds, the
same accounting as utils/evaluation.evaluate.

After each reset the arm holds its reset pose for `--settle-steps` steps
before the policy runs: `num_rerenders_on_reset` is 0, so the observation
returned by reset() can still show the previous episode's camera frame.

Examples:
    python scripts/eval_dp_sim.py --checkpoint outputs/dp_train/run0/dp_so101.pt --headless
    python scripts/eval_dp_sim.py --checkpoint ... --inference-steps 4 --video-dir outputs/dp_eval/videos
    python scripts/eval_dp_sim.py --checkpoint ... --action-horizon 4 --no-warmstart
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

parser = argparse.ArgumentParser(description="Evaluate a Diffusion Policy checkpoint in simulation.")
parser.add_argument("--checkpoint", type=Path, required=True, help="dp_so101.pt or step_*.pt from train_dp.py")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--num-rounds", type=int, default=5, help="episodes = num-envs x num-rounds")
parser.add_argument(
    "--action-horizon",
    type=int,
    default=None,
    help="actions executed per plan (default: the checkpoint's action_horizon, 8)",
)
parser.add_argument(
    "--inference-steps",
    type=int,
    default=None,
    help="DDIM steps (default: the checkpoint's, 16; the paper's sim eval used 4, its real robot 8)",
)
parser.add_argument("--no-warmstart", action="store_true", help="plan from pure noise instead of the warm-started previous plan")
parser.add_argument("--no-ema", action="store_true", help="use raw weights even if the checkpoint has EMA weights")
parser.add_argument("--max-steps", type=int, default=None, help="steps per round (default: the task's episode limit)")
parser.add_argument("--settle-steps", type=int, default=1, help="hold steps after reset before the policy acts")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--render-randomization", action="store_true",
                    help="resample the synthetic-data rendering randomization every round")
parser.add_argument("--image-gain", default="", help="policy-input brightness gain, e.g. front=1.4,wrist=2.3")
parser.add_argument("--no-backdrop", action="store_true", help="bare scene without the floor/walls")
parser.add_argument("--env-spacing", type=float, default=30.0)
parser.add_argument("--video-dir", type=Path, default=None, help="write env 0's front|wrist view per round as mp4")
parser.add_argument("--out", type=Path, default=None, help="metrics JSON (default: <checkpoint dir>/eval_sim.json)")
parser.add_argument("--small", type=float, nargs=3, metavar=("X_CM", "Y_CM", "YAW_DEG"),
                    help="optional fixed small-cube centre, yellow-base frame")
parser.add_argument("--large", type=float, nargs=3, metavar=("X_CM", "Y_CM", "YAW_DEG"),
                    help="optional fixed large-cube centre, yellow-base frame; requires --small")
parser.add_argument("--realtime", action="store_true", help="pace the visible demo at simulation control frequency")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
IMAGE_GAIN = {
    f"observation.images.{name.strip()}": float(value)
    for name, value in (item.split("=") for item in args_cli.image_gain.split(",") if item.strip())
}
if args_cli.num_envs < 1 or args_cli.num_rounds < 1:
    parser.error("--num-envs and --num-rounds must be >= 1")
if (args_cli.small is None) != (args_cli.large is None):
    parser.error("--small and --large must be supplied together")
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import so101.tasks  # noqa: E402,F401  (registers environments)
from so101.configs import make_env_cfg  # noqa: E402
from so101.tasks.render_randomization import RenderRandomizer, add_backdrop  # noqa: E402
from so101.learning.act.data import CAMERA_OBSERVATION_KEYS  # noqa: E402
from so101.learning.dp.inference import DPRunner  # noqa: E402


def _images(observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    images = {key: observation[obs_key] for key, obs_key in CAMERA_OBSERVATION_KEYS.items()}
    # --image-gain: brighten/darken what the policy sees (not the video), to
    # test sensitivity to camera exposure, e.g. front=1.4,wrist=2.3.
    for key, gain in IMAGE_GAIN.items():
        images[key] = (images[key] * gain).clamp(0.0, 1.0)
    return images


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
    if args_cli.realtime:
        env_cfg.viewer.eye = (0.85, -0.55, 0.65)
        env_cfg.viewer.lookat = (0.24, 0.42, 0.13)
    # Match the synthetic-data scenes (generate_teacher_demos.py): neighbours
    # out of the front view, and the backdrop the student has always seen.
    env_cfg.scene.env_spacing = args_cli.env_spacing
    env_cfg.scene.wrist_camera.data_types = ["rgb"]
    env_cfg.scene.external_camera.data_types = ["rgb"]
    if not args_cli.no_backdrop:
        add_backdrop(env_cfg.scene)
    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    randomizer = None
    unwrapped = env.unwrapped
    max_steps = args_cli.max_steps or unwrapped.max_episode_length
    control_hz = 1.0 / unwrapped.step_dt

    runner = DPRunner(
        args_cli.checkpoint,
        device=unwrapped.device,
        action_horizon=args_cli.action_horizon,
        inference_steps=args_cli.inference_steps,
        warmstart_timestep=None if args_cli.no_warmstart else "checkpoint",
        use_ema=not args_cli.no_ema,
    )
    config = runner.config
    print(
        f"[INFO] {args_cli.checkpoint}: pred_horizon={config.pred_horizon} "
        f"action_horizon={config.action_horizon} inference_steps={config.inference_steps} "
        f"warmstart_timestep={config.warmstart_timestep} image_sizes={runner.image_sizes}"
    )

    episodes = []
    policy_times = []
    for round_index in range(args_cli.num_rounds):
        with torch.inference_mode():
            observation, _ = env.reset()
            if args_cli.small is not None:
                from fixed_layout_check import layout_world, place_cubes

                place_cubes(unwrapped, layout_world(args_cli))
            if args_cli.render_randomization:
                if randomizer is None:
                    randomizer = RenderRandomizer(unwrapped, seed=args_cli.seed)
                randomizer.apply(unwrapped)
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
            if args_cli.realtime:
                time.sleep(max(0.0, unwrapped.step_dt - (time.perf_counter() - started)))
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
        "seed": args_cli.seed,
        "max_steps": max_steps,
        "fixed_layout_cm": {"small": args_cli.small, "large": args_cli.large}
        if args_cli.small is not None else None,
        "render_randomization": args_cli.render_randomization,
        "image_gain": args_cli.image_gain,
        "backdrop": not args_cli.no_backdrop,
        "episodes": len(episodes),
        "success_rate": len(successes) / len(episodes),
        "stacked_rate": sum(e["stacked"] for e in episodes) / len(episodes),
        "mean_success_step": float(np.mean([e["success_step"] for e in successes])) if successes else None,
        "pred_horizon": config.pred_horizon,
        "action_horizon": config.action_horizon,
        "inference_steps": config.inference_steps,
        "warmstart_timestep": config.warmstart_timestep,
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
