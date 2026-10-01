"""Render ResiP teacher rollouts into image demonstrations (ResiP §II-D).

The state-based teacher (scripts/export_resip_teacher.py) drives the visual
StackCube scene; every step records what a real-robot student can observe --
6 joint positions and front/wrist RGB -- next to the teacher's joint-target
action, in the same trajectory_*.pkl format scripts/teleop_task.py writes, so
the student trainers read synthetic and real demos the same way.

Per round (all envs reset together):
  reset -> resample rendering randomization -> one hold step (the frame
  env.reset() returns is stale, see check_camera_lag.py) -> teacher rollout.
An env is recorded from that first fresh frame until `--post-success-steps`
after its first success (so the release/retreat is in the demo), and only
successful episodes are saved, as in the paper.
Acceptance additionally requires a complete post-success tail and consecutive
success on its final frames. --motion-substeps interpolates each teacher
target in the simulator; images and states are recorded at every 30 Hz tick.

Images are stored at 240x320 via so101.learning.act.data.to_uint8_rgb, the
same resize teleop and inference use.

Example:
    python -u scripts/generate_teacher_demos.py --headless --num-episodes 600 --out outputs/synthetic/resip_v1
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher  # noqa: E402

TASK = "so101-visual-StackCube-v0"
parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--teacher", type=Path, default=REPO_ROOT / "outputs/teachers/resip_sd042_20260924_210715_e2000")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--num-episodes", type=int, default=400, help="successful episodes to save")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=0, help="round r resets with seed + r")
parser.add_argument("--post-success-steps", type=int, default=40)
parser.add_argument("--stable-success-steps", type=int, default=15,
                    help="require success on this many final consecutive frames")
parser.add_argument("--motion-substeps", type=int, default=1,
                    help="interpolate each teacher target over N control ticks, recording actual simulated states")
parser.add_argument("--wrist-pos-jitter-m", type=float, default=0.01)
parser.add_argument("--wrist-rot-jitter-deg", type=float, default=3.0)
parser.add_argument("--max-rounds", type=int, default=1000)
parser.add_argument("--image-height", type=int, default=240)
parser.add_argument("--image-width", type=int, default=320)
parser.add_argument("--no-randomization", action="store_true", help="render the nominal scene only")
parser.add_argument("--no-backdrop", action="store_true", help="no randomized floor/walls")
parser.add_argument("--env-spacing", type=float, default=30.0, help="keeps neighbouring envs out of the front view")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import so101.tasks  # noqa: E402,F401
from so101.configs import make_env_cfg  # noqa: E402
from so101.learning.act.data import to_uint8_rgb  # noqa: E402
from so101.learning.resip_teacher import ResiPTeacher  # noqa: E402
from so101.tasks.render_randomization import add_backdrop, RenderRandomizer, RenderRandomizationCfg  # noqa: E402
from so101.tasks.teacher import teacher_observation  # noqa: E402

INDEXED = re.compile(r"^trajectory_(\d+)\.pkl$")


def next_index(directory: Path) -> int:
    indices = [int(m.group(1)) for p in directory.glob("trajectory_*.pkl") if (m := INDEXED.match(p.name))]
    return max(indices, default=-1) + 1


def main() -> None:
    if args_cli.motion_substeps < 1 or not 1 <= args_cli.stable_success_steps <= args_cli.post_success_steps:
        raise ValueError("motion-substeps must be >= 1; stable-success-steps must be between 1 and post-success-steps")
    cfg = make_env_cfg(TASK, num_envs=args_cli.num_envs, device=args_cli.device)
    cfg.terminate_on_success = False
    cfg.truncate_on_timeout = False
    cfg.scene.env_spacing = args_cli.env_spacing
    # Depth is not recorded; dropping it roughly halves the render memory.
    cfg.scene.wrist_camera.data_types = ["rgb"]
    cfg.scene.external_camera.data_types = ["rgb"]
    if not args_cli.no_backdrop:
        add_backdrop(cfg.scene)
    env = gym.make(TASK, cfg=cfg).unwrapped
    n = env.num_envs
    teacher = ResiPTeacher(args_cli.teacher, device=env.device)
    generator = torch.Generator(device=env.device).manual_seed(args_cli.seed)
    size = (args_cli.image_height, args_cli.image_width)
    args_cli.out.mkdir(parents=True, exist_ok=True)
    index = next_index(args_cli.out)
    (args_cli.out / "generation.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args_cli).items()}, indent=2))
    for name in ("wrist", "front"):
        (args_cli.out / f"calibration_{name}.yaml").write_text(
            (REPO_ROOT / "calibration" / "cameras" / f"{name}.yaml").read_text())

    saved = attempted = succeeded = 0
    round_index = 0
    randomizer = None
    started = time.perf_counter()
    horizon = env.max_episode_length * args_cli.motion_substeps
    while saved < args_cli.num_episodes:
        if round_index >= args_cli.max_rounds:
            raise RuntimeError(f"Reached max-rounds with only {saved} accepted episodes")
        with torch.inference_mode():
            observation, _ = env.reset(seed=args_cli.seed + round_index)
            if randomizer is None and not args_cli.no_randomization:
                randomizer = RenderRandomizer(env, cfg=RenderRandomizationCfg(
                    wrist_pos_jitter_m=args_cli.wrist_pos_jitter_m,
                    wrist_rot_jitter_deg=args_cli.wrist_rot_jitter_deg,
                ), seed=args_cli.seed)
            if randomizer is not None:
                randomizer.apply(env)
            observation, *_ = env.step(env.robot.data.default_joint_pos.clone())
            teacher.reset()

            buffers = [{k: [] for k in ("obs", "act", "front", "wrist", "teacher", "success")} for _ in range(n)]
            success = torch.zeros(n, dtype=torch.bool, device=env.device)
            stop_at = torch.full((n,), horizon + args_cli.post_success_steps, device=env.device)
            for step in range(horizon + args_cli.post_success_steps):
                recording = (step <= stop_at) & (success | (step < horizon))
                if not recording.any():
                    break
                t_obs = teacher_observation(env)
                if step % args_cli.motion_substeps == 0:
                    segment_start = observation["state"].clone()
                    segment_target = teacher.act(t_obs, generator=generator)
                fraction = (step % args_cli.motion_substeps + 1) / args_cli.motion_substeps
                action = segment_start + fraction * (segment_target - segment_start)
                state = observation["state"].cpu().numpy()
                act = action.cpu().numpy()
                t_obs_np = t_obs.cpu().numpy()
                for i in torch.nonzero(recording).flatten().tolist():
                    b = buffers[i]
                    b["obs"].append(state[i].copy())
                    b["act"].append(act[i].copy())
                    b["teacher"].append(t_obs_np[i].copy())
                    b["front"].append(to_uint8_rgb(observation["front_image"][i], size))
                    b["wrist"].append(to_uint8_rgb(observation["wrist_image"][i], size))
                observation, _, _, _, info = env.step(action)
                step_success = info["success"].bool()
                newly = step_success & ~success & recording
                stop_at[newly] = step + args_cli.post_success_steps
                success |= step_success & recording
                for i in torch.nonzero(recording).flatten().tolist():
                    buffers[i]["success"].append(bool(step_success[i]))

        attempted += n
        accepted = [i for i in torch.nonzero(success).flatten().tolist()
                    if len(buffers[i]["success"]) == int(stop_at[i]) + 1
                    and all(buffers[i]["success"][-args_cli.stable_success_steps:])]
        succeeded += len(accepted)
        for i in accepted:
            if saved >= args_cli.num_episodes:
                break
            b = buffers[i]
            length = len(b["act"])
            terminals = np.zeros(length, dtype=np.bool_)
            terminals[-1] = True
            obs = np.stack(b["obs"])
            data = {
                "observations": obs,
                "actions": np.stack(b["act"]),
                "rewards": np.asarray(b["success"], dtype=np.float32),
                "terminals": terminals,
                "successes": np.asarray(b["success"], dtype=np.bool_),
                "next_observations": np.concatenate([obs[1:], obs[-1:]]),
                "front_images": np.stack(b["front"]),
                "wrist_images": np.stack(b["wrist"]),
                "image_format": "uint8_rgb",
                "image_shape": [*size, 3],
                "image_size_source": [640, 480],
                "teacher_observations": np.stack(b["teacher"]),
                "source": "resip_teacher",
                "teacher": str(args_cli.teacher),
                "seed": args_cli.seed + round_index,
                "env_index": i,
                "render_randomization": randomizer is not None,
                "motion_substeps": args_cli.motion_substeps,
                "post_success_steps": args_cli.post_success_steps,
                "stable_success_steps": args_cli.stable_success_steps,
                "first_success_step": next(j for j, value in enumerate(b["success"]) if value),
                "fps": 30,
            }
            path = args_cli.out / f"trajectory_{index:06d}.pkl"
            with path.open("xb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            index += 1
            saved += 1
        round_index += 1
        elapsed = time.perf_counter() - started
        print(
            f"[INFO] round {round_index}: ever success {int(success.sum())}/{n}, stable accepted {len(accepted)}/{n}, saved {saved}/{args_cli.num_episodes} "
            f"(teacher success {succeeded / attempted:.1%}), {elapsed / 60:.1f} min",
        )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
