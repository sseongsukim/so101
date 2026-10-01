"""Roll out the PyTorch ResiP teacher in this branch's StackCube scene.

Checks that the exported teacher (scripts/export_resip_teacher.py) still
reaches its training success rate in the calibrated scene, through the
robot-relative observation adapter (so101.tasks.teacher). Accounting matches
online.py's evaluation: every env runs the full horizon with auto-reset off,
success is latched on the first step it holds.

Example:
    python -u scripts/eval_resip_teacher.py --headless --num-envs 256
    python -u scripts/eval_resip_teacher.py --headless --no-root-shift   # shows the adapter is needed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Evaluate the ResiP teacher in simulation.")
parser.add_argument("--teacher", type=Path,
                    default=REPO_ROOT / "outputs/teachers/resip_sd042_20260924_210715_e2000")
parser.add_argument("--task", default="so101-StackCube-v0",
                    choices=["so101-StackCube-v0", "so101-visual-StackCube-v0"])
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=1042)
parser.add_argument("--no-root-shift", action="store_true", help="feed world-frame positions (no adapter)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = args_cli.task == "so101-visual-StackCube-v0"
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import so101.tasks  # noqa: E402,F401
from so101.configs import make_env_cfg  # noqa: E402
from so101.learning.resip_teacher import ResiPTeacher  # noqa: E402
from so101.tasks.env import SO101TaskEnv  # noqa: E402
from so101.tasks.teacher import teacher_observation  # noqa: E402


def main() -> None:
    cfg = make_env_cfg(args_cli.task, num_envs=args_cli.num_envs, device=args_cli.device)
    cfg.terminate_on_success = False
    cfg.truncate_on_timeout = False
    cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    teacher = ResiPTeacher(args_cli.teacher, device=env.device)
    generator = torch.Generator(device=env.device).manual_seed(args_cli.seed)
    observe = (lambda: SO101TaskEnv._get_observations(env)["state"]) if args_cli.no_root_shift \
        else (lambda: teacher_observation(env))

    with torch.inference_mode():
        env.reset(seed=args_cli.seed)
        teacher.reset()
        success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        stacked = torch.zeros_like(success)
        first = torch.full((env.num_envs,), -1, device=env.device)
        for step in range(env.max_episode_length):
            _, _, _, _, info = env.step(teacher.act(observe(), generator=generator))
            new = info["success"].bool() & ~success
            first[new] = step + 1
            success |= info["success"].bool()
            stacked |= info["stacked"].bool()
    solved = first[first > 0].float()
    result = {
        "teacher": str(args_cli.teacher), "task": args_cli.task, "root_shift": not args_cli.no_root_shift,
        "episodes": env.num_envs, "success": success.float().mean().item(),
        "stacked": stacked.float().mean().item(),
        "mean_first_success_step": solved.mean().item() if len(solved) else None,
    }
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
