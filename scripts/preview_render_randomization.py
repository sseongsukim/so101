"""Render samples of the synthetic-data rendering randomization.

Writes one image per draw: the top row is front|wrist with the nominal
(calibrated, unrandomized) scene, the other rows are randomized draws, one
per environment. Compare them with real frames (view_cameras.py --snapshot)
to see whether the real cameras fall inside the randomized range.

Example:
    python -u scripts/preview_render_randomization.py --headless
    python -u scripts/preview_render_randomization.py --headless --camera-jitter-scale 10   # make camera jitter obvious
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher  # noqa: E402

TASK = "so101-visual-StackCube-v0"
parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--draws", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--camera-jitter-scale", type=float, default=1.0)
parser.add_argument("--no-backdrop", action="store_true", help="no randomized floor/walls")
parser.add_argument("--env-spacing", type=float, default=30.0)
parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/render_randomization")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import so101.tasks  # noqa: E402,F401
from so101.configs import make_env_cfg  # noqa: E402
from so101.tasks.render_randomization import add_backdrop, RenderRandomizationCfg, RenderRandomizer  # noqa: E402


def frames(env, observation) -> list[np.ndarray]:
    rows = []
    for i in range(env.num_envs):
        views = [observation[k][i].clamp(0, 1).mul(255).byte().cpu().numpy() for k in ("front_image", "wrist_image")]
        rows.append(np.concatenate(views, axis=1))
    return rows


def settle(env, steps: int = 2):
    hold = env.robot.data.default_joint_pos.clone()
    for _ in range(steps):
        observation, *_ = env.step(hold)
    return observation


def main() -> None:
    cfg = make_env_cfg(TASK, num_envs=args_cli.num_envs, device=args_cli.device)
    cfg.terminate_on_success = False
    cfg.truncate_on_timeout = False
    cfg.scene.wrist_camera.data_types = ["rgb"]
    cfg.scene.external_camera.data_types = ["rgb"]
    # Neighbouring envs otherwise sit in the front camera's background.
    cfg.scene.env_spacing = args_cli.env_spacing
    if not args_cli.no_backdrop:
        add_backdrop(cfg.scene)
    env = gym.make(TASK, cfg=cfg).unwrapped
    dr_cfg = RenderRandomizationCfg()
    for name in ("front_pos_jitter_m", "front_rot_jitter_deg", "wrist_pos_jitter_m", "wrist_rot_jitter_deg"):
        setattr(dr_cfg, name, getattr(dr_cfg, name) * args_cli.camera_jitter_scale)
    args_cli.out.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        env.reset(seed=args_cli.seed)
        randomizer = RenderRandomizer(env, dr_cfg, seed=args_cli.seed)
        for draw in range(args_cli.draws):
            env.reset()
            randomizer.restore()
            nominal = frames(env, settle(env))[0]
            randomizer.apply(env)
            rows = [nominal, *frames(env, settle(env))]
            grid = cv2.resize(np.concatenate(rows, axis=0), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            path = args_cli.out / f"draw_{draw:02d}.png"
            cv2.imwrite(str(path), grid[..., ::-1])
            print(f"[INFO] wrote {path} (row 0 nominal, rows 1..{env.num_envs} randomized)")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
