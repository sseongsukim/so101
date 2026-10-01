"""Fit the simulated cube material colors so the front camera renders them like the real cubes.

A material's diffuse color is not what the camera records -- lighting, the
renderer's tone mapping and the real camera's exposure sit in between -- so
the nominal cube colors are fitted to *pixels*: render the nominal
(unrandomized) scene, measure the mean sRGB of each cube's pixels, and scale
the diffuse color in linear space by target/measured, a few times. Rendering
randomization then varies appearance around this matched nominal.

Targets are mean RGB of each cube in a real front-camera frame (e.g. from
view_cameras.py --snapshot, measured with a color mask).

Example:
    python -u scripts/match_cube_colors.py --headless --small 71 86 46 --large 167 43 61
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
parser.add_argument("--small", nargs=3, type=float, required=True, metavar=("R", "G", "B"), help="real small-cube mean sRGB 0-255")
parser.add_argument("--large", nargs=3, type=float, required=True, metavar=("R", "G", "B"), help="real large-cube mean sRGB 0-255")
parser.add_argument("--iterations", type=int, default=5)
parser.add_argument("--num-envs", type=int, default=8, help="layouts averaged per measurement")
parser.add_argument("--out", type=Path, default=None, help="optional PNG of the final nominal render")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import so101.tasks  # noqa: E402,F401
from so101.configs import make_env_cfg  # noqa: E402
from so101.tasks.render_randomization import add_backdrop  # noqa: E402


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def main() -> None:
    import omni.usd
    from pxr import Gf, UsdShade

    cfg = make_env_cfg(TASK, num_envs=args_cli.num_envs, device=args_cli.device)
    cfg.terminate_on_success = False
    cfg.truncate_on_timeout = False
    cfg.scene.env_spacing = 30.0
    cfg.scene.wrist_camera.data_types = ["rgb"]
    cfg.scene.external_camera.data_types = ["rgb"]
    add_backdrop(cfg.scene)
    env = gym.make(TASK, cfg=cfg).unwrapped
    stage = omni.usd.get_context().get_stage()
    shaders = {
        name: [UsdShade.Shader(stage.GetPrimAtPath(f"/World/envs/env_{i}/{name}/geometry/material/Shader"))
               for i in range(env.num_envs)]
        for name in ("HeldAsset", "FixedAsset")
    }
    targets = {"HeldAsset": np.asarray(args_cli.small), "FixedAsset": np.asarray(args_cli.large)}
    colors = {name: np.array(s[0].GetInput("diffuseColor").Get(), dtype=np.float64) for name, s in shaders.items()}

    def set_colors(values: dict[str, np.ndarray]) -> None:
        for name, value in values.items():
            for shader in shaders[name]:
                shader.GetInput("diffuseColor").Set(Gf.Vec3f(*np.clip(value, 0, 1).tolist()))

    def render() -> np.ndarray:
        hold = env.robot.data.default_joint_pos.clone()
        for _ in range(3):
            observation, *_ = env.step(hold)
        return observation["front_image"].clamp(0, 1).mul(255).cpu().numpy()  # (N, H, W, 3) sRGB

    with torch.inference_mode():
        env.reset(seed=0)
        for iteration in range(args_cli.iterations + 1):
            # Masks: pixels that change when one cube's color flips between two extremes.
            measured = {}
            for name in shaders:
                others = {n: c for n, c in colors.items() if n != name}
                set_colors({**others, name: np.array([1.0, 1.0, 1.0])})
                bright = render()
                set_colors({**others, name: np.array([0.0, 0.0, 0.0])})
                dark = render()
                mask = (bright - dark).sum(-1) > 60
                set_colors(colors)
                frame = render()
                measured[name] = frame[mask].mean(0)
            report = "  ".join(
                f"{'small' if n == 'HeldAsset' else 'large'}: diffuse {np.round(colors[n], 3).tolist()} -> "
                f"pixels {np.round(measured[n]).tolist()} (target {targets[n].tolist()})"
                for n in shaders
            )
            print(f"[iter {iteration}] {report}")
            if iteration == args_cli.iterations:
                break
            for name in shaders:
                ratio = srgb_to_linear(targets[name]) / np.maximum(srgb_to_linear(measured[name]), 1e-4)
                colors[name] = np.clip(colors[name] * ratio, 0.0, 1.0)
        if args_cli.out is not None:
            import cv2

            args_cli.out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args_cli.out), frame[0].astype(np.uint8)[..., ::-1])
    print(f"SMALL_CUBE_COLOR = {tuple(round(float(v), 4) for v in colors['HeldAsset'])}")
    print(f"LARGE_CUBE_COLOR = {tuple(round(float(v), 4) for v in colors['FixedAsset'])}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
