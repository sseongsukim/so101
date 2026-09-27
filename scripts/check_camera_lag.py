"""Measure whether simulated camera frames lag the joint state they are paired with.

`teleop_task.py --visual` stores observations[t] (joint positions) next to
front_images[t] / wrist_images[t] from the same env observation. If the RTX
render trails physics by a step, every image shows the arm one step earlier
than its state says, and ACT learns from misaligned pairs.

Test 1 (step lag): the arm is teleported (joint state written directly, and
the same pose commanded, so there is no controller transient) between two
poses A and B in a fixed pseudo-random order, one pose per env step. Each
returned frame is classified as "A" or "B" by comparing it to reference frames
captured after holding each pose, and the classification is matched against
the commanded sequence shifted by 0, 1, 2, 3 steps. The shift that matches
~100% is the lag; 0 means frame t shows the pose in state t.

Test 2 (reset): whether the observation returned by env.reset() still shows
the pre-reset scene (num_rerenders_on_reset = 0 in the task config).

Example:
    python scripts/check_camera_lag.py --headless
    python scripts/check_camera_lag.py --headless --rerenders-on-reset 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
# simulation_app.close() ends the process without flushing a redirected stdout.
sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher  # noqa: E402

TASK = "so101-visual-StackCube-v0"
CAMERAS = ("front_image", "wrist_image")

parser = argparse.ArgumentParser(description="Check simulated camera/state frame alignment.")
parser.add_argument("--pan-delta", type=float, default=0.35, help="shoulder_pan offset of pose B from A (rad)")
parser.add_argument("--sequence-length", type=int, default=40)
parser.add_argument("--hold-steps", type=int, default=5, help="steps held before capturing a reference frame")
parser.add_argument("--max-lag", type=int, default=3)
parser.add_argument(
    "--rerenders-on-reset",
    type=int,
    default=None,
    help="override env_cfg.num_rerenders_on_reset to see whether it fixes test 2",
)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import so101.tasks  # noqa: E402,F401  (registers environments)
from so101.configs import make_env_cfg  # noqa: E402


def _frames(observation: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {key: observation[key][0].float().cpu().numpy() for key in CAMERAS}


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def main() -> None:
    env_cfg = make_env_cfg(TASK, num_envs=1, device=args_cli.device)
    env_cfg.terminate_on_success = False
    env_cfg.truncate_on_timeout = False
    if args_cli.rerenders_on_reset is not None:
        env_cfg.num_rerenders_on_reset = args_cli.rerenders_on_reset
    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    robot = env.unwrapped.robot
    print(f"[INFO] num_rerenders_on_reset = {env.unwrapped.cfg.num_rerenders_on_reset}", flush=True)

    with torch.inference_mode():
        env.reset()
        pose_a = robot.data.default_joint_pos.clone()
        pose_b = pose_a.clone()
        pose_b[:, 0] += args_cli.pan_delta
        poses = (pose_a, pose_b)
        zeros = torch.zeros_like(pose_a)

        def teleport_step(pose: torch.Tensor) -> dict[str, torch.Tensor]:
            robot.write_joint_state_to_sim(pose, zeros)
            observation, *_ = env.step(pose)
            return observation

        # Reference frames, after holding each pose long enough for any lag to flush.
        refs = []
        for pose in poses:
            for _ in range(args_cli.hold_steps):
                observation = teleport_step(pose)
            refs.append(_frames(observation))
        for key in CAMERAS:
            print(f"[INFO] {key}: A-vs-B reference difference {_diff(refs[0][key], refs[1][key]):.4f}", flush=True)

        # ---- Test 1: step lag ------------------------------------------------
        rng = np.random.default_rng(args_cli.seed)
        sequence = rng.integers(0, 2, args_cli.sequence_length)
        labels = {key: [] for key in CAMERAS}
        state_errors = []
        for pose_index in sequence:
            observation = teleport_step(poses[pose_index])
            state_errors.append(float((observation["state"] - poses[pose_index]).abs().max()))
            frames = _frames(observation)
            for key in CAMERAS:
                d_a, d_b = (_diff(frames[key], ref[key]) for ref in refs)
                labels[key].append(0 if d_a < d_b else 1)

        print(f"\n[TEST 1] max |state - commanded pose| over the sequence: {max(state_errors):.2e} rad", flush=True)
        print("         (should be ~0: the state is the teleported pose, so it has no lag)", flush=True)
        verdicts = []
        for key in CAMERAS:
            label = np.asarray(labels[key])
            rates = {
                lag: float((label[lag:] == sequence[: len(sequence) - lag]).mean())
                for lag in range(args_cli.max_lag + 1)
            }
            best = max(rates, key=rates.get)
            verdicts.append(best)
            rate_text = "  ".join(f"lag {lag}: {rate:6.1%}" for lag, rate in rates.items())
            print(f"  {key:12s} {rate_text}   -> image lags state by {best} step(s)", flush=True)

        # ---- Test 2: reset ---------------------------------------------------
        for _ in range(args_cli.hold_steps):
            observation = teleport_step(pose_b)
        before_reset = _frames(observation)
        observation, _ = env.reset()
        at_reset = _frames(observation)
        observation = teleport_step(robot.data.default_joint_pos.clone())
        after_one_step = _frames(observation)
        print("\n[TEST 2] frame returned by env.reset():", flush=True)
        stale = []
        for key in CAMERAS:
            to_before = _diff(at_reset[key], before_reset[key])
            to_after = _diff(at_reset[key], after_one_step[key])
            is_stale = to_before < to_after
            stale.append(is_stale)
            print(
                f"  {key:12s} diff to pre-reset frame {to_before:.4f}, to first post-reset step {to_after:.4f}"
                f"   -> {'STALE (shows pre-reset scene)' if is_stale else 'fresh'}"
            )

    print("\n[SUMMARY]", flush=True)
    if all(v == 0 for v in verdicts):
        print("  images and joint state come from the same step: teleop pairs are aligned.", flush=True)
    else:
        print(
            f"  images lag the joint state by {verdicts} step(s) (front, wrist). Recorded pairs are "
            "misaligned; shift images forward by that many steps in the dataset or fix the render order."
        )
    if any(stale):
        print(
            "  reset() returns a stale frame. teleop_task.py never records it (the first tick after "
            "reset is its unrecorded baseline) and eval_act_sim.py settles one step first."
        )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
