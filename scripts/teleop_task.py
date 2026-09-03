"""Drive an SO-101 simulation task with a physical SO-101 leader arm.

Examples:

    python scripts/teleop_task.py so101-StackCube-v0
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

TASK_NAMES = (
    "so101-StackCube-v0",
)

parser = argparse.ArgumentParser(
    description="Teleoperate an SO-101 task with a physical SO-101 leader arm."
)
parser.add_argument(
    "task_name",
    nargs="?",
    choices=TASK_NAMES,
    help="Registered SO-101 environment name.",
)
parser.add_argument(
    "--task",
    "--task-name",
    "--task_name",
    dest="task_option",
    choices=TASK_NAMES,
    help="Registered SO-101 environment name (alternative to the positional argument).",
)
parser.add_argument(
    "--port",
    default=os.getenv("TELEOP_PORT", "/dev/ttyACM0"),
    help="Serial port of the SO-101 leader arm.",
)
parser.add_argument(
    "--robot-id",
    default="my_leader",
    help="LeRobot calibration ID of the leader arm.",
)
parser.add_argument(
    "--print-every",
    type=int,
    default=30,
    help="Print diagnostics every N environment steps (default: 30).",
)
parser.add_argument(
    "--rate",
    type=float,
    default=30.0,
    help="Maximum wall-clock control rate in Hz; 0 disables pacing (default: 30).",
)
parser.add_argument(
    "--dataset-dir",
    type=Path,
    default=Path("outputs/teleop"),
    help="Directory for collected .pkl trajectories (default: outputs/teleop).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

selected_task = args_cli.task_option or args_cli.task_name
if selected_task is None:
    parser.error("a task name is required (positional or via --task)")
if args_cli.print_every < 1:
    parser.error("--print-every must be at least 1")
if args_cli.rate < 0.0:
    parser.error("--rate must be non-negative")
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import so101.tasks  # noqa: E402,F401  (registers environments)
from so101.configs import make_env_cfg  # noqa: E402
from so101.real.collect_enum import CollectEnum  # noqa: E402
from so101.real.interface import LeRobotSO101Interface  # noqa: E402
from so101.real.keyboard import KeyboardInterface  # noqa: E402


class TrajectoryBuffer:
    """In-memory buffer for one state-only teleoperation trajectory."""

    _INDEXED_FILENAME = re.compile(r"^trajectory_(\d+)\.pkl$")

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.next_index = self._find_next_index()
        self.clear()

    def _find_next_index(self) -> int:
        """Continue after existing pickle trajectories."""
        if not self.directory.exists():
            return 0
        trajectory_files = list(self.directory.glob("trajectory_*.pkl"))
        indexed = []
        for path in trajectory_files:
            match = self._INDEXED_FILENAME.match(path.name)
            if match is not None:
                indexed.append(int(match.group(1)))
        after_largest_index = max(indexed, default=-1) + 1
        return max(after_largest_index, len(trajectory_files))

    def clear(self) -> None:
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.terminals: list[bool] = []
        self.successes: list[bool] = []
        self.next_observations: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.actions)

    @staticmethod
    def _state(observation: dict[str, torch.Tensor]) -> np.ndarray:
        """Copy the single environment's state observation to host memory."""
        return observation["state"][0].detach().cpu().numpy().copy()

    def append(
        self,
        observation: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        terminal: bool,
        success: bool,
        next_observation: dict[str, torch.Tensor],
    ) -> None:
        self.observations.append(self._state(observation))
        self.actions.append(action[0].detach().cpu().numpy().copy())
        self.rewards.append(float(reward.reshape(-1)[0].item()))
        self.terminals.append(terminal)
        self.successes.append(success)
        self.next_observations.append(self._state(next_observation))

    def save(
        self,
        *,
        force_terminal: bool = False,
        force_success: bool = False,
    ) -> Path | None:
        if not self.actions:
            return None
        if force_terminal:
            self.terminals[-1] = True
        if force_success:
            self.successes[-1] = True

        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"trajectory_{self.next_index:06d}.pkl"
        while path.exists():
            self.next_index += 1
            path = self.directory / f"trajectory_{self.next_index:06d}.pkl"
        data = {
            "observations": np.stack(self.observations),
            "actions": np.stack(self.actions),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "terminals": np.asarray(self.terminals, dtype=np.bool_),
            "successes": np.asarray(self.successes, dtype=np.bool_),
            "next_observations": np.stack(self.next_observations),
        }
        with path.open("xb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        self.next_index += 1
        self.clear()
        return path


def _reset_observation(env: gym.Env) -> dict[str, torch.Tensor]:
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def _first_value(value: Any, default: Any = None) -> Any:
    """Convert a scalar or first vectorized-environment value for logging."""
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return value.reshape(-1)[0].item()
    return value


def _print_diagnostics(step: int, reward: torch.Tensor, info: dict[str, Any]) -> None:
    """Print reward, success, and filtered jaw-to-held contact diagnostics."""
    reward_value = float(_first_value(reward, 0.0))
    phase = int(_first_value(info.get("reward_phase"), 0))
    success = bool(_first_value(info.get("success"), False))
    reward_success = float(_first_value(info.get("reward_success"), 0.0))
    contact = bool(_first_value(info.get("jaw_contact"), False))
    contact_force = float(_first_value(info.get("jaw_contact_force"), 0.0))
    lift_height = float(_first_value(info.get("lift_height"), 0.0))
    has_lifted = bool(_first_value(info.get("has_lifted"), False))
    reach_distance = float(_first_value(info.get("reach_distance"), float("nan")))
    target_distance = float(
        _first_value(info.get("held_target_distance"), float("nan"))
    )

    print(
        f"[TELEOP] step={step:06d} reward={reward_value:7.4f} phase={phase} "
        f"success={success} reward_success={reward_success:.0f} "
        f"jaw_contact={contact} contact_force={contact_force:7.3f}N "
        f"lift={lift_height * 1000.0:7.2f}mm has_lifted={has_lifted} "
        f"reach_dist={reach_distance * 1000.0:7.2f}mm "
        f"target_dist={target_distance * 1000.0:7.2f}mm",
        flush=True,
    )


def teleop_task() -> None:
    env_cfg = make_env_cfg(selected_task, num_envs=1, device=args_cli.device)
    # Data collection episodes end only through the keyboard. Isaac Lab resets
    # environments internally whenever terminated or truncated is returned, so
    # suppress both signals for teleoperation while retaining success in info.
    env_cfg.terminate_on_success = False
    env_cfg.truncate_on_timeout = False

    env = gym.make(selected_task, cfg=env_cfg, render_mode=None)
    leader = LeRobotSO101Interface(
        device=env.unwrapped.device,
        port=args_cli.port,
        id=args_cli.robot_id,
        cameras={},
        fps=30,
        kind="leader",
    )
    leader_connected = False
    keyboard: KeyboardInterface | None = None
    trajectory = TrajectoryBuffer(args_cli.dataset_dir)

    try:
        keyboard = KeyboardInterface()
        observation = _reset_observation(env)
        leader.init_device()
        leader.connect()
        leader_connected = True

        actions = env.unwrapped.robot.data.default_joint_pos.clone()
        last_recorded_action: torch.Tensor | None = None
        collecting = True
        print(f"[INFO] Teleoperating {selected_task} with leader {args_cli.robot_id}")
        print(f"[INFO] Leader serial port: {args_cli.port}")
        print(f"[INFO] Dataset directory: {args_cli.dataset_dir.resolve()}")
        print(f"[INFO] Next trajectory index: {trajectory.next_index:06d}")
        print("[INFO] Recording only when a mapped leader joint value changes.")
        print("[INFO] Keyboard: t = save and pause, r = reset and resume collection.")
        print("[INFO] Automatic success/timeout resets are disabled for teleoperation.")
        print("[INFO] Press Ctrl+C or close the Isaac Sim window to stop.")
        print(
            "[INFO] Diagnostics: reward/phase/success and jaw-to-held contact/lift/target distance."
        )

        step = 0
        control_period = 1.0 / args_cli.rate if args_cli.rate > 0.0 else 0.0
        while simulation_app.is_running():
            step_started = time.perf_counter()
            with torch.inference_mode():
                _, key_command, _ = keyboard.get_action()
                if key_command == CollectEnum.SUCCESS:
                    path = trajectory.save(
                        force_terminal=True,
                        force_success=True,
                    )
                    if path is None:
                        print("[DATA] Nothing to save.", flush=True)
                    else:
                        print(f"[DATA] Saved {path}; collection paused.", flush=True)
                    collecting = False
                    last_recorded_action = None
                    continue
                if key_command == CollectEnum.RESET:
                    discarded = len(trajectory)
                    trajectory.clear()
                    observation = _reset_observation(env)
                    actions[:] = env.unwrapped.robot.data.default_joint_pos
                    collecting = True
                    last_recorded_action = None
                    print(
                        f"[DATA] Reset; discarded {discarded} transitions and "
                        "resumed collection.",
                        flush=True,
                    )
                    continue

                leader_action = leader.robot.get_action()
                _, mapped_action = leader.real_to_sim_obs_processor(leader_action)

                actions[:] = mapped_action
                next_observation, reward, terminated, truncated, info = env.step(actions)
                done = bool(_first_value(terminated, False)) or bool(
                    _first_value(truncated, False)
                )
                success = bool(_first_value(info.get("success"), False))
                if not collecting:
                    # Keep following the leader so it can be returned to its
                    # initial pose, but do not put that motion in the next episode.
                    pass
                elif last_recorded_action is None:
                    # Establish the stationary leader pose as the baseline;
                    # connecting or resetting alone must not create a sample.
                    last_recorded_action = actions.clone()
                else:
                    if not torch.equal(actions, last_recorded_action):
                        trajectory.append(
                            observation,
                            actions,
                            reward,
                            done,
                            success,
                            next_observation,
                        )
                        last_recorded_action.copy_(actions)
                        joint_values = env.unwrapped.robot.data.joint_pos[0]
                        joint_text = ", ".join(
                            f"{value:.5f}" for value in joint_values.tolist()
                        )
                        print(
                            f"[DATA] step={len(trajectory):06d} "
                            f"joints=[{joint_text}] "
                            f"reward={float(_first_value(reward, 0.0)):.6f} "
                            f"success={success}",
                            flush=True,
                        )
                observation = next_observation

                if collecting and done:
                    path = trajectory.save(
                        force_terminal=True, force_success=success
                    )
                    if path is None:
                        print(
                            "[DATA] Episode ended without leader motion; nothing saved.",
                            flush=True,
                        )
                    else:
                        print(f"[DATA] Episode ended; saved {path}", flush=True)
                    last_recorded_action = None

            step += 1
            if step == 1 or step % args_cli.print_every == 0:
                _print_diagnostics(step, reward, info)

            remaining = control_period - (time.perf_counter() - step_started)
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n[INFO] Teleoperation stopped by user.")
    finally:
        if keyboard is not None:
            keyboard.listener.stop()
        if leader_connected:
            leader.robot.disconnect()
        env.close()


def main() -> None:
    teleop_task()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
