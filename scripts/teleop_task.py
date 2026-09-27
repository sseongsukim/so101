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
    "so101-visual-StackCube-v0",
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_leader_calibration_dir() -> Path:
    """Use the project calibration, or the standard LeRobot cache fallback."""
    project_dir = REPO_ROOT / "calibration/teleoperators/so_leader"
    cache_dir = Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/so_leader"
    if (project_dir / "my_leader.json").is_file():
        return project_dir
    if (cache_dir / "my_leader.json").is_file():
        return cache_dir
    return project_dir


DEFAULT_LEADER_CALIBRATION_DIR = _default_leader_calibration_dir()

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
    default=os.getenv("TELEOP_PORT", "/dev/so101-leader"),
    help="Serial port of the SO-101 leader arm.",
)
parser.add_argument(
    "--robot-id",
    default="my_leader",
    help="LeRobot calibration ID of the leader arm.",
)
parser.add_argument(
    "--calibration-dir",
    type=Path,
    default=DEFAULT_LEADER_CALIBRATION_DIR,
    help="Directory containing the leader calibration JSON.",
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
parser.add_argument(
    "--visual",
    action="store_true",
    help="Collect wrist/front RGB images along with joint state and actions.",
)
parser.add_argument(
    "--image-width",
    type=int,
    default=320,
    help="Stored RGB width in visual mode; source camera remains 640x480.",
)
parser.add_argument(
    "--image-height",
    type=int,
    default=240,
    help="Stored RGB height in visual mode; source camera remains 640x480.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

selected_task = args_cli.task_option or args_cli.task_name
if args_cli.visual:
    if selected_task is None:
        selected_task = "so101-visual-StackCube-v0"
    elif selected_task == "so101-StackCube-v0":
        selected_task = "so101-visual-StackCube-v0"
if selected_task is None:
    parser.error("a task name is required (positional or via --task)")
if args_cli.rate < 0.0:
    parser.error("--rate must be non-negative")
if args_cli.image_width <= 0 or args_cli.image_height <= 0:
    parser.error("--image-width and --image-height must be positive")
args_cli.enable_cameras = selected_task == "so101-visual-StackCube-v0"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import so101.tasks  # noqa: E402,F401  (registers environments)
from so101.configs import make_env_cfg  # noqa: E402
from so101.learning.act.data import to_uint8_rgb  # noqa: E402
from so101.real.collect_enum import CollectEnum  # noqa: E402
from so101.real.interface import LeRobotSO101Interface  # noqa: E402
from so101.real.keyboard import KeyboardInterface  # noqa: E402


class TrajectoryBuffer:
    """In-memory buffer for one teleoperation trajectory.

    Visual trajectories additionally contain uint8 RGB frames under
    ``front_images`` and ``wrist_images``. Keeping images separate from the
    state arrays preserves compatibility with the existing state-only pickle
    reader while making the files directly usable by an ACT data adapter.
    """

    _INDEXED_FILENAME = re.compile(r"^trajectory_(\d+)\.pkl$")

    def __init__(self, directory: Path, image_size: tuple[int, int]) -> None:
        self.directory = directory
        self.image_width, self.image_height = image_size
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
        self.front_images: list[np.ndarray] = []
        self.wrist_images: list[np.ndarray] = []
        self.wall_times: list[float] = []

    def __len__(self) -> int:
        return len(self.actions)

    @staticmethod
    def _state(observation: dict[str, torch.Tensor]) -> np.ndarray:
        """Copy the single environment's state observation to host memory."""
        return observation["state"][0].detach().cpu().numpy().copy()

    def _image(
        self, observation: dict[str, torch.Tensor], key: str
    ) -> np.ndarray:
        """Resize a visual observation and convert it to uint8 RGB.

        Shared with ACT inference so live frames match the training data.
        """
        return to_uint8_rgb(
            observation[key][0], (self.image_height, self.image_width)
        )

    def append(
        self,
        observation: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        terminal: bool,
        success: bool,
        next_observation: dict[str, torch.Tensor],
        *,
        visual: bool = False,
    ) -> None:
        self.observations.append(self._state(observation))
        self.actions.append(action[0].detach().cpu().numpy().copy())
        self.rewards.append(float(reward.reshape(-1)[0].item()))
        self.terminals.append(terminal)
        self.successes.append(success)
        self.next_observations.append(self._state(next_observation))
        if visual:
            self.front_images.append(self._image(observation, "front_image"))
            self.wrist_images.append(self._image(observation, "wrist_image"))
        # Each env.step advances 1/30 s of sim time regardless of how long it
        # took; the wall-clock stamps show whether the demo was slowed down.
        self.wall_times.append(time.perf_counter())

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
            "wall_times": np.asarray(self.wall_times, dtype=np.float64),
        }
        if self.front_images:
            data.update(
                {
                    "front_images": np.stack(self.front_images),
                    "wrist_images": np.stack(self.wrist_images),
                    "image_format": "uint8_rgb",
                    "image_shape": [self.image_height, self.image_width, 3],
                    "image_size_source": [640, 480],
                }
            )
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
        calibration_dir=args_cli.calibration_dir,
    )
    leader_connected = False
    keyboard: KeyboardInterface | None = None
    trajectory = TrajectoryBuffer(
        args_cli.dataset_dir,
        image_size=(args_cli.image_width, args_cli.image_height),
    )
    visual_collection = selected_task == "so101-visual-StackCube-v0"

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
        print(f"[INFO] RGB cameras: {'enabled' if visual_collection else 'disabled'}")
        if visual_collection:
            print(
                f"[INFO] Stored RGB size: "
                f"{args_cli.image_width}x{args_cli.image_height} "
                "(source 640x480)"
            )
        if visual_collection:
            print("[INFO] Recording every 30 Hz control tick for RGB/ACT data.")
        else:
            print("[INFO] Recording only when a mapped leader joint value changes.")
        print("[INFO] Keyboard: t = save and pause, r = reset and resume collection.")
        print("[INFO] Automatic success/timeout resets are disabled for teleoperation.")
        print("[INFO] Press Ctrl+C or close the Isaac Sim window to stop.")

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
                        print("[INFO] Nothing to save.", flush=True)
                    else:
                        print(f"[INFO] Saved {path}; collection paused.", flush=True)
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
                        f"[INFO] Reset; discarded {discarded} transitions and "
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
                    # ACT needs a fixed-rate image/state sequence.  Keep the
                    # historical motion-only filtering for state-only data,
                    # but record every control tick when RGB cameras are on.
                    should_record = visual_collection or not torch.equal(
                        actions, last_recorded_action
                    )
                    if should_record:
                        trajectory.append(
                            observation,
                            actions,
                            reward,
                            done,
                            success,
                            next_observation,
                            visual=visual_collection,
                        )
                        last_recorded_action.copy_(actions)
                        # Keep the terminal/action/image data in the pickle,
                        # but do not print every collected sample.
                observation = next_observation

                # Do not save automatically when the simulator reports done.
                # The operator explicitly finalizes a trajectory with `t`, so
                # incomplete or accidental episodes cannot silently become
                # dataset files.

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
