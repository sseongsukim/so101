"""Record real SO-101 demonstrations (leader -> follower) for sim+real co-training.

The ResiP paper co-trains the image student on ~10-40 real teleoperated demos
next to the rendered teacher rollouts. Each 30 Hz tick here stores what the
student sees and does on the real rig, in the simulator's units:

    observations  follower joints, Isaac radians (follower joint mapping)
    actions       the joint target sent to the follower, Isaac radians
    front/wrist   rectified RGB, resized to 240x320 with the training resize

The leader is mapped to Isaac radians the same way teleop_task.py maps it into
simulation, and the follower is commanded to that target through the follower
mapping, so a real demo means the same thing as a simulated one.

Every episode starts exactly at the simulated start pose, which is where the
policy will start on deployment. Hand-matching that pose is impractical, so
both arms are driven there:

    r   follower AND leader move to the start pose (hands off the leader) and
        hold it -- the leader under torque. Place the cubes now. Pressing r
        while recording discards the episode and starts over.
    s   the leader's torque is released and recording starts at once: take
        the leader and do the task.
    t   save the episode as a success; back to free teleoperation
    b   discard the episode; back to free teleoperation

Keys are read from the desktop session (pynput), so any focused window works.
Ctrl+C stops; the follower is folded to its home pose and both arms' torque
is released.

Example:
    python scripts/record_real_demos.py --out outputs/real_demos/v1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)

import queue  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

# One intra-op thread: with the default pool (one thread per core) small CPU
# tensor ops stalled this 30 Hz loop for ~100 ms every few ticks whenever
# anything else (camera threads, Isaac) was using the CPU.
torch.set_num_threads(1)

from so101.real.constants import SO101_HOME_POSE, SO101_JOINT_ORDER  # noqa: E402
from so101.real.follower import CameraThread, FollowerArm, Rate, SafetyLimits, START_POSE  # noqa: E402
from so101.real.recording import EpisodeRecorder  # noqa: E402

MOTORS = [joint.split(".")[0] for joint in SO101_JOINT_ORDER]
TELEOP, MOVING, HOLDING, RECORDING = "teleop", "moving", "holding", "recording"


def default_leader_calibration_dir() -> Path:
    project = REPO_ROOT / "calibration/teleoperators/so_leader"
    cache = Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/so_leader"
    return project if (project / "my_leader.json").is_file() else cache


class Keys:
    """Single-character key presses from the desktop session."""

    def __init__(self, accepted: str):
        from pynput.keyboard import Listener

        self.accepted = set(accepted)
        self.queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.listener = Listener(on_press=self._on_press)
        self.listener.start()

    def _on_press(self, key) -> None:
        char = getattr(key, "char", None)
        if char is not None and char.lower() in self.accepted:
            self.queue.put(char.lower())

    def get(self) -> str | None:
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        self.listener.stop()


class LeaderArm:
    """The leader as an input device, which can also be driven to a pose."""

    def __init__(self, port: str, robot_id: str, calibration_dir: Path):
        from so101.real.interface import LeRobotSO101Interface

        self.interface = LeRobotSO101Interface(device="cpu", port=port, id=robot_id, cameras={}, fps=30,
                                               kind="leader", calibration_dir=calibration_dir)
        self.connected = False
        self.torque = False

    @property
    def bus(self):
        return self.interface.robot.bus

    def connect(self) -> None:
        self.interface.init_device()
        self.interface.connect()
        self.connected = True

    def read_raw(self) -> np.ndarray:
        action = self.interface.robot.get_action()
        return np.array([float(action[j]) for j in SO101_JOINT_ORDER])

    def read_sim(self) -> torch.Tensor:
        """(6,) leader pose in Isaac radians (the teleop target)."""
        _, target = self.interface.real_to_sim_obs_processor(self.interface.robot.get_action())
        return target.float()

    def hold(self, raw: np.ndarray) -> None:
        """Enable torque holding `raw` (the goal is written first so enabling
        torque cannot recall a stale goal and jump)."""
        self.command(raw)
        self.bus.enable_torque()
        self.torque = True
        self.command(raw)

    def command(self, raw: np.ndarray) -> None:
        self.bus.sync_write("Goal_Position", {m: float(v) for m, v in zip(MOTORS, raw)})

    def release(self) -> None:
        self.bus.disable_torque()
        self.torque = False

    def close(self) -> None:
        if self.connected:
            try:
                if self.torque:
                    self.release()
            finally:
                self.interface.robot.disconnect()
                self.connected = False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--follower-port", default="/dev/so101-follower")
    parser.add_argument("--follower-id", default="my_follower")
    parser.add_argument("--leader-port", default=os.getenv("TELEOP_PORT", "/dev/so101-leader"))
    parser.add_argument("--leader-id", default="my_leader")
    parser.add_argument("--leader-calibration-dir", type=Path, default=default_leader_calibration_dir())
    parser.add_argument("--max-step-rad", type=float, default=0.15,
                        help="follower rate limit per tick; above normal teleop speed")
    parser.add_argument("--move-seconds", type=float, default=3.0, help="duration of the move to the start pose")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    leader = LeaderArm(args.leader_port, args.leader_id, args.leader_calibration_dir)
    arm = FollowerArm(args.follower_port, args.follower_id, SafetyLimits(max_step_rad=args.max_step_rad))
    recorder = EpisodeRecorder(args.out, source="real_teleop", follower_mapping=arm.interface.joint_mapping.kind
                               if arm.interface.joint_mapping is not None else "linear")
    cameras: list[CameraThread] = []
    keys = None
    try:
        leader.connect()
        arm.connect()
        cameras = [CameraThread("front"), CameraThread("wrist")]
        keys = Keys("rstb")
        print("[info] moving follower to the leader's current pose ...")
        arm.move_to(leader.read_sim(), seconds=3.0)
        # Leader goal for the start pose, in its own calibrated units.
        leader_start = leader.interface.get_raw_actions_from_radians(START_POSE).numpy()
        print(f"[info] saving to {args.out.resolve()} (next {recorder.next_path().name})")
        print("[info] r = both arms to start pose, s = release leader + record, t = save, b = discard; Ctrl+C = stop")

        state_name = TELEOP
        move = None  # (follower_from, leader_from, t0) while MOVING
        rate = Rate(args.fps)
        while True:
            key = keys.get()
            if key == "r":
                if state_name == RECORDING:
                    print(f"\n[info] discarded {len(recorder)} steps")
                    recorder.clear()
                print("\n[info] moving both arms to the start pose -- hands off the leader")
                follower_from = arm.read()
                leader_from = leader.read_raw()
                leader.hold(leader_from)
                arm.command.reset(follower_from)
                move = (follower_from, leader_from, time.monotonic())
                state_name = MOVING
            elif key == "s" and state_name == HOLDING:
                leader.release()
                recorder.clear()
                rate.overruns = 0
                state_name = RECORDING
                print("[info] leader released, recording ... (t = save, b = discard)")
            elif key == "s" and state_name != HOLDING:
                print("\n[info] press r first: s starts recording from the start pose")
            elif key in ("t", "b") and state_name == RECORDING:
                if key == "t":
                    recorded = len(recorder)
                    cut_start, cut_mid, cut_end = recorder.trim_idle()
                    steps = len(recorder)
                    path = recorder.save(success=True)
                    print(f"[info] saved {path}: {steps} steps ({steps / args.fps:.1f} s; recorded {recorded}, "
                          f"idle removed {cut_start} start / {cut_mid} pauses / {cut_end} end), "
                          f"{rate.overruns} loop overruns")
                else:
                    print(f"[info] discarded {len(recorder)} steps")
                    recorder.clear()
                state_name = TELEOP
                print("[info] free teleoperation; r = next episode")

            state = arm.read()
            front, wrist = (c.latest() for c in cameras)
            if state_name == MOVING:
                follower_from, leader_from, t0 = move
                alpha = min((time.monotonic() - t0) / args.move_seconds, 1.0)
                eased = 0.5 - 0.5 * np.cos(np.pi * alpha)
                arm.send(follower_from + (START_POSE - follower_from) * float(eased))
                leader.command(leader_from + (leader_start - leader_from) * eased)
                if alpha >= 1.0:
                    state_name = HOLDING
                    print("[info] both at the start pose. Place the cubes, take the leader, press s")
            elif state_name == HOLDING:
                arm.send(START_POSE)
            else:  # TELEOP or RECORDING: follower mirrors the leader
                sent = arm.send(leader.read_sim())
                if state_name == RECORDING:
                    recorder.append(state, sent, front, wrist, arm.last_raw_read, arm.last_raw_sent)
            rate.sleep()
    except KeyboardInterrupt:
        print("\n[info] stopping")
    finally:
        if keys is not None:
            keys.stop()
        for camera in cameras:
            camera.close()
        leader.close()
        if arm.connected:
            arm.go_home()
            arm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
