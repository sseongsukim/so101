"""Real follower arm + cameras in the simulator's units, for the on-site scripts.

Everything a policy exchanges with the real rig goes through here so the
conventions cannot drift between recording real demos, system identification,
and deployment:

* joints are Isaac radians in SO101_SIM_JOINT_ORDER, converted with the
  follower mapping in force (so101.real.joint_mapping.follower_mapping);
* every command is clamped to the USD joint limits and rate-limited, so a bad
  policy output or a mapping mistake moves the arm slowly instead of slamming;
* camera frames are rectified (matching the simulated pinhole cameras),
  converted BGR -> RGB, and read by background threads so a 30 Hz loop never
  waits on USB.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import torch

from so101.real.constants import SO101_HOME_POSE, SO101_JOINT_ORDER, SO101_SIM_JOINT_ORDER, STACK_CUBE_DEFAULT_JOINT_POS
from so101.real.joint_mapping import USD_MAX, USD_MIN

# Servo P gain for every follower motor. LeRobot's configure() writes 16 (half
# the STS3215 default, "to avoid shakiness"); with I = 0 that left the
# gravity-loaded elbow 3.7 deg below its target and 5 ticks late. At 32 the
# same sys-id run measured 2.1 deg and 3 ticks -- matching the simulated
# elbow's tracking error -- and no jitter at rest (2026-09-30,
# outputs/sysid/real_p32.npz). The real demos in outputs/real_demos/v1 were
# recorded at 16.
DEFAULT_P_GAIN = {m: 32 for m in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")}

START_POSE = torch.tensor([STACK_CUBE_DEFAULT_JOINT_POS[j] for j in SO101_SIM_JOINT_ORDER], dtype=torch.float32)
JOINT_LOW = torch.tensor(np.deg2rad(USD_MIN), dtype=torch.float32)
JOINT_HIGH = torch.tensor(np.deg2rad(USD_MAX), dtype=torch.float32)


@dataclass
class SafetyLimits:
    # Per-tick change of the commanded target, rad, arm joints / gripper.
    # Must stay ABOVE how fast the policies were trained to move: the ResiP
    # teacher's targets change up to 0.175 rad/tick on the arm (p99 ~0.13)
    # and 0.45 on the gripper, so the earlier 0.06 clipped 45% of its ticks
    # and left the arm behind its own plan -- on 2026-09-30 DP closed the
    # gripper before reaching the cube. The servos' own top speed (~5 rad/s,
    # 0.17 rad/tick) is the effective limit; this only stops wild jumps.
    max_step_rad: float = 0.25
    max_gripper_step_rad: float = 0.5
    # Margin kept inside the USD joint limits, rad.
    limit_margin_rad: float = 0.02


class RateLimitedCommand:
    """Clamp targets to joint limits and limit how fast they may change."""

    def __init__(self, limits: SafetyLimits):
        self.limits = limits
        self.last: torch.Tensor | None = None

    def reset(self, current: torch.Tensor) -> None:
        self.last = current.clone()

    def __call__(self, target: torch.Tensor) -> torch.Tensor:
        m = self.limits.limit_margin_rad
        target = torch.maximum(torch.minimum(target, JOINT_HIGH - m), JOINT_LOW + m)
        if self.last is not None:
            step = torch.full_like(target, self.limits.max_step_rad)
            step[-1] = max(self.limits.max_gripper_step_rad, self.limits.max_step_rad)
            target = self.last + torch.maximum(torch.minimum(target - self.last, step), -step)
        self.last = target.clone()
        return target


class FollowerArm:
    def __init__(self, port: str = "/dev/so101-follower", robot_id: str = "my_follower",
                 limits: SafetyLimits | None = None, fps: int = 30,
                 p_gain: dict[str, int] | None = None):
        """`p_gain`: servo P_Coefficient per motor name, written after
        connecting; None means DEFAULT_P_GAIN (32 everywhere). Pass
        {m: 16 for m in ...} to keep LeRobot's setting."""
        from so101.real.interface import LeRobotSO101Interface

        self.p_gain = dict(DEFAULT_P_GAIN if p_gain is None else p_gain)

        self.interface = LeRobotSO101Interface(
            device="cpu", port=port, id=robot_id, cameras={}, fps=fps, kind="follower"
        )
        self.command = RateLimitedCommand(limits or SafetyLimits())
        self.connected = False
        # Raw LeRobot values (body [-100, 100], gripper [0, 100]) of the last
        # read / command. Recorders keep them so data can be re-mapped when
        # the follower joint mapping is refitted (scripts/remap_real_episodes.py).
        self.last_raw_read: torch.Tensor | None = None
        self.last_raw_sent: torch.Tensor | None = None

    def connect(self) -> None:
        self.interface.init_device()
        self.interface.connect()
        self.connected = True
        if self.p_gain:
            bus = self.interface.robot.bus
            # P_Coefficient is an EEPROM register: writable only with torque off
            # (LeRobot's configure() does the same on every connect). The arm is
            # normally folded at rest here, so the brief release is harmless.
            with bus.torque_disabled():
                for motor, value in self.p_gain.items():
                    bus.write("P_Coefficient", motor, int(value))
            print(f"[follower] P_Coefficient set: {self.p_gain}")
        self.command.reset(self.read())

    def read_gains(self) -> dict[str, int]:
        bus = self.interface.robot.bus
        return {m: int(bus.read("P_Coefficient", m, normalize=False)) for m in bus.motors}

    def read(self) -> torch.Tensor:
        """(6,) measured joints, Isaac radians."""
        observation = self.interface.robot.get_observation()
        raw = torch.tensor([float(observation[j]) for j in SO101_JOINT_ORDER], dtype=torch.float32)
        self.last_raw_read = raw
        return self.interface.get_mapped_actions_vectorized(raw).float()

    def send(self, target: torch.Tensor) -> torch.Tensor:
        """Command (6,) Isaac radians after clamping/rate limiting; returns what was sent."""
        safe = self.command(target.float().cpu())
        raw = self.interface.get_raw_actions_from_radians(safe)
        self.last_raw_sent = raw.float().clone()
        self.interface.robot.send_action({j: float(v) for j, v in zip(SO101_JOINT_ORDER, raw.tolist())})
        return safe

    def move_to(self, target: torch.Tensor, seconds: float = 3.0, fps: float = 30.0) -> None:
        """Cosine-eased move from the current pose (bypasses the per-step limit
        by construction: the easing is slower than it)."""
        start = self.read()
        self.command.reset(start)
        steps = max(int(seconds * fps), 1)
        for k in range(1, steps + 1):
            alpha = 0.5 - 0.5 * np.cos(np.pi * k / steps)
            self.send(start + (target - start) * float(alpha))
            time.sleep(1.0 / fps)

    def rest_pose(self) -> torch.Tensor:
        """Folded rest pose (Isaac radians): SO101_HOME_POSE's arm joints, but the
        wrist roll and gripper of the start pose. The workshop home pose puts
        the roll at -93 deg on this rig, which swung the wrist camera into the
        forearm on the way down (2026-09-30)."""
        raw = torch.tensor([SO101_HOME_POSE[j] for j in SO101_JOINT_ORDER], dtype=torch.float32)
        pose = self.interface.get_mapped_actions_vectorized(raw).float()
        pose[4] = START_POSE[4]
        return pose

    def go_home(self, seconds: float = 3.0) -> None:
        """Start pose first (arm raised, clear of the table and itself), then
        fold into the rest pose: only lift/elbow/flex move in the second leg."""
        self.move_to(START_POSE, seconds=seconds)
        self.move_to(self.rest_pose(), seconds=seconds * 0.7)

    def close(self) -> None:
        if self.connected:
            try:
                self.interface.robot.disconnect()
            finally:
                self.connected = False


class CameraThread:
    """Keeps the newest rectified frame of one camera, as (H, W, 3) uint8 RGB."""

    def __init__(self, name: str, meter_to: float | None = None):
        from so101.real.cameras import open_camera

        self.name = name
        self.camera = open_camera(name, rectify=True, meter_to=meter_to)
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            frame = self.camera.read()  # rectified BGR
            with self._lock:
                self._frame = np.ascontiguousarray(frame[..., ::-1])
                self._stamp = time.monotonic()

    def latest(self, max_age_s: float = 0.2) -> np.ndarray:
        deadline = time.monotonic() + 2.0
        while True:
            with self._lock:
                frame, stamp = self._frame, self._stamp
            if frame is not None and time.monotonic() - stamp <= max_age_s:
                return frame
            if time.monotonic() > deadline:
                raise RuntimeError(f"camera {self.name!r} produced no fresh frame for 2 s")
            time.sleep(0.005)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.camera.close()


class Rate:
    """Fixed-rate loop timer that reports overruns instead of hiding them."""

    def __init__(self, hz: float):
        self.period = 1.0 / hz
        self.next = time.perf_counter() + self.period
        self.overruns = 0

    def sleep(self) -> None:
        remaining = self.next - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        else:
            self.overruns += 1
        self.next = max(self.next + self.period, time.perf_counter())
