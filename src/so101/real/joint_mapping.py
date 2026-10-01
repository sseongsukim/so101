"""LeRobot calibrated joint values <-> Isaac joint radians.

Both mappings here take the values LeRobot reports with ``use_degrees=False``
(body joints in [-100, 100], gripper in [0, 100]) and have the same form

    sim_deg = offset_deg + scale_deg * (value - center)

with ``center`` = 0 for body joints and 50 for the gripper (the calibrated
range's midpoint). They differ only in ``scale_deg``:

* ``linear`` -- what ``LeRobotSO101Interface`` does today: the calibrated
  range is stretched onto the USD joint limits, so ``scale_deg`` is
  (USD range) / 200. It is only right if the arm was swept exactly to the USD
  limits during ``lerobot-calibrate``.
* ``physical`` -- the calibrated value is turned back into motor ticks with
  the calibration file's range_min/range_max, and ticks into degrees at the
  STS3215's 4096 ticks/turn. That is the scale LeRobot's ``use_degrees=True``
  uses (its DEGREES mode also measures from the range midpoint), extended to
  the gripper, without changing how LeRobot is configured. Only the per-joint
  ``offset_deg`` (where the range midpoint sits in Isaac) is then unknown;
  ``scripts/fit_follower_joint_mapping.py`` measures it.

``drive_mode`` needs no handling: LeRobot's range modes already apply it
before these values are produced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from so101.real.constants import SO101_USD_MAPPING

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TICKS_PER_TURN = 4096  # STS3215; LeRobot divides by (resolution - 1)
DEG_PER_TICK = 360.0 / (TICKS_PER_TURN - 1)
CENTER = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0])
# Normalized units spanning the calibrated range: 200 for [-100, 100], 100 for [0, 100].
SPAN = np.array([200.0, 200.0, 200.0, 200.0, 200.0, 100.0])
USD_MIN = np.array([SO101_USD_MAPPING[j]["joint_min"] for j in JOINTS])
USD_MAX = np.array([SO101_USD_MAPPING[j]["joint_max"] for j in JOINTS])
USD_MID = (USD_MIN + USD_MAX) / 2.0


def load_lerobot_calibration(path: str | Path) -> dict[str, dict]:
    return json.loads(Path(path).read_text())


def calibrated_sweep_deg(calibration: dict[str, dict]) -> np.ndarray:
    """Physical angle between range_min and range_max, per joint."""
    return np.array(
        [(calibration[j]["range_max"] - calibration[j]["range_min"]) * DEG_PER_TICK for j in JOINTS]
    )


@dataclass
class JointMapping:
    kind: str
    scale_deg: np.ndarray                       # Isaac degrees per normalized unit
    offset_deg: np.ndarray                      # Isaac degrees at the range midpoint
    calibration: dict[str, dict] | None = field(default=None, repr=False)

    @classmethod
    def linear(cls) -> "JointMapping":
        """The current `LeRobotSO101Interface` mapping, for comparison."""
        return cls("linear", (USD_MAX - USD_MIN) / SPAN, USD_MID.copy())

    @classmethod
    def physical(cls, calibration: dict[str, dict], offset_deg=None) -> "JointMapping":
        """Physical degrees from a LeRobot calibration file. With no offsets,
        the range midpoint lands where the linear mapping puts it (the USD
        range midpoint), so the two differ only in scale."""
        offset = USD_MID.copy() if offset_deg is None else np.asarray(offset_deg, dtype=np.float64)
        return cls("physical", calibrated_sweep_deg(calibration) / SPAN, offset, calibration)

    # -- conversion ---------------------------------------------------------

    def _coefficients(self, like):
        if hasattr(like, "device"):  # torch tensor
            import torch

            as_tensor = lambda a: torch.as_tensor(a, dtype=like.dtype, device=like.device)  # noqa: E731
            return as_tensor(self.scale_deg), as_tensor(self.offset_deg), as_tensor(CENTER)
        return self.scale_deg, self.offset_deg, CENTER

    def to_sim(self, values):
        """(..., 6) LeRobot calibrated values -> Isaac radians. numpy or torch."""
        scale, offset, center = self._coefficients(values)
        return (offset + scale * (values - center)) * (np.pi / 180.0)

    def to_lerobot(self, radians):
        """(..., 6) Isaac radians -> LeRobot calibrated values. numpy or torch."""
        scale, offset, center = self._coefficients(radians)
        return (radians * (180.0 / np.pi) - offset) / scale + center

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path, **metadata) -> None:
        import yaml

        if self.kind != "physical" or self.calibration is None:
            raise ValueError("only a physical mapping with its calibration is saved")
        data = {
            "kind": "physical",
            "offset_deg": {j: float(v) for j, v in zip(JOINTS, self.offset_deg)},
            # The offsets are only valid for this motor calibration; `load`
            # refuses a file whose calibration no longer matches.
            "calibration_ranges": {
                j: [int(self.calibration[j]["range_min"]), int(self.calibration[j]["range_max"])] for j in JOINTS
            },
            **metadata,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False))

    @classmethod
    def load(cls, path: str | Path, calibration: dict[str, dict]) -> "JointMapping":
        import yaml

        data = yaml.safe_load(Path(path).read_text())
        for j in JOINTS:
            saved = data["calibration_ranges"][j]
            current = [calibration[j]["range_min"], calibration[j]["range_max"]]
            if list(saved) != current:
                raise ValueError(
                    f"{path}: {j} was fitted for calibration range {saved} but the motor "
                    f"calibration now says {current}; re-run fit_follower_joint_mapping.py"
                )
        return cls.physical(calibration, [data["offset_deg"][j] for j in JOINTS])


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FOLLOWER_MAPPING = REPO_ROOT / "calibration/joint_mapping/follower.yaml"


def follower_calibration_path(robot_id: str = "my_follower", calibration_dir: str | Path | None = None) -> Path:
    """Where LeRobot keeps the follower's motor calibration: an explicit
    directory, then this repo's calibration/, then LeRobot's cache."""
    candidates = [Path(calibration_dir).expanduser()] if calibration_dir is not None else []
    candidates += [
        REPO_ROOT / "calibration/robots/so_follower",
        Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower",
    ]
    for directory in candidates:
        if (directory / f"{robot_id}.json").is_file():
            return directory / f"{robot_id}.json"
    raise FileNotFoundError(f"no {robot_id}.json in {[str(c) for c in candidates]}")


def follower_mapping(
    robot_id: str = "my_follower",
    calibration_dir: str | Path | None = None,
    mapping_path: str | Path | None = None,
    verbose: bool = True,
) -> JointMapping:
    """The follower mapping in force: the fitted physical mapping if
    `mapping_path` (default calibration/joint_mapping/follower.yaml) exists,
    otherwise the historical linear one.

    Everything that turns follower readings into Isaac joints -- the robot
    interface, hand-eye solving, the policy deployment loop -- goes through
    this, so adopting a fitted mapping is one file and cannot be applied to
    some of them and not others.
    """
    path = Path(mapping_path) if mapping_path is not None else DEFAULT_FOLLOWER_MAPPING
    if not path.is_file():
        if verbose:
            print(f"[joint_mapping] {path} not found; follower uses the linear USD-range mapping")
        return JointMapping.linear()
    calibration_file = follower_calibration_path(robot_id, calibration_dir)
    mapping = JointMapping.load(path, load_lerobot_calibration(calibration_file))
    if verbose:
        print(f"[joint_mapping] follower uses the physical mapping from {path} (calibration {calibration_file})")
    return mapping
