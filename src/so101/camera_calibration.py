"""Camera calibration records shared by the simulation and the real robot.

A single YAML file per camera is the one source of truth for that camera's
geometry.  The real capture layer reads it to build rectification maps and the
Isaac Lab scene reads it to configure the matching sensor, so the two can never
drift apart.

Why the "virtual" camera exists: Isaac Lab renders an ideal pinhole only.
``Camera._update_intrinsic_matrices`` hardcodes ``f_y = f_x``, ``c_x = W / 2``
and ``c_y = H / 2``, and ``spawn_camera`` drops aperture offsets outright
(NVIDIA ticket OM-42611).  Rather than bending the renderer, real frames are
remapped onto an ideal pinhole -- ``K_virtual`` -- that Isaac can reproduce
exactly.  Lens distortion disappears in the same ``remap`` call.

This module depends only on numpy and PyYAML so the simulation side can import
it without pulling in the real-robot stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_DIR = REPO_ROOT / "calibration" / "cameras"

# Isaac Lab's PinholeCameraCfg default, in cm.  Kept here because the sim-side
# focal length is derived from it and the derivation must match the renderer.
ISAAC_HORIZONTAL_APERTURE = 20.955

CAMERA_NAMES = ("wrist", "front")


class CalibrationError(RuntimeError):
    """Raised when a calibration file is missing, malformed or unusable."""


def quat_wxyz_to_matrix(quat: tuple[float, float, float, float]) -> np.ndarray:
    """Convert a (w, x, y, z) quaternion to a 3x3 rotation matrix."""
    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise CalibrationError(f"degenerate quaternion: {quat}")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a (w, x, y, z) quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise CalibrationError(f"expected a 3x3 rotation matrix, got {m.shape}")
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    if w < 0.0:  # canonical sign, so saved values compare cleanly
        w, x, y, z = -w, -x, -y, -z
    return (float(w), float(x), float(y), float(z))


@dataclass
class CameraExtrinsic:
    """Camera pose relative to ``parent``, in the OpenCV/ROS camera convention.

    ``convention`` is always ``"ros"``: ``+Z`` along the optical axis, ``+Y``
    down the image rows.  That is exactly what ``cv2.solvePnP`` and
    ``cv2.calibrateHandEye`` return and exactly what Isaac Lab's
    ``CameraCfg.OffsetCfg`` means by ``convention="ros"``, so the numbers move
    between the two without a conversion layer to get a sign wrong in.
    """

    parent: str
    pos: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    convention: str = "ros"

    def __post_init__(self) -> None:
        if self.convention != "ros":
            raise CalibrationError(
                f"only the 'ros' convention is supported, got {self.convention!r}"
            )
        self.pos = tuple(float(v) for v in self.pos)  # type: ignore[assignment]
        self.quat_wxyz = tuple(float(v) for v in self.quat_wxyz)  # type: ignore[assignment]
        if len(self.pos) != 3 or len(self.quat_wxyz) != 4:
            raise CalibrationError("extrinsic needs 3 position and 4 quaternion values")

    @property
    def matrix(self) -> np.ndarray:
        """The 4x4 homogeneous transform ``T_parent_camera``."""
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = quat_wxyz_to_matrix(self.quat_wxyz)
        transform[:3, 3] = self.pos
        return transform

    @classmethod
    def from_matrix(cls, transform: np.ndarray, parent: str) -> CameraExtrinsic:
        transform = np.asarray(transform, dtype=np.float64)
        if transform.shape != (4, 4):
            raise CalibrationError(f"expected a 4x4 transform, got {transform.shape}")
        return cls(
            parent=parent,
            pos=tuple(float(v) for v in transform[:3, 3]),
            quat_wxyz=matrix_to_quat_wxyz(transform[:3, :3]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent": self.parent,
            "convention": self.convention,
            "pos": [float(v) for v in self.pos],
            "quat_wxyz": [float(v) for v in self.quat_wxyz],
        }


@dataclass
class CameraCalibration:
    """Everything known about one physical camera and its simulated twin."""

    name: str
    device: str
    resolution: tuple[int, int]  # (width, height)
    fps: int
    camera_matrix: np.ndarray  # measured K, 3x3
    distortion: np.ndarray  # measured OpenCV coefficients
    virtual_matrix: np.ndarray  # rectification target K_virtual, 3x3
    alpha: float = 0.0
    extrinsic: CameraExtrinsic | None = None
    exposure: dict[str, Any] = field(default_factory=dict)
    calib_rms_px: float | None = None
    handeye_rmse_px: float | None = None
    hfov_loss_frac: float | None = None
    date: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        self.resolution = (int(self.resolution[0]), int(self.resolution[1]))
        self.fps = int(self.fps)
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64).reshape(3, 3)
        self.distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        self.virtual_matrix = np.asarray(self.virtual_matrix, dtype=np.float64).reshape(3, 3)
        self._validate_virtual_matrix()

    def _validate_virtual_matrix(self) -> None:
        """Reject a K_virtual that Isaac Lab cannot actually render.

        Catching this here turns a silent few-pixel misalignment -- the kind
        that shows up much later as an unexplained gate failure -- into an
        immediate, located error.
        """
        width, height = self.resolution
        fx, fy = self.virtual_matrix[0, 0], self.virtual_matrix[1, 1]
        cx, cy = self.virtual_matrix[0, 2], self.virtual_matrix[1, 2]
        if not math.isclose(fx, fy, rel_tol=1e-6):
            raise CalibrationError(
                f"{self.name}: K_virtual must have fx == fy (Isaac Lab averages them); "
                f"got fx={fx:.6f}, fy={fy:.6f}"
            )
        if not (
            math.isclose(cx, width / 2.0, abs_tol=1e-6)
            and math.isclose(cy, height / 2.0, abs_tol=1e-6)
        ):
            raise CalibrationError(
                f"{self.name}: K_virtual must be centred at ({width / 2}, {height / 2}) "
                f"(Isaac Lab hardcodes the principal point); got ({cx:.6f}, {cy:.6f})"
            )

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]

    @property
    def focal_px(self) -> float:
        """The virtual camera's focal length in pixels."""
        return float(self.virtual_matrix[0, 0])

    def isaac_focal_length(
        self, horizontal_aperture: float = ISAAC_HORIZONTAL_APERTURE
    ) -> float:
        """``PinholeCameraCfg.focal_length`` (cm) reproducing this camera.

        Isaac derives ``fx = width * focal_length / horizontal_aperture``, so
        inverting that is all it takes.  ``from_intrinsic_matrix`` is
        deliberately not used: it discards ``cx``/``cy`` and averages
        ``fx``/``fy`` without the caller seeing it.
        """
        return float(self.focal_px * horizontal_aperture / self.width)

    def horizontal_fov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.focal_px)))

    def rectify_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """Build the ``cv2.remap`` lookup tables taking raw frames to ``K_virtual``."""
        import cv2  # imported lazily so the sim side never needs OpenCV

        return cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.distortion,
            None,
            self.virtual_matrix,
            self.resolution,
            cv2.CV_16SC2,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "device": self.device,
            "resolution": [self.width, self.height],
            "fps": self.fps,
            "K": self.camera_matrix.tolist(),
            "dist": self.distortion.tolist(),
            "K_virtual": self.virtual_matrix.tolist(),
            "alpha": float(self.alpha),
            "extrinsic": self.extrinsic.to_dict() if self.extrinsic else None,
            "exposure": dict(self.exposure),
            "calib_rms_px": self.calib_rms_px,
            "handeye_rmse_px": self.handeye_rmse_px,
            "hfov_loss_frac": self.hfov_loss_frac,
            "date": self.date or _date.today().isoformat(),
        }
        if self.notes:
            payload["notes"] = self.notes
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CameraCalibration:
        missing = [
            key
            for key in ("name", "device", "resolution", "fps", "K", "dist", "K_virtual")
            if key not in payload
        ]
        if missing:
            raise CalibrationError(f"calibration is missing required keys: {missing}")
        extrinsic_payload = payload.get("extrinsic")
        extrinsic = None
        if extrinsic_payload:
            extrinsic = CameraExtrinsic(
                parent=extrinsic_payload["parent"],
                pos=tuple(extrinsic_payload["pos"]),
                quat_wxyz=tuple(extrinsic_payload["quat_wxyz"]),
                convention=extrinsic_payload.get("convention", "ros"),
            )
        return cls(
            name=payload["name"],
            device=payload["device"],
            resolution=tuple(payload["resolution"]),
            fps=payload["fps"],
            camera_matrix=np.asarray(payload["K"], dtype=np.float64),
            distortion=np.asarray(payload["dist"], dtype=np.float64),
            virtual_matrix=np.asarray(payload["K_virtual"], dtype=np.float64),
            alpha=float(payload.get("alpha", 0.0)),
            extrinsic=extrinsic,
            exposure=dict(payload.get("exposure") or {}),
            calib_rms_px=payload.get("calib_rms_px"),
            handeye_rmse_px=payload.get("handeye_rmse_px"),
            hfov_loss_frac=payload.get("hfov_loss_frac"),
            date=payload.get("date", ""),
            notes=payload.get("notes", ""),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                self.to_dict(),
                sort_keys=False,
                allow_unicode=True,
                # Keep matrices and vectors on one line each; these files are
                # meant to be read and sanity-checked by a person.
                default_flow_style=None,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def from_yaml(cls, path: str | Path) -> CameraCalibration:
        path = Path(path)
        if not path.is_file():
            raise CalibrationError(f"no calibration file at {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise CalibrationError(f"{path} does not contain a calibration mapping")
        return cls.from_dict(payload)


def calibration_path(name: str, directory: str | Path | None = None) -> Path:
    return Path(directory or DEFAULT_CALIBRATION_DIR) / f"{name}.yaml"


def load_calibration(
    name: str, directory: str | Path | None = None
) -> CameraCalibration:
    """Load one camera's calibration by name (``"wrist"`` or ``"front"``)."""
    return CameraCalibration.from_yaml(calibration_path(name, directory))


def try_load_calibration(
    name: str, directory: str | Path | None = None
) -> CameraCalibration | None:
    """Load a calibration, or return ``None`` if it has not been produced yet.

    Callers use this to stay runnable before the on-site calibration session:
    the capture layer falls back to raw frames and the scene falls back to its
    historical constants, both with a warning.
    """
    try:
        return load_calibration(name, directory)
    except CalibrationError:
        return None
