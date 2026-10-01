"""Capture layer for the physical cameras used by the SO-101 rig.

Both cameras are opened through V4L2 with :class:`cv2.VideoCapture`, so the
wrist (Innomaker U20CAM-720P) and the front (Orbbec Gemini colour stream) go
down one identical code path.  Depth is deliberately out of scope, which is
what lets the Orbbec avoid a vendor SDK dependency.

Two properties of this module matter for sim-to-real:

* :meth:`Camera.read` always returns a **rectified** frame.  Rectification
  lives inside the capture layer rather than in each caller so that the
  training, inference and verification paths cannot disagree about
  preprocessing -- a mismatch there is the classic silent sim-to-real break.
  :meth:`Camera.read_raw` exists for the calibration scripts, which need the
  distortion they are trying to measure.
* Auto exposure and auto white balance are locked.  Isaac's renderer has no
  concept of auto exposure, so leaving it enabled on the real side injects a
  source of variation that has no counterpart in simulation.  The values that
  were locked in are recorded in the calibration YAML.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

import cv2
import numpy as np

from so101.camera_calibration import CameraCalibration, try_load_calibration

logger = logging.getLogger(__name__)

# UVC menu value shared by both cameras; 1 is "Manual Mode" on each.
AUTO_EXPOSURE_MANUAL = 1


@dataclass
class CameraSpec:
    """Which device to open and in what mode."""

    name: str
    device: str
    width: int = 640
    height: int = 480
    fps: int = 30
    # YUYV is uncompressed, so checkerboard corners are not softened by JPEG
    # ringing.  Switch to "MJPG" if USB bandwidth cannot carry two uncompressed
    # streams at once; calibration itself runs one camera at a time.
    fourcc: str = "YUYV"
    lock_exposure: bool = True
    lock_white_balance: bool = True
    target_brightness: float = 100.0
    # Fixed manual exposure (exposure_time_absolute, 100 us units) and sensor
    # gain; None meters the exposure / leaves the gain alone.
    exposure: int | None = None
    gain: int | None = None
    # Snap metered exposures to multiples of this (exposure_time_absolute
    # units, 100 us): 83 = 8.3 ms, one period of the 120 Hz flicker of 60 Hz
    # mains lighting. Other exposure times show dark horizontal bands.
    flicker_step: int | None = None


DEFAULT_SPECS: dict[str, CameraSpec] = {
    # This Innomaker's manual white-balance temperature does not reproduce
    # the gains selected by its automatic mode: switching auto off at the
    # reported 4600 K gives a strong cyan cast.  Its manual exposure behaves
    # similarly poorly on this unit, so keep both controls automatic.  MJPEG
    # is also the camera's cleanest 30 fps output at 640x480.
    # Exposure, though, is fixed (2026-09-29): in auto mode this camera also
    # raises its sensor gain, and at the policy's start pose it lifted the
    # black tabletop to mean gray ~95 with heavy noise, while the simulated
    # wrist view sits at ~42 (29-69 under rendering randomization). Manual
    # exposure 120 with gain 0 measured ~47 there, with visibly less noise.
    # Colour balance stays automatic for the reason above.
    "wrist": CameraSpec(
        name="wrist",
        device="/dev/video0",
        fourcc="MJPG",
        lock_exposure=True,
        lock_white_balance=False,
        # 83 (8.3 ms), not the 120 first chosen: 12 ms is not a multiple of the
        # lights' 8.3 ms flicker period and banded the image.
        exposure=83,
        gain=0,
        flicker_step=83,
    ),
    # Fixed rather than re-metered on every open (2026-09-29): metering
    # depends on what is in view -- arm pose, cubes, daylight through the
    # windows -- so demo and deployment sessions would each get a different
    # exposure. 6187 is what metering chose with the cubes on the table, the
    # frame the simulated cube colors were fitted to (match_cube_colors.py).
    # Re-measure if the room lighting changes a lot.
    "front": CameraSpec(name="front", device="/dev/video6", exposure=6187),
}


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened or configured as requested."""


def _v4l2_available() -> bool:
    return shutil.which("v4l2-ctl") is not None


def _v4l2_controls(device: str) -> set[str]:
    """Names of the V4L2 controls this device exposes."""
    if not _v4l2_available():
        return set()
    try:
        out = subprocess.run(
            ["v4l2-ctl", "-d", device, "--list-ctrls"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError) as err:
        logger.warning("could not list controls for %s: %s", device, err)
        return set()
    names: set[str] = set()
    for line in out.splitlines():
        line = line.strip()
        if " 0x" in line:
            names.add(line.split(" ", 1)[0])
    return names


def _v4l2_get(device: str, control: str) -> int | None:
    if not _v4l2_available():
        return None
    try:
        out = subprocess.run(
            ["v4l2-ctl", "-d", device, f"--get-ctrl={control}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if ":" not in out:
        return None
    try:
        return int(out.split(":", 1)[1].strip())
    except ValueError:
        return None


def _v4l2_range(device: str, control: str) -> tuple[int, int] | None:
    """The ``min``/``max`` a control accepts, parsed from ``--list-ctrls``."""
    if not _v4l2_available():
        return None
    try:
        out = subprocess.run(
            ["v4l2-ctl", "-d", device, "--list-ctrls"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        stripped = line.strip()
        if not stripped.startswith(control):
            continue
        low = high = None
        for token in stripped.split():
            if token.startswith("min="):
                low = int(token[4:])
            elif token.startswith("max="):
                high = int(token[4:])
        if low is not None and high is not None:
            return (low, high)
    return None


def _v4l2_set(device: str, control: str, value: int) -> bool:
    if not _v4l2_available():
        return False
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, f"--set-ctrl={control}={value}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as err:
        logger.warning("failed to set %s=%s on %s: %s", control, value, device, err)
        return False
    if result.returncode != 0:
        logger.warning(
            "failed to set %s=%s on %s: %s",
            control,
            value,
            device,
            result.stderr.strip(),
        )
        return False
    return True


@dataclass
class ExposureLock:
    """What was actually locked, and what the device refused to lock.

    ``unsupported`` is not a failure: the Orbbec Gemini exposes no white
    balance control over V4L2 at all, so its colour balance stays automatic and
    that fact belongs in the record rather than in a silent assumption.
    """

    applied: dict[str, int] = field(default_factory=dict)
    unsupported: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = dict(self.applied)
        if self.unsupported:
            payload["unsupported_controls"] = list(self.unsupported)
        return payload


class Camera:
    """One physical camera, optionally rectified onto its virtual twin."""

    def __init__(
        self,
        spec: CameraSpec,
        calibration: CameraCalibration | None = None,
        *,
        rectify: bool = True,
        lock_exposure: bool | None = None,
        lock_white_balance: bool | None = None,
        exposure: int | None = None,
        white_balance: int | None = None,
        settle_frames: int = 30,
        target_brightness: float | None = None,
        brightness_tolerance: float = 10.0,
        metering_fraction: float = 0.5,
    ) -> None:
        self.spec = spec
        self.calibration = calibration
        self._want_rectify = rectify
        self._lock_exposure = (
            spec.lock_exposure if lock_exposure is None else lock_exposure
        )
        self._lock_white_balance = (
            spec.lock_white_balance
            if lock_white_balance is None
            else lock_white_balance
        )
        self._exposure = exposure if exposure is not None else spec.exposure
        self._white_balance = white_balance
        self._settle_frames = settle_frames
        self._target_brightness = (
            spec.target_brightness
            if target_brightness is None
            else target_brightness
        )
        self._brightness_tolerance = brightness_tolerance
        self._metering_fraction = metering_fraction
        self._capture: cv2.VideoCapture | None = None
        self._maps: tuple[np.ndarray, np.ndarray] | None = None
        self.exposure_lock = ExposureLock()

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> Camera:
        capture = cv2.VideoCapture(self.spec.device, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise CameraError(f"could not open {self.spec.device}")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.spec.fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.spec.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec.height)
        capture.set(cv2.CAP_PROP_FPS, self.spec.fps)
        # CAP_PROP_BUFFERSIZE is deliberately left alone.  Setting it to 1 looks
        # like the right way to keep frames fresh, but measured on this rig it
        # drops the wrist camera from 30.1 to 21.2 fps while the driver itself
        # sustains 30.15.  Freshness is handled explicitly instead, by
        # read_fresh(), which only the capture-at-a-pose paths need.
        self._capture = capture

        actual = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if actual != (self.spec.width, self.spec.height):
            self.close()
            raise CameraError(
                f"{self.spec.device} gave {actual[0]}x{actual[1]}, "
                f"expected {self.spec.width}x{self.spec.height}"
            )

        # Apply the requested policy explicitly.  Merely "leaving" an auto
        # control alone is not enough: UVC controls persist after a process
        # exits, so a previous manual capture can otherwise poison the next
        # session.
        self._apply_exposure_lock()
        self._prepare_rectification()
        return self

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> Camera:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- configuration -----------------------------------------------------

    def _apply_exposure_lock(self) -> None:
        device = self.spec.device
        controls = _v4l2_controls(device)
        lock = ExposureLock()

        if not self._lock_exposure:
            if "auto_exposure" in controls:
                _v4l2_set(device, "auto_exposure", 3)
                lock.applied["auto_exposure"] = 3
            logger.info("%s: using automatic exposure", device)
        elif "auto_exposure" in controls:
            _v4l2_set(device, "auto_exposure", AUTO_EXPOSURE_MANUAL)
            lock.applied["auto_exposure"] = AUTO_EXPOSURE_MANUAL
            # Gain first: metering below must see the gain it will run with.
            if self.spec.gain is not None and "gain" in controls:
                if _v4l2_set(device, "gain", int(self.spec.gain)):
                    lock.applied["gain"] = int(self.spec.gain)
            exposure = self._exposure
            if exposure is None and "exposure_time_absolute" in controls and self.spec.flicker_step:
                exposure = self._search_exposure(device)
                if exposure is not None:
                    step = int(self.spec.flicker_step)
                    exposure = max(step, int(round(exposure / step)) * step)
            if exposure is None and "exposure_time_absolute" in controls:
                # Reading exposure_time_absolute while auto exposure is running
                # returns the driver's default, not what the metering chose --
                # the control is flagged inactive.  Locking that value produces
                # an image unrelated to the actual lighting, which on this rig
                # blew the wrist camera out to pure white and would have made
                # board detection fail on site.  Meter it here instead.
                exposure = self._search_exposure(device)
            if exposure is not None and "exposure_time_absolute" in controls:
                if _v4l2_set(device, "exposure_time_absolute", int(exposure)):
                    lock.applied["exposure_time_absolute"] = int(exposure)
        else:
            lock.unsupported.append("auto_exposure")

        # Holding the frame rate steady matters as much as holding exposure:
        # this control lets the driver drop below 30 fps in dim light.
        if self._lock_exposure and "exposure_dynamic_framerate" in controls:
            if _v4l2_set(device, "exposure_dynamic_framerate", 0):
                lock.applied["exposure_dynamic_framerate"] = 0

        if not self._lock_white_balance:
            if "white_balance_automatic" in controls:
                _v4l2_set(device, "white_balance_automatic", 1)
                lock.applied["white_balance_automatic"] = 1
                # This camera needs more than one second for its colour gains
                # to converge after manual mode has been left behind.
                self._drain(self._settle_frames * 3)
            logger.info("%s: using automatic white balance", device)
        elif "white_balance_automatic" in controls:
            white_balance = self._white_balance
            if white_balance is None:
                # Let automatic white balance settle and read what it chose.
                # Unlike exposure this control reports a live value, so the
                # read is meaningful.
                self._drain(self._settle_frames)
                white_balance = _v4l2_get(device, "white_balance_temperature")
            _v4l2_set(device, "white_balance_automatic", 0)
            if white_balance is not None and "white_balance_temperature" in controls:
                if _v4l2_set(device, "white_balance_temperature", int(white_balance)):
                    lock.applied["white_balance_temperature"] = int(white_balance)
            lock.applied["white_balance_automatic"] = 0
        else:
            lock.unsupported.append("white_balance_automatic")
            logger.warning(
                "%s exposes no white balance control over V4L2; colour balance "
                "stays automatic for this camera",
                device,
            )

        self.exposure_lock = lock

    def _mean_brightness(self, settle: int = 30) -> float:
        """Mean brightness of the central region.

        Metering the whole frame lets bright background -- lab windows and
        ceiling in the front camera's case -- pull the exposure down until the
        tabletop, which is the only part that has to be readable, is too dark.
        The centre is where the board and the cubes are.
        """
        for _ in range(settle):
            self._capture.read()  # type: ignore[union-attr]
        ok, frame = self._capture.read()  # type: ignore[union-attr]
        if not ok or frame is None:
            return float("nan")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        height, width = gray.shape[:2]
        margin = (1.0 - self._metering_fraction) / 2.0
        y0, y1 = int(height * margin), int(height * (1.0 - margin))
        x0, x1 = int(width * margin), int(width * (1.0 - margin))
        return float(gray[y0:y1, x0:x1].mean())

    def _search_exposure(self, device: str) -> int | None:
        """Pick a manual exposure that lands near ``self._target_brightness``.

        Brightness rises monotonically with exposure time, so a (log-scale)
        bisection is enough.  Doing the metering ourselves means the chosen value is
        reproducible and gets recorded, instead of depending on a driver's
        hidden auto-exposure state.
        """
        bounds = _v4l2_range(device, "exposure_time_absolute")
        if bounds is None:
            return None
        low, high = bounds
        low = max(low, 1)
        best: int | None = None
        best_error = float("inf")
        for _ in range(10):
            if high - low <= 1:
                break
            # Geometric midpoint: brightness scales roughly with exposure
            # time, and the wrist camera's working exposure (~20-120) sits at
            # the bottom of a 1..5000 range, where a linear bisection spends
            # most of its 10 steps far above the answer.
            mid = int(round((low * high) ** 0.5))
            mid = min(max(mid, low + 1), high - 1)
            if not _v4l2_set(device, "exposure_time_absolute", mid):
                return None
            brightness = self._mean_brightness()
            if not np.isfinite(brightness):
                return None
            error = abs(brightness - self._target_brightness)
            if error < best_error:
                best_error, best = error, mid
            if error <= self._brightness_tolerance:
                return mid
            if brightness < self._target_brightness:
                low = mid
            else:
                high = mid
        if best is not None:
            logger.info(
                "%s: metered exposure %d (mean brightness off target by %.1f)",
                device,
                best,
                best_error,
            )
        return best

    def _prepare_rectification(self) -> None:
        if not self._want_rectify:
            self._maps = None
            return
        if self.calibration is None:
            logger.warning(
                "no calibration for camera %r; read() returns raw frames and the "
                "simulation will not match",
                self.spec.name,
            )
            self._maps = None
            return
        if self.calibration.resolution != (self.spec.width, self.spec.height):
            raise CameraError(
                f"{self.spec.name}: calibration is for "
                f"{self.calibration.resolution[0]}x{self.calibration.resolution[1]} "
                f"but the camera is opened at {self.spec.width}x{self.spec.height}. "
                "Calibrate at the resolution you run at -- USB cameras crop or bin "
                "differently per mode, so a rescaled K is quietly wrong."
            )
        self._maps = self.calibration.rectify_maps()

    # -- capture -----------------------------------------------------------

    @property
    def is_rectified(self) -> bool:
        return self._maps is not None

    def _drain(self, frames: int) -> None:
        if self._capture is None or frames <= 0:
            return
        for _ in range(frames):
            self._capture.read()
            time.sleep(1.0 / max(self.spec.fps, 1))

    def read_raw(self) -> np.ndarray:
        """Return the unmodified sensor frame (BGR).

        The calibration scripts use this: they exist to measure the distortion
        that :meth:`read` removes.
        """
        if self._capture is None:
            raise CameraError(f"camera {self.spec.name!r} is not open")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError(f"failed to read a frame from {self.spec.device}")
        return frame

    def read(self) -> np.ndarray:
        """Return a rectified frame (BGR), matching the simulated camera."""
        frame = self.read_raw()
        if self._maps is None:
            return frame
        return cv2.remap(frame, self._maps[0], self._maps[1], cv2.INTER_LINEAR)

    def read_fresh(self, *, raw: bool = False, discard: int = 4) -> np.ndarray:
        """Discard queued frames, then return the current one.

        Used where an image has to correspond to the robot's *present* pose --
        hand-eye capture, intrinsic shots -- because the driver's queue can
        otherwise hand back an image from before the arm stopped moving.  The
        arm is stationary at these moments, so the extra grabs cost nothing
        that matters.
        """
        if self._capture is None:
            raise CameraError(f"camera {self.spec.name!r} is not open")
        for _ in range(max(discard, 0)):
            self._capture.grab()
        return self.read_raw() if raw else self.read()

    def rectify(self, frame: np.ndarray) -> np.ndarray:
        """Apply this camera's rectification to an already-captured frame."""
        if self._maps is None:
            return frame
        return cv2.remap(frame, self._maps[0], self._maps[1], cv2.INTER_LINEAR)


def calibration_name(camera: str, width: int, height: int) -> str:
    """Calibration file stem for a camera at a given readout size.

    Intrinsics are resolution-specific -- a USB camera crops or bins
    differently per mode -- so each one gets its own record.  The *extrinsic*
    is not: where the camera sits is a physical fact, so a pose measured from a
    high-resolution shot applies to the resolution the task runs at.
    """
    default = DEFAULT_SPECS[camera]
    if (width, height) == (default.width, default.height):
        return camera
    return f"{camera}_{width}x{height}"


# Session metering targets: centre-50% gray mean (Camera._mean_brightness) of
# the policies' training frames at the start pose (2026-09-30): synthetic
# front 88 / wrist 55, real demos front 106 / wrist 58. A fixed exposure set
# the day before read front 151 / wrist 148 in the next morning's sunlight,
# so deployment re-meters at the start pose instead (open_camera(meter_to=)).
METER_TARGETS = {"front": 100.0, "wrist": 57.0}


def open_camera(
    name: str,
    *,
    device: str | None = None,
    calibration_dir: str | None = None,
    rectify: bool = True,
    lock_exposure: bool | None = None,
    fourcc: str | None = None,
    width: int | None = None,
    height: int | None = None,
    meter_to: float | None = None,
) -> Camera:
    """Open one of the rig's cameras by role name (``"wrist"`` or ``"front"``).

    ``meter_to`` replaces the spec's fixed exposure with a manual exposure
    metered to that centre brightness on opening (see METER_TARGETS); point
    the camera at the scene it will work in first.
    """
    if name not in DEFAULT_SPECS:
        raise CameraError(
            f"unknown camera {name!r}; expected one of {sorted(DEFAULT_SPECS)}"
        )
    spec = DEFAULT_SPECS[name]
    if any(v is not None for v in (device, fourcc, width, height)):
        spec = CameraSpec(
            name=spec.name,
            device=device or spec.device,
            width=width or spec.width,
            height=height or spec.height,
            fps=spec.fps,
            fourcc=fourcc or spec.fourcc,
            lock_exposure=spec.lock_exposure,
            lock_white_balance=spec.lock_white_balance,
            target_brightness=spec.target_brightness,
            exposure=spec.exposure,
            gain=spec.gain,
            flicker_step=spec.flicker_step,
        )
    if meter_to is not None:
        from dataclasses import replace

        spec = replace(spec, exposure=None, lock_exposure=True, target_brightness=float(meter_to))
    calibration = try_load_calibration(
        calibration_name(name, spec.width, spec.height), calibration_dir
    )
    return Camera(
        spec,
        calibration,
        rectify=rectify,
        lock_exposure=lock_exposure,
    ).open()
