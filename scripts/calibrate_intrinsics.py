"""Measure one camera's intrinsics and write its calibration YAML.

Runs on the rig. During capture it shows a live preview with the detected
ChArUco corners, frame coverage, and board tilt. After every shot it also
reports the same guidance in the terminal, because coverage and tilt are what
separate a calibration that converges from one that quietly does not.

Capture and solve are separate so a session can be re-solved without
re-shooting -- useful when the printed square turns out not to be exactly
30 mm and the real dimension has to be supplied.

Examples:

    python scripts/calibrate_intrinsics.py --camera front
    python scripts/calibrate_intrinsics.py --camera front --width 1280 --height 720
    python scripts/calibrate_intrinsics.py --camera front --solve-only --square-mm 31.7
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from so101.camera_calibration import (  # noqa: E402
    CameraCalibration,
    calibration_path,
)
from so101.charuco import (  # noqa: E402
    board_corner_count,
    board_object_points,
    charuco_board,
    detect_board,
    load_board_spec,
)
from so101.intrinsics import (  # noqa: E402
    centered_virtual_matrix,
    distortion_pixel_magnitude,
    horizontal_fov_loss,
    raw_horizontal_fov_deg,
    virtual_horizontal_fov_deg,
)
from so101.real.cameras import (  # noqa: E402
    DEFAULT_SPECS,
    Camera,
    CameraSpec,
    calibration_name,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DEFAULT_CAPTURE_ROOT = REPO_ROOT / "outputs" / "intrinsics"
COVERAGE_GRID = 3
RMS_GATE_PX = 0.3
FOV_LOSS_REVIEW = 0.10
MIN_TILT_DEG = 25.0
MIN_TILTED_SHOTS = 5


def coverage_cells(corners: np.ndarray, size: tuple[int, int]) -> set[tuple[int, int]]:
    width, height = size
    cells = set()
    for point in corners.reshape(-1, 2):
        col = min(int(point[0] / width * COVERAGE_GRID), COVERAGE_GRID - 1)
        row = min(int(point[1] / height * COVERAGE_GRID), COVERAGE_GRID - 1)
        cells.add((row, col))
    return cells


def approximate_tilt_deg(
    corners: np.ndarray, ids: np.ndarray, board, size: tuple[int, int]
) -> float | None:
    """Rough angle between the board normal and the optical axis.

    Only for guiding the operator, so a generic lens guess is good enough --
    the real intrinsics are what this session is trying to find.
    """
    width, height = size
    guess = np.array(
        [[0.9 * width, 0, width / 2], [0, 0.9 * width, height / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    object_points = board_object_points(board, ids)
    if len(object_points) < 6:
        return None
    ok, rvec, _ = cv2.solvePnP(
        object_points,
        corners.reshape(-1, 2).astype(np.float64),
        guess,
        np.zeros(5),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    normal = rotation @ np.array([0.0, 0.0, 1.0])
    cosine = abs(float(normal @ np.array([0.0, 0.0, 1.0])))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def describe_gaps(covered: set[tuple[int, int]]) -> str:
    names = {
        (0, 0): "top-left", (0, 1): "top", (0, 2): "top-right",
        (1, 0): "left", (1, 1): "centre", (1, 2): "right",
        (2, 0): "bottom-left", (2, 1): "bottom", (2, 2): "bottom-right",
    }
    missing = [names[cell] for cell in sorted(names) if cell not in covered]
    return ", ".join(missing) if missing else "none"


def gui_available() -> bool:
    """Return whether Matplotlib selected an interactive window backend."""
    import matplotlib

    backend = matplotlib.get_backend().lower()
    return not backend.endswith("agg") or backend in {"tkagg", "qtagg", "qt5agg"}


class PreviewWindow:
    """Small Matplotlib-backed viewer that also works with headless OpenCV."""

    def __init__(self, title: str) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self._key: str | None = None
        self._closed = False
        self._figure, self._axes = plt.subplots()
        self._figure.canvas.manager.set_window_title(title)
        self._figure.canvas.mpl_connect("key_press_event", self._on_key)
        self._figure.canvas.mpl_connect("close_event", self._on_close)
        self._axes.axis("off")
        self._image = None
        plt.show(block=False)

    def _on_key(self, event) -> None:
        self._key = event.key

    def _on_close(self, _event) -> None:
        self._closed = True

    def show(self, frame: np.ndarray) -> str | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self._image is None:
            self._image = self._axes.imshow(rgb)
            self._figure.tight_layout(pad=0)
        else:
            self._image.set_data(rgb)
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()
        self._plt.pause(0.001)
        key, self._key = self._key, None
        return "q" if self._closed else key

    def close(self) -> None:
        self._plt.close(self._figure)


def preview_frame(
    frame: np.ndarray,
    detection,
    camera_name: str,
    index: int,
    wanted_shots: int,
    covered: set[tuple[int, int]],
    tilt: float | None,
) -> np.ndarray:
    canvas = frame.copy()
    if detection.count:
        cv2.aruco.drawDetectedCornersCharuco(
            canvas, detection.charuco_corners, detection.charuco_ids, (0, 255, 0)
        )

    lines = [
        f"{camera_name}  shots: {index}/{wanted_shots}  corners: {detection.count}/{board_corner_count()}",
        f"tilt: {tilt:.0f} deg" if tilt is not None else "tilt: --",
        f"uncovered: {describe_gaps(covered)}",
        "SPACE/ENTER capture   D drop last   Q/ESC finish",
    ]
    for line_index, line in enumerate(lines):
        y = 22 + line_index * 22
        cv2.putText(
            canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    return canvas


def capture_session(
    camera_name: str,
    capture_dir: Path,
    auto: int,
    interval: float,
    square_mm: float | None,
    spec: CameraSpec,
    wanted_shots: int,
    preview: bool,
) -> None:
    board = charuco_board(square_mm) if square_mm else charuco_board()
    size = (spec.width, spec.height)
    capture_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(capture_dir.glob("shot_*.png"))
    index = len(existing)
    if existing:
        print(f"[info] {index} existing shots in {capture_dir}; continuing from there")

    covered: set[tuple[int, int]] = set()
    tilts: list[float] = []
    preview_window = None

    # Rectification is off: this script exists to measure the distortion that
    # rectification removes.
    with Camera(spec, calibration=None, rectify=False) as camera:
        if preview:
            preview_window = PreviewWindow(f"Intrinsic calibration - {camera_name}")
        print(f"[info] {camera_name} open on {spec.device} at {size[0]}x{size[1]}")
        print(f"[info] exposure locked: {camera.exposure_lock.as_dict()}")
        print(
            f"\nAim for about {wanted_shots} shots covering every part of the "
            f"frame, including at\nleast {MIN_TILTED_SHOTS} tilted "
            f"{MIN_TILT_DEG:.0f} deg or more. Head-on shots alone cannot "
            "separate\nfocal length from distance.\n"
        )
        if preview:
            print("[info] preview: Space/Enter=capture, d=drop last, q/Esc=finish")
        next_auto_capture = time.monotonic() + interval
        try:
            while True:
                if auto and index >= auto:
                    break

                if not preview and not auto:
                    answer = input("[Enter] capture, 'q' finish, 'd' drop last: ").strip()
                    if answer.lower() == "q":
                        break
                    if answer.lower() == "d":
                        if index > 0:
                            index -= 1
                            (capture_dir / f"shot_{index:03d}.png").unlink(missing_ok=True)
                            print(f"  dropped shot_{index:03d}.png")
                        continue

                # The preview loop continuously consumes the stream, so its
                # current frame is already fresh. Terminal mode blocks between
                # shots and must explicitly discard frames queued meanwhile.
                frame = camera.read_raw() if preview else camera.read_fresh(raw=True)
                detection = detect_board(frame, board)
                tilt = None
                if detection.usable():
                    tilt = approximate_tilt_deg(
                        detection.charuco_corners, detection.charuco_ids, board, size
                    )

                capture_requested = not preview and not auto
                if preview_window is not None:
                    key = preview_window.show(
                        preview_frame(
                            frame,
                            detection,
                            camera_name,
                            index,
                            wanted_shots,
                            covered,
                            tilt,
                        )
                    )
                    if key in ("q", "escape"):
                        break
                    if key == "d":
                        if index > 0:
                            index -= 1
                            (capture_dir / f"shot_{index:03d}.png").unlink(missing_ok=True)
                            print(f"  dropped shot_{index:03d}.png")
                        continue
                    capture_requested = key in (" ", "enter")

                if auto:
                    now = time.monotonic()
                    capture_requested = now >= next_auto_capture
                    if capture_requested:
                        next_auto_capture = now + interval

                if not capture_requested:
                    if not preview and auto:
                        time.sleep(
                            min(0.02, max(0.0, next_auto_capture - time.monotonic()))
                        )
                    continue

                if not detection.usable():
                    print(f"  rejected: only {detection.count} corners found")
                    continue

                covered |= coverage_cells(detection.charuco_corners, size)
                if tilt is not None:
                    tilts.append(tilt)

                path = capture_dir / f"shot_{index:03d}.png"
                cv2.imwrite(str(path), frame)
                index += 1
                tilted = sum(1 for t in tilts if t >= MIN_TILT_DEG)
                print(
                    f"  saved {path.name}: {detection.count}/{board_corner_count()} corners, "
                    f"tilt~{tilt:.0f} deg" if tilt is not None else f"  saved {path.name}"
                )
                print(
                    f"    shots={index}  uncovered regions: {describe_gaps(covered)}  "
                    f"tilted>={MIN_TILT_DEG:.0f}deg: {tilted}/{MIN_TILTED_SHOTS}"
                )
        finally:
            if preview_window is not None:
                preview_window.close()

    print(f"\n[info] captured {index} shots into {capture_dir}")


def solve(
    camera_name: str,
    capture_dir: Path,
    square_mm: float | None,
    output: Path,
    rms_gate: float,
    spec: CameraSpec,
) -> int:
    board = charuco_board(square_mm) if square_mm else charuco_board()
    size = (spec.width, spec.height)

    shots = sorted(capture_dir.glob("shot_*.png"))
    if len(shots) < 6:
        print(f"[fail] only {len(shots)} shots in {capture_dir}; need at least 6")
        return 1

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    covered: set[tuple[int, int]] = set()
    tilts: list[float] = []
    used = 0
    for path in shots:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        if (frame.shape[1], frame.shape[0]) != size:
            print(
                f"[fail] {path.name} is {frame.shape[1]}x{frame.shape[0]}, "
                f"expected {size[0]}x{size[1]}. Calibrate at the resolution you "
                "run at -- USB cameras crop or bin differently per mode."
            )
            return 1
        detection = detect_board(frame, board)
        if not detection.usable():
            print(f"  skipping {path.name}: {detection.count} corners")
            continue
        object_points.append(board_object_points(board, detection.charuco_ids))
        image_points.append(
            detection.charuco_corners.reshape(-1, 2).astype(np.float64)
        )
        covered |= coverage_cells(detection.charuco_corners, size)
        tilt = approximate_tilt_deg(
            detection.charuco_corners, detection.charuco_ids, board, size
        )
        if tilt is not None:
            tilts.append(tilt)
        used += 1

    if used < 6:
        print(f"[fail] only {used} usable shots")
        return 1

    board_square_mm = square_mm or load_board_spec().square_mm
    # Tangential distortion (p1, p2) aliases with the principal point when few
    # images are available: fitting it from a modest set couples cx/cy to noise
    # in p1/p2 rather than constraining them independently.  Measured on
    # synthetic data, this alone roughly doubled the fraction of 24-shot
    # sessions landing cx within 2 px (48% -> 84%).  Consumer/webcam lenses
    # have tangential distortion small enough that fixing it at zero costs
    # negligible accuracy on the radial terms that matter.
    rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        [p.astype(np.float32) for p in object_points],
        [p.astype(np.float32) for p in image_points],
        size,
        None,
        None,
        flags=cv2.CALIB_ZERO_TANGENT_DIST,
    )
    distortion = np.asarray(distortion).reshape(-1)

    virtual = centered_virtual_matrix(camera_matrix, distortion, size)
    fov_loss = horizontal_fov_loss(camera_matrix, distortion, virtual, size)
    raw_fov = raw_horizontal_fov_deg(camera_matrix, distortion, size)
    virtual_fov = virtual_horizontal_fov_deg(virtual, size)
    distortion_px = distortion_pixel_magnitude(camera_matrix, distortion, size)

    print("\n" + "=" * 68)
    print(
        f"INTRINSICS: {camera_name} at {size[0]}x{size[1]}  "
        f"({used} shots, square {board_square_mm:.2f} mm)"
    )
    print("=" * 68)
    print(f"  RMS reprojection : {rms:.4f} px   (gate < {rms_gate})")
    print(
        f"  fx={camera_matrix[0,0]:.3f}  fy={camera_matrix[1,1]:.3f}  "
        f"cx={camera_matrix[0,2]:.2f}  cy={camera_matrix[1,2]:.2f}"
    )
    print(f"  distortion       : {np.round(distortion, 5).tolist()}")
    print(f"  max distortion   : {distortion_px:.2f} px displacement")
    print(f"  virtual focal    : {virtual[0,0]:.3f} px  (centred, square pixels)")
    print(f"  hFOV raw/virtual : {raw_fov:.2f} / {virtual_fov:.2f} deg")
    print(f"  hFOV loss        : {fov_loss * 100:.2f} %   (review above {FOV_LOSS_REVIEW * 100:.0f} %)")
    print(f"  uncovered regions: {describe_gaps(covered)}")
    tilted = sum(1 for t in tilts if t >= MIN_TILT_DEG)
    print(f"  tilted shots     : {tilted} at >= {MIN_TILT_DEG:.0f} deg")

    problems = []
    if rms > rms_gate:
        problems.append(f"RMS {rms:.4f} px exceeds the {rms_gate} px gate")
    if describe_gaps(covered) != "none":
        problems.append(f"frame regions with no coverage: {describe_gaps(covered)}")
    if tilted < MIN_TILTED_SHOTS:
        problems.append(
            f"only {tilted} shots tilted >= {MIN_TILT_DEG:.0f} deg "
            f"(need {MIN_TILTED_SHOTS}); focal length and distance stay entangled"
        )

    calibration = CameraCalibration(
        name=calibration_name(camera_name, spec.width, spec.height),
        device=spec.device,
        resolution=size,
        fps=spec.fps,
        camera_matrix=camera_matrix,
        distortion=distortion,
        virtual_matrix=virtual,
        alpha=0.0,
        extrinsic=None,
        exposure={},
        calib_rms_px=float(rms),
        hfov_loss_frac=float(fov_loss),
        date=date.today().isoformat(),
        notes=f"{used} shots at {size[0]}x{size[1]}, square {board_square_mm:.2f} mm",
    )

    if problems:
        print("\n  NOT ACCEPTED:")
        for problem in problems:
            print(f"    - {problem}")
        draft = capture_dir / f"{camera_name}.rejected.yaml"
        calibration.save(draft)
        print(f"\n  draft written to {draft} for inspection; {output} left untouched")
        return 1

    calibration.save(output)
    print(f"\n  ACCEPTED -> {output}")
    if fov_loss > FOV_LOSS_REVIEW:
        print(
            f"  NOTE: rectification gives up {fov_loss * 100:.1f} % of horizontal FOV. "
            "Per the plan this triggers a review of whether to model distortion in "
            "Isaac Sim instead (see the conditional-response issue)."
        )
    print(
        "\n  CAVEAT: a low RMS here does not by itself prove cx/cy are right. On\n"
        "  synthetic data with this same board and detector, an RMS well under\n"
        f"  {rms_gate} px still came with the principal point off by more than 2 px\n"
        "  in roughly 1 in 3 sessions -- fx/distortion/principal-point can trade off\n"
        "  against each other while reprojection error stays low. This gate cannot\n"
        "  see that trade-off; the alignment gate downstream (verify_alignment.py\n"
        "  --gate) is what actually has to pass, and the diagnosis order in that\n"
        "  script's failure message starts here for a reason."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate one camera's intrinsics and write its YAML."
    )
    parser.add_argument("--camera", required=True, choices=sorted(DEFAULT_SPECS))
    parser.add_argument("--capture-dir", type=Path, default=None)
    parser.add_argument(
        "--square-mm",
        type=float,
        default=None,
        help="measured printed square size in mm; defaults to calibration/board.yaml",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="readout size; intrinsics are per-resolution, so a non-default size "
        "is written to its own file",
    )
    parser.add_argument("--solve-only", action="store_true")
    parser.add_argument(
        "--auto",
        type=int,
        default=0,
        help="capture this many shots automatically instead of prompting",
    )
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="capture through terminal prompts without opening a live window",
    )
    parser.add_argument("--rms-gate", type=float, default=RMS_GATE_PX)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.solve_only and not args.no_preview and not gui_available():
        print(
            "[fail] Matplotlib has no interactive GUI backend.\n"
            "       Run this command from a desktop session,\n"
            "       or use --no-preview for terminal-only capture."
        )
        return 1

    default_spec = DEFAULT_SPECS[args.camera]
    spec = CameraSpec(
        name=default_spec.name,
        device=default_spec.device,
        width=args.width or default_spec.width,
        height=args.height or default_spec.height,
        fps=default_spec.fps,
        fourcc=default_spec.fourcc,
        lock_exposure=default_spec.lock_exposure,
        lock_white_balance=default_spec.lock_white_balance,
        target_brightness=default_spec.target_brightness,
    )
    stem = calibration_name(args.camera, spec.width, spec.height)

    capture_dir = args.capture_dir or (DEFAULT_CAPTURE_ROOT / stem)
    output = args.output or calibration_path(stem)

    board_spec = load_board_spec()
    # A small board constrains the principal point weakly, and measured on
    # synthetic data (60 independent pose sets per count, same board and
    # detector as here) the odds of landing cx within 2 px climb with more
    # shots and then plateau: 28 shots -> 58%, 32 -> 68%, 36 -> 77%,
    # 40 -> 82%, 48 -> 78% (no further gain -- this board's ceiling).  40 is
    # therefore the target, not a number to stop short of.  Disabling
    # tangential-distortion estimation below (p1/p2 alias with the principal
    # point when views are limited) does not let the count come down much,
    # but it does cut the worst case roughly in half (max cx error ~5 px ->
    # ~3 px at 40 shots).  No shot count is a guarantee -- the alignment gate
    # downstream, not this number, is what actually has to pass.
    wanted_shots = 40 if board_spec.width_mm < 300 else 22
    print(
        f"[info] board {board_spec.cols}x{board_spec.rows}, square "
        f"{board_spec.square_mm:.1f} mm, {board_spec.corner_count} corners"
    )
    if args.square_mm:
        print(
            f"[info] using measured square {args.square_mm} mm instead of the "
            f"recorded {board_spec.square_mm} mm"
        )

    if not args.solve_only:
        capture_session(
            args.camera, capture_dir, args.auto, args.interval, args.square_mm,
            spec, wanted_shots, preview=not args.no_preview,
        )
    return solve(args.camera, capture_dir, args.square_mm, output, args.rms_gate, spec)


if __name__ == "__main__":
    raise SystemExit(main())
