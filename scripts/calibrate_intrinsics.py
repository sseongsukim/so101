"""Measure one camera's intrinsics and write its calibration YAML.

Runs on the rig, and deliberately gives its feedback through the terminal:
the installed OpenCV is the headless wheel, so there is no preview window.
After every shot it reports which parts of the frame still have no coverage
and how much the board has been tilted, because those two things are what
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


def capture_session(
    camera_name: str,
    capture_dir: Path,
    auto: int,
    interval: float,
    square_mm: float | None,
    spec: CameraSpec,
    wanted_shots: int,
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

    # Rectification is off: this script exists to measure the distortion that
    # rectification removes.
    with Camera(spec, calibration=None, rectify=False) as camera:
        print(f"[info] {camera_name} open on {spec.device} at {size[0]}x{size[1]}")
        print(f"[info] exposure locked: {camera.exposure_lock.as_dict()}")
        print(
            f"\nAim for about {wanted_shots} shots covering every part of the "
            f"frame, including at\nleast {MIN_TILTED_SHOTS} tilted "
            f"{MIN_TILT_DEG:.0f} deg or more. Head-on shots alone cannot "
            "separate\nfocal length from distance.\n"
        )
        while True:
            if auto:
                if index >= auto:
                    break
                time.sleep(interval)
            else:
                answer = input("[Enter] capture, 'q' finish, 'd' drop last: ").strip()
                if answer.lower() == "q":
                    break
                if answer.lower() == "d":
                    if index > 0:
                        index -= 1
                        (capture_dir / f"shot_{index:03d}.png").unlink(missing_ok=True)
                        print(f"  dropped shot_{index:03d}.png")
                    continue

            frame = camera.read_fresh(raw=True)
            detection = detect_board(frame, board)
            if not detection.usable():
                print(f"  rejected: only {detection.count} corners found")
                continue

            cells = coverage_cells(detection.charuco_corners, size)
            covered |= cells
            tilt = approximate_tilt_deg(
                detection.charuco_corners, detection.charuco_ids, board, size
            )
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
    rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        [p.astype(np.float32) for p in object_points],
        [p.astype(np.float32) for p in image_points],
        size,
        None,
        None,
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
    parser.add_argument("--rms-gate", type=float, default=RMS_GATE_PX)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    default_spec = DEFAULT_SPECS[args.camera]
    spec = CameraSpec(
        name=default_spec.name,
        device=default_spec.device,
        width=args.width or default_spec.width,
        height=args.height or default_spec.height,
        fps=default_spec.fps,
        fourcc=default_spec.fourcc,
    )
    stem = calibration_name(args.camera, spec.width, spec.height)

    capture_dir = args.capture_dir or (DEFAULT_CAPTURE_ROOT / stem)
    output = args.output or calibration_path(stem)

    board_spec = load_board_spec()
    # A small board constrains the principal point weakly, and that is fixed by
    # taking more images rather than by taking better ones.
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
            spec, wanted_shots,
        )
    return solve(args.camera, capture_dir, args.square_mm, output, args.rms_gate, spec)


if __name__ == "__main__":
    raise SystemExit(main())
