"""Check the calibration pipeline against data whose answer is already known.

Every number this project produces is measured, which means nothing about it
looks wrong when it is wrong.  So the maths is first run on synthetic data
built from a ground truth: a camera with chosen intrinsics, a board at chosen
poses, a gripper on a chosen arc.  If the pipeline cannot recover what it was
given here, the problem is the code, and no amount of care at the rig will
help.

Two checks, neither of which needs the simulator or the robot:

* **intrinsics** -- render a ChArUco board through known ``K`` and distortion,
  then calibrate and compare.
* **hand-eye** -- synthesise gripper poses and the matching ``T_cam_target``
  for both mountings, solve, and compare.  The eye-to-hand case is included
  deliberately: inverting the robot transforms the wrong way produces a
  confident, wrong pose that nothing downstream would flag.

Examples:

    python scripts/selftest_pipeline.py
    python scripts/selftest_pipeline.py --keep-images /tmp/selftest
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from so101.charuco import (  # noqa: E402
    BOARD_COLS,
    BOARD_ROWS,
    SQUARE_MM,
    board_object_points,
    charuco_board,
    detect_board,
    tag_object_points,
)
from so101.handeye import (  # noqa: E402
    inter_solver_spread,
    pose_matrix,
    rotation_angle_deg,
    solve_hand_eye,
)
from so101.intrinsics import centered_virtual_matrix  # noqa: E402

BOARD_W = BOARD_COLS * SQUARE_MM / 1000.0
BOARD_H = BOARD_ROWS * SQUARE_MM / 1000.0
SIZE = (640, 480)

K_TRUE = np.array([[598.0, 0.0, 325.0], [0.0, 602.0, 236.0], [0.0, 0.0, 1.0]])
D_TRUE = np.array([-0.095, 0.021, 0.0004, -0.0003, 0.0])

FX_TOLERANCE = 0.01  # 1 %
CENTRE_TOLERANCE_PX = 2.0
HANDEYE_POS_TOLERANCE_MM = 2.0
HANDEYE_ROT_TOLERANCE_DEG = 0.5


def board_texture(scale: int = 40) -> np.ndarray:
    board = charuco_board()
    return board.generateImage(
        (int(BOARD_W * 1000 * scale / 10), int(BOARD_H * 1000 * scale / 10)),
        marginSize=0,
        borderBits=1,
    )


def render_board(
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    texture: np.ndarray,
) -> np.ndarray:
    """Image of the board seen by a camera with these intrinsics and pose.

    Inverse mapping: every output pixel is traced back through the distortion
    model onto the board plane, which keeps the rendered geometry consistent
    with the very model the calibration is expected to recover.
    """
    width, height = SIZE
    tex_h, tex_w = texture.shape
    us, vs = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    pixels = np.stack([us.ravel(), vs.ravel()], axis=1)
    normalised = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2), camera_matrix, distortion
    ).reshape(-1, 2)
    rays = np.concatenate([normalised, np.ones((len(normalised), 1))], axis=1)

    system = np.zeros((len(rays), 3, 3))
    system[:, :, 0] = rotation[:, 0]
    system[:, :, 1] = rotation[:, 1]
    system[:, :, 2] = -rays
    rhs = np.repeat(-translation.reshape(1, 3), len(rays), axis=0)
    solved = np.linalg.solve(system, rhs[..., None])[..., 0]
    x, y, depth = solved[:, 0], solved[:, 1], solved[:, 2]

    px = x / BOARD_W * tex_w
    py = y / BOARD_H * tex_h
    valid = (depth > 0) & (px >= 0) & (px < tex_w - 1) & (py >= 0) & (py < tex_h - 1)
    map_x = np.where(valid, px, -1).reshape(height, width).astype(np.float32)
    map_y = np.where(valid, py, -1).reshape(height, width).astype(np.float32)
    rendered = cv2.remap(
        texture, map_x, map_y, cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT, borderValue=128,
    )
    return cv2.cvtColor(rendered, cv2.COLOR_GRAY2BGR)


def check_intrinsics(keep: Path | None) -> bool:
    print("\n" + "=" * 68)
    print("SELF-TEST 1: intrinsics")
    print("=" * 68)
    texture = board_texture()
    rng = np.random.default_rng(7)
    centre = np.array([BOARD_W / 2, BOARD_H / 2, 0.0])

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    board = charuco_board()
    for index in range(24):
        angles = np.radians(
            [rng.uniform(-42, 42), rng.uniform(-42, 42), rng.uniform(-25, 25)]
        )
        rotation = (
            cv2.Rodrigues(np.array([angles[0], 0, 0]))[0]
            @ cv2.Rodrigues(np.array([0, angles[1], 0]))[0]
            @ cv2.Rodrigues(np.array([0, 0, angles[2]]))[0]
        )
        shift = np.array([rng.uniform(-0.09, 0.09), rng.uniform(-0.07, 0.07), 0.0])
        translation = -rotation @ (centre + shift) + np.array(
            [0, 0, rng.uniform(0.35, 0.65)]
        )
        frame = render_board(K_TRUE, D_TRUE, rotation, translation, texture)
        if keep is not None:
            cv2.imwrite(str(keep / f"shot_{index:03d}.png"), frame)
        detection = detect_board(frame, board)
        if not detection.usable():
            continue
        object_points.append(
            board_object_points(board, detection.charuco_ids).astype(np.float32)
        )
        image_points.append(
            detection.charuco_corners.reshape(-1, 2).astype(np.float32)
        )

    print(f"  usable renders: {len(object_points)}/24")
    if len(object_points) < 8:
        print("  FAIL: too few detections; the renderer or detector is broken")
        return False

    rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, SIZE, None, None
    )
    distortion = np.asarray(distortion).reshape(-1)
    virtual = centered_virtual_matrix(camera_matrix, distortion, SIZE)

    fx_error = abs(camera_matrix[0, 0] - K_TRUE[0, 0]) / K_TRUE[0, 0]
    fy_error = abs(camera_matrix[1, 1] - K_TRUE[1, 1]) / K_TRUE[1, 1]
    cx_error = abs(camera_matrix[0, 2] - K_TRUE[0, 2])
    cy_error = abs(camera_matrix[1, 2] - K_TRUE[1, 2])

    print(f"  RMS reprojection : {rms:.4f} px")
    print(f"  fx error         : {fx_error * 100:.3f} %   (tolerance {FX_TOLERANCE * 100:.0f} %)")
    print(f"  fy error         : {fy_error * 100:.3f} %")
    print(f"  cx / cy error    : {cx_error:.2f} / {cy_error:.2f} px "
          f"(tolerance {CENTRE_TOLERANCE_PX} px)")
    print(f"  virtual focal    : {virtual[0, 0]:.2f} px")

    passed = (
        fx_error < FX_TOLERANCE
        and fy_error < FX_TOLERANCE
        and cx_error < CENTRE_TOLERANCE_PX
        and cy_error < CENTRE_TOLERANCE_PX
    )
    print(f"  -> {'PASS' if passed else 'FAIL'}")
    return passed


def synth_hand_eye(
    eye_in_hand: bool,
    seed: int = 3,
    pixel_noise_px: float = 0.0,
    joint_noise_deg: float = 0.0,
):
    """Build gripper poses and T_cam_target pairs from a known camera pose."""
    rng = np.random.default_rng(seed)

    if eye_in_hand:
        truth = pose_matrix(
            cv2.Rodrigues(np.array([0.78, -0.12, 0.05]))[0],
            np.array([-0.005, -0.060, -0.062]),
        )
        rig = pose_matrix(  # board pose in the environment frame
            cv2.Rodrigues(np.array([0.0, 0.0, 0.3]))[0],
            np.array([0.25, 0.0, 0.03]),
        )
    else:
        truth = pose_matrix(  # camera pose in the environment frame
            cv2.Rodrigues(np.array([-1.9, 0.35, 0.9]))[0],
            np.array([0.62, -0.50, 0.42]),
        )
        rig = pose_matrix(  # tag pose on the gripper
            cv2.Rodrigues(np.array([0.2, 0.9, -0.1]))[0],
            np.array([0.01, -0.03, -0.02]),
        )

    gripper_poses: list[np.ndarray] = []
    target_poses: list[np.ndarray] = []
    observations: list[tuple[np.ndarray, np.ndarray]] = []
    object_points = (
        board_object_points(charuco_board(), np.arange(24).reshape(-1, 1))
        if eye_in_hand
        else tag_object_points()
    )

    attempts = 0
    while len(gripper_poses) < 20 and attempts < 20000:
        attempts += 1
        rotation = cv2.Rodrigues(rng.normal(0.0, 0.9, size=3))[0]
        position = np.array([0.25, 0.0, 0.18]) + rng.normal(0.0, 0.06, size=3)
        gripper = pose_matrix(rotation, position)

        if eye_in_hand:
            target = np.linalg.inv(truth) @ np.linalg.inv(gripper) @ rig
        else:
            target = np.linalg.inv(truth) @ gripper @ rig

        # Every object point -- not just the target origin -- has to sit in
        # front of the lens at a plausible distance.  The board spans 0.21 m,
        # so a tilted one can put corners behind the camera even when its
        # origin is comfortably in view.  Without this the solve still comes
        # out exact while the reprojection residual diverges and stops meaning
        # anything.
        in_camera = (target[:3, :3] @ object_points.T).T + target[:3, 3]
        depths = in_camera[:, 2]
        if depths.min() < 0.10 or depths.max() > 1.00:
            continue

        rvec, _ = cv2.Rodrigues(target[:3, :3])
        projected, _ = cv2.projectPoints(
            object_points.astype(np.float64), rvec, target[:3, 3], K_TRUE, D_TRUE
        )
        projected = projected.reshape(-1, 2)
        if pixel_noise_px:
            projected = projected + rng.normal(0.0, pixel_noise_px, projected.shape)
        if joint_noise_deg:
            # Stand-in for forward-kinematics error: the pose handed to the
            # solver is not quite the pose the image was taken at, which is
            # exactly what cheap servos produce.
            perturb = cv2.Rodrigues(
                rng.normal(0.0, np.radians(joint_noise_deg), size=3)
            )[0]
            gripper = pose_matrix(
                perturb @ gripper[:3, :3],
                gripper[:3, 3] + rng.normal(0.0, 0.0008, size=3),
            )
        gripper_poses.append(gripper)
        target_poses.append(target)
        observations.append((object_points, projected))

    if len(gripper_poses) < 10:
        raise RuntimeError(
            f"synthetic sampler found only {len(gripper_poses)} valid poses"
        )
    return truth, gripper_poses, target_poses, observations


def check_hand_eye() -> bool:
    print("\n" + "=" * 68)
    print("SELF-TEST 2: hand-eye")
    print("=" * 68)
    passed = True
    for eye_in_hand, label in ((True, "wrist / eye-in-hand"), (False, "front / eye-to-hand")):
        truth, gripper_poses, target_poses, observations = synth_hand_eye(eye_in_hand)
        results = solve_hand_eye(
            gripper_poses, target_poses, observations, K_TRUE, D_TRUE, eye_in_hand
        )
        print(f"\n  [{label}]  {len(results)} solvers returned")
        if not results:
            print("    FAIL: no solver produced an answer")
            passed = False
            continue
        for result in results:
            position_error = float(
                np.linalg.norm(result.transform[:3, 3] - truth[:3, 3]) * 1000.0
            )
            rotation_error = rotation_angle_deg(result.transform, truth)
            print(
                f"    {result.method:<11} pos err {position_error:6.3f} mm   "
                f"rot err {rotation_error:6.3f} deg   "
                f"reproj {result.reprojection_rmse_px:7.4f} px   "
                f"scatter {result.rig_scatter_mm:6.3f} mm"
            )
        spread_mm, spread_deg = inter_solver_spread(results)
        print(f"    inter-solver spread: {spread_mm:.3f} mm, {spread_deg:.3f} deg")

        chosen = min(results, key=lambda r: r.reprojection_rmse_px)
        position_error = float(
            np.linalg.norm(chosen.transform[:3, 3] - truth[:3, 3]) * 1000.0
        )
        rotation_error = rotation_angle_deg(chosen.transform, truth)
        ok = (
            position_error < HANDEYE_POS_TOLERANCE_MM
            and rotation_error < HANDEYE_ROT_TOLERANCE_DEG
        )
        print(
            f"    chosen {chosen.method}: {position_error:.3f} mm, "
            f"{rotation_error:.3f} deg -> {'PASS' if ok else 'FAIL'}"
        )
        passed = passed and ok

        # The inversion is what makes eye-to-hand work; solving it the other
        # way must visibly fail, otherwise this test proves nothing.
        wrong = solve_hand_eye(
            gripper_poses,
            target_poses,
            observations,
            K_TRUE,
            D_TRUE,
            not eye_in_hand,
        )
        if wrong:
            best_wrong = min(wrong, key=lambda r: r.reprojection_rmse_px)
            wrong_error = float(
                np.linalg.norm(best_wrong.transform[:3, 3] - truth[:3, 3]) * 1000.0
            )
            print(
                f"    control (wrong mounting): {wrong_error:.1f} mm off -- "
                f"{'detected' if wrong_error > 10.0 else 'NOT DETECTED, test is weak'}"
            )
            passed = passed and wrong_error > 10.0

    print("\n  [noisy data -- does picking by reprojection actually discriminate?]")
    for eye_in_hand, label in ((True, "wrist"), (False, "front")):
        truth, gripper_poses, target_poses, observations = synth_hand_eye(
            eye_in_hand, seed=11, pixel_noise_px=0.3, joint_noise_deg=0.4
        )
        results = solve_hand_eye(
            gripper_poses, target_poses, observations, K_TRUE, D_TRUE, eye_in_hand
        )
        if not results:
            print(f"    {label}: FAIL, no solver returned")
            passed = False
            continue
        errors = {
            r.method: float(np.linalg.norm(r.transform[:3, 3] - truth[:3, 3]) * 1000.0)
            for r in results
        }
        chosen = min(results, key=lambda r: r.reprojection_rmse_px)
        best_possible = min(errors.values())
        spread_mm, _ = inter_solver_spread(results)
        print(
            f"    {label}: chosen {chosen.method} at {errors[chosen.method]:.2f} mm; "
            f"best available {best_possible:.2f} mm; "
            f"worst {max(errors.values()):.2f} mm; spread {spread_mm:.2f} mm"
        )
        # The criterion does not have to find the single best solver, but it
        # must not land on a bad one.
        if errors[chosen.method] > best_possible + 5.0:
            print("      FAIL: reprojection picked a solver far from the best")
            passed = False
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-test the calibration pipeline.")
    parser.add_argument("--keep-images", type=Path, default=None)
    args = parser.parse_args()

    keep = args.keep_images
    if keep is not None:
        keep.mkdir(parents=True, exist_ok=True)
    else:
        temp = tempfile.TemporaryDirectory()
        keep = None

    intrinsics_ok = check_intrinsics(keep)
    hand_eye_ok = check_hand_eye()

    print("\n" + "=" * 68)
    print(f"intrinsics : {'PASS' if intrinsics_ok else 'FAIL'}")
    print(f"hand-eye   : {'PASS' if hand_eye_ok else 'FAIL'}")
    print("=" * 68)
    return 0 if (intrinsics_ok and hand_eye_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
