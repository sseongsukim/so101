"""Hand-eye solving, kept separate from the capture workflow so it is testable.

The two things most likely to be silently wrong live here: the eye-to-hand
inversion, and the residual that decides which solver to trust.  Both can be
checked against synthetic data with a known answer, which is what
``scripts/selftest_pipeline.py`` does.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SOLVERS: dict[str, int] = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def pose_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return matrix


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a[:3, :3].T @ b[:3, :3]
    cosine = (np.trace(relative) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


@dataclass
class HandEyeResult:
    """One solver's answer and how well it explains the data."""

    method: str
    transform: np.ndarray
    reprojection_rmse_px: float
    rig_scatter_mm: float


def evaluate(
    solution: np.ndarray,
    gripper_poses: list[np.ndarray],
    target_poses: list[np.ndarray],
    observations: list[tuple[np.ndarray, np.ndarray]],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    eye_in_hand: bool,
) -> tuple[float, float]:
    """Reprojection RMSE, and how rigidly the target actually held still.

    One transform in the chain must be identical at every pose: the board's
    pose in the environment (eye-in-hand), or the tag's pose on the gripper
    (eye-to-hand).  Recomputing it per pose and measuring its scatter is a
    physical check -- it catches a target that shifted or kinematics that does
    not match the real arm, neither of which a reprojection number alone
    distinguishes from lens noise.
    """
    rigs = []
    for gripper, target in zip(gripper_poses, target_poses):
        if eye_in_hand:
            rigs.append(gripper @ solution @ target)
        else:
            rigs.append(np.linalg.inv(gripper) @ solution @ target)

    positions = np.array([rig[:3, 3] for rig in rigs])
    scatter_mm = float(np.linalg.norm(positions.std(axis=0)) * 1000.0)
    # Use the most central sample rather than an average of rotations, which
    # would need care to stay a rotation.
    central = int(np.argmin(np.linalg.norm(positions - positions.mean(0), axis=1)))
    rig = rigs[central]

    squared = []
    for (object_points, image_points), gripper in zip(observations, gripper_poses):
        if eye_in_hand:
            predicted = np.linalg.inv(solution) @ np.linalg.inv(gripper) @ rig
        else:
            predicted = np.linalg.inv(solution) @ gripper @ rig
        rvec, _ = cv2.Rodrigues(predicted[:3, :3])
        projected, _ = cv2.projectPoints(
            np.asarray(object_points, dtype=np.float64),
            rvec,
            predicted[:3, 3],
            camera_matrix,
            distortion,
        )
        squared.append(
            np.sum((projected.reshape(-1, 2) - np.asarray(image_points)) ** 2, axis=1)
        )
    rmse = float(np.sqrt(np.concatenate(squared).mean()))
    return rmse, scatter_mm


def solve_hand_eye(
    gripper_poses: list[np.ndarray],
    target_poses: list[np.ndarray],
    observations: list[tuple[np.ndarray, np.ndarray]],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    eye_in_hand: bool,
) -> list[HandEyeResult]:
    """Run every solver and score each one.

    ``gripper_poses`` are the gripper link's poses in the environment frame,
    ``target_poses`` are ``T_cam_target`` from solvePnP.

    For eye-in-hand the answer is ``T_gripper_camera``.  For eye-to-hand the
    robot transforms are inverted first, which turns OpenCV's eye-in-hand
    formulation into the eye-to-hand one and makes the answer
    ``T_env_camera``.  Getting that inversion backwards yields a plausible and
    completely wrong pose, so it is exercised by the self-test.
    """
    chain = (
        gripper_poses
        if eye_in_hand
        else [np.linalg.inv(pose) for pose in gripper_poses]
    )
    r_chain = [pose[:3, :3] for pose in chain]
    t_chain = [pose[:3, 3] for pose in chain]
    r_target = [pose[:3, :3] for pose in target_poses]
    t_target = [pose[:3, 3] for pose in target_poses]

    results: list[HandEyeResult] = []
    for name, method in SOLVERS.items():
        try:
            rotation, translation = cv2.calibrateHandEye(
                r_chain, t_chain, r_target, t_target, method=method
            )
        except cv2.error:
            continue
        solution = pose_matrix(rotation, translation)
        if not np.all(np.isfinite(solution)):
            continue
        rmse, scatter = evaluate(
            solution,
            gripper_poses,
            target_poses,
            observations,
            camera_matrix,
            distortion,
            eye_in_hand,
        )
        results.append(HandEyeResult(name, solution, rmse, scatter))
    return results


def inter_solver_spread(results: list[HandEyeResult]) -> tuple[float, float]:
    """Worst pairwise disagreement, in millimetres and degrees.

    Methods built on different assumptions agreeing is the cheapest evidence
    the data is sound; disagreement indicts the poses or the kinematics rather
    than any one solver.
    """
    worst_mm = 0.0
    worst_deg = 0.0
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i].transform, results[j].transform
            worst_mm = max(worst_mm, float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000.0))
            worst_deg = max(worst_deg, rotation_angle_deg(a, b))
    return worst_mm, worst_deg


def board_pose_in_env(
    env_gripper_poses: list[np.ndarray],
    gripper_camera: np.ndarray,
    camera_board_poses: list[np.ndarray],
) -> list[np.ndarray]:
    """Where a static board sits, as measured through a wrist-mounted camera.

    ``T_env_board = T_env_gripper · T_gripper_camera · T_camera_board``

    One estimate per observation.  They should all agree, because the board
    does not move; the spread between them is the error bar on everything
    derived from it.
    """
    return [
        gripper @ gripper_camera @ board
        for gripper, board in zip(env_gripper_poses, camera_board_poses)
    ]


def camera_pose_from_board(
    env_board: np.ndarray, camera_board: np.ndarray
) -> np.ndarray:
    """A fixed camera's pose, from its view of a board whose pose is known.

    ``T_env_camera = T_env_board · T_camera_board⁻¹``

    This is what removes the gripper-mounted tag: the board is the shared
    reference, so the camera never has to see anything attached to the robot.
    """
    return env_board @ np.linalg.inv(camera_board)


def consensus(poses: list[np.ndarray]) -> tuple[np.ndarray, float, float]:
    """The most central pose, plus how far the others scatter from it.

    Picking a member rather than averaging keeps the result a valid rigid
    transform without having to average rotations carefully.
    """
    positions = np.array([p[:3, 3] for p in poses])
    scatter_mm = float(np.linalg.norm(positions.std(axis=0)) * 1000.0)
    central = int(np.argmin(np.linalg.norm(positions - positions.mean(0), axis=1)))
    best = poses[central]
    spread_deg = max((rotation_angle_deg(best, p) for p in poses), default=0.0)
    return best, scatter_mm, spread_deg
