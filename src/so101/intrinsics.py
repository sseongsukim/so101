"""Intrinsic calibration maths, separated from the capture workflow.

The part worth isolating is choosing the virtual camera.  Isaac Lab can only
render a pinhole with square pixels and a centred principal point, so the
rectification target is not whatever ``getOptimalNewCameraMatrix`` returns --
that generally has ``fx != fy`` and an off-centre principal point.  The focal
length is instead searched for directly: the largest field of view whose every
output pixel still has a real source pixel behind it.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def _virtual_frame_is_covered(
    focal: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    size: tuple[int, int],
    samples: int = 96,
) -> bool:
    """Does every border pixel of the virtual frame come from inside the raw one?

    Checking the border is enough: the mapping is continuous, so if the whole
    boundary lands inside the source image the interior does too.
    """
    width, height = size
    cx, cy = width / 2.0, height / 2.0

    us = np.linspace(0.0, width - 1.0, samples)
    vs = np.linspace(0.0, height - 1.0, samples)
    border = np.concatenate(
        [
            np.stack([us, np.zeros_like(us)], axis=1),
            np.stack([us, np.full_like(us, height - 1.0)], axis=1),
            np.stack([np.zeros_like(vs), vs], axis=1),
            np.stack([np.full_like(vs, width - 1.0), vs], axis=1),
        ]
    )

    # Virtual pixel -> normalised ray -> distort -> raw pixel.
    rays = np.stack(
        [
            (border[:, 0] - cx) / focal,
            (border[:, 1] - cy) / focal,
            np.ones(len(border)),
        ],
        axis=1,
    )
    projected, _ = cv2.projectPoints(
        rays.astype(np.float64),
        np.zeros(3),
        np.zeros(3),
        camera_matrix,
        distortion,
    )
    projected = projected.reshape(-1, 2)
    inside = (
        (projected[:, 0] >= 0.0)
        & (projected[:, 0] <= width - 1.0)
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] <= height - 1.0)
    )
    return bool(inside.all())


def centered_virtual_matrix(
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    size: tuple[int, int],
    tolerance: float = 0.05,
) -> np.ndarray:
    """Widest centred, square-pixel K with no invalid pixels (alpha=0 equivalent).

    Invalid border pixels are not a cosmetic problem: they would put a pattern
    into the policy's input that has no counterpart in simulation, which is the
    very gap this alignment exists to close.
    """
    width, height = size
    measured = float((camera_matrix[0, 0] + camera_matrix[1, 1]) / 2.0)

    # Widen until coverage fails, so the bracket is valid even for strong
    # pincushion where the measured focal length already covers the frame.
    low = measured * 0.25
    high = measured * 4.0
    if not _virtual_frame_is_covered(high, camera_matrix, distortion, size):
        raise ValueError(
            "no centred virtual camera covers the frame even at 4x the measured "
            "focal length; the distortion estimate is probably bad"
        )
    if _virtual_frame_is_covered(low, camera_matrix, distortion, size):
        focal = low
    else:
        while high - low > tolerance:
            mid = (low + high) / 2.0
            if _virtual_frame_is_covered(mid, camera_matrix, distortion, size):
                high = mid
            else:
                low = mid
        focal = high

    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def raw_horizontal_fov_deg(
    camera_matrix: np.ndarray, distortion: np.ndarray, size: tuple[int, int]
) -> float:
    """True horizontal field of view of the raw image, distortion included."""
    width, height = size
    edges = np.array([[0.0, height / 2.0], [width - 1.0, height / 2.0]])
    undistorted = cv2.undistortPoints(
        edges.reshape(-1, 1, 2), camera_matrix, distortion
    ).reshape(-1, 2)
    left = np.array([undistorted[0, 0], undistorted[0, 1], 1.0])
    right = np.array([undistorted[1, 0], undistorted[1, 1], 1.0])
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    return float(math.degrees(math.acos(float(np.clip(left @ right, -1.0, 1.0)))))


def virtual_horizontal_fov_deg(
    virtual_matrix: np.ndarray, size: tuple[int, int]
) -> float:
    width = size[0]
    return float(math.degrees(2.0 * math.atan(width / (2.0 * virtual_matrix[0, 0]))))


def horizontal_fov_loss(
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    virtual_matrix: np.ndarray,
    size: tuple[int, int],
) -> float:
    """Fraction of horizontal field of view given up by rectifying.

    Above roughly 0.10 the plan calls for reconsidering the approach: Isaac
    Sim 5.1 can express full OpenCV distortion through
    ``OmniLensDistortionOpenCvPinholeAPI``, at the cost of bypassing Isaac Lab
    and leaving its reported intrinsics wrong.
    """
    raw = raw_horizontal_fov_deg(camera_matrix, distortion, size)
    virtual = virtual_horizontal_fov_deg(virtual_matrix, size)
    if raw <= 0.0:
        return 0.0
    return float(max(0.0, 1.0 - virtual / raw))


def distortion_pixel_magnitude(
    camera_matrix: np.ndarray, distortion: np.ndarray, size: tuple[int, int]
) -> float:
    """Largest pixel displacement distortion causes across the frame.

    Reported so the decision to rectify is backed by a measured number rather
    than an assumption about the lens.
    """
    width, height = size
    us, vs = np.meshgrid(
        np.linspace(0.0, width - 1.0, 24), np.linspace(0.0, height - 1.0, 24)
    )
    points = np.stack([us.ravel(), vs.ravel()], axis=1)
    undistorted = cv2.undistortPoints(
        points.reshape(-1, 1, 2), camera_matrix, distortion, P=camera_matrix
    ).reshape(-1, 2)
    return float(np.max(np.linalg.norm(undistorted - points, axis=1)))
