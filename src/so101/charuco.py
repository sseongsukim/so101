"""The calibration targets, defined once.

The printed sheet, the detector, the hand-eye solver and the simulated board
must all agree on the dictionary, the grid and the physical dimensions.  If
they drift apart the symptom is not an error but a plausible-looking wrong
answer, so the numbers live here and nowhere else.

Marker ids do not overlap: the 7x5 ChArUco board consumes ids 0-16 and the
gripper tags start at 20, so both targets can be in frame at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DICTIONARY_NAME = "DICT_4X4_50"

BOARD_COLS = 7
BOARD_ROWS = 5
SQUARE_MM = 30.0
MARKER_MM = 22.0

GRIPPER_TAG_MM = 30.0
GRIPPER_TAG_IDS = (20, 21, 22)


def dictionary() -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY_NAME))


def charuco_board(square_mm: float = SQUARE_MM, marker_mm: float = MARKER_MM):
    """The table board, with dimensions in metres.

    ``square_mm`` is overridable because a printer may not honour 100% scale;
    the on-site instruction is to measure the printed square and pass the real
    value rather than trust the nominal one.
    """
    return cv2.aruco.CharucoBoard(
        (BOARD_COLS, BOARD_ROWS),
        square_mm / 1000.0,
        marker_mm / 1000.0,
        dictionary(),
    )


def board_corner_count() -> int:
    """Number of interior chessboard corners the board can yield."""
    return (BOARD_COLS - 1) * (BOARD_ROWS - 1)


@dataclass
class BoardDetection:
    """What one image gave up."""

    charuco_corners: np.ndarray | None
    charuco_ids: np.ndarray | None

    @property
    def count(self) -> int:
        return 0 if self.charuco_ids is None else int(len(self.charuco_ids))

    def usable(self, minimum: int = 8) -> bool:
        return self.count >= minimum


def detect_board(image: np.ndarray, board) -> BoardDetection:
    """Find ChArUco corners in a BGR or grayscale image."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) == 0:
        return BoardDetection(None, None)
    return BoardDetection(charuco_corners, charuco_ids)


def detect_gripper_tags(
    image: np.ndarray, tag_mm: float = GRIPPER_TAG_MM
) -> dict[int, np.ndarray]:
    """Find the gripper's ArUco tags, returning ``{id: 4x2 corner array}``.

    Only ids in :data:`GRIPPER_TAG_IDS` are returned, so board markers in the
    same frame are ignored rather than silently mixed in.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(dictionary(), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    found: dict[int, np.ndarray] = {}
    if ids is None:
        return found
    for corner, tag_id in zip(corners, ids.flatten()):
        if int(tag_id) in GRIPPER_TAG_IDS:
            found[int(tag_id)] = corner.reshape(4, 2)
    return found


def tag_object_points(tag_mm: float = GRIPPER_TAG_MM) -> np.ndarray:
    """Corners of one ArUco tag in its own frame (metres), OpenCV order."""
    half = tag_mm / 2000.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def board_object_points(board, charuco_ids: np.ndarray) -> np.ndarray:
    """3-D coordinates of the detected ChArUco corners in the board frame."""
    all_corners = board.getChessboardCorners()
    return np.array(
        [all_corners[int(i)] for i in charuco_ids.flatten()], dtype=np.float64
    )
