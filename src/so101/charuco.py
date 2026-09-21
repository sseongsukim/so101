"""The calibration targets, defined once.

The printed sheet, the detector, the hand-eye solver and the simulated board
must all agree on the dictionary, the grid and the physical dimensions.  If
they drift apart the symptom is not an error but a plausible-looking wrong
answer, so the numbers live here and nowhere else.

There are two boards, not one:

* the **table board** -- handheld for both cameras' intrinsic calibration,
  left flat on the table for the wrist camera's eye-in-hand hand-eye solve.
  Marker ids 0-19.
* the **gripper board** -- taped flat to the gripper, read directly by the
  front camera for its own eye-to-hand solve.  Marker ids 20-29, so both
  boards can be in the same frame without the detector confusing one for the
  other, even though they share the same DICT_4X4_50 dictionary.

The id separation works because each board's ``CharucoDetector`` is built
from that board's own (id-shifted) dictionary: a detector holding only the
byte patterns for ids 20-29 simply does not match anything belonging to ids
0-19, and vice versa, regardless of what else is in frame.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BOARD_SPEC_PATH = REPO_ROOT / "calibration" / "board.yaml"
GRIPPER_BOARD_SPEC_PATH = REPO_ROOT / "calibration" / "gripper_board.yaml"


@dataclass(frozen=True)
class BoardSpec:
    """One printed ChArUco board's actual geometry.

    The generator writes this alongside the PDF and every consumer reads it
    back, so the detector can never be configured for a board different from
    the one that was printed.  That mismatch does not raise -- it returns a
    confident, wrong pose -- which is why the spec is a file rather than a
    constant someone remembers to change in two places.

    ``id_offset`` shifts which slice of the dictionary this board's markers
    come from, so two boards sharing one dictionary never collide.
    """

    cols: int = 7
    rows: int = 5
    square_mm: float = 30.0
    marker_mm: float = 22.0
    dictionary: str = "DICT_4X4_50"
    id_offset: int = 0

    @property
    def width_mm(self) -> float:
        return self.cols * self.square_mm

    @property
    def height_mm(self) -> float:
        return self.rows * self.square_mm

    @property
    def corner_count(self) -> int:
        return (self.cols - 1) * (self.rows - 1)

    @property
    def marker_count(self) -> int:
        return (self.cols * self.rows) // 2

    @property
    def marker_ids(self) -> range:
        return range(self.id_offset, self.id_offset + self.marker_count)

    def scaled(self, measured_square_mm: float | None) -> BoardSpec:
        """The same board as actually printed, if the printer missed the scale."""
        if not measured_square_mm or measured_square_mm == self.square_mm:
            return self
        factor = measured_square_mm / self.square_mm
        return replace(
            self,
            square_mm=measured_square_mm,
            marker_mm=self.marker_mm * factor,
        )

    def to_dict(self) -> dict:
        return {
            "cols": self.cols,
            "rows": self.rows,
            "square_mm": float(self.square_mm),
            "marker_mm": float(self.marker_mm),
            "dictionary": self.dictionary,
            "id_offset": self.id_offset,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )
        return path


def load_board_spec(path: str | Path = BOARD_SPEC_PATH) -> BoardSpec:
    """The board that was actually generated, or the default if none was."""
    path = Path(path)
    if not path.is_file():
        return BoardSpec()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return BoardSpec(**{k: payload[k] for k in BoardSpec().to_dict() if k in payload})


def load_gripper_board_spec(path: str | Path = GRIPPER_BOARD_SPEC_PATH) -> BoardSpec:
    """The gripper board that was actually generated, or the default if none was.

    The default matches the table board's marker size (30 mm) rather than its
    own square size, because marker size is what the front camera's readable
    range at a typical hand-eye capture distance is measured against -- see
    the tag-sizing analysis this default was chosen from.
    """
    default = BoardSpec(
        cols=4, rows=5, square_mm=41.0, marker_mm=30.0, id_offset=20
    )
    path = Path(path)
    if not path.is_file():
        return default
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return BoardSpec(**{k: payload[k] for k in default.to_dict() if k in payload})


# Module-level names kept for callers that only need the table board's defaults.
_DEFAULT = BoardSpec()
DICTIONARY_NAME = _DEFAULT.dictionary
BOARD_COLS = _DEFAULT.cols
BOARD_ROWS = _DEFAULT.rows
SQUARE_MM = _DEFAULT.square_mm
MARKER_MM = _DEFAULT.marker_mm


def dictionary(name: str = DICTIONARY_NAME) -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _dictionary_for(spec: BoardSpec) -> cv2.aruco.Dictionary:
    """The (possibly id-shifted) dictionary this board's markers are drawn from.

    A ``CharucoBoard`` always numbers its own markers ``0..N-1`` internally
    (there is no constructor argument for a starting id), so an id offset is
    made by handing it a dictionary whose entry 0 is really the base
    dictionary's entry ``id_offset``.  Detected against the *full* base
    dictionary, such a marker still reports its true global id (verified:
    slicing bytesList[20:30] and detecting with the full DICT_4X4_50 dictionary
    returns ids 20-29, not 0-9) -- and a ``CharucoDetector`` built from this
    board only ever holds the shifted slice, so it cannot match a marker that
    belongs to a different id range at all.
    """
    base = dictionary(spec.dictionary)
    if spec.id_offset == 0:
        return base
    n = spec.marker_count
    shifted = cv2.aruco.Dictionary(
        base.bytesList[spec.id_offset : spec.id_offset + n], base.markerSize
    )
    return shifted


def charuco_board(
    square_mm: float | None = None,
    marker_mm: float | None = None,
    spec: BoardSpec | None = None,
):
    """The table board, with dimensions in metres.

    With no arguments this is the board described by ``calibration/board.yaml``
    -- the one that was printed.  ``square_mm`` overrides the size for the case
    the on-site instructions call out: the printer missed 100% scale and the
    measured square is the truth.
    """
    spec = spec or load_board_spec()
    if square_mm is not None:
        spec = spec.scaled(square_mm) if marker_mm is None else replace(
            spec, square_mm=square_mm, marker_mm=marker_mm
        )
    return cv2.aruco.CharucoBoard(
        (spec.cols, spec.rows),
        spec.square_mm / 1000.0,
        spec.marker_mm / 1000.0,
        _dictionary_for(spec),
    )


def gripper_board(
    square_mm: float | None = None,
    marker_mm: float | None = None,
    spec: BoardSpec | None = None,
):
    """The gripper board (taped flat to the gripper), with dimensions in metres."""
    spec = spec or load_gripper_board_spec()
    if square_mm is not None:
        spec = spec.scaled(square_mm) if marker_mm is None else replace(
            spec, square_mm=square_mm, marker_mm=marker_mm
        )
    return cv2.aruco.CharucoBoard(
        (spec.cols, spec.rows),
        spec.square_mm / 1000.0,
        spec.marker_mm / 1000.0,
        _dictionary_for(spec),
    )


def board_corner_count(spec: BoardSpec | None = None) -> int:
    """Number of interior chessboard corners the board can yield."""
    return (spec or load_board_spec()).corner_count


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
    """Find ChArUco corners in a BGR or grayscale image.

    Works for either board: each is detected with a ``CharucoDetector`` built
    from that specific board object, so passing the wrong board simply finds
    nothing rather than mixing the two up.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) == 0:
        return BoardDetection(None, None)
    return BoardDetection(charuco_corners, charuco_ids)


def board_object_points(board, charuco_ids: np.ndarray) -> np.ndarray:
    """3-D coordinates of the detected ChArUco corners in the board frame."""
    all_corners = board.getChessboardCorners()
    return np.array(
        [all_corners[int(i)] for i in charuco_ids.flatten()], dtype=np.float64
    )
