"""Generate printable calibration targets for the SO-101 camera rig.

Two targets are needed, and they have opposite size constraints:

* a **ChArUco board** that sits on the table -- used for intrinsic calibration
  of both cameras and for the wrist camera's eye-in-hand solve;
* **single ArUco tags** that mount on the gripper -- used for the front
  camera's eye-to-hand solve.  Several tags at different angles, because a
  lone planar tag seen near head-on suffers the planar pose ambiguity and its
  rotation estimate flips, which would land straight in the hand-eye solution.

Marker ids do not overlap: the 7x5 board consumes ids 0-16, the gripper tags
start at 20, so both can be in frame at once without confusing the detector.

Examples:

    python scripts/make_calibration_targets.py
    python scripts/make_calibration_targets.py --out-dir calibration/targets
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from so101.charuco import (
    BOARD_COLS,
    BOARD_ROWS,
    DICTIONARY_NAME,
    GRIPPER_TAG_IDS,
    GRIPPER_TAG_MM,
    MARKER_MM,
    SQUARE_MM,
    charuco_board,
    dictionary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "calibration" / "targets"

# 24 px/mm == 609.6 dpi, chosen so every millimetre is a whole number of
# pixels.  That keeps the printed square exactly 30 mm rather than 30 mm plus
# a rounding error that would propagate into every measured distance.
PX_PER_MM = 24
A4_MM = (210.0, 297.0)


def _mm(value: float) -> int:
    return int(round(value * PX_PER_MM))


def _dpi() -> float:
    return PX_PER_MM * 25.4


def _font(size_px: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size_px)
    return ImageFont.load_default()


def _blank_page(landscape: bool) -> Image.Image:
    width_mm, height_mm = (A4_MM[1], A4_MM[0]) if landscape else A4_MM
    return Image.new("L", (_mm(width_mm), _mm(height_mm)), color=255)


def _draw_ruler(draw: ImageDraw.ImageDraw, x_mm: float, y_mm: float) -> None:
    """A 100 mm scale bar, so a misprint is caught before it costs a session.

    If this bar does not measure 100 mm on the printout, the page was scaled
    and every dimension derived from it is wrong.
    """
    font = _font(_mm(3.0))
    x0, y0 = _mm(x_mm), _mm(y_mm)
    draw.line([(x0, y0), (x0 + _mm(100.0), y0)], fill=0, width=_mm(0.4))
    for tick in range(0, 101, 10):
        height = 4.0 if tick % 50 == 0 else 2.5
        tx = x0 + _mm(float(tick))
        draw.line([(tx, y0), (tx, y0 - _mm(height))], fill=0, width=_mm(0.3))
    draw.text(
        (x0, y0 + _mm(1.5)),
        "100 mm reference - measure this. If it is not 100 mm, reprint at 100% scale.",
        fill=0,
        font=font,
    )


def _paste_array(page: Image.Image, array: np.ndarray, x_mm: float, y_mm: float) -> None:
    page.paste(Image.fromarray(array), (_mm(x_mm), _mm(y_mm)))


def build_charuco_board() -> tuple[cv2.aruco.CharucoBoard, np.ndarray]:
    board = charuco_board()
    size_px = (_mm(BOARD_COLS * SQUARE_MM), _mm(BOARD_ROWS * SQUARE_MM))
    image = board.generateImage(size_px, marginSize=0, borderBits=1)
    return board, image


def render_table_board() -> Image.Image:
    _, board_image = build_charuco_board()
    page = _blank_page(landscape=True)
    draw = ImageDraw.Draw(page)

    board_w_mm = BOARD_COLS * SQUARE_MM
    board_h_mm = BOARD_ROWS * SQUARE_MM
    page_w_mm = A4_MM[1]
    x_mm = (page_w_mm - board_w_mm) / 2.0
    # Keep the whole layout clear of the ~5 mm non-printable edge most printers
    # have; the ruler is useless if it lands in that band and gets clipped.
    y_mm = 17.0
    _paste_array(page, board_image, x_mm, y_mm)

    # Corner crop marks make it obvious if the printer clipped an edge.
    for corner_x, corner_y in (
        (x_mm, y_mm),
        (x_mm + board_w_mm, y_mm),
        (x_mm, y_mm + board_h_mm),
        (x_mm + board_w_mm, y_mm + board_h_mm),
    ):
        cx, cy = _mm(corner_x), _mm(corner_y)
        draw.line([(cx - _mm(5), cy), (cx + _mm(5), cy)], fill=0, width=_mm(0.3))
        draw.line([(cx, cy - _mm(5)), (cx, cy + _mm(5))], fill=0, width=_mm(0.3))

    title = _font(_mm(4.5))
    body = _font(_mm(3.2))
    draw.text((_mm(x_mm), _mm(8.0)), "SO-101 table calibration board", fill=0, font=title)
    lines = [
        f"{DICTIONARY_NAME} | ChArUco {BOARD_COLS}x{BOARD_ROWS} | "
        f"square {SQUARE_MM:.0f} mm | marker {MARKER_MM:.0f} mm | "
        f"ids 0-{(BOARD_COLS * BOARD_ROWS) // 2 - 1} | {date.today().isoformat()}",
        "Print at 100% scale (turn OFF 'fit to page'). Matte paper only - gloss reflects and detection fails.",
        "Mount flat on a rigid board with no bubbles; a warped board breaks the planar assumption.",
    ]
    for index, line in enumerate(lines):
        draw.text(
            (_mm(x_mm), _mm(y_mm + board_h_mm + 8.0 + index * 5.0)),
            line,
            fill=0,
            font=body,
        )
    _draw_ruler(draw, x_mm, y_mm + board_h_mm + 26.0)
    return page


def render_gripper_tags() -> Image.Image:
    tag_dictionary = dictionary()
    page = _blank_page(landscape=False)
    draw = ImageDraw.Draw(page)

    title = _font(_mm(4.5))
    body = _font(_mm(3.2))
    draw.text((_mm(20.0), _mm(15.0)), "SO-101 gripper hand-eye tags", fill=0, font=title)
    lines = [
        f"{DICTIONARY_NAME} | {GRIPPER_TAG_MM:.0f} mm tags | "
        f"ids {', '.join(str(i) for i in GRIPPER_TAG_IDS)} | {date.today().isoformat()}",
        "Print at 100% scale. Mount the three tags on a small cube or bracket at",
        "DIFFERENT angles, then fix that rigidly to the gripper.",
        "A single head-on planar tag flips its rotation estimate (planar pose",
        "ambiguity) and that error lands directly in the hand-eye solution.",
        "The tags must not move relative to the gripper during capture.",
    ]
    for index, line in enumerate(lines):
        draw.text((_mm(20.0), _mm(24.0 + index * 5.0)), line, fill=0, font=body)

    tag_px = _mm(GRIPPER_TAG_MM)
    y_mm = 62.0
    for tag_id in GRIPPER_TAG_IDS:
        marker = cv2.aruco.generateImageMarker(
            tag_dictionary, tag_id, tag_px, borderBits=1
        )
        # A quiet zone is mandatory: without white margin the detector cannot
        # find the tag border.
        _paste_array(page, marker, 30.0, y_mm)
        draw.rectangle(
            [
                (_mm(30.0 - 8.0), _mm(y_mm - 8.0)),
                (_mm(30.0 + GRIPPER_TAG_MM + 8.0), _mm(y_mm + GRIPPER_TAG_MM + 8.0)),
            ],
            outline=0,
            width=_mm(0.25),
        )
        draw.text(
            (_mm(30.0 + GRIPPER_TAG_MM + 16.0), _mm(y_mm + GRIPPER_TAG_MM / 2 - 2.0)),
            f"id {tag_id}   {GRIPPER_TAG_MM:.0f} x {GRIPPER_TAG_MM:.0f} mm"
            "   (cut on the outer line, keep the white border)",
            fill=0,
            font=body,
        )
        y_mm += GRIPPER_TAG_MM + 26.0

    _draw_ruler(draw, 30.0, y_mm + 6.0)
    return page


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate printable ChArUco/ArUco calibration targets."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="also write PNG copies alongside the PDFs",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pages = {
        "table_charuco": render_table_board(),
        "gripper_aruco": render_gripper_tags(),
    }
    for name, page in pages.items():
        pdf_path = args.out_dir / f"{name}.pdf"
        page.save(pdf_path, "PDF", resolution=_dpi())
        print(f"wrote {pdf_path}")
        if args.png:
            png_path = args.out_dir / f"{name}.png"
            page.save(png_path, "PNG", dpi=(_dpi(), _dpi()))
            print(f"wrote {png_path}")

    print(
        f"\nSpecs: {DICTIONARY_NAME}, ChArUco {BOARD_COLS}x{BOARD_ROWS} "
        f"(square {SQUARE_MM:.0f} mm, marker {MARKER_MM:.0f} mm, ids 0-"
        f"{(BOARD_COLS * BOARD_ROWS) // 2 - 1}); gripper tags "
        f"{GRIPPER_TAG_MM:.0f} mm ids {list(GRIPPER_TAG_IDS)}."
    )
    print("Verify the 100 mm bar on each printout before using the targets.")


if __name__ == "__main__":
    main()
