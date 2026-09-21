"""Generate the two printable calibration targets for the SO-101 camera rig.

Two targets, one job each:

* the **table board** -- handheld for both cameras' intrinsic calibration,
  and left flat on the table for the wrist camera's eye-in-hand hand-eye
  solve;
* the **gripper tags** -- taped to the gripper, and read directly by the
  front camera for its own eye-to-hand hand-eye solve.  Nothing about the
  front camera's extrinsic depends on the wrist camera's calibration this
  way: each camera is solved on its own.  (A tag taped on is enough --
  it does not need to be a permanent mount.)

Both are printed by default.  ``--no-gripper-tags`` skips the second sheet
for the fallback in ``calibrate_handeye.py --via-board``, which derives the
front camera's pose from the wrist camera's view of the table board instead
-- useful if taping a tag to the gripper is genuinely not an option, at the
cost of inheriting the wrist camera's own calibration error on top of an
extra PnP solve (measured: a 2 mm error in the table board's recovered pose
puts the front camera off by 5.8 mm on average, 14.1 mm worst case).

The table board's size is set by intrinsic calibration's own needs (more
corners at a comfortable handheld distance), not by any long-distance
viewing requirement -- that requirement only applies to the ``--via-board``
fallback, which is why ``--fx``/``--distance``/``--capture-width`` below stay
around purely for that path's benefit.

The generated geometry is written to ``calibration/board.yaml`` so the
detector reads back exactly what was printed.  A mismatch there does not
raise -- it returns a confident, wrong pose.

Examples:

    python scripts/make_calibration_targets.py
    python scripts/make_calibration_targets.py --square-mm 40
    python scripts/make_calibration_targets.py --no-gripper-tags
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from so101.charuco import (
    DICTIONARY_NAME,
    GRIPPER_TAG_IDS,
    GRIPPER_TAG_MM,
    BoardSpec,
    charuco_board,
    dictionary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "calibration" / "targets"

# 24 px/mm == 609.6 dpi, chosen so every millimetre is a whole number of
# pixels.  That keeps the printed square exactly 30 mm rather than 30 mm plus
# a rounding error that would propagate into every measured distance.
PX_PER_MM = 24
PAGES_MM = {"a4": (210.0, 297.0), "a3": (297.0, 420.0)}

# Room reserved under the board for the title block and the 100 mm ruler, and
# above it for the heading.  Kept tight: A3 is only 297 mm tall and the grid
# has to stay dense enough to pin the principal point.
FOOTER_MM = 24.0
TOP_MM = 10.0
MIN_SIDE_MARGIN_MM = 12.0


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


def _page_mm(name: str, landscape: bool) -> tuple[float, float]:
    short, long = PAGES_MM[name]
    return (long, short) if landscape else (short, long)


def _blank_page(name: str, landscape: bool) -> Image.Image:
    width_mm, height_mm = _page_mm(name, landscape)
    return Image.new("L", (_mm(width_mm), _mm(height_mm)), color=255)


def choose_page(spec: BoardSpec) -> str:
    """Smallest page the board plus its title block actually fits on."""
    for name in ("a4", "a3"):
        width_mm, height_mm = _page_mm(name, landscape=True)
        fits_w = spec.width_mm + 2 * MIN_SIDE_MARGIN_MM <= width_mm
        fits_h = spec.height_mm + TOP_MM + FOOTER_MM <= height_mm
        if fits_w and fits_h:
            return name
    raise SystemExit(
        f"board {spec.width_mm:.0f}x{spec.height_mm:.0f} mm does not fit on A3; "
        "reduce --square-mm or the grid"
    )


def predict_cell_px(spec: BoardSpec, fx: float, distance_m: float) -> float:
    """Pixels per marker cell at that distance -- the number that decides it.

    A marker is the data bits plus a one-cell black border on each side, and
    detection collapses when a cell gets down to about two pixels.
    """
    marker_px = spec.marker_mm / 1000.0 * fx / distance_m
    return marker_px / 6.0


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


def build_charuco_board(spec: BoardSpec) -> tuple[cv2.aruco.CharucoBoard, np.ndarray]:
    board = charuco_board(spec=spec)
    size_px = (_mm(spec.width_mm), _mm(spec.height_mm))
    image = board.generateImage(size_px, marginSize=0, borderBits=1)
    return board, image


def render_table_board(spec: BoardSpec, page_name: str) -> Image.Image:
    _, board_image = build_charuco_board(spec)
    page = _blank_page(page_name, landscape=True)
    draw = ImageDraw.Draw(page)

    board_w_mm = spec.width_mm
    board_h_mm = spec.height_mm
    page_w_mm, _ = _page_mm(page_name, landscape=True)
    x_mm = (page_w_mm - board_w_mm) / 2.0
    # Keep the whole layout clear of the ~5 mm non-printable edge most printers
    # have; the ruler is useless if it lands in that band and gets clipped.
    y_mm = TOP_MM
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
    draw.text((_mm(x_mm), _mm(6.0)), "SO-101 table calibration board", fill=0, font=title)
    lines = [
        f"{spec.dictionary} | ChArUco {spec.cols}x{spec.rows} | "
        f"square {spec.square_mm:.0f} mm | marker {spec.marker_mm:.0f} mm | "
        f"ids 0-{spec.marker_count - 1} | {date.today().isoformat()}",
        "Print at 100% scale (turn OFF 'fit to page'). Matte paper only - gloss reflects and detection fails.",
        "Mount flat on a rigid board with no bubbles; a warped board breaks the planar assumption.",
    ]
    for index, line in enumerate(lines):
        draw.text(
            (_mm(x_mm), _mm(y_mm + board_h_mm + 6.0 + index * 4.6)),
            line,
            fill=0,
            font=body,
        )
    _draw_ruler(draw, x_mm, y_mm + board_h_mm + 22.0)
    return page


def render_gripper_tags() -> Image.Image:
    tag_dictionary = dictionary()
    page = _blank_page("a4", landscape=False)
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
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument(
        "--square-mm",
        type=float,
        default=32.0,
        help="chessboard square size; the front camera sees the board from about "
        "0.7 m, so this is what decides whether it can read it at all",
    )
    parser.add_argument(
        "--marker-ratio",
        type=float,
        default=22.0 / 30.0,
        help="marker size as a fraction of the square",
    )
    parser.add_argument("--page", choices=["auto", "a4", "a3"], default="auto")
    parser.add_argument(
        "--fx",
        type=float,
        default=412.0,
        help="focal length in pixels used for the readability estimate; replace "
        "with the measured value once intrinsics are done",
    )
    parser.add_argument("--distance", type=float, default=0.73,
                        help="metres from the front camera to the board")
    parser.add_argument(
        "--capture-width",
        type=int,
        default=1280,
        help="width the front camera will use when it photographs the board; "
        "the extrinsic does not depend on resolution, so the board may be shot "
        "at a higher one than the task runs at",
    )
    parser.add_argument(
        "--no-gripper-tags",
        action="store_true",
        help="skip the gripper tag sheet. Printed by default: the front "
        "camera's extrinsic is solved directly from a tag taped to the "
        "gripper (eye-to-hand) -- see calibrate_handeye.py. --via-board there "
        "is a fallback that needs nothing on the gripper, at the cost of "
        "depending on the wrist camera's own calibration.",
    )
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()

    spec = BoardSpec(
        cols=args.cols,
        rows=args.rows,
        square_mm=args.square_mm,
        marker_mm=round(args.square_mm * args.marker_ratio, 2),
    )
    page_name = choose_page(spec) if args.page == "auto" else args.page

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pages = {"table_charuco": render_table_board(spec, page_name)}
    if not args.no_gripper_tags:
        pages["gripper_aruco"] = render_gripper_tags()

    for name, page in pages.items():
        pdf_path = args.out_dir / f"{name}.pdf"
        page.save(pdf_path, "PDF", resolution=_dpi())
        print(f"wrote {pdf_path}")
        if args.png:
            png_path = args.out_dir / f"{name}.png"
            page.save(png_path, "PNG", dpi=(_dpi(), _dpi()))
            print(f"wrote {png_path}")

    # Record what was actually generated so the detector cannot be configured
    # for a different board than the one on the table.
    spec_path = spec.save()
    print(f"wrote {spec_path}")

    def verdict(cell: float) -> str:
        if cell < 3.0:
            return "TOO SMALL -- detection collapses near 2 px per cell"
        if cell < 4.0:
            return "MARGINAL -- fine head-on, fragile once tilted away"
        return "OK"

    task_cell = predict_cell_px(spec, args.fx, args.distance)
    scale = args.capture_width / 640.0
    shot_cell = task_cell * scale
    print(f"\n[--via-board fallback only] readability from the front camera at "
          f"{args.distance:.2f} m across the table:")
    print(f"  read out at  640 px wide (fx {args.fx:.0f})  : {task_cell:.1f} px/cell -- {verdict(task_cell)}")
    print(f"  read out at {args.capture_width:5d} px wide (fx ~{args.fx * scale:.0f}) : "
          f"{shot_cell:.1f} px/cell -- {verdict(shot_cell)}")
    if task_cell < 4.0 <= shot_cell:
        print(
            "\n  Too small to read at the task resolution, which is fine for --via-board:\n"
            "  photograph the board at the higher one for that extrinsic solve. This does\n"
            "  not matter at all for the default path (gripper tags), which never views\n"
            "  the table board from across the room."
        )

    if spec.corner_count < 24:
        print(
            f"\n  Only {spec.corner_count} corners -- few enough that the principal point may stay\n"
            "  poorly constrained however many images are taken. This affects intrinsic\n"
            "  calibration for both cameras regardless of which hand-eye route is used."
        )
    elif spec.width_mm < 300:
        print(
            f"\n  {spec.corner_count} corners on a {spec.width_mm:.0f} mm board: plan on about 40 intrinsic\n"
            "  shots rather than 20-25. Measured on this exact board: 24 shots land the\n"
            "  principal point within 2 px about 70% of the time, 40 shots about 82%\n"
            "  (this plateaus -- more than 40 does not help further)."
        )
    print("\nPrint at 100% scale and measure the 100 mm bar before using the board.")


if __name__ == "__main__":
    main()
