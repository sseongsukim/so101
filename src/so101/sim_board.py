"""Put a ChArUco board into the Isaac scene, as a textured quad.

Used for two things that both need the renderer in the loop:

* the self-test, which places the board at a pose it already knows and checks
  the calibration pipeline recovers it -- a failure there is a code problem,
  not a measurement one;
* the alignment gate, which places the board where the real one actually is
  and compares rendered pixels against captured ones.

Isaac Lab's ``MdlFileCfg`` exposes no texture field, so the OmniPBR material
is bound through it and the texture input is set on the shader prim directly.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from so101.charuco import BOARD_COLS, BOARD_ROWS, MARKER_MM, SQUARE_MM, charuco_board

BOARD_WIDTH_M = BOARD_COLS * SQUARE_MM / 1000.0
BOARD_HEIGHT_M = BOARD_ROWS * SQUARE_MM / 1000.0
BOARD_THICKNESS_M = 0.003

# Enough texels that a 30 mm square stays sharp when the board fills the frame;
# blur here shows up as corner-detection noise and would be mistaken for a
# calibration error.
TEXELS_PER_MM = 8


def board_size_m(square_mm: float = SQUARE_MM) -> tuple[float, float]:
    return (BOARD_COLS * square_mm / 1000.0, BOARD_ROWS * square_mm / 1000.0)


def write_board_texture(path: str | Path, square_mm: float = SQUARE_MM) -> Path:
    """Render the board to a PNG that the material can reference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    board = charuco_board(square_mm, MARKER_MM * square_mm / SQUARE_MM)
    width_m, height_m = board_size_m(square_mm)
    image = board.generateImage(
        (
            int(width_m * 1000 * TEXELS_PER_MM),
            int(height_m * 1000 * TEXELS_PER_MM),
        ),
        marginSize=0,
        borderBits=1,
    )
    # Written unflipped.  Flipping the image to suit USD's bottom-up texture
    # origin mirrors the ArUco markers, and mirrored markers do not decode at
    # all -- the board rendered perfectly and detected zero corners.  The
    # orientation is handled in the mesh's UV coordinates instead.
    cv2.imwrite(str(path), image)
    return path


def board_corners_in_board_frame() -> np.ndarray:
    """ChArUco corner coordinates, in the board's own frame (metres)."""
    return np.asarray(charuco_board().getChessboardCorners(), dtype=np.float64)


def board_centre_offset() -> np.ndarray:
    """Vector from the board frame origin to the quad's geometric centre."""
    return np.array([BOARD_WIDTH_M / 2.0, BOARD_HEIGHT_M / 2.0, 0.0])


def spawn_board(
    prim_path: str,
    texture_path: str | Path,
    position: np.ndarray,
    orientation_wxyz: np.ndarray,
    square_mm: float = SQUARE_MM,
) -> None:
    """Create the textured board quad with its own frame at ``position``.

    The quad is an explicit mesh with UV coordinates rather than a primitive
    with a projected material.  Isaac Lab's CuboidCfg produces geometry with no
    texture coordinates, so OmniPBR had nothing to map the board onto and it
    rendered as a blank grey panel.  Triplanar projection would fix the
    blankness but leaves the texture's orientation up to a world-space
    projection, and orientation is precisely what this board is measuring.

    The mesh's local frame is the board frame: origin at the corner that
    ``getChessboardCorners`` measures from, +x across the columns, +y down the
    rows, so ``position``/``orientation_wxyz`` are T_world_board directly.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    width, height = board_size_m(square_mm)
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(width, 0.0, 0.0),
                Gf.Vec3f(width, height, 0.0),
                Gf.Vec3f(0.0, height, 0.0),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 1.0)] * 4))
    mesh.CreateDoubleSidedAttr(True)

    # Board +y runs down the image rows while USD's v runs up from the bottom,
    # so v is inverted here.  Doing it in the UVs keeps the texture itself
    # unmirrored, which the ArUco markers require.
    uvs = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
    )
    uvs.Set(
        Vt.Vec2fArray(
            [Gf.Vec2f(0.0, 1.0), Gf.Vec2f(1.0, 1.0), Gf.Vec2f(1.0, 0.0), Gf.Vec2f(0.0, 0.0)]
        )
    )

    xform = UsdGeom.Xformable(mesh)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(
        Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
    )
    quat = [float(v) for v in orientation_wxyz]
    xform.AddOrientOp().Set(Gf.Quatf(quat[0], Gf.Vec3f(quat[1], quat[2], quat[3])))

    material_path = f"{prim_path}/Material"
    material_cfg = sim_utils.MdlFileCfg(mdl_path="OmniPBR.mdl", project_uvw=False)
    material_cfg.func(material_path, material_cfg)

    texture = str(Path(texture_path).resolve())
    bound = False
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(material_path) or prim.GetTypeName() != "Shader":
            continue
        shader = UsdShade.Shader(prim)
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(texture)
        # OmniPBR multiplies the texture by the tint, so a non-white tint would
        # wash the board out and cost corner contrast.
        shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(1.0, 1.0, 1.0)
        )
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.9)
        bound = True
    if not bound:
        raise RuntimeError(
            f"no Shader prim under {material_path}; the board would render "
            "untextured and every corner check would fail for the wrong reason"
        )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath(material_path))
    )
