"""Rendering domain randomization for synthetic sim-to-real data (ResiP §II-D).

The paper re-renders teacher rollouts with randomized lighting (position,
brightness, hue), object colors, and camera poses, so that a student trained
on the images does not latch onto one exact rendering. This applies the same
kinds of variation to the visual StackCube scene through the USD stage:

    lights    key (distant) light direction / intensity / tint, dome fill
              intensity / tint -- global, one draw for all envs
    colors    per env: cube hue/saturation/value jitter around the nominal
              colors, tabletop brightness and tint, roughness of both
    cameras   per env: small position/orientation jitter of the front and
              wrist camera prims around their calibrated poses
    backdrop  per env (when `add_backdrop` put one in the scene): floor and
              wall colors. The real front camera sees the lab around the
              table -- floor, walls, furniture -- where the bare scene shows
              an empty sky, so the student would otherwise never have seen a
              background at all.

The cameras are calibrated here (the paper's were not), so their jitter is
kept to calibration-error size rather than the paper's wide range. Physics is
untouched: only appearance changes.

Call `RenderRandomizer.apply(env)` after every `env.reset()`; it resamples
everything. `scripts/preview_render_randomization.py` renders samples of the
configured ranges for eyeballing against the real cameras.
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RenderRandomizationCfg:
    # key light: extra rotation (deg) about world z / a horizontal axis, and scales
    key_yaw_deg: float = 60.0
    key_tilt_deg: float = 20.0
    key_intensity_scale: tuple[float, float] = (0.4, 1.6)
    dome_intensity_scale: tuple[float, float] = (0.4, 1.6)
    light_tint: float = 0.12              # max per-channel multiplicative tint deviation
    # cubes: HSV jitter around each cube's nominal color
    cube_hue_jitter: float = 0.05         # fraction of the hue circle
    cube_saturation_scale: tuple[float, float] = (0.7, 1.2)
    cube_value_scale: tuple[float, float] = (0.6, 1.3)
    roughness: tuple[float, float] = (0.25, 0.9)
    # tabletop: dark gray with a slight tint (the real table is black)
    table_value: tuple[float, float] = (0.004, 0.04)
    table_tint: float = 0.15
    # camera pose jitter
    front_pos_jitter_m: float = 0.01
    front_rot_jitter_deg: float = 1.5
    wrist_pos_jitter_m: float = 0.003
    wrist_rot_jitter_deg: float = 1.0
    # backdrop (add_backdrop): gray levels and tint
    floor_value: tuple[float, float] = (0.03, 0.35)
    wall_value: tuple[float, float] = (0.25, 0.9)
    backdrop_tint: float = 0.2


# Visual-only backdrop around the tabletop, in env coordinates. The tabletop
# spans x 0..0.7, y 0..1.2 with its top at z 0.03; the front camera looks from
# (0.62, 0.70, 0.30) back past the robot toward -x/-y, so walls stand there.
BACKDROP_PARTS = {
    "Floor": {"size": (8.0, 8.0, 0.02), "pos": (0.35, 0.6, -0.72)},
    "WallBack": {"size": (0.05, 8.0, 3.0), "pos": (-1.6, 0.6, 0.5)},
    "WallSide": {"size": (8.0, 0.05, 3.0), "pos": (0.35, -1.8, 0.5)},
}


def add_backdrop(scene_cfg) -> None:
    """Add the randomizable floor and walls to a visual StackCube scene cfg
    (before gym.make). Visual only: no collision, no rigid body."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    for name, part in BACKDROP_PARTS.items():
        setattr(scene_cfg, f"backdrop_{name.lower()}", AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Backdrop{name}",
            spawn=sim_utils.CuboidCfg(
                size=part["size"],
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), roughness=0.8),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=part["pos"]),
        ))


def _axis_angle_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return np.array([math.cos(angle_rad / 2), *(axis * math.sin(angle_rad / 2))])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _random_small_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    return _axis_angle_quat(axis, math.radians(rng.uniform(-max_deg, max_deg)))


class _XformPose:
    """Read-once nominal local pose of a prim, re-set with a perturbation."""

    def __init__(self, prim):
        from pxr import UsdGeom

        self.xform = UsdGeom.Xformable(prim)
        ops = {op.GetOpName(): op for op in self.xform.GetOrderedXformOps()}
        self.translate = ops.get("xformOp:translate")
        self.orient = ops.get("xformOp:orient")
        if self.translate is None or self.orient is None:
            raise RuntimeError(f"{prim.GetPath()}: expected xformOp:translate and xformOp:orient")
        self.pos = np.array(self.translate.Get(), dtype=np.float64)
        q = self.orient.Get()
        self.quat = np.array([q.GetReal(), *q.GetImaginary()], dtype=np.float64)

    def set(self, pos: np.ndarray, quat: np.ndarray) -> None:
        from pxr import Gf

        self.translate.Set(Gf.Vec3d(*pos.tolist()))
        quat = quat / np.linalg.norm(quat)
        orient_type = type(self.orient.Get())
        self.orient.Set(orient_type(float(quat[0]), *[float(v) for v in quat[1:]]))

    def restore(self) -> None:
        self.set(self.pos, self.quat)


class RenderRandomizer:
    def __init__(self, env, cfg: RenderRandomizationCfg | None = None, seed: int = 0):
        import omni.usd
        from pxr import UsdLux, UsdShade

        self.cfg = cfg or RenderRandomizationCfg()
        self.rng = np.random.default_rng(seed)
        stage = omni.usd.get_context().get_stage()
        self.num_envs = env.num_envs

        def shader(path):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"missing prim {path}")
            return UsdShade.Shader(prim)

        env_root = "/World/envs/env_{}"
        self.cube_shaders = {
            name: [shader(f"{env_root.format(i)}/{name}/geometry/material/Shader") for i in range(self.num_envs)]
            for name in ("HeldAsset", "FixedAsset")
        }
        self.cube_nominal = {
            name: np.array(shaders[0].GetInput("diffuseColor").Get(), dtype=np.float64)
            for name, shaders in self.cube_shaders.items()
        }
        self.table_shaders = [shader(f"{env_root.format(i)}/Tabletop/geometry/material/Shader")
                              for i in range(self.num_envs)]
        self.backdrop_shaders = {
            name: [shader(f"{env_root.format(i)}/Backdrop{name}/geometry/material/Shader") for i in range(self.num_envs)]
            for name in BACKDROP_PARTS
            if stage.GetPrimAtPath(f"{env_root.format(0)}/Backdrop{name}").IsValid()
        }

        self.key = UsdLux.DistantLight(stage.GetPrimAtPath("/World/KeyLight"))
        self.dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/FillLight"))
        self.key_pose = _XformPose(self.key.GetPrim())
        self.key_nominal = (self.key.GetIntensityAttr().Get(), np.array(self.key.GetColorAttr().Get()))
        self.dome_nominal = (self.dome.GetIntensityAttr().Get(), np.array(self.dome.GetColorAttr().Get()))

        self.front = [_XformPose(stage.GetPrimAtPath(f"{env_root.format(i)}/ExternalCamera"))
                      for i in range(self.num_envs)]
        self.wrist = [_XformPose(stage.GetPrimAtPath(f"{env_root.format(i)}/Robot/gripper/gripper_cam"))
                      for i in range(self.num_envs)]

    # -- sampling ---------------------------------------------------------------

    def _tint(self, amount: float) -> np.ndarray:
        return np.clip(1.0 + self.rng.uniform(-amount, amount, 3), 0.0, None)

    def _cube_color(self, nominal: np.ndarray) -> np.ndarray:
        c = self.cfg
        h, s, v = colorsys.rgb_to_hsv(*np.clip(nominal, 0, 1))
        h = (h + self.rng.uniform(-c.cube_hue_jitter, c.cube_hue_jitter)) % 1.0
        s = np.clip(s * self.rng.uniform(*c.cube_saturation_scale), 0, 1)
        v = np.clip(v * self.rng.uniform(*c.cube_value_scale), 0, 1)
        return np.array(colorsys.hsv_to_rgb(h, s, v))

    def apply(self, env=None) -> dict:
        """Resample every randomized property. Returns what was drawn."""
        from pxr import Gf

        del env
        c = self.cfg
        drawn = {}

        yaw = _axis_angle_quat(np.array([0.0, 0.0, 1.0]), math.radians(self.rng.uniform(-c.key_yaw_deg, c.key_yaw_deg)))
        tilt = _axis_angle_quat(np.array([1.0, 0.0, 0.0]), math.radians(self.rng.uniform(-c.key_tilt_deg, c.key_tilt_deg)))
        self.key_pose.set(self.key_pose.pos, _quat_mul(yaw, _quat_mul(tilt, self.key_pose.quat)))
        for light, (intensity, color), scale in (
            (self.key, self.key_nominal, c.key_intensity_scale),
            (self.dome, self.dome_nominal, c.dome_intensity_scale),
        ):
            light.GetIntensityAttr().Set(float(intensity * self.rng.uniform(*scale)))
            light.GetColorAttr().Set(Gf.Vec3f(*np.clip(color * self._tint(c.light_tint), 0, 1).tolist()))

        for i in range(self.num_envs):
            for name, shaders in self.cube_shaders.items():
                shaders[i].GetInput("diffuseColor").Set(Gf.Vec3f(*self._cube_color(self.cube_nominal[name]).tolist()))
                shaders[i].GetInput("roughness").Set(float(self.rng.uniform(*c.roughness)))
            gray = self.rng.uniform(*c.table_value)
            self.table_shaders[i].GetInput("diffuseColor").Set(
                Gf.Vec3f(*np.clip(gray * self._tint(c.table_tint), 0, 1).tolist()))
            self.table_shaders[i].GetInput("roughness").Set(float(self.rng.uniform(*c.roughness)))
            for name, shaders in self.backdrop_shaders.items():
                value = self.rng.uniform(*(c.floor_value if name == "Floor" else c.wall_value))
                shaders[i].GetInput("diffuseColor").Set(
                    Gf.Vec3f(*np.clip(value * self._tint(c.backdrop_tint), 0, 1).tolist()))
            for pose, pos_jitter, rot_jitter in (
                (self.front[i], c.front_pos_jitter_m, c.front_rot_jitter_deg),
                (self.wrist[i], c.wrist_pos_jitter_m, c.wrist_rot_jitter_deg),
            ):
                pose.set(pose.pos + self.rng.uniform(-pos_jitter, pos_jitter, 3),
                         _quat_mul(pose.quat, _random_small_rotation(self.rng, rot_jitter)))
        return drawn

    def restore(self) -> None:
        """Put every property back to the scene's nominal values."""
        from pxr import Gf

        self.key_pose.restore()
        for light, (intensity, color) in ((self.key, self.key_nominal), (self.dome, self.dome_nominal)):
            light.GetIntensityAttr().Set(float(intensity))
            light.GetColorAttr().Set(Gf.Vec3f(*color.tolist()))
        for i in range(self.num_envs):
            for name, shaders in self.cube_shaders.items():
                shaders[i].GetInput("diffuseColor").Set(Gf.Vec3f(*self.cube_nominal[name].tolist()))
            for pose in (self.front[i], self.wrist[i]):
                pose.restore()
