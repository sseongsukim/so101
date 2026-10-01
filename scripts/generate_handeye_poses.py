"""Design the robot poses that hand-eye calibration will be captured at.

Replaying a scripted joint sequence on the real arm is what makes hand-eye
capture possible without a second person driving the leader, but it raises two
questions a script alone cannot answer: does the arm hit anything, and is the
target actually in frame at that pose.  Both are questions simulation answers
exactly, and the simulation already exists.

Designing the set also beats sampling it.  The conditioning of the hand-eye
solution depends on how varied the *rotations* are, so poses are selected
greedily for rotational spread rather than taken as they come.

Two sequences are produced, because the two cameras need opposite things:

* ``wrist`` -- eye-in-hand.  The table board must be in the wrist camera's view.
* ``front`` -- eye-to-hand.  The gripper board must be in the front camera's view.

Examples:

    python -u scripts/generate_handeye_poses.py
    python -u scripts/generate_handeye_poses.py --camera wrist --count 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path

# so101 is installed editable from a sibling checkout, so an unqualified
# import silently resolves there instead of to this working tree.  The .pth
# only appends to sys.path, so putting this repo's src first wins.
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate hand-eye capture poses.")
parser.add_argument("--task", default="so101-visual-StackCube-v0")
parser.add_argument(
    "--camera",
    action="append",
    choices=["wrist", "front"],
    help="which sequence to build; repeatable, defaults to both",
)
parser.add_argument("--count", type=int, default=24, help="poses to select")
parser.add_argument("--candidates", type=int, default=4000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--out-dir",
    type=Path,
    default=Path("calibration") / "handeye_poses",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import so101.tasks  # noqa: E402,F401  (registers environments)
from so101.charuco import load_board_spec, load_gripper_board_spec  # noqa: E402
from so101.configs import make_env_cfg  # noqa: E402
from so101.scenes.tabletop import ROBOT_BASE_BOTTOM_Z  # noqa: E402

# Read back what was actually printed rather than BoardSpec's bare defaults --
# calibration/board.yaml has long since diverged from those defaults (8x5 at
# 32 mm vs the dataclass's 7x5 at 30 mm), and this file used the wrong ones
# silently until now.
_TABLE_SPEC = load_board_spec()
_GRIPPER_SPEC = load_gripper_board_spec()
BOARD_W = _TABLE_SPEC.width_mm / 1000.0
BOARD_H = _TABLE_SPEC.height_mm / 1000.0
GRIPPER_BOARD_W = _GRIPPER_SPEC.width_mm / 1000.0
GRIPPER_BOARD_H = _GRIPPER_SPEC.height_mm / 1000.0

# Where the board is assumed to sit for planning purposes: flat on the table,
# centred on the cube workspace.  Expressed in the ENVIRONMENT frame, which is
# rotated 90 degrees about Z from the robot base frame -- see the frame audit.
# The real board goes in the same place; the
# solve reads its measured pose from the images, so a few centimetres of
# difference here only affects pose selection, not the result.
BOARD_CENTRE_ENV = np.array([0.25, 0.0, ROBOT_BASE_BOTTOM_Z])

# Clearance kept between every robot link and the tabletop.  The real cell has
# obstacles the simulation does not model -- cables, clamps, the front camera
# mount -- so this is a floor, not a guarantee.
TABLE_CLEARANCE = 0.012
MIN_TARGET_MARGIN_PX = 40
MIN_VISIBLE_BOARD_POINTS = 3
MIN_DEPTH_M = 0.08
MAX_DEPTH_M = 1.2


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def board_points() -> np.ndarray:
    """Board outline plus centre, in the environment frame."""
    half_w, half_h = BOARD_W / 2.0, BOARD_H / 2.0
    offsets = [
        (-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h),
        (0.0, 0.0),
    ]
    return np.array([BOARD_CENTRE_ENV + np.array([dx, dy, 0.0]) for dx, dy in offsets])


def gripper_board_points(gripper_pos: np.ndarray, gripper_rot: np.ndarray) -> np.ndarray:
    """Outline and centre of the gripper board, approximated as flat on the link.

    The real mount offset is unknown until hand-eye solves for it, which is the
    point of the exercise.  For pose selection only the rough location matters.
    """
    half_w, half_h = GRIPPER_BOARD_W / 2.0, GRIPPER_BOARD_H / 2.0
    local = [
        (-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h),
        (0.0, 0.0),
    ]
    return np.array(
        [gripper_pos + gripper_rot @ np.array([dx, dy, 0.0]) for dx, dy in local]
    )


def visible_mask(
    points: np.ndarray,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    intrinsics: np.ndarray,
    size: tuple[int, int],
    margin: int,
) -> np.ndarray:
    """Which of ``points`` land inside the frame with a margin to spare."""
    width, height = size
    mask = np.zeros(len(points), dtype=bool)
    for index, point in enumerate(points):
        in_cam = cam_rot.T @ (point - cam_pos)
        depth = float(in_cam[2])
        if not (MIN_DEPTH_M < depth < MAX_DEPTH_M):
            continue
        pixel = intrinsics @ (in_cam / depth)
        mask[index] = (
            margin <= pixel[0] <= width - margin
            and margin <= pixel[1] <= height - margin
        )
    return mask


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    relative = a.T @ b
    cosine = (np.trace(relative) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def select_diverse(rotations: list[np.ndarray], count: int) -> list[int]:
    """Greedy farthest-point selection in rotation space.

    Hand-eye recovers rotation from how the *changes* in pose relate; a cluster
    of similar orientations leaves the solution poorly conditioned no matter how
    many samples it contains.
    """
    if not rotations:
        return []
    chosen = [0]
    while len(chosen) < min(count, len(rotations)):
        best_index, best_distance = None, -1.0
        for index, rotation in enumerate(rotations):
            if index in chosen:
                continue
            distance = min(rotation_angle(rotation, rotations[c]) for c in chosen)
            if distance > best_distance:
                best_index, best_distance = index, distance
        if best_index is None:
            break
        chosen.append(best_index)
    return chosen


def main() -> None:
    cfg = make_env_cfg(args_cli.task, num_envs=1, device=args_cli.device)
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    device = robot.data.joint_pos.device
    lower = robot.data.soft_joint_pos_limits[0, :, 0].cpu().numpy()
    upper = robot.data.soft_joint_pos_limits[0, :, 1].cpu().numpy()
    joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)
    gripper_index = body_names.index("gripper")
    # The base link sits at the origin by construction, below the tabletop
    # surface constant, so including it would reject every pose regardless of
    # the arm's configuration.  Only the links that actually move can collide.
    moving_indices = [i for i, n in enumerate(body_names) if n != "base"]
    env_origin = env.scene.env_origins[0].cpu().numpy()

    cameras = {"wrist": env.scene["wrist_camera"], "front": env.scene["external_camera"]}
    wanted = args_cli.camera or ["wrist", "front"]

    # Sample the camera rig once.  The front camera is static, and the wrist
    # camera is rigidly attached to the gripper, so its pose at any joint
    # configuration follows from the gripper's.  Deriving it beats calling
    # Camera.update() per candidate, which would re-render the scene thousands
    # of times to learn nothing but a transform.
    env.sim.step()
    env.scene.update(dt=env.physics_dt)
    rig: dict[str, dict] = {}
    for name, camera in cameras.items():
        cam_pos = camera.data.pos_w[0].cpu().numpy()
        cam_rot = quat_to_matrix(camera.data.quat_w_ros[0].cpu().numpy())
        rig[name] = {
            "intrinsics": camera.data.intrinsic_matrices[0].cpu().numpy(),
            "size": (camera.image_shape[1], camera.image_shape[0]),
            "pos": cam_pos,
            "rot": cam_rot,
        }
    gripper_pos0 = robot.data.body_pos_w[0, gripper_index].cpu().numpy()
    gripper_rot0 = quat_to_matrix(robot.data.body_quat_w[0, gripper_index].cpu().numpy())
    # T_gripper_cam for the wrist camera, held fixed from here on.
    wrist_rel_rot = gripper_rot0.T @ rig["wrist"]["rot"]
    wrist_rel_pos = gripper_rot0.T @ (rig["wrist"]["pos"] - gripper_pos0)

    print(f"joints      : {joint_names}")
    print(f"lower limits: {np.round(lower, 4).tolist()}")
    print(f"upper limits: {np.round(upper, 4).tolist()}")
    print(f"bodies      : {body_names}")

    rng = np.random.default_rng(args_cli.seed)
    # The jaw is not part of the arm's geometry for these purposes; hold it
    # closed so the tags stay put and nothing swings.
    jaw_index = joint_names.index("Jaw") if "Jaw" in joint_names else None

    board = board_points() + env_origin
    results: dict[str, list[dict]] = {name: [] for name in wanted}
    rotations: dict[str, list[np.ndarray]] = {name: [] for name in wanted}
    rejected = {name: {"table": 0, "visibility": 0} for name in wanted}

    # Uniform sampling almost never aims the wrist camera at the board -- it
    # found 14 poses in 6000 tries, several nearly identical.  So the search
    # runs in two phases: sample uniformly to find seeds, then resample around
    # whatever worked.  Diversity is not lost, because the selection step still
    # picks for rotational spread across everything found.
    seeds: dict[str, list[np.ndarray]] = {name: [] for name in wanted}
    refine_from = args_cli.candidates // 3
    jitter = 0.25

    for candidate_index in range(args_cli.candidates):
        pool = [s for name in wanted for s in seeds[name]]
        if candidate_index >= refine_from and pool:
            base_sample = pool[rng.integers(len(pool))]
            sample = np.clip(
                base_sample + rng.normal(0.0, jitter, size=len(lower)), lower, upper
            )
        else:
            sample = rng.uniform(lower, upper)
        if jaw_index is not None:
            sample[jaw_index] = float(lower[jaw_index])
        joint_tensor = torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(0)
        robot.write_joint_state_to_sim(
            joint_tensor, torch.zeros_like(joint_tensor)
        )
        robot.set_joint_position_target(joint_tensor)
        robot.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

        body_pos = robot.data.body_pos_w[0].cpu().numpy() - env_origin
        if float(body_pos[moving_indices, 2].min()) < ROBOT_BASE_BOTTOM_Z + TABLE_CLEARANCE:
            for name in wanted:
                rejected[name]["table"] += 1
            continue

        gripper_pos = robot.data.body_pos_w[0, gripper_index].cpu().numpy()
        gripper_rot = quat_to_matrix(
            robot.data.body_quat_w[0, gripper_index].cpu().numpy()
        )

        for name in wanted:
            if name == "wrist":
                cam_rot = gripper_rot @ wrist_rel_rot
                cam_pos = gripper_pos + gripper_rot @ wrist_rel_pos
            else:
                cam_pos, cam_rot = rig[name]["pos"], rig[name]["rot"]
            intrinsics = rig[name]["intrinsics"]
            size = rig[name]["size"]

            points = (
                board if name == "wrist"
                else gripper_board_points(gripper_pos, gripper_rot)
            )
            mask = visible_mask(
                points, cam_pos, cam_rot, intrinsics, size, MIN_TARGET_MARGIN_PX
            )
            # Both targets are now full ChArUco boards, which resolve pose from
            # partial views -- that is the point of ChArUco over a plain
            # checkerboard or a single marker.  Demanding the whole outline be
            # visible would force the camera to stand back far enough that few
            # or no reachable poses qualify.  The last sample point is the
            # board's centre, required regardless.
            ok = bool(mask[-1]) and int(mask.sum()) >= MIN_VISIBLE_BOARD_POINTS
            if not ok:
                rejected[name]["visibility"] += 1
                continue

            results[name].append(
                {
                    "joint_positions": [float(v) for v in sample],
                    "gripper_pos_env": [float(v) for v in (gripper_pos - env_origin)],
                }
            )
            rotations[name].append(gripper_rot)
            seeds[name].append(sample.copy())

    args_cli.out_dir.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        found = len(results[name])
        print(f"\n[{name}] {found} candidate poses out of {args_cli.candidates}")
        print(
            f"  rejected: {rejected[name]['table']} for table clearance, "
            f"{rejected[name]['visibility']} for target visibility"
        )
        if found == 0:
            print("  -> nothing to write; widen the sampling or check the assumed")
            print("     board position")
            continue

        picked = select_diverse(rotations[name], args_cli.count)
        chosen_rotations = [rotations[name][i] for i in picked]
        spread = [
            rotation_angle(chosen_rotations[i], chosen_rotations[j])
            for i in range(len(chosen_rotations))
            for j in range(i + 1, len(chosen_rotations))
        ]
        print(f"  selected {len(picked)} for rotational spread")
        if spread:
            print(
                f"  pairwise rotation: min {min(spread):.1f} deg, "
                f"mean {float(np.mean(spread)):.1f} deg, max {max(spread):.1f} deg"
            )

        payload = {
            "camera": name,
            "joint_names": joint_names,
            "board_centre_env": [float(v) for v in BOARD_CENTRE_ENV],
            "table_clearance_m": TABLE_CLEARANCE,
            "min_pairwise_rotation_deg": float(min(spread)) if spread else None,
            "mean_pairwise_rotation_deg": float(np.mean(spread)) if spread else None,
            "poses": [results[name][i] for i in picked],
            "warning": (
                "Simulation does not model cables, clamps or the front camera "
                "mount. Watch the first replay with a hand on the stop."
            ),
        }
        path = args_cli.out_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  wrote {path}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
