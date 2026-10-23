"""Render ResiP teacher rollouts into image demonstrations (ResiP §II-D).

The state-based teacher (scripts/export_resip_teacher.py) drives the visual
StackCube scene; every step records what a real-robot student can observe --
6 joint positions and front/wrist RGB -- next to the teacher's joint-target
action, in the same trajectory_*.pkl format scripts/teleop_task.py writes, so
the student trainers read synthetic and real demos the same way.

Per round (all envs reset together):
  reset -> resample rendering randomization -> one hold step (the frame
  env.reset() returns is stale, see check_camera_lag.py) -> teacher rollout.
An env is recorded from that first fresh frame until `--post-success-steps`
after its first success (so the release/retreat is in the demo), and only
successful episodes are saved, as in the paper.
Acceptance additionally requires a complete post-success tail and consecutive
success on its final frames. --motion-substeps interpolates each teacher
target in the simulator; images and states are recorded at every 30 Hz tick.

Images are stored at 240x320 via so101.learning.act.data.to_uint8_rgb, the
same resize teleop and inference use.

Example:
    python -u scripts/generate_teacher_demos.py --headless --num-episodes 600 --out outputs/synthetic/resip_v1
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher  # noqa: E402

TASK = "so101-visual-StackCube-v0"
parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--teacher", type=Path, default=REPO_ROOT / "outputs/teachers/resip_sd042_20260924_210715_e2000")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--num-episodes", type=int, default=400, help="successful episodes to save")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=0, help="round r resets with seed + r")
parser.add_argument("--post-success-steps", type=int, default=40)
parser.add_argument("--stable-success-steps", type=int, default=15,
                    help="require success on this many final consecutive frames")
parser.add_argument("--motion-substeps", type=int, default=1,
                    help="interpolate each teacher target over N control ticks, recording actual simulated states")
parser.add_argument("--wrist-pos-jitter-m", type=float, default=0.01)
parser.add_argument("--wrist-rot-jitter-deg", type=float, default=3.0)
parser.add_argument("--max-rounds", type=int, default=1000)
parser.add_argument("--image-height", type=int, default=240)
parser.add_argument("--image-width", type=int, default=320)
parser.add_argument("--no-randomization", action="store_true", help="render the nominal scene only")
parser.add_argument("--no-backdrop", action="store_true", help="no randomized floor/walls")
parser.add_argument("--env-spacing", type=float, default=30.0, help="keeps neighbouring envs out of the front view")
parser.add_argument("--diverse", action="store_true",
                    help="save failed and perturbed rollouts as well as successes")
parser.add_argument("--perturbation-std", type=float, default=0.04,
                    help="temporally coherent joint-target noise std in radians")
parser.add_argument("--perturbation-hold", type=int, default=8,
                    help="number of ticks between perturbation updates")
parser.add_argument("--record-cube-instance-masks", action="store_true",
                    help="record path-resolved held/fixed cube instance masks (diagnostics/targets only)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import so101.tasks  # noqa: E402,F401
from so101.configs import make_env_cfg  # noqa: E402
from so101.learning.act.data import to_uint8_rgb  # noqa: E402
from so101.learning.simdist.instance_masks import resolve_cube_instance_masks  # noqa: E402
from so101.learning.resip_teacher import ResiPTeacher  # noqa: E402
from so101.tasks.render_randomization import add_backdrop, RenderRandomizer, RenderRandomizationCfg  # noqa: E402
from so101.tasks.teacher import teacher_observation  # noqa: E402
from so101.tasks.assets import SMALL_CUBE_SIZE, LARGE_CUBE_SIZE  # noqa: E402

INDEXED = re.compile(r"^trajectory_(\d+)\.pkl$")


def next_index(directory: Path) -> int:
    indices = [int(m.group(1)) for p in directory.glob("trajectory_*.pkl") if (m := INDEXED.match(p.name))]
    return max(indices, default=-1) + 1


def _cube_instance_masks_pair(front_camera, wrist_camera, env_index: int):
    """Resolve each camera's raw IDs through its current per-frame path map."""
    key = "instance_id_segmentation_fast"
    packed_masks, paths_by_camera = [], []
    for camera in (front_camera, wrist_camera):
        data = camera.data
        if key not in data.output:
            raise RuntimeError(f"camera output lacks {key}; enable the collector annotation flag")
        info = data.info[env_index].get(key)
        if not isinstance(info, dict) or not isinstance(info.get("idToLabels"), dict):
            raise RuntimeError(f"camera env {env_index} lacks {key}.info.idToLabels path map")
        image = data.output[key][env_index, ..., 0].detach().cpu().numpy()
        packed, id_to_path = resolve_cube_instance_masks(image, info["idToLabels"], env_index)
        packed_masks.append(packed)
        paths_by_camera.append(id_to_path)
    return np.stack(packed_masks, axis=0), paths_by_camera


def main() -> None:
    if (args_cli.motion_substeps < 1 or not 1 <= args_cli.stable_success_steps <= args_cli.post_success_steps
            or args_cli.perturbation_std < 0 or args_cli.perturbation_hold < 1):
        raise ValueError("motion-substeps must be >= 1; stable-success-steps must be between 1 and post-success-steps")
    cfg = make_env_cfg(TASK, num_envs=args_cli.num_envs, device=args_cli.device)
    cfg.terminate_on_success = False
    cfg.truncate_on_timeout = False
    cfg.scene.env_spacing = args_cli.env_spacing
    # Depth is not recorded; dropping it roughly halves the render memory.
    # Instance IDs are an opt-in teacher target. Uncolorized int32 IDs are
    # resolved through the per-frame idToLabels USD-path mapping below.
    camera_types = (["rgb", "instance_id_segmentation_fast"]
                    if args_cli.record_cube_instance_masks else ["rgb"])
    cfg.scene.wrist_camera.data_types = list(camera_types)
    cfg.scene.external_camera.data_types = list(camera_types)
    if args_cli.record_cube_instance_masks:
        cfg.scene.wrist_camera.colorize_instance_id_segmentation = False
        cfg.scene.external_camera.colorize_instance_id_segmentation = False
    if not args_cli.no_backdrop:
        add_backdrop(cfg.scene)
    env = gym.make(TASK, cfg=cfg).unwrapped
    n = env.num_envs
    teacher = ResiPTeacher(args_cli.teacher, device=env.device)
    generator = torch.Generator(device=env.device).manual_seed(args_cli.seed)
    size = (args_cli.image_height, args_cli.image_width)
    args_cli.out.mkdir(parents=True, exist_ok=True)
    index = next_index(args_cli.out)
    (args_cli.out / "generation.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args_cli).items()}, indent=2))
    for name in ("wrist", "front"):
        (args_cli.out / f"calibration_{name}.yaml").write_text(
            (REPO_ROOT / "calibration" / "cameras" / f"{name}.yaml").read_text())

    saved = attempted = ever_succeeded = 0
    round_index = 0
    randomizer = None
    started = time.perf_counter()
    # The Gym task horizon is already expressed in simulator control ticks.
    # motion_substeps changes how often teacher targets are refreshed, not how
    # many seconds the task lasts.
    horizon = int(env.max_episode_length)
    while saved < args_cli.num_episodes:
        if round_index >= args_cli.max_rounds:
            raise RuntimeError(f"Reached max-rounds with only {saved} accepted episodes")
        with torch.inference_mode():
            observation, _ = env.reset(seed=args_cli.seed + round_index)
            if randomizer is None and not args_cli.no_randomization:
                randomizer = RenderRandomizer(env, cfg=RenderRandomizationCfg(
                    wrist_pos_jitter_m=args_cli.wrist_pos_jitter_m,
                    wrist_rot_jitter_deg=args_cli.wrist_rot_jitter_deg,
                ), seed=args_cli.seed)
            if randomizer is not None:
                randomizer.apply(env)
            observation, *_ = env.step(env.robot.data.default_joint_pos.clone())
            teacher.reset()

            keys = ("obs", "act", "front", "wrist", "teacher", "cube_positions",
                    "next_cube_positions", "next_grasp_points", "table_height", "camera_pose",
                    "camera_intrinsics", "success", "reward", "phase", "expert",
                    "terminated", "truncated", "next_obs", "next_front", "next_wrist", "tick")
            if args_cli.record_cube_instance_masks:
                keys = (*keys, "cube_instance_masks", "camera_frame_counters")
            buffers = [{k: [] for k in keys} for _ in range(n)]
            instance_id_paths_by_camera = [{"front": {}, "wrist": {}} for _ in range(n)]
            success = torch.zeros(n, dtype=torch.bool, device=env.device)
            perturbation = torch.zeros((n, 6), dtype=torch.float32, device=env.device)
            stop_at = torch.full((n,), horizon + args_cli.post_success_steps, device=env.device)
            for step in range(horizon + args_cli.post_success_steps):
                recording = (step <= stop_at) & (success | (step < horizon))
                if not recording.any():
                    break
                t_obs = teacher_observation(env)
                if step % args_cli.motion_substeps == 0:
                    segment_start = observation["state"].clone()
                    segment_target = teacher.act(t_obs, generator=generator)
                fraction = (step % args_cli.motion_substeps + 1) / args_cli.motion_substeps
                action = segment_start + fraction * (segment_target - segment_start)
                if args_cli.diverse:
                    if step % args_cli.perturbation_hold == 0:
                        perturbation = torch.randn((n, 6), generator=generator, device=env.device) * args_cli.perturbation_std
                    limits = env.robot.data.soft_joint_pos_limits
                    action = torch.maximum(torch.minimum(action + perturbation, limits[:, :, 1]), limits[:, :, 0])
                state = observation["state"].cpu().numpy()
                act = action.cpu().numpy()
                t_obs_np = t_obs.cpu().numpy()
                cube_positions = t_obs_np[:, (4, 5, 6, 11, 12, 13)].reshape(n, 2, 3)
                # Record the actual rendered camera geometry for this same
                # current observation. Pose uses robot-root-relative world
                # translation and world-forward quaternion (w, x, y, z).
                camera_positions = torch.stack((
                    env.external_camera.data.pos_w,
                    env.wrist_camera.data.pos_w,
                ), dim=1) - env.robot.data.root_pos_w[:, None, :]
                camera_quaternions = torch.stack((
                    env.external_camera.data.quat_w_world,
                    env.wrist_camera.data.quat_w_world,
                ), dim=1)
                camera_pose = torch.cat((camera_positions, camera_quaternions), dim=-1)
                camera_intrinsics = torch.stack((
                    env.external_camera.data.intrinsic_matrices,
                    env.wrist_camera.data.intrinsic_matrices,
                ), dim=1)
                for i in torch.nonzero(recording).flatten().tolist():
                    b = buffers[i]
                    b["obs"].append(state[i].copy())
                    b["act"].append(act[i].copy())
                    b["teacher"].append(t_obs_np[i].copy())
                    b["cube_positions"].append(cube_positions[i].copy())
                    b["camera_pose"].append(camera_pose[i].cpu().numpy().copy())
                    b["camera_intrinsics"].append(camera_intrinsics[i].cpu().numpy().copy())
                    if args_cli.record_cube_instance_masks:
                        masks, camera_paths = _cube_instance_masks_pair(
                            env.external_camera, env.wrist_camera, i
                        )
                        b["cube_instance_masks"].append(masks)
                        b["camera_frame_counters"].append(torch.stack((
                            env.external_camera.frame[i], env.wrist_camera.frame[i],
                        )).cpu().numpy().astype(np.int64, copy=True))
                        for camera_name, entries in zip(("front", "wrist"), camera_paths):
                            known = instance_id_paths_by_camera[i][camera_name]
                            for instance_id, path in entries.items():
                                if instance_id in known and known[instance_id] != path:
                                    raise RuntimeError("instance ID changed USD path within one episode")
                                known[instance_id] = path
                    b["front"].append(to_uint8_rgb(observation["front_image"][i], size))
                    b["wrist"].append(to_uint8_rgb(observation["wrist_image"][i], size))
                observation, reward, terminated, truncated, info = env.step(action)
                # Privileged labels are captured at the actual successor tick.
                # They supervise training only; policy inference receives RGB + q.
                next_teacher = teacher_observation(env).cpu().numpy()
                next_cube_positions = next_teacher[:, (4, 5, 6, 11, 12, 13)].reshape(n, 2, 3)
                midpoint, fixed_fingertip, moving_fingertip = env._get_grasp_points()
                robot_root = env.robot.data.root_pos_w
                next_grasp_points = torch.stack(
                    (midpoint, fixed_fingertip, moving_fingertip), dim=1
                ) - robot_root[:, None, :]
                table_height = (
                    env._held_initial_root_z - SMALL_CUBE_SIZE / 2.0 - robot_root[:, 2]
                )
                step_success = info["success"].bool()
                newly = step_success & ~success & recording
                stop_at[newly] = step + args_cli.post_success_steps
                success |= step_success & recording
                for i in torch.nonzero(recording).flatten().tolist():
                    buffers[i]["success"].append(bool(step_success[i]))
                    buffers[i]["phase"].append("post_success" if bool(success[i]) else "pre_success")
                    buffers[i]["expert"].append(
                        bool(not args_cli.diverse or perturbation[i].abs().max().item() < 1e-8)
                    )
                    buffers[i]["reward"].append(float(reward[i]))
                    buffers[i]["terminated"].append(bool(terminated[i]))
                    buffers[i]["truncated"].append(bool(truncated[i]))
                    buffers[i]["next_obs"].append(observation["state"][i].cpu().numpy().copy())
                    buffers[i]["next_cube_positions"].append(next_cube_positions[i].copy())
                    buffers[i]["next_grasp_points"].append(next_grasp_points[i].cpu().numpy().copy())
                    buffers[i]["table_height"].append(float(table_height[i].item()))
                    buffers[i]["next_front"].append(to_uint8_rgb(observation["front_image"][i], size))
                    buffers[i]["next_wrist"].append(to_uint8_rgb(observation["wrist_image"][i], size))
                    buffers[i]["tick"].append(step)

        attempted += n
        accepted = ([i for i in range(n) if buffers[i]["act"]] if args_cli.diverse else
                    [i for i in torch.nonzero(success).flatten().tolist()
                     if len(buffers[i]["success"]) == int(stop_at[i]) + 1
                     and all(buffers[i]["success"][-args_cli.stable_success_steps:])])
        ever_succeeded += int(success.sum())
        phase_counts = {"pre_success": 0, "post_success": 0}
        for i in accepted:
            if saved >= args_cli.num_episodes:
                break
            b = buffers[i]
            for phase in b["phase"]:
                phase_counts[phase] += 1
            length = len(b["act"])
            # Recording disables auto-reset to retain the post-success tail.
            # After the required stable-success hold, the final captured
            # transition is a task terminal; otherwise it is censored timeout.
            stable_success = (
                len(b["success"]) >= args_cli.stable_success_steps
                and all(b["success"][-args_cli.stable_success_steps:])
            )
            termination_reason = "stable_success_hold" if stable_success else "collector_timeout"
            if length:
                b["terminated"][-1] = stable_success
                b["truncated"][-1] = not stable_success
            obs = np.stack(b["obs"])
            data = {
                "observations": obs,
                "actions": np.stack(b["act"]),
                # Keep the established imitation dataset's sparse success
                # field intact; diverse SimDist rollouts store dense task reward.
                "rewards": np.asarray(b["reward"] if args_cli.diverse else b["success"], dtype=np.float32),
                "terminated": np.asarray(b["terminated"], dtype=np.bool_),
                "truncated": np.asarray(b["truncated"], dtype=np.bool_),
                "terminals": np.asarray(b["terminated"], dtype=np.bool_),
                "successes": np.asarray(b["success"], dtype=np.bool_),
                "next_observations": np.stack(b["next_obs"]),
                "next_front_images": np.stack(b["next_front"]),
                "next_wrist_images": np.stack(b["next_wrist"]),
                "front_images": np.stack(b["front"]),
                "wrist_images": np.stack(b["wrist"]),
                "image_format": "uint8_rgb",
                "image_shape": [*size, 3],
                "image_size_source": [640, 480],
                "teacher_observations": np.stack(b["teacher"]),
                "cube_positions_robot_relative_m": np.stack(b["cube_positions"]).astype(np.float32),
                "next_cube_positions_robot_relative_m": np.stack(b["next_cube_positions"]).astype(np.float32),
                "cube_position_label_semantics": "simulator_xyz_robot_root_relative_training_only",
                "cube_position_order": ["held_cube", "fixed_cube"],
                "camera_poses_robot_relative_m_wxyz": np.stack(b["camera_pose"]).astype(np.float32),
                "camera_intrinsics_source_px": np.stack(b["camera_intrinsics"]).astype(np.float32),
                "camera_order": ["front", "wrist"],
                "camera_pose_semantics": "robot_root_relative_translation_world_axes_quaternion_wxyz_forward_x_up_z",
                "camera_intrinsics_semantics": "pinhole_3x3_source_sensor_pixel_coordinates",
                "camera_source_image_shape_hw": [list(env.external_camera.data.image_shape),
                                                  list(env.wrist_camera.data.image_shape)],
                **({
                    "cube_instance_masks_packed": np.stack(b["cube_instance_masks"]).astype(np.uint8),
                    "cube_instance_mask_camera_order": ["front", "wrist"],
                    "cube_instance_mask_object_order": ["held_cube", "fixed_cube"],
                    "cube_instance_mask_semantics": "uncolorized_instance_id_segmentation_fast_resolved_by_usd_prim_path",
                    "cube_instance_mask_storage": "np.packbits_axis_last_bitorder_big",
                    "cube_instance_mask_source_shape_hw": [list(env.external_camera.data.image_shape),
                                                              list(env.wrist_camera.data.image_shape)],
                    "camera_sensor_frame_counters": np.stack(b["camera_frame_counters"]).astype(np.int64),
                    "camera_sensor_frame_counter_semantics": "same_current_render_used_for_rgb_and_instance_masks",
                    "cube_instance_id_to_usd_path_by_camera": instance_id_paths_by_camera[i],
                    "cube_instance_masks_training_only": True,
                } if args_cli.record_cube_instance_masks else {}),
                "next_grasp_points_robot_relative_m": np.stack(b["next_grasp_points"]).astype(np.float32),
                "grasp_point_order": ["midpoint", "fixed_fingertip", "moving_fingertip"],
                "table_height_robot_relative_m": np.asarray(b["table_height"], dtype=np.float32),
                "geometry_reward_config": {
                    "small_cube_size_m": SMALL_CUBE_SIZE,
                    "large_cube_size_m": LARGE_CUBE_SIZE,
                    "distance_gain": float(env.cfg.stack_distance_gain),
                    "lift_clearance_m": float(env.cfg.stack_lift_clearance),
                    "success_xy_threshold_m": float(env.cfg.success_xy_threshold),
                    "success_height_threshold_m": float(env.cfg.success_height_threshold),
                    "gripper_away_threshold_m": float(env.cfg.stack_gripper_away_threshold),
                    "reward_distance_weight": 0.1,
                    "reward_lift_weight": 1.5,
                    "reward_align_weight": 2.0,
                    "reward_success": 16.0,
                },
                "source": "resip_teacher",
                "teacher": str(args_cli.teacher),
                "seed": args_cli.seed + round_index,
                "env_index": i,
                "render_randomization": randomizer is not None,
                "motion_substeps": args_cli.motion_substeps,
                "task_horizon_ticks": horizon,
                "collection_limit_ticks": horizon,
                "tick_origin": "first_recorded_action_after_reset_settle_hold",
                "post_success_steps": args_cli.post_success_steps,
                "stable_success_steps": args_cli.stable_success_steps,
                "first_success_step": next((j for j, value in enumerate(b["success"]) if value), None),
                "fps": 30,
                "tick": np.asarray(b["tick"], dtype=np.int64),
                "action_semantics": "applied_absolute_joint_target_radians",
                "reward_semantics": "dense_environment_reward" if args_cli.diverse else "sparse_success_indicator",
                "expert_policy_flag": np.asarray(b["expert"], dtype=np.bool_),
                "trajectory_quality": (
                    "perturbed" if args_cli.diverse and args_cli.perturbation_std > 0
                    else "expert_teacher" if args_cli.diverse else "successful_expert"
                ),
                "termination_reason": termination_reason,
                "phase_labels": b["phase"],
            }
            path = args_cli.out / f"trajectory_{index:06d}.pkl"
            with path.open("xb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            index += 1
            saved += 1
        round_index += 1
        elapsed = time.perf_counter() - started
        quality_counts = {"success": int(success.sum()), "failure": n - int(success.sum()),
                          "perturbed": n if args_cli.diverse and args_cli.perturbation_std > 0 else 0}
        print(
            f"[INFO] round {round_index}: quality={quality_counts}, accepted {len(accepted)}/{n}, saved {saved}/{args_cli.num_episodes} "
            f"phase_steps={phase_counts}, cumulative success {ever_succeeded / attempted:.1%}, {elapsed / 60:.1f} min",
        )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
