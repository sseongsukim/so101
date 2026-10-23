#!/usr/bin/env python3
"""Shared-seed Isaac evaluation of visual WM policy / reward MPPI / MC MPPI.

Uses simulator limits and six absolute joint targets. This script has no robot
connection or hardware actuation code. Short untrained smoke runs verify wiring,
and must not be reported as learned policy quality or real robustness evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
checkpoint_argument = parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--teacher", type=Path, default=Path("outputs/teachers/resip_sd042_20260924_210715_e2000"))
parser.add_argument("--dp-checkpoint", type=Path)
parser.add_argument("--post-success-steps", type=int, default=40)
parser.add_argument("--stable-success-steps", type=int, default=15)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--num-rounds", type=int, default=1)
parser.add_argument("--max-steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--methods", nargs="+", choices=("resip", "image_dp", "wm_policy", "mppi_reward", "mppi_mc",
                                                     "mppi_dp_reward", "mppi_dp_mc"),
                    default=("wm_policy", "mppi_reward", "mppi_mc"))
parser.add_argument("--samples", type=int, default=64)
parser.add_argument("--iterations", type=int, default=3)
parser.add_argument("--elites", type=int, default=16)
parser.add_argument("--max-policy-deviation-rad", type=float, default=0.25,
                    help="trust-region radius around native policy or external proposal")
parser.add_argument("--policy-std-rad", type=float, default=0.03,
                    help="proposal noise scale in absolute joint radians")
parser.add_argument("--init-std-rad", type=float, default=0.10,
                    help="MPPI refinement noise scale in absolute joint radians")
parser.add_argument("--max-velocity-rad-s", type=float, nargs=6, default=[7.5]*5+[15.0])
parser.add_argument("--model-device", default="cpu")
parser.add_argument("--render-randomization", action="store_true")
parser.add_argument("--env-spacing", type=float, default=30)
out_argument = parser.add_argument("--out", type=Path, required=True)
# IsaacLab temporarily disables help and calls parse_known_args() while adding
# its flags. Defer required-file validation during that preliminary parse so
# --help can display the completed parser without checkpoint/output arguments.
checkpoint_argument.required = out_argument.required = False
AppLauncher.add_app_launcher_args(parser)
checkpoint_argument.required = out_argument.required = True
args = parser.parse_args()
if min(args.num_envs, args.num_rounds, args.max_steps) < 1:
    parser.error("positive environments, rounds and steps required")
if any(method in args.methods for method in ("image_dp", "mppi_dp_reward", "mppi_dp_mc")) and args.dp_checkpoint is None:
    parser.error("image_dp and DP-proposal MPPI comparisons require --dp-checkpoint")
if not 1 <= args.stable_success_steps <= args.post_success_steps:
    parser.error("stable success hold must be positive and within post-success tail")
args.enable_cameras = True
app = AppLauncher(args).app


def main():
    import gymnasium as gym
    import numpy as np
    import torch

    import so101.tasks  # noqa: F401
    from so101.configs import make_env_cfg
    from so101.real.constants import SO101_SIM_JOINT_ORDER
    from so101.tasks.render_randomization import RenderRandomizer, add_backdrop
    from so101.learning.simdist.checkpoint import load_worldmodel
    from so101.learning.simdist.mppi import MppiConfig, MppiController, project_actions
    from so101.learning.resip_teacher import ResiPTeacher
    from so101.learning.dp.inference import DPRunner
    from so101.learning.act.data import CAMERA_OBSERVATION_KEYS, to_uint8_rgb
    from so101.tasks.teacher import teacher_observation

    if args.model_device == "cpu":
        torch.set_num_threads(1)
    model, saved = load_worldmodel(args.checkpoint, args.model_device)
    cfg = make_env_cfg("so101-visual-StackCube-v0", num_envs=args.num_envs, device=args.device)
    cfg.terminate_on_success = False; cfg.truncate_on_timeout = False
    cfg.seed = args.seed; cfg.scene.env_spacing = args.env_spacing
    cfg.scene.wrist_camera.data_types = ["rgb"]; cfg.scene.external_camera.data_types = ["rgb"]
    add_backdrop(cfg.scene)
    env = gym.make("so101-visual-StackCube-v0", cfg=cfg).unwrapped
    try:
        if list(env.robot.joint_names) != SO101_SIM_JOINT_ORDER:
            raise ValueError(f"joint order mismatch: {env.robot.joint_names}")
        if not np.isclose(env.step_dt, model.config.tick_seconds):
            raise ValueError("simulator step duration differs from trained WM")
        limits = env.robot.data.soft_joint_pos_limits.cpu().numpy()
        lower = torch.as_tensor(limits[:,:,0], device=env.device)
        upper = torch.as_tensor(limits[:,:,1], device=env.device)
        max_step = torch.as_tensor(args.max_velocity_rad_s, device=env.device) * env.step_dt
        teacher = ResiPTeacher(args.teacher, device=env.device) if "resip" in args.methods else None
        needs_dp = any(method in args.methods for method in ("image_dp", "mppi_dp_reward", "mppi_dp_mc"))
        dp = DPRunner(args.dp_checkpoint, device=env.device) if needs_dp else None
        # The diffusion runner owns a mutable action queue and warm start. Hybrid
        # MPPI asks for a proposal per environment, so each environment needs an
        # independent runner; sharing `dp` would consume env 0's queue for env 1.
        dp_hybrid = ([DPRunner(args.dp_checkpoint, device=env.device)
                      for _ in range(args.num_envs)]
                     if any(method in args.methods for method in ("mppi_dp_reward", "mppi_dp_mc")) else None)
        randomizer = RenderRandomizer(env, seed=args.seed) if args.render_randomization else None
        def observation_dict(observation, i, tick):
            return {"joints": observation["state"][i].cpu().numpy(),
                    "front": to_uint8_rgb(observation["front_image"][i], model.config.stored_image_shape),
                    "wrist": to_uint8_rgb(observation["wrist_image"][i], model.config.stored_image_shape),
                    "timestamp_seconds": tick * env.step_dt,
                    "remaining_ticks": model.config.task_horizon_ticks - tick}
        episodes, latencies = [], []
        layouts = {}
        settled_joints = {}
        dr_snapshots = {}
        def snapshot_dr(randomizer):
            def pose(p):
                q = p.orient.Get()
                return {"position": list(p.translate.Get()), "quaternion": [q.GetReal(), *q.GetImaginary()]}
            def material(shader):
                return {"color": list(shader.GetInput("diffuseColor").Get()),
                        "roughness": float(shader.GetInput("roughness").Get()) if shader.GetInput("roughness") and shader.GetInput("roughness").Get() is not None else None}
            return {"lights": [{"intensity": float(light.GetIntensityAttr().Get()), "color": list(light.GetColorAttr().Get())}
                               for light in (randomizer.key, randomizer.dome)],
                    "key_pose": pose(randomizer.key_pose),
                    "front": [pose(p) for p in randomizer.front], "wrist": [pose(p) for p in randomizer.wrist],
                    "cubes": {k: [material(s) for s in v] for k,v in randomizer.cube_shaders.items()},
                    "table": [material(s) for s in randomizer.table_shaders],
                    "backdrop": {k: [material(s) for s in v] for k,v in randomizer.backdrop_shaders.items()}}

        for method in args.methods:
            arm_latencies = []
            for round_idx in range(args.num_rounds):
                seed = args.seed + round_idx
                torch.manual_seed(seed)
                obs, _ = env.reset(seed=seed)
                layout = {"held": env.held_asset.data.root_pose_w.cpu().numpy().tolist(),
                          "fixed": env.fixed_asset.data.root_pose_w.cpu().numpy().tolist()}
                if round_idx in layouts:
                    for k in layout:
                        if not np.allclose(layout[k], layouts[round_idx][k], atol=1e-6):
                            raise RuntimeError("same-seed arm resets produced different cube layouts")
                else:
                    layouts[round_idx] = layout
                if randomizer is not None:
                    randomizer.rng = np.random.default_rng(seed)
                    randomizer.apply(env)
                    snapshot = snapshot_dr(randomizer)
                    if round_idx in dr_snapshots and snapshot != dr_snapshots[round_idx]:
                        raise RuntimeError("paired arms have unequal actual rendering randomization parameters")
                    dr_snapshots[round_idx] = snapshot
                hold = env.robot.data.default_joint_pos.clone()
                # One actual render tick replaces reset's potentially stale RGB.
                obs, *_ = env.step(hold)
                controllers = [MppiController(model, limits[i,:,0], limits[i,:,1], args.max_velocity_rad_s,
                                MppiConfig(num_samples=args.samples, num_elites=args.elites,
                                           iterations=args.iterations,
                                           value_weight=0 if method in ("mppi_reward", "mppi_dp_reward") else 1,
                                           max_policy_deviation_rad=args.max_policy_deviation_rad,
                                           policy_std_rad=args.policy_std_rad,
                                           init_std_rad=args.init_std_rad,
                                           seed=seed+i)) for i in range(args.num_envs)]
                for i,c in enumerate(controllers):
                    c.reset(observation_dict(obs,i,0), hold[i].cpu().numpy())
                startup_rewards = np.zeros(args.num_envs)
                startup_discounted = np.zeros(args.num_envs)
                for tick in range(1, model.config.history_length+1):
                    obs, hold_reward, *_ = env.step(hold)
                    startup_rewards += hold_reward.cpu().numpy()
                    startup_discounted += model.config.gamma ** (tick-1) * hold_reward.cpu().numpy()
                    for i,c in enumerate(controllers):
                        c.update(observation_dict(obs,i,tick), hold[i].cpu().numpy())
                if round_idx in settled_joints:
                    if not np.allclose(obs["state"].cpu().numpy(), settled_joints[round_idx], atol=1e-5):
                        raise RuntimeError("same-seed arms have unequal post-settle joint state")
                else:
                    settled_joints[round_idx] = obs["state"].cpu().numpy().copy()
                if teacher is not None:
                    teacher.reset()
                if dp is not None:
                    dp.reset()
                if dp_hybrid is not None:
                    for runner in dp_hybrid:
                        runner.reset()
                torch.manual_seed(seed)  # DPRunner/DDIM samples from Torch's global RNG.
                teacher_rng = torch.Generator(device=env.device).manual_seed(seed)
                success = np.zeros(args.num_envs, bool); stacked = success.copy()
                instant_success = success.copy(); done = success.copy()
                first = np.full(args.num_envs, -1); terminal_tick = np.full(args.num_envs, -1)
                stable_count = np.zeros(args.num_envs, np.int64)
                rewards = startup_rewards.copy(); discounted_rewards = startup_discounted.copy()
                sent = hold.clone()
                sent_action_trace = [[] for _ in range(args.num_envs)]
                for step in range(min(args.max_steps, model.config.task_horizon_ticks - model.config.history_length)):
                    tick = model.config.history_length + step + 1
                    if method in ("resip", "image_dp"):
                        torch.cuda.synchronize() if str(env.device).startswith("cuda") else None
                        started = perf_counter()
                        if method == "resip":
                            proposed = teacher.act(teacher_observation(env), generator=teacher_rng)
                        else:
                            images = {k: obs[obs_key] for k,obs_key in CAMERA_OBSERVATION_KEYS.items()}
                            proposed = dp.act(obs["state"], images)
                        proposed = project_actions(proposed.unsqueeze(1), sent, lower, upper, max_step)[:,0]
                        active = torch.as_tensor(~done, device=env.device)
                        sent = torch.where(active[:,None], proposed, sent)
                        torch.cuda.synchronize() if str(env.device).startswith("cuda") else None
                        elapsed = perf_counter()-started
                        row = {"method": method, "round": round_idx, "step": step,
                               "latency_seconds": elapsed, "batch_size": args.num_envs,
                               "latency_scope": "policy+common_constraints; batch latency",
                               "within_tick_budget": elapsed <= env.step_dt}
                        latencies.append(row); arm_latencies.append(elapsed)
                    else:
                        for i,c in enumerate(controllers):
                            if done[i]:
                                continue
                            started = perf_counter()
                            proposal_chunk = None
                            if method in ("mppi_dp_reward", "mppi_dp_mc"):
                                runner = dp_hybrid[i]
                                dp_action = runner.act(obs["state"][i:i+1], {
                                    key: obs[obs_key][i:i+1] for key, obs_key in CAMERA_OBSERVATION_KEYS.items()
                                })
                                queued = list(runner.policy.actions)
                                proposal_rows = [dp_action[0]] + [row[0] for row in queued[:model.config.prediction_length-1]]
                                proposal_rows.extend([proposal_rows[-1]] * (model.config.prediction_length-len(proposal_rows)))
                                proposal_chunk = torch.stack(proposal_rows)
                            plan = c.plan(policy_only=method=="wm_policy", proposal_chunk=proposal_chunk)
                            elapsed = perf_counter() - started
                            if proposal_chunk is not None:
                                c.last_report["latency_seconds"] = elapsed
                                c.last_report["within_tick_budget"] = elapsed <= model.config.tick_seconds
                            proposed = torch.as_tensor(plan[0], device=env.device)
                            # Same final safeguard for all five arms; MPPI already
                            # applies these constraints inside every candidate.
                            sent[i] = project_actions(proposed.unsqueeze(0), sent[i], lower[i], upper[i], max_step)[0]
                            row = {"method": method, "round": round_idx, "env": i, "step": step,
                                   "latency_scope": ("DP inference+preprocessing+per-controller MPPI planner; host image transfer excluded"
                                                     if proposal_chunk is not None else
                                                     "per-controller encoded-history planner; host image transfer excluded"),
                                   **c.last_report}
                            latencies.append(row); arm_latencies.append(row["latency_seconds"])
                    previous_done = done.copy()
                    for i in range(args.num_envs):
                        sent_action_trace[i].append(sent[i].detach().cpu().tolist())
                    obs, reward, _, _, info = env.step(sent)
                    step_success = info["success"].cpu().numpy().astype(bool)
                    first[step_success & ~instant_success] = step+1
                    instant_success |= step_success
                    stable_count = np.where(step_success, stable_count+1, 0)
                    raw_reward = reward.cpu().numpy() * ~previous_done
                    rewards += raw_reward
                    discounted_rewards += model.config.gamma ** (tick-1) * raw_reward
                    # A true success terminal requires the same tail/hold rule
                    # as collected labels; instant success remains a diagnostic.
                    held_success = ((first >= 0) & (step+1 >= first + args.post_success_steps)
                                    & (stable_count >= args.stable_success_steps))
                    newly_done = held_success & ~done
                    terminal_tick[newly_done] = tick
                    success |= held_success; done |= held_success
                    stacked |= info["stacked"].cpu().numpy().astype(bool)
                    for i,c in enumerate(controllers):
                        c.update(observation_dict(obs,i,tick), sent[i].cpu().numpy())
                    if done.all():
                        break
                for i in range(args.num_envs):
                    episodes.append({"method": method, "round": round_idx, "env": i, "seed": seed,
                                     "success": bool(success[i]), "stacked": bool(stacked[i]),
                                     "instant_success": bool(instant_success[i]),
                                     "first_success_step": int(first[i]), "terminal_tick": int(terminal_tick[i]),
                                     "reward_sum": float(rewards[i]), "discounted_reward": float(discounted_rewards[i]),
                                     "sent_actions_rad": sent_action_trace[i],
                                     "steps": int(terminal_tick[i] - model.config.history_length) if terminal_tick[i]>=0 else step+1,
                                     "history_hold_ticks": model.config.history_length,
                                     "objective_ticks": int(terminal_tick[i]) if terminal_tick[i]>=0 else model.config.history_length + step+1,
                                     "full_horizon_completed": model.config.history_length + step+1 == model.config.task_horizon_ticks})
                print(json.dumps({"method": method, "round": round_idx, "success": int(success.sum()),
                                  "episodes": args.num_envs}), flush=True)
            print(json.dumps({"method": method, "mean_plan_seconds": float(np.mean(arm_latencies)),
                              "p95_plan_seconds": float(np.percentile(arm_latencies,95))}), flush=True)
        result = {"checkpoint": str(args.checkpoint.resolve()), "training_steps": saved["step"],
                  "research_status": "simulation evaluation only; real robustness unmeasured",
                  "config": vars(args).copy(), "shared_layouts": layouts, "shared_rendering_parameters": dr_snapshots,
                  "objective_horizon_includes_history_hold_ticks": True, "episodes": episodes,
                  "plan_latencies": latencies, "tick_seconds": env.step_dt}
        for key in ("checkpoint", "out", "teacher", "dp_checkpoint"):
            result["config"][key] = str(getattr(args,key)) if getattr(args,key) is not None else None
        result["common_constraints"] = {"joint_order": SO101_SIM_JOINT_ORDER,
                                        "lower_radians": limits[:,:,0].tolist(), "upper_radians": limits[:,:,1].tolist(),
                                        "max_velocity_rad_s": args.max_velocity_rad_s}
        result["success_rule"] = {"post_success_steps": args.post_success_steps,
                                  "stable_success_steps": args.stable_success_steps}
        result["summary"] = {m: {"success_rate": float(np.mean([e["success"] for e in episodes if e["method"]==m])),
                                    "stacked_rate": float(np.mean([e["stacked"] for e in episodes if e["method"]==m]))}
                             for m in args.methods}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2)+"\n")
        print(f"wrote {args.out}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
