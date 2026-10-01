"""Run an image student policy (DP or ACT) on the real SO-101.

Per 30 Hz tick: follower joints (Isaac radians, follower mapping) + rectified
front/wrist RGB -> runner.act -> clamp to joint limits and rate-limit ->
follower. The runner does the training-identical resize/normalization
(DPRunner / ACTRunner), so nothing here touches images beyond BGR -> RGB.

Safety:
  --dry-run        move to the start pose, then read + infer only: the policy's
                   targets are printed, never sent
  rate limit       --max-step-rad per tick (default 0.25 rad; the teacher moves up
                   to 0.175 rad/tick, so a tighter limit makes the arm lag its plan)
  start            eased move to the simulated start pose before each episode
  stop             Ctrl+C (or --max-seconds) -> hold, then eased move to the
                   folded home pose before torque is released

Every episode is recorded (same trajectory_*.pkl format as the demos) under
--record, so failures can be replayed and inspected.

Example:
    python scripts/deploy_policy_real.py --policy dp --checkpoint outputs/dp_train/v1/dp_final.pt --dry-run
    python scripts/deploy_policy_real.py --policy dp --checkpoint outputs/dp_train/v1/dp_final.pt --episodes 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)

import torch  # noqa: E402

# One intra-op thread: with the default pool (one thread per core) small CPU
# tensor ops stalled this 30 Hz loop for ~100 ms every few ticks whenever
# anything else (camera threads, Isaac) was using the CPU.
torch.set_num_threads(1)

from so101.learning.act.data import CAMERA_SOURCES  # noqa: E402
from so101.real.constants import SO101_HOME_POSE, SO101_JOINT_ORDER  # noqa: E402
from so101.real.follower import CameraThread, FollowerArm, Rate, SafetyLimits, START_POSE  # noqa: E402
from so101.real.cameras import METER_TARGETS  # noqa: E402
from so101.real.recording import EpisodeRecorder  # noqa: E402

CAMERA_OF_KEY = {"observation.images.front": "front", "observation.images.wrist": "wrist"}


def make_runner(policy: str, checkpoint: Path, device: str, action_steps: int | None, inference_steps: int | None = None):
    if policy == "act":
        from so101.learning.act.inference import ACTRunner

        return ACTRunner(checkpoint, device=device, n_action_steps=action_steps)
    from so101.learning.dp.inference import DPRunner

    kwargs = {} if action_steps is None else {"action_horizon": action_steps}
    if inference_steps is not None:
        kwargs["inference_steps"] = inference_steps
    return DPRunner(checkpoint, device=device, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policy", choices=["dp", "act"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--action-steps", type=int, default=None, help="executed actions per query (runner default if unset)")
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=4,
        help="DP only: DDIM steps per plan. Measured on the RTX 2080 Ti: 16 -> ~100 ms, 8 -> ~54 ms, "
        "4 -> ~31 ms per plan; at 30 Hz a plan must fit in ~33 ms or the arm pauses every chunk",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--max-step-rad", type=float, default=0.25,
                        help="per-tick arm target change limit (gripper: 0.5); keep above the teacher's ~0.175")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-meter", action="store_true", help="keep the fixed camera exposures (cameras.DEFAULT_SPECS)")
    parser.add_argument("--record", type=Path, default=REPO_ROOT / "outputs/real_rollouts")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--robot-id", default="my_follower")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    runner = make_runner(args.policy, args.checkpoint, args.device, args.action_steps,
                         args.inference_steps if args.policy == "dp" else None)
    arm = FollowerArm(args.port, args.robot_id, SafetyLimits(max_step_rad=args.max_step_rad))
    recorder = EpisodeRecorder(args.record / args.checkpoint.parent.name, source=f"real_rollout_{args.policy}",
                               checkpoint=str(args.checkpoint), dry_run=args.dry_run)
    cameras: dict[str, CameraThread] = {}
    try:
        arm.connect()
        # Meter the cameras on the scene the policy starts from, to the
        # brightness of its training frames (daylight changes the fixed
        # exposure's result a lot; cameras.METER_TARGETS).
        arm.move_to(START_POSE, seconds=3.0)
        cameras = {name: CameraThread(name, None if args.no_meter else METER_TARGETS[name]) for name in ("front", "wrist")}
        for name, camera in cameras.items():
            print(f"[info] {name} exposure: {camera.camera.exposure_lock.as_dict()}")
        for episode in range(args.episodes):
            answer = input(f"[episode {episode + 1}/{args.episodes}] place the cubes, Enter to start, q to quit: ")
            if answer.strip().lower() == "q":
                break
            # Also in --dry-run: the policy should see the pose it was trained
            # to start from; only its own outputs are withheld from the arm.
            arm.move_to(START_POSE, seconds=3.0)
            time.sleep(0.5)
            runner.reset()
            arm.command.reset(arm.read())
            rate = Rate(args.fps)
            started = time.perf_counter()
            latencies = []
            try:
                while time.perf_counter() - started < args.max_seconds:
                    state = arm.read()
                    frames = {key: cameras[CAMERA_OF_KEY[key]].latest() for key in CAMERA_SOURCES}
                    t0 = time.perf_counter()
                    target = runner.act(
                        state.unsqueeze(0),
                        {key: torch.from_numpy(frame).unsqueeze(0) for key, frame in frames.items()},
                    )[0].float().cpu()
                    latencies.append(time.perf_counter() - t0)
                    if args.dry_run:
                        sent = arm.command(target)
                        if len(latencies) % 15 == 1:
                            print(f"  t={time.perf_counter() - started:5.1f}s state {state.numpy().round(3)} -> {sent.numpy().round(3)}")
                    else:
                        sent = arm.send(target)
                    recorder.append(state, sent, frames["observation.images.front"], frames["observation.images.wrist"],
                                    arm.last_raw_read, None if args.dry_run else arm.last_raw_sent)
                    rate.sleep()
            except KeyboardInterrupt:
                print("\n[info] interrupted; holding")
                if not args.dry_run:
                    arm.send(arm.read())
                recorder.save(success=False, interrupted=True)
                raise
            lat = torch.tensor(latencies)
            print(f"[info] policy latency mean {lat.mean() * 1e3:.1f} ms, max {lat.max() * 1e3:.1f} ms; "
                  f"loop overruns {rate.overruns}")
            ok = input("success? [y/N]: ").strip().lower() == "y"
            print(f"[info] saved {recorder.save(success=ok)}")
    except KeyboardInterrupt:
        pass
    finally:
        for camera in cameras.values():
            camera.close()
        if arm.connected:
            # Fold to home before releasing torque (also after --dry-run, which moved the arm).
            arm.go_home()
            arm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
