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
  stop             q / Ctrl+C -> hold, record interruption, then eased home

Keys (no Enter): r = return to start and hold; s = begin the prepared episode;
y/n = hold and save success/failure (also while running); q = quit.
r during an episode keeps an interrupted record and prepares the next attempt.
At --max-seconds the arm holds and waits for y/n. Default key mode reads the
focused terminal; --keys desktop uses the desktop's pynput listener.

Every episode is recorded (same trajectory_*.pkl format as the demos) under
--record, so failures can be replayed and inspected.

Example:
    python scripts/deploy_policy_real.py --policy dp --checkpoint outputs/dp_train/v1/dp_final.pt --dry-run
    python scripts/deploy_policy_real.py --policy dp --checkpoint outputs/dp_train/v1/dp_final.pt --episodes 10
"""

from __future__ import annotations

import argparse
import math
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
from so101.real.evaluation_controls import EvaluationLog, SingleKeyInput  # noqa: E402

CAMERA_OF_KEY = {"observation.images.front": "front", "observation.images.wrist": "wrist"}


class EvaluationQuit(Exception):
    pass


def wait_key(keys, accepted):
    while True:
        key = keys.get()
        if key == "q":
            raise EvaluationQuit()
        if key is not None and key in accepted:
            return key
        time.sleep(0.01)


def move_to_start(arm, keys, seconds=3.0):
    """Existing cosine-eased start move with q checked between motor commands."""
    start = arm.read()
    arm.command.reset(start)
    steps = max(int(seconds * 30), 1)
    for i in range(1, steps + 1):
        if keys.get() == "q":
            arm.send(arm.read())
            raise EvaluationQuit()
        alpha = 0.5 - 0.5 * math.cos(math.pi * i / steps)
        arm.send(start + (START_POSE - start) * alpha)
        time.sleep(1 / 30)


def prepare_episode(arm, keys, episode, total, reset_requested=False):
    if not reset_requested:
        print(f"[attempt {episode}; target {total} rated episodes] r=initial pose, q=quit (no Enter)")
        wait_key(keys, "r")
    while True:
        move_to_start(arm, keys)
        print("[ready] arm at initial pose; place cubes, then s=start; r=reset again, q=quit")
        if wait_key(keys, "rs") == "s":
            return


def run_evaluation(args, runner, arm, cameras, recorder, keys, results):
    reset_requested = False
    while results.summary()["rated_episodes"] < args.episodes:
        episode = results.summary()["attempts"]
        prepare_episode(arm, keys, episode + 1, args.episodes, reset_requested)
        reset_requested = False
        latencies = []
        started = None
        rate = None
        rollout_elapsed = 0.0

        def save(label, reason):
            n = len(recorder)
            path = recorder.save(
                success=label == "success", interrupted=label == "interrupted",
                evaluation_label=label, episode_index=episode + 1,
                evaluation_session=results.session_id, termination_reason=reason,
            )
            results.add(episode + 1, path, label, frames=n, termination_reason=reason,
                        elapsed_s=rollout_elapsed,
                        mean_policy_time_s=sum(latencies) / len(latencies) if latencies else None,
                        loop_overruns=rate.overruns if rate is not None else 0)
            summary = results.summary()
            print(f"[saved] {label}: {path}; successes {summary['successes']}/{summary['rated_episodes']}, "
                  f"interrupted {summary['interrupted']}")

        try:
            time.sleep(0.5)
            runner.reset()
            arm.command.reset(arm.read())
            rate = Rate(args.fps)
            started = time.perf_counter()
            label_key = None
            reason = "timeout"
            print("[running] y=success, n=failure (stop and save); r=reset attempt, q=quit")
            while time.perf_counter() - started < args.max_seconds:
                key = keys.get()
                if key == "q":
                    raise EvaluationQuit()
                if key == "r":
                    reset_requested, reason = True, "reset"
                    break
                if key in ("y", "n") and len(recorder):
                    label_key, reason = key, "operator_label"
                    break
                state = arm.read()
                frames = {key: cameras[CAMERA_OF_KEY[key]].latest() for key in CAMERA_SOURCES}
                t0 = time.perf_counter()
                target = runner.act(
                    state.unsqueeze(0),
                    {key: torch.from_numpy(frame).unsqueeze(0) for key, frame in frames.items()},
                )[0].float().cpu()
                latencies.append(time.perf_counter() - t0)
                # q pressed during inference prevents the next policy target from being sent.
                after_inference = keys.get()
                if after_inference == "q":
                    raise EvaluationQuit()
                if after_inference == "r":
                    reset_requested, reason = True, "reset"
                    break
                sent = arm.command(target) if args.dry_run else arm.send(target)
                recorder.append(state, sent, frames["observation.images.front"], frames["observation.images.wrist"],
                                arm.last_raw_read, None if args.dry_run else arm.last_raw_sent)
                if after_inference in ("y", "n"):
                    label_key, reason = after_inference, "operator_label"
                    break
                rate.sleep()
            rollout_elapsed = time.perf_counter() - started
            if not args.dry_run:
                arm.send(arm.read())
            if reset_requested:
                save("interrupted", "reset")
                continue
            if latencies:
                print(f"[info] policy latency mean {sum(latencies) / len(latencies) * 1e3:.1f} ms, "
                      f"max {max(latencies) * 1e3:.1f} ms; loop overruns {rate.overruns}")
            if label_key is None:
                print("[held] success? y=yes, n=no; r=reset unrated attempt; q=quit")
                label_key = wait_key(keys, "ryn")
                if label_key == "r":
                    save("interrupted", "reset")
                    reset_requested = True
                    continue
            save("success" if label_key == "y" else "failure", reason)
        except (EvaluationQuit, KeyboardInterrupt, EOFError) as error:
            rollout_elapsed = time.perf_counter() - started if started is not None else 0.0
            if not args.dry_run:
                arm.send(arm.read())
            save("interrupted", "q" if isinstance(error, EvaluationQuit) else type(error).__name__)
            raise
        except Exception:
            save("interrupted", "error")
            raise


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
    parser.add_argument("--episodes", type=int, default=1, help="target y/n-rated episodes; reset/quit interruptions logged separately")
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
    parser.add_argument("--keys", choices=["terminal", "desktop"], default="terminal",
                        help="terminal: focus terminal, single keys without Enter; desktop: pynput global keys (X11)")
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_seconds <= 0 or args.fps <= 0:
        parser.error("episodes, max-seconds and fps must be positive")

    # Fail before connecting to hardware if the key backend cannot be opened.
    keys = SingleKeyInput(args.keys)
    arm = None
    cameras: dict[str, CameraThread] = {}
    results = None
    try:
        runner = make_runner(args.policy, args.checkpoint, args.device, args.action_steps,
                             args.inference_steps if args.policy == "dp" else None)
        arm = FollowerArm(args.port, args.robot_id, SafetyLimits(max_step_rad=args.max_step_rad))
        deployment_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        directory = args.record / args.checkpoint.parent.name
        results = EvaluationLog(directory, args.checkpoint, deployment_args)
        recorder = EpisodeRecorder(directory, source=f"real_rollout_{args.policy}",
                                   checkpoint=str(args.checkpoint.resolve()), dry_run=args.dry_run,
                                   deployment_args=deployment_args)
        arm.connect()
        move_to_start(arm, keys)
        # Meter exposures at the initial pose as in the existing deployment.
        for name in ("front", "wrist"):
            cameras[name] = CameraThread(name, None if args.no_meter else METER_TARGETS[name])
            print(f"[info] {name} exposure: {cameras[name].camera.exposure_lock.as_dict()}")
        print(f"[info] result log: {results.path}")
        print(f"[info] key mode: {args.keys}; no Enter required")
        run_evaluation(args, runner, arm, cameras, recorder, keys, results)
    except (EvaluationQuit, KeyboardInterrupt, EOFError):
        print("\n[info] stopping evaluation; returning home")
    finally:
        keys.stop()
        for camera in cameras.values():
            camera.close()
        if arm is not None and arm.connected:
            try:
                arm.go_home()
            finally:
                arm.close()
        if results is not None:
            summary = results.summary()
            print(f"[summary] successes {summary['successes']}/{summary['rated_episodes']}; "
                  f"failures {summary['failures']}; interrupted {summary['interrupted']}")
            if summary["attempts"]:
                print(f"[summary] {results.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
