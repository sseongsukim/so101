"""Compare how the real follower and the simulated arm track the same joint targets.

The student imitates teacher trajectories that were executed by Isaac's joint
PD drives. If the real STS3215 servos lag or overshoot very differently, the
same target sequence produces different motion and the policy is off-
distribution on the real arm. This measures that gap before any policy runs.

Three modes, sharing one command sequence (steps and sinusoids around the
StackCube start pose, one joint at a time, 30 Hz):

    --real    (on site) command the follower, record measured joints
    --sim     replay the sequence in Isaac (so101-StackCube-v0), record joints
    --compare real.npz sim.npz: per joint, tracking delay (cross-correlation),
              RMS tracking error, step overshoot and 90% rise time; writes a plot

Example:
    python scripts/sysid_joint_tracking.py --real --out outputs/sysid/real.npz          # on site
    python -u scripts/sysid_joint_tracking.py --sim --headless --out outputs/sysid/sim.npz
    python scripts/sysid_joint_tracking.py --compare outputs/sysid/real.npz outputs/sysid/sim.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402

from so101.real.constants import SO101_SIM_JOINT_ORDER, STACK_CUBE_DEFAULT_JOINT_POS  # noqa: E402

FPS = 30.0
START = np.array([STACK_CUBE_DEFAULT_JOINT_POS[j] for j in SO101_SIM_JOINT_ORDER])
# Excitation per joint (rad). Kept small around the start pose so the arm stays
# above the table; the jaw gets a larger open/close swing.
AMPLITUDE = np.array([0.25, 0.20, 0.20, 0.25, 0.30, 0.45])


def command_sequence(hold_s: float = 1.0) -> tuple[np.ndarray, list[dict]]:
    """(T, 6) targets and per-segment labels: for each joint, a +step, a
    -step back, then 0.5 Hz and 1.5 Hz sinusoids."""
    rows, segments = [], []
    hold = int(hold_s * FPS)

    def add(kind: str, joint: int, values: np.ndarray) -> None:
        segments.append({"kind": kind, "joint": joint, "start": len(rows), "length": len(values)})
        for v in values:
            q = START.copy()
            q[joint] = v
            rows.append(q)

    for j in range(6):
        base, amp = START[j], AMPLITUDE[j]
        add("rest", j, np.full(hold, base))
        add("step_up", j, np.full(hold, base + amp))
        add("step_down", j, np.full(hold, base))
        for freq in (0.5, 1.5):
            t = np.arange(int(2.0 / freq * FPS)) / FPS
            add(f"sine_{freq}", j, base + amp * np.sin(2 * np.pi * freq * t))
        add("rest", j, np.full(hold, base))
    return np.stack(rows).astype(np.float32), segments


def run_real(args) -> None:
    import torch

    torch.set_num_threads(1)  # see record_real_demos.py: keeps the 30 Hz loop from stalling

    from so101.real.follower import FollowerArm, Rate, SafetyLimits

    targets, segments = command_sequence()
    print(f"[info] {len(targets)} steps ({len(targets) / FPS:.0f} s), amplitudes {AMPLITUDE.tolist()} rad")
    if not args.yes and input("The follower will move around the start pose. Type MOVE: ").strip() != "MOVE":
        print("aborted")
        return
    # Rate limit above the sequence's own speed so it never alters the command.
    arm = FollowerArm(args.port, args.robot_id, SafetyLimits(max_step_rad=0.2),
                      p_gain=parse_p_gain(args.p_gain) if args.p_gain else None)
    measured, sent, stamps, raw_measured, raw_sent = [], [], [], [], []
    try:
        arm.connect()
        print(f"[info] servo P gains: {arm.read_gains()}")
        arm.move_to(torch.from_numpy(START).float(), seconds=4.0)
        time.sleep(1.0)
        rate = Rate(FPS)
        for q in targets:
            measured.append(arm.read().numpy())
            raw_measured.append(arm.last_raw_read.numpy())
            sent.append(arm.send(torch.from_numpy(q)).numpy())
            raw_sent.append(arm.last_raw_sent.numpy())
            stamps.append(time.perf_counter())
            rate.sleep()
        print(f"[info] loop overruns: {rate.overruns}")
        arm.move_to(torch.from_numpy(START).float(), seconds=2.0)
    finally:
        arm.close()
    save(args.out, targets, np.stack(sent), np.stack(measured), segments, "real", np.asarray(stamps),
         raw_sent=np.stack(raw_sent), raw_measured=np.stack(raw_measured))


def parse_p_gain(spec: str | None) -> dict[str, int]:
    """"32" -> every motor 32; "elbow_flex=32,shoulder_lift=24" -> those motors."""
    motors = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    if not spec:
        return {}
    if "=" not in spec:
        return {m: int(spec) for m in motors}
    out = {}
    for item in spec.split(","):
        name, value = item.split("=")
        if name.strip() not in motors:
            raise SystemExit(f"unknown motor {name!r}; expected one of {motors}")
        out[name.strip()] = int(value)
    return out


def run_sim(args) -> None:
    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app
    try:
        import gymnasium as gym
        import torch

        import so101.tasks  # noqa: F401
        from so101.configs import make_env_cfg

        cfg = make_env_cfg("so101-StackCube-v0", num_envs=1, device=args.device)
        cfg.terminate_on_success = False
        cfg.truncate_on_timeout = False
        env = gym.make("so101-StackCube-v0", cfg=cfg).unwrapped
        targets, segments = command_sequence()
        env.reset()
        start = torch.from_numpy(START).float().to(env.device).unsqueeze(0)
        for _ in range(60):
            env.step(start)
        measured = []
        with torch.inference_mode():
            for q in targets:
                # Measured before the command, as in the real loop.
                measured.append(env.robot.data.joint_pos[0].cpu().numpy().copy())
                env.step(torch.from_numpy(q).to(env.device).unsqueeze(0))
        env.close()
        save(args.out, targets, targets, np.stack(measured), segments, "sim", None)
    finally:
        app.close()


def save(path: Path, targets, sent, measured, segments, source, stamps, **raw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {} if stamps is None else {"stamps": stamps}
    extra.update(raw)
    np.savez(path, targets=targets, sent=sent, measured=measured,
             segments=json.dumps(segments), source=source, **extra)
    print(f"[info] wrote {path}")


def analyse(data) -> dict:
    sent, measured = data["sent"], data["measured"]
    segments = json.loads(str(data["segments"]))
    result = {}
    for j, name in enumerate(SO101_SIM_JOINT_ORDER):
        segs = [s for s in segments if s["joint"] == j]
        lo = segs[0]["start"]
        hi = segs[-1]["start"] + segs[-1]["length"]
        cmd, meas = sent[lo:hi, j], measured[lo:hi, j]
        # measured[k] is read before command k is sent, so the best match of
        # measured[k + d] to cmd[k] is the delay d in steps.
        lags = range(0, 16)
        errors = [np.sqrt(np.mean((meas[d:] - cmd[: len(cmd) - d]) ** 2)) for d in lags]
        delay = int(np.argmin(errors))
        up = next(s for s in segs if s["kind"] == "step_up")
        s0, n = up["start"], up["length"]
        before, after = measured[s0 - 1, j], measured[s0 : s0 + n, j]
        target = sent[s0, j]
        span = target - before
        progress = (after - before) / span if abs(span) > 1e-6 else np.zeros_like(after)
        rise = next((k for k, p in enumerate(progress) if p >= 0.9), None)
        result[name] = {
            "delay_steps": delay,
            "rms_error_rad": float(np.sqrt(np.mean((meas - cmd) ** 2))),
            "rms_error_after_delay_rad": float(errors[delay]),
            "step_overshoot_frac": float(max(progress.max() - 1.0, 0.0)),
            "step_rise90_steps": rise,
            "step_steady_error_rad": float(abs(after[-5:].mean() - target)),
        }
    return result


def compare(real_path: Path, sim_path: Path, plot: Path, robot_id: str = "my_follower") -> None:
    real, sim = dict(np.load(real_path)), dict(np.load(sim_path))
    if "raw_measured" in real:
        # Re-express the real run in the follower mapping in force now, so a
        # recording made before the mapping was refitted stays comparable.
        from so101.real.joint_mapping import follower_mapping

        mapping = follower_mapping(robot_id)
        real["measured"] = mapping.to_sim(real["raw_measured"].astype(np.float64))
        real["sent"] = mapping.to_sim(real["raw_sent"].astype(np.float64))
    a, b = analyse(real), analyse(sim)
    keys = ["delay_steps", "step_rise90_steps", "step_overshoot_frac", "step_steady_error_rad", "rms_error_after_delay_rad"]
    print(f"{'joint':12s} " + "  ".join(f"{k:>26s}" for k in keys))
    for name in SO101_SIM_JOINT_ORDER:
        cells = []
        for k in keys:
            ra, sb = a[name][k], b[name][k]
            fmt = (lambda v: "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v))
            cells.append(f"{'real ' + fmt(ra) + ' / sim ' + fmt(sb):>26s}")
        print(f"{name:12s} " + "  ".join(cells))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
    t = np.arange(len(real["sent"])) / FPS
    for j, ax in enumerate(axes):
        ax.plot(t, real["sent"][:, j], color="0.6", lw=1, label="command")
        ax.plot(t, real["measured"][:, j], lw=1.2, label="real")
        ax.plot(t[: len(sim["measured"])], sim["measured"][:, j], lw=1.2, label="sim")
        ax.set_ylabel(SO101_SIM_JOINT_ORDER[j])
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("s")
    fig.tight_layout()
    plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot, dpi=110)
    print(f"[info] wrote {plot}")
    (plot.with_suffix(".json")).write_text(json.dumps({"real": a, "sim": b}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--real", action="store_true")
    mode.add_argument("--sim", action="store_true")
    mode.add_argument("--compare", nargs=2, type=Path, metavar=("REAL", "SIM"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=REPO_ROOT / "outputs/sysid/compare.png")
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--robot-id", default="my_follower")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--p-gain", default=None,
                        help='--real: servo P_Coefficient, "16" for all motors or "elbow_flex=32,..." (default: follower.DEFAULT_P_GAIN, 32; LeRobot sets 16)')
    if "--sim" in sys.argv:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.real:
        run_real(args) if args.out else parser.error("--out is required")
    elif args.sim:
        run_sim(args) if args.out else parser.error("--out is required")
    else:
        compare(*args.compare, args.plot, args.robot_id)


if __name__ == "__main__":
    main()
