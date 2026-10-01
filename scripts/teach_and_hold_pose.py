"""Hand-guide the SO-101 to a pose, save it, and hold that exact pose."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from so101.real.interface import LeRobotSO101Interface  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--robot-id", default="my_follower")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/taught_poses/wrist_intrinsics.json",
    )
    args = parser.parse_args()

    interface = LeRobotSO101Interface(
        device="cpu",
        port=args.port,
        id=args.robot_id,
        cameras={},
        fps=30,
        kind="follower",
    )
    interface.init_device()
    calibration_path = interface.robot.calibration_fpath
    if not calibration_path.is_file():
        raise SystemExit(f"motor calibration not found: {calibration_path}")

    print("Motor calibration:", calibration_path)
    print("Connecting does not command a new pose.")
    interface.connect()
    holding = False
    try:
        print("\nSupport the arm with your hand: torque will now be released.")
        interface.robot.bus.disable_torque()
        print("Torque OFF. Move the arm by hand to the desired camera pose.")
        answer = input("Press ENTER or type 's' then ENTER to save and hold: ").strip().lower()
        if answer not in ("", "s"):
            print("Aborted. Support the arm before it is disconnected.")
            input("Press ENTER when you are supporting the arm: ")
            return 1

        observation = interface.robot.get_observation()
        pose = {
            key: float(value)
            for key, value in observation.items()
            if key.endswith(".pos")
        }
        if not pose:
            raise RuntimeError("robot returned no joint positions")

        # Set the current position as the goal while torque is still off.  This
        # prevents enabling torque from recalling a stale goal and jumping.
        interface.robot.send_action(pose)
        interface.robot.bus.enable_torque()
        interface.robot.send_action(pose)
        holding = True

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "robot_id": args.robot_id,
                    "port": args.port,
                    "saved_at": datetime.now().astimezone().isoformat(),
                    "positions": pose,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("\nTorque ON. Holding the taught pose.")
        print("Saved:", args.output)
        print(json.dumps(pose, indent=2))
        print("Run wrist intrinsic calibration in another terminal now.")
        input("When calibration is finished, press ENTER here: ")

        print("Support the arm before releasing torque.")
        while input("Type 'release' to disable torque and exit: ").strip().lower() != "release":
            print("Still holding.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Torque is still holding if it had been enabled.")
        if holding:
            print("Support the arm before release.")
            try:
                while input("Type 'release' to disable torque and exit: ").strip().lower() != "release":
                    print("Still holding.")
            except (KeyboardInterrupt, EOFError):
                print("Cannot confirm support; leaving connection open is impossible on exit.")
        return 130
    finally:
        interface.robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
