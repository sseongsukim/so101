"""Show, and optionally release, the torque state of the SO-101 arms.

Reads Torque_Enable / Present_Position / Present_Load straight from the
servos (no LeRobot robot object, so nothing is enabled or moved by the
check). A script that dies without its cleanup (closed terminal, kill -9)
leaves the follower holding its last pose with torque on; this is the quick
way to see and undo that.

    python scripts/release_torque.py            # report both arms
    python scripts/release_torque.py --release  # also disable follower torque

Releasing lets the arm fall under gravity: support it first unless it is
folded in its home pose (the script says which, from shoulder_lift/elbow).
"""

from __future__ import annotations

import argparse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def open_bus(port: str) -> FeetechMotorsBus:
    bus = FeetechMotorsBus(port=port, motors={n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100)
                                              for i, n in enumerate(NAMES)})
    bus.connect(handshake=True)
    return bus


def report(bus: FeetechMotorsBus, who: str) -> dict:
    torque = {n: bus.read("Torque_Enable", n, normalize=False) for n in NAMES}
    position = {n: bus.read("Present_Position", n, normalize=False) for n in NAMES}
    load = {n: bus.read("Present_Load", n, normalize=False) for n in NAMES}
    on = [n for n, t in torque.items() if t]
    print(f"{who}: torque {'ON for ' + ', '.join(on) if on else 'off'}")
    print(f"  position (ticks) {position}")
    print(f"  load             {load}")
    return position


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--follower-port", default="/dev/so101-follower")
    parser.add_argument("--leader-port", default="/dev/so101-leader")
    parser.add_argument("--release", action="store_true", help="disable follower torque")
    parser.add_argument("--yes", action="store_true", help="release without asking, even if the arm is raised")
    args = parser.parse_args()

    for port, who in ((args.leader_port, "leader"), (args.follower_port, "follower")):
        try:
            bus = open_bus(port)
        except Exception as error:  # noqa: BLE001 - surface the reason, keep going
            print(f"{who}: cannot read ({type(error).__name__}: {str(error).splitlines()[0]})")
            continue
        try:
            position = report(bus, who)
            if who == "follower" and args.release:
                # Home pose (SO101_HOME_POSE) is shoulder_lift near its low end and
                # elbow near its high end; anywhere else the arm will drop.
                folded = position["shoulder_lift"] < 1000 and position["elbow_flex"] > 2900
                if not folded and not args.yes:
                    answer = input("  arm is NOT folded: support it, then type RELEASE: ").strip()
                    if answer != "RELEASE":
                        print("  not released")
                        continue
                bus.disable_torque(num_retry=5)
                print(f"  released -> torque {[bus.read('Torque_Enable', n, normalize=False) for n in NAMES]}")
        finally:
            bus.port_handler.closePort()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
