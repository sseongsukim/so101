"""Re-express recorded real episodes in the follower joint mapping in force now.

Real demos and rollouts store the follower's raw LeRobot values
(`raw_observations`, `raw_actions`) next to the Isaac-radian `observations` /
`actions` computed at recording time. When the follower mapping is refitted
(fit_follower_joint_mapping.py --write), the raw values are the ground truth
and the radians are recomputed from them -- so real data can be recorded
before the final mapping exists.

Writes the remapped episodes to --out (never in place); episodes without raw
values are reported and skipped.

Example:
    python scripts/remap_real_episodes.py --src outputs/real_demos/v1 --out outputs/real_demos/v1_remapped
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from so101.real.joint_mapping import follower_mapping  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--robot-id", default="my_follower")
    parser.add_argument("--mapping", type=Path, default=None, help="default: calibration/joint_mapping/follower.yaml")
    args = parser.parse_args()
    if args.out.resolve() == args.src.resolve():
        parser.error("--out must differ from --src")

    mapping = follower_mapping(args.robot_id, mapping_path=args.mapping)
    args.out.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for path in sorted(args.src.glob("trajectory_*.pkl")):
        with path.open("rb") as f:
            data = pickle.load(f)
        if "raw_observations" not in data or "raw_actions" not in data:
            print(f"skip {path.name}: no raw values")
            skipped += 1
            continue
        obs = mapping.to_sim(data["raw_observations"].astype(np.float64)).astype(np.float32)
        act = mapping.to_sim(data["raw_actions"].astype(np.float64)).astype(np.float32)
        shift = np.abs(obs - data["observations"]).max()
        data.update(observations=obs, actions=act, next_observations=np.concatenate([obs[1:], obs[-1:]]),
                    follower_mapping=mapping.kind, remapped_from=str(path))
        with (args.out / path.name).open("xb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{path.name}: max |new - old| joint {np.rad2deg(shift):.2f} deg")
        done += 1
    print(f"remapped {done}, skipped {skipped} -> {args.out}")
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
