# so101

SO-101 robot assets and real-robot interface helpers for Isaac Lab experiments.

This repository intentionally does not copy the workshop task/environment. It
keeps only the parts that are useful when building a custom Isaac Lab scene:

- `so101.assets.SO101_CFG`: camera-free Isaac Lab `ArticulationCfg` using the
  SO-101 USD with a black printed body.
- `so101.real.interface.LeRobotSO101Interface`: LeRobot bridge utilities for
  mapping real SO-101 joint values to Isaac Lab radians and back.
- `so101.real.control.SO101Control`: a small real-robot control wrapper with
  initial/home poses and optional Rerun logging.
- calibration helper scripts for checking and summarizing SO-101 calibration
  files.

## Install

Use the Isaac Lab virtual environment named `isaaclab`.

```bash
conda activate isaaclab
cd /home/seongsu/workspace/research/so101
pip install -e .
```

Install the real-robot dependencies in the same environment when you need the
physical robot interface.

```bash
pip install -e ".[real]"
```

## Isaac Lab Usage

```python
from so101.assets import SO101_CFG

robot_cfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

Use `SO101_CONTACT_GRASP_CFG` when contact sensors are needed for grasp logic.

### Tabletop scene viewer

The package includes a minimal scene with the robot base at `(0, 0, 0)` and a
fixed 50 cm square tabletop extending along +X. The tabletop has collision but
no rigid body, so it remains fixed without legs or gravity settings. Its top
surface is at `z=0.030081 m`, matching the actual bottom of the robot base mesh.
The robot articulation root itself remains at `(0, 0, 0)`.
The robot uses true black while the tabletop uses a slightly lighter, rough
charcoal black so their silhouettes remain distinguishable under scene lighting.

```bash
conda activate isaaclab
cd /home/seongsu/workspace/research/so101
pip install -e .
python scripts/view_tabletop_scene.py
```

Close the Isaac Sim window to stop the script. Standard `AppLauncher` options
are available; for example, use `--device cpu` when a CUDA physics device is
not desired.

### StackCube environment

Environment configuration is defined explicitly under `src/so101/configs/`.
Scripts create configurations through the public helper instead of reading
private values from `gym.spec()`:

```python
from so101.configs import make_env_cfg

env_cfg = make_env_cfg("so101-StackCube-v0", num_envs=1, device="cuda:0")
env = gym.make("so101-StackCube-v0", cfg=env_cfg)
```

`configs/base.py` contains shared simulation, control, viewer, observation, and
reward parameters. `configs/tasks.py` contains the task-specific overrides,
and `configs/registry.py` maps every Gym task ID to its configuration class.
These use Isaac Lab's dataclass-compatible `@configclass` so nested scene and
simulation configurations retain `copy`, `replace`, and validation behavior.

The registered Gymnasium tasks are `so101-StackCube-v0` (state-only,
camera-free robot) and `so101-visual-StackCube-v0` (camera-equipped robot with
wrist and external RGB-D sensors). Both use exact Isaac Lab cuboid primitives:
a movable 2.5 cm cube and a movable 4 cm target cube.

StackCube randomizes both cube poses on every episode reset. It samples the
large cube first, then rejection-samples the small cube until their XY centers
are at least 6 cm apart. Both centers use `x=0.22..0.30 m` and
`y=-0.10..0.10 m`; each cube receives a random yaw while remaining flat on the
tabletop.

```bash
python scripts/view_task.py --task so101-StackCube-v0
python scripts/view_task.py --task so101-visual-StackCube-v0
```

StackCube instead returns the IsaacGym Franka cube-stack common state plus
SO-101 joint state: cube-A quaternion (4), cube-A position (3), cube-A to
cube-B position (3), end-effector position (3), end-effector quaternion (4),
joint position (6), and joint velocity (6), for 29 dimensions.

For StackCube, the EEF position is the midpoint of two explicit distal grasp
points derived from the SO-101 colliders: one fixed to `/Robot/gripper` and one
fixed to `/Robot/jaw`. The EEF quaternion remains the `/Robot/gripper`
orientation.

StackCube uses the IsaacGym Franka cube-stack reward instead: distance `0.1`,
lift `1.5`, alignment `2.0`, and exclusive stack-success reward `16.0`. Stack
success requires cube-center XY error below 1 cm, height error below 0.5 cm,
and the end effector to be more than 2 cm from the small cube. Its distance
gain is doubled to 20 and its lift-clearance threshold is reduced to 2 cm to
match cubes half the size of the Franka example. Success sets
`terminated=True`; the 350-step (approximately 11.67-second) time limit sets
`truncated=True` when success has not occurred.

### Leader-arm task teleoperation

A calibrated physical SO-101 leader arm can directly command StackCube's six
absolute simulation joint targets:

```bash
lerobot-calibrate --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 --teleop.id=leader_arm_1

python scripts/teleop_task.py so101-StackCube-v0 --print-every 1
```

The default control-rate cap is 30 Hz and diagnostics are printed every 30
steps. Each line reports total reward, reward phase, geometric and
lift-qualified success, filtered jaw-to-held contact force, lift height, and
reach/target distances. Set `--print-every 1` to inspect every environment
step, or `--rate 0` to disable wall-clock pacing. `TELEOP_PORT` and `TELEOP_ID`
can be used instead of the corresponding command-line options.

The environment action is a six-dimensional absolute SO-101 joint-position
target in radians, matching the Sim-to-Real SO-101 Workshop. The current
implementation supplies task geometry, physics, joint observations/actions,
default-pose reset behavior, and cube-stack rewards.

The workshop changes the robot color by editing the USD shader at
`Looks/material_a_3d_printed/Shader`. This package uses the same mechanism and
sets `SO101_CFG` to black by default. If you want reset-time color
randomization in your own environment:

```python
from isaaclab.managers import EventTerm
from so101.mdp.randomization import ROBOT_COLORS, randomize_robot_color, set_robot_color

reset_set_robot_visual_material = EventTerm(
    func=set_robot_color,
    mode="reset",
    params={"color": "black"},
)

reset_randomize_robot_visual_material = EventTerm(
    func=randomize_robot_color,
    mode="reset",
    params={"color_names": list(ROBOT_COLORS.keys())},
)
```

## Real Robot Usage

```bash
so101-control --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower_arm_1
so101-manual-control --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower_arm_1
```

Calibration helpers:

```bash
so101-calibration-stats
ROBOT_PORT=/dev/ttyACM0 ROBOT_ID=follower_arm_1 so101-check-calibration
```

## Source

The SO-101 USD, Isaac Lab robot config, and LeRobot/real-robot helper code were
adapted from `reference/Sim-to-Real-SO-101-Workshop`. Files copied from that
workshop keep their original Apache-2.0 SPDX headers.
