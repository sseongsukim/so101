# so101

SO-101 robot assets and real-robot interface helpers for Isaac Lab experiments.

This repository intentionally does not copy the workshop task/environment. It
keeps only the parts that are useful when building a custom Isaac Lab scene:

- `so101.assets.SO101_CFG`: Isaac Lab `ArticulationCfg` using the SO-101 USD.
  The printed robot body is set to black by default to match the local real
  robot.
- `so101.assets.SO101_NO_CAMERA_CFG`: the same robot without the camera mesh.
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

Use `SO101_NO_CAMERA_CFG` if your environment should not include the camera USD.
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

### Cameras

The visual tabletop scene variant includes two adjustable 640x480 RGB-D cameras:

- `wrist_camera`, attached under `Robot/gripper/gripper_cam`
- `external_camera`, fixed above and to the side of the table

Their default poses are grouped near the top of
`src/so101/scenes/tabletop.py`. They can also be changed per environment
without editing the shared defaults:

```python
env_cfg.scene.wrist_camera.offset.pos = (x, y, z)
env_cfg.scene.wrist_camera.offset.rot = (w, x, y, z)
env_cfg.scene.external_camera.offset.pos = (x, y, z)
env_cfg.scene.external_camera.offset.rot = (w, x, y, z)
```

State environments do not spawn either camera. Visual environments spawn both
and expose normalized RGB images in the policy observation.

This checkout is Isaac Lab v2.2.0 and expects Isaac Sim 5.0.0. Camera render
products may fail during startup when it is run against an incompatible Isaac
Sim package version; keep those versions aligned for visual environments.

### Scaled contact-task environments

Three Gymnasium environments build directly on the SO-101 tabletop scene and
load only the required Isaac Factory assets. PegInsert and NutThread use the
original Factory asset scale. GearMesh alone uses 75% scale, with its masses
scaled by `0.75 ** 3`.

```bash
python scripts/view_task.py --task so101-PegInsert-v0
python scripts/view_task.py --task so101-GearMesh-v0
python scripts/view_task.py --task so101-NutThread-v0
```

Visual variants use the same task geometry and additionally return two camera
images plus robot proprioception:

```bash
python scripts/view_task.py --task so101-visual-PegInsert-v0
python scripts/view_task.py --task so101-visual-GearMesh-v0
python scripts/view_task.py --task so101-visual-NutThread-v0
```

Their policy observation is a dictionary:

```python
{
    "proprio": float_tensor,       # (num_envs, 12), joint position + velocity
    "rgb_wrist": float_tensor,     # (num_envs, 480, 640, 3), range [0, 1]
    "rgb_external": float_tensor,  # (num_envs, 480, 640, 3), range [0, 1]
}
```

The environment action is a normalized six-dimensional SO-101 joint-position
offset. The current implementation supplies task geometry, physics, cameras,
joint observations/actions, default-pose reset behavior, and a basic
asset-proximity reward. SO-101-specific in-gripper reset initialization and
final insertion/meshing/threading reward shaping are the next layer; the Franka
reset cannot be reused because its fingertip frame, gripper, and 7-DoF IK differ.

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
