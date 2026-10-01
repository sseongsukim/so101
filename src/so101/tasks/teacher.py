"""The 36-D state the ResiP teacher was trained on, from any StackCube env.

The teacher was trained on origin/main's scene, where the robot root sits at
the environment origin, so its cube / grasp-point positions are effectively
robot-relative. This branch places the robot at the calibrated
ROBOT_ROOT_POS (translation only, no rotation) and spawns the cubes relative
to it, so subtracting the robot root recovers exactly the frame the teacher
knows. The visual env returns only encoder positions + images, so the state
is rebuilt from the simulator with the state task's own observation code.
"""

from __future__ import annotations

import torch

from so101.tasks.env import SO101TaskEnv

# Position slices of the 36-D state (cube_a, cube_b, grasp point). The
# cube_b - cube_a slice (14:17) is a difference and needs no shift.
POSITION_SLICES = (slice(4, 7), slice(11, 14), slice(17, 20))


def teacher_observation(env: SO101TaskEnv) -> torch.Tensor:
    state = SO101TaskEnv._get_observations(env)["state"].clone()
    robot_root = env.robot.data.root_pos_w - env.scene.env_origins
    for sl in POSITION_SLICES:
        state[:, sl] -= robot_root
    return state
