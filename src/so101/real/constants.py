"""Shared joint naming and range constants for SO-101."""

SO101_USD_MAPPING = {
    "shoulder_pan": {"joint_min": -110.0, "joint_max": 110.0},
    "shoulder_lift": {"joint_min": -100.0, "joint_max": 100.0},
    "elbow_flex": {"joint_min": -100.0, "joint_max": 90.0},
    "wrist_flex": {"joint_min": -95.0, "joint_max": 95.0},
    "wrist_roll": {"joint_min": -160.0, "joint_max": 160.0},
    "gripper": {"joint_min": -10.0, "joint_max": 100.0},
}

SO101_JOINT_ORDER = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

SO101_SIM_JOINT_ORDER = [
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
]

SO101_INITIAL_POSE = {
    "shoulder_pan.pos": -6.3602,
    "shoulder_lift.pos": -51.9492,
    "elbow_flex.pos": 16.7276,
    "wrist_flex.pos": 89.2483,
    "wrist_roll.pos": -51.8499,
    "gripper.pos": 0.0,
}

SO101_HOME_POSE = {
    "shoulder_pan.pos": -6.2835,
    "shoulder_lift.pos": -91.4407,
    "elbow_flex.pos": 93.1444,
    "wrist_flex.pos": 69.1434,
    "wrist_roll.pos": -51.9027,
    "gripper.pos": 0.0707,
}

# Reset pose of the StackCube tasks, in Isaac joint names and radians
# (order: SO101_SIM_JOINT_ORDER).
STACK_CUBE_DEFAULT_JOINT_POS = {
    "Rotation": -0.0685,
    "Pitch": -1.3674,
    "Elbow": 1.3919,
    "Wrist_Pitch": 1.0408,
    "Wrist_Roll": -0.0211,
    "Jaw": 0.0808,
}
