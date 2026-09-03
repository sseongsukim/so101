"""Reference: https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/devices/keyboard.py"""

import numpy as np
from queue import Empty, SimpleQueue
from pynput.keyboard import Key, Listener


from so101.real.collect_enum import CollectEnum
import so101.utils.transform as T


class KeyboardInterface(object):
    """Define keyboard interface to control franka."""

    POSE_ACTIONS = ["s", "w", "a", "d", "e", "q"]
    GRIP_ACTIONS = ["z"]
    ROT_ACTIONS = ["i", "k", "j", "l", "u", "o"]
    REWARD_ACTIONS = ["x", "c"]

    # Only these actions are exposed to gym environment.
    ACTIONS = POSE_ACTIONS + GRIP_ACTIONS + ROT_ACTIONS + REWARD_ACTIONS

    ADJUST_DELTA = ["[", "]"]

    INIT_POS_DELTA = 0.02
    INIT_ROT_DELTA = 0.2  # Radian.

    MAX_POS_DELTA = 0.1
    MAX_ROT_DELTA = 0.2  # Radian.

    intr_z_limit = (-15, 17)

    def __init__(self):
        self._command_queue: SimpleQueue[CollectEnum] = SimpleQueue()
        self.reset()

        # Make a thread to listen to keyboard and register callback functions.
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def reset(self):
        self.pos_delta = KeyboardInterface.INIT_POS_DELTA
        self.rot_delta = KeyboardInterface.INIT_ROT_DELTA
        self.pos = np.zeros(3)  # (x, y, z)
        self.last_pos = self.pos.copy()
        self.ori = np.zeros(3)  # (Roll, Pitch, Yaw)
        self.last_ori = self.ori.copy()
        self.grasp = np.array([-1])
        self.reward = 0

        self.key_enum = CollectEnum.DONE_FALSE

    def _queue_command(self, command: CollectEnum) -> None:
        """Queue a command so the control loop cannot lose keyboard events."""
        self.key_enum = command
        self._command_queue.put(command)

    def on_press(self, k):
        try:
            k = k.char

            # Moving arm.
            if k in KeyboardInterface.ACTIONS:
                if k in KeyboardInterface.POSE_ACTIONS:
                    self._pose_action(k)
                elif k in KeyboardInterface.GRIP_ACTIONS:
                    self._grip_action(k)
                elif k in KeyboardInterface.ROT_ACTIONS:
                    self._rot_action(k)
                elif k in KeyboardInterface.REWARD_ACTIONS:
                    self._rew_action(k)

            # Data labelling and debugging.
            elif k == "t":
                self._queue_command(CollectEnum.SUCCESS)
            elif k.isdigit():
                self.rew_key = int(k)
                self._queue_command(CollectEnum.REWARD)
                print(f"[KEYBOARD] Reward pressed: {k}", flush=True)
            elif k == "y":
                self._queue_command(CollectEnum.SKILL)
                print("[KEYBOARD] Skill complete pressed", flush=True)
            elif k == "r":
                self._queue_command(CollectEnum.RESET)
                print("[KEYBOARD] Reset pressed", flush=True)
            elif k == "b":
                self._queue_command(CollectEnum.UNDO)
                print("[KEYBOARD] Undo pressed", flush=True)
        except AttributeError as e:
            pass

    def on_release(self, k):
        try:
            # Terminates keyboard monitoring.
            if k == Key.esc:
                return False
        except AttributeError as e:
            pass

    def _rew_action(self, k):
        if k == "x":
            self.reward += 1
        elif k == "c":
            self.reward -= 1

    def _pose_action(self, k):
        if k == "w":
            self.pos[0] -= self.pos_delta
        elif k == "s":
            self.pos[0] += self.pos_delta
        elif k == "a":
            self.pos[1] -= self.pos_delta
        elif k == "d":
            self.pos[1] += self.pos_delta
        elif k == "q":
            self.pos[2] -= self.pos_delta
        elif k == "e":
            self.pos[2] += self.pos_delta

    def _grip_action(self, k):
        if k == "z":
            self.grasp = -self.grasp

    def _rot_action(self, k):
        if k == "k":
            self.ori[1] += self.rot_delta
        elif k == "i":
            self.ori[1] -= self.rot_delta
        elif k == "j":
            self.ori[0] += self.rot_delta
        elif k == "l":
            self.ori[0] -= self.rot_delta
        elif k == "o":
            self.ori[2] -= self.rot_delta
        elif k == "u":
            self.ori[2] += self.rot_delta

    @property
    def rot_fraction(self):
        return (self.ori[2] - self.intr_z_limit[0]) / (
            self.intr_z_limit[1] - self.intr_z_limit[0]
        )

    def _adjust_delta(self, k):
        if k == "]":
            # Use larger step size of movement.
            self.pos_delta += 0.001
            self.rot_delta += 0.05
        elif k == "[":
            # Use smaller step size of movement.
            self.pos_delta -= 0.001
            self.rot_delta -= 0.05
            # Prevent becomming negative value.
            self.pos_delta = min(self.pos_delta, KeyboardInterface.MAX_POS_DELTA)
            self.rot_delta = min(self.rot_delta, KeyboardInterface.MAX_ROT_DELTA)
        print(
            "pose delta: {:.3f}, rotation delta: {:.3f}".format(
                self.pos_delta, self.rot_delta
            ),
            flush=True,
        )

    def get_action(self, use_quat=True):
        dpos = self.pos - self.last_pos
        dori = self.ori - self.last_ori

        self.last_pos = self.pos.copy()
        self.last_ori = self.ori.copy()

        if use_quat:
            dquat = T.mat2quat(T.euler2mat(dori))
            # Use positive element for the first element of quaternion (ease of learning).
            action = np.concatenate([dpos, dquat, self.grasp])
        else:
            action = np.concatenate([dpos, dori, self.grasp])
        try:
            command = self._command_queue.get_nowait()
        except Empty:
            command = CollectEnum.DONE_FALSE
        self.key_enum = CollectEnum.DONE_FALSE
        return action, command, self.reward

    def print_usage(self):
        print("==============Keyboard Usage=================")
        print("Positional movements in base frame")
        print("q (- z-axis) w (- x-axis) e (+ z-axis)")
        print("a (- y-axis) s (+ x-axis) d (+ y-axis)")

        print("Rotational movements in base frame")
        print("u (- z-axis-rot) i (neg y-axis-rot)  o (+ z-axis)")
        print("j (pos x-axis-rot) k (pos y-axis-rot)  l (neg x-axis-rot)")
        print("Data collection")
        print("t (save successful trajectory)  r (discard trajectory and reset)")

        print("Toggle gripper open and close")
        print("z")

        print("Save toggle: t")
        print("Save subtask toggle: u")
        print("Breakdown toggle: u")
        print("Reset toggle: r")
        print("===============================")

    def close(self):
        self.listener.stop()
