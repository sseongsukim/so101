"""Single-key operator controls and per-session real evaluation results."""

from __future__ import annotations

import json
import os
import queue
import select
import sys
import termios
import tty
from datetime import datetime
from pathlib import Path


class SingleKeyInput:
    """r/s/y/n/q without Enter; terminal mode also works over SSH and Wayland."""

    def __init__(self, mode="terminal", stream=None):
        self.mode = mode
        self.stream = sys.stdin if stream is None else stream
        self.fd = None
        self.original = None
        self.listener = None
        self.queue = queue.SimpleQueue()
        self.pressed = set()
        if mode == "terminal" and not self.stream.isatty():
            raise RuntimeError("hotkeys require an interactive terminal; run without stdin redirection")
        try:
            if self.stream.isatty():
                self.fd = self.stream.fileno()
                self.original = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
            if mode == "desktop":
                from pynput.keyboard import Listener

                self.listener = Listener(on_press=self._press, on_release=self._release)
                self.listener.start()
            elif mode != "terminal":
                raise ValueError(f"unknown key mode: {mode}")
        except BaseException:
            self.stop()
            raise

    def _press(self, key):
        char = (getattr(key, "char", None) or "").lower()
        if char in ("r", "s", "y", "n", "q") and char not in self.pressed:
            self.pressed.add(char)
            self.queue.put(char)

    def _release(self, key):
        self.pressed.discard((getattr(key, "char", None) or "").lower())

    def get(self):
        if self.mode == "desktop":
            try:
                return self.queue.get_nowait()
            except queue.Empty:
                return None
        if not select.select([self.fd], [], [], 0)[0]:
            return None
        char = os.read(self.fd, 1).decode("utf-8", errors="ignore").lower()
        if not char:
            raise EOFError("operator terminal closed")
        return char if char in ("r", "s", "y", "n", "q") else None

    def stop(self):
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        if self.original is not None:
            try:
                termios.tcflush(self.fd, termios.TCIFLUSH)
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original)
            finally:
                self.original = None


class EvaluationLog:
    """Keep interrupted attempts distinct from operator-labelled failures."""

    def __init__(self, directory, checkpoint, deployment_args):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.path = self.directory / f"evaluation_{self.session_id}.jsonl"
        self.summary_path = self.directory / f"evaluation_{self.session_id}_summary.json"
        self.metadata = {"session_id": self.session_id, "checkpoint": str(Path(checkpoint).resolve()),
                         "deployment_args": deployment_args}
        self.results = []

    def add(self, episode_index, trajectory, label, **details):
        if label not in ("success", "failure", "interrupted"):
            raise ValueError(f"invalid evaluation label: {label}")
        row = {**self.metadata, "episode_index": episode_index,
               "recorded_at": datetime.now().astimezone().isoformat(),
               "trajectory": str(trajectory.resolve()) if trajectory is not None else None,
               "label": label, "success": {"success": True, "failure": False, "interrupted": None}[label],
               **details}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
        self.results.append(row)
        self.summary_path.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")

    def summary(self):
        successes = sum(row["label"] == "success" for row in self.results)
        failures = sum(row["label"] == "failure" for row in self.results)
        rated = successes + failures
        return {**self.metadata, "attempts": len(self.results), "rated_episodes": rated,
                "successes": successes, "failures": failures,
                "interrupted": len(self.results) - rated,
                "success_rate": successes / rated if rated else None,
                "results": self.results}
