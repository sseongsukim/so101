"""A live preview window with key events that works with headless OpenCV.

LeRobot pins opencv-python-headless, which has no namedWindow/imshow/waitKey.
`PreviewWindow` uses OpenCV's GUI when this build has one and otherwise a
matplotlib (TkAgg) figure, behind the same two calls:

    window = PreviewWindow("title")
    window.show(bgr_frame)        # update the picture
    key = window.poll_key()       # like cv2.waitKey(1) & 0xFF; -1 if none

Keys come back as cv2.waitKey codes: printable characters as ord(c), Enter
13, Space 32, Esc 27. The matplotlib window must have focus to receive keys.
"""

from __future__ import annotations

import os
from collections import deque

import numpy as np

_SPECIAL_KEYS = {"enter": 13, "return": 13, " ": 32, "space": 32, "escape": 27}


def opencv_has_gui() -> bool:
    import cv2

    try:
        cv2.namedWindow("__gui_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__gui_probe__")
        return True
    except cv2.error:
        return False


def preview_available() -> bool:
    return opencv_has_gui() or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class PreviewWindow:
    def __init__(self, title: str, size: tuple[int, int] = (960, 720)):
        self.title = title
        self.use_cv2 = opencv_has_gui()
        if self.use_cv2:
            import cv2

            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, *size)
            return
        import matplotlib

        if matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "ps", "cairo", "template"):
            matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        for key in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[key] = []  # keep 's', 'q', 'f', ... from triggering matplotlib actions
        self._plt = plt
        self._keys: deque[int] = deque()
        self._closed = False
        self.fig, self.ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
        self.fig.canvas.manager.set_window_title(title)
        self.ax.axis("off")
        self.fig.subplots_adjust(0, 0, 1, 1)
        self._image = None
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        plt.ion()
        plt.show(block=False)

    def _on_key(self, event) -> None:
        if event.key is None:
            return
        key = event.key.lower()
        if key in _SPECIAL_KEYS:
            self._keys.append(_SPECIAL_KEYS[key])
        elif len(event.key) == 1:
            self._keys.append(ord(event.key))

    def _on_close(self, _event) -> None:
        self._closed = True
        self._keys.append(27)  # closing the window finishes, like Esc

    def show(self, bgr: np.ndarray) -> None:
        if self.use_cv2:
            import cv2

            cv2.imshow(self.title, bgr)
            return
        if self._closed:
            return
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        if self._image is None:
            self._image = self.ax.imshow(rgb)
        else:
            self._image.set_data(rgb)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def poll_key(self) -> int:
        if self.use_cv2:
            import cv2

            key = cv2.waitKey(1)
            return -1 if key < 0 else key & 0xFF
        self._plt.pause(0.001)
        return self._keys.popleft() if self._keys else -1

    def close(self) -> None:
        if self.use_cv2:
            import cv2

            cv2.destroyWindow(self.title)
        elif not self._closed:
            self._plt.close(self.fig)
            self._closed = True
