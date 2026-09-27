"""`scripts/teleop_task.py --visual` pickles -> ACT training samples.

Replaces the LeRobotDataset bridge from so101-ros: the teleop script already
records a fixed-rate (one sample per 30 Hz control step) image/state/action
sequence per file, so the trajectories are read directly instead of being
converted to another on-disk format first.

One `trajectory_*.pkl` is one episode. Per step t it holds

    observations[t]   (6,) simulated joint positions (rad)  -> observation.state
    actions[t]        (6,) absolute joint targets (rad)     -> action
    front_images[t]   (H, W, 3) uint8 RGB                   -> observation.images.front
    wrist_images[t]   (H, W, 3) uint8 RGB                   -> observation.images.wrist

and a sample at frame t is the step-t observation plus the action chunk
`actions[t : t + chunk_size]`. Entries that run past the episode end repeat the
last action and are flagged in `action_is_pad`, which is what LeRobot's
`delta_timestamps` query produced and what `ACTPolicy.forward` masks out of
the loss.

Images are returned CHW float32 in [0, 1], the same as LeRobot returned them;
`to_uint8_rgb` is the one resize path shared with the teleop recorder, so a
policy fed live simulator frames sees exactly what it was trained on.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.utils.data import Dataset

from so101.learning.act.config import ACTION, OBS_STATE

FPS = 30
# State/action dimensions whose std falls below this are left unscaled (std
# 1.0), the same rule as utils/datasets.Normalizer. Otherwise a joint that
# barely moved in the demos (e.g. wrist_roll held still) is divided by ~0, and
# any sim-to-real offset on it reaches the policy as a huge input.
MIN_STD = 1e-3
TRAJECTORY_GLOB = "trajectory_*.pkl"
# Dataset feature key -> key inside a teleop pickle.
CAMERA_SOURCES = {
    "observation.images.front": "front_images",
    "observation.images.wrist": "wrist_images",
}
# Simulator observation key (env `_get_observations`) for each camera feature.
CAMERA_OBSERVATION_KEYS = {
    "observation.images.front": "front_image",
    "observation.images.wrist": "wrist_image",
}


def to_uint8_rgb(image: Tensor, size: tuple[int, int]) -> np.ndarray:
    """Resize one (H, W, 3) simulator frame to `size` = (height, width) uint8 RGB.

    Float frames are taken to be in [0, 1]; integer frames in [0, 255].
    """
    image = image.detach().cpu()
    if torch.is_floating_point(image):
        image = image.clamp(0.0, 1.0).mul(255.0).round()
    image = image.to(dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
    return image[0].permute(1, 2, 0).clamp(0.0, 255.0).round().to(dtype=torch.uint8).numpy().copy()


def image_to_tensor(image: np.ndarray) -> Tensor:
    """(H, W, 3) uint8 RGB -> (3, H, W) float32 in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)


@dataclass
class Episode:
    path: Path
    state: np.ndarray                 # (T, state_dim) float32
    action: np.ndarray                # (T, action_dim) float32
    images: dict[str, np.ndarray]     # feature key -> (T, H, W, 3) uint8

    def __len__(self) -> int:
        return len(self.action)


def _load_episode(path: Path) -> Episode:
    with path.open("rb") as file:
        data = pickle.load(file)  # local, trusted teleop output
    missing = [key for key in ("observations", "actions", *CAMERA_SOURCES.values()) if key not in data]
    if missing:
        raise ValueError(
            f"{path}: missing {missing}. ACT needs RGB trajectories; "
            "record them with `scripts/teleop_task.py --visual`."
        )
    state = np.asarray(data["observations"], dtype=np.float32)
    action = np.asarray(data["actions"], dtype=np.float32)
    images = {key: np.asarray(data[src]) for key, src in CAMERA_SOURCES.items()}
    length = len(action)
    for name, array in (("observations", state), *images.items()):
        if len(array) != length:
            raise ValueError(f"{path}: {name} has {len(array)} steps but actions has {length}")
    for key, array in images.items():
        if array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError(f"{path}: {key} must be (T, H, W, 3) uint8, got {array.shape} {array.dtype}")
    if "wall_times" in data and length > 1:
        wall_dt = float(np.median(np.diff(data["wall_times"])))
        if wall_dt > 1.1 / FPS:
            # Leader was sampled on the wall clock but the sim advanced
            # 1/FPS per step, so the demo plays back faster than it was moved.
            print(
                f"warning: {path.name} was recorded at {1.0 / wall_dt:.1f} Hz wall-clock "
                f"(< {FPS} Hz); its motion is {FPS * wall_dt:.2f}x faster in sim time"
            )
    return Episode(path=path, state=state, action=action, images=images)


def load_episodes(data_dir: str | Path, min_length: int = 1) -> list[Episode]:
    """Load every `trajectory_*.pkl` under `data_dir`, in file-name order.

    Episodes shorter than `min_length` steps are skipped and reported -- the
    teleop script saves whatever was recorded when `t` is pressed, so a
    stray keypress leaves a few-step file behind. Shape disagreements between
    files are errors, not warnings.
    """
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob(TRAJECTORY_GLOB))
    if not paths:
        raise FileNotFoundError(f"no {TRAJECTORY_GLOB} under {data_dir}")

    episodes: list[Episode] = []
    for path in paths:
        episode = _load_episode(path)
        if len(episode) < min_length:
            print(f"skip {path.name}: {len(episode)} steps < --min-length {min_length}")
            continue
        if episodes:
            ref = episodes[0]
            for name, a, b in (
                ("observations", episode.state, ref.state),
                ("actions", episode.action, ref.action),
                *((key, episode.images[key], ref.images[key]) for key in CAMERA_SOURCES),
            ):
                if a.shape[1:] != b.shape[1:]:
                    raise ValueError(
                        f"{path.name}: {name} per-step shape {a.shape[1:]} differs from "
                        f"{ref.path.name}'s {b.shape[1:]}"
                    )
        episodes.append(episode)
    if not episodes:
        raise ValueError(f"every trajectory under {data_dir} is shorter than {min_length} steps")
    return episodes


def compute_stats(episodes: list[Episode]) -> dict:
    """MEAN_STD stats in the `stats.json` schema of `checkpoint.save_stats`.

    State/action: per dimension over all frames, with std below `MIN_STD`
    replaced by 1.0. Images: per channel over all pixels, in the [0, 1] scale
    the model receives.
    """
    stats = {}
    for key, attr in ((OBS_STATE, "state"), (ACTION, "action")):
        values = np.concatenate([getattr(ep, attr) for ep in episodes]).astype(np.float64)
        std = values.std(0)
        flat = np.flatnonzero(std < MIN_STD)
        if flat.size:
            print(f"warning: {key} dims {flat.tolist()} have std < {MIN_STD}; left unscaled")
        stats[key] = {"mean": values.mean(0).tolist(), "std": np.where(std < MIN_STD, 1.0, std).tolist()}
    for key in CAMERA_SOURCES:
        total = np.zeros(3)
        total_sq = np.zeros(3)
        count = 0
        for ep in episodes:
            # Chunked so the float64 copy stays small for long episodes.
            for start in range(0, len(ep), 64):
                pixels = ep.images[key][start : start + 64].reshape(-1, 3).astype(np.float64) / 255.0
                total += pixels.sum(0)
                total_sq += np.square(pixels).sum(0)
                count += len(pixels)
        mean = total / count
        std = np.sqrt(np.maximum(total_sq / count - np.square(mean), 0.0))
        stats[key] = {"mean": mean.tolist(), "std": std.tolist()}
    return stats


class TeleopACTDataset(Dataset):
    """Frame-indexed view over loaded episodes. Every frame of every episode is
    one sample; nothing is copied out of the episode arrays until indexed."""

    def __init__(self, episodes: list[Episode], chunk_size: int):
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.episodes = episodes
        self.chunk_size = chunk_size
        self.image_keys = tuple(CAMERA_SOURCES)
        lengths = np.array([len(ep) for ep in episodes])
        self._episode_index = np.repeat(np.arange(len(episodes)), lengths)
        self._frame_index = np.concatenate([np.arange(n) for n in lengths])

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def __len__(self) -> int:
        return len(self._frame_index)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        episode = self.episodes[self._episode_index[index]]
        t = int(self._frame_index[index])
        chunk = t + np.arange(self.chunk_size)
        is_pad = chunk >= len(episode)
        chunk = np.minimum(chunk, len(episode) - 1)
        sample = {
            OBS_STATE: torch.from_numpy(episode.state[t].copy()),
            ACTION: torch.from_numpy(episode.action[chunk]),
            "action_is_pad": torch.from_numpy(is_pad),
        }
        for key in self.image_keys:
            sample[key] = image_to_tensor(episode.images[key][t])
        return sample
