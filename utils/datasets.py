"""Dataset and MultistepDataset from MjDex, with SO101 normalization."""

from functools import partial
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=("padding",))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0)), mode="edge"
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=("padding",))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert "observations" in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = (
            False  # Whether to additionally return next actions; set outside the class.
        )

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self["terminals"] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[
                np.searchsorted(self.initial_locs, idxs, side="right") - 1
            ]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(
                    jax.tree_util.tree_map(
                        lambda arr: arr[cur_idxs], self["observations"]
                    )
                )
                if i != self.frame_stack - 1:
                    next_obs.append(
                        jax.tree_util.tree_map(
                            lambda arr: arr[cur_idxs], self["observations"]
                        )
                    )
            next_obs.append(
                jax.tree_util.tree_map(lambda arr: arr[idxs], self["next_observations"])
            )

            batch["observations"] = jax.tree_util.tree_map(
                lambda *args: np.concatenate(args, axis=-1), *obs
            )
            batch["next_observations"] = jax.tree_util.tree_map(
                lambda *args: np.concatenate(args, axis=-1), *next_obs
            )
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                self.augment(batch, ["observations", "next_observations"])
        return batch

    def sample_sequence(self, batch_size, sequence_length, discount):
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)

        data = {k: v[idxs] for k, v in self.items()}

        # Pre-compute all required indices
        all_idxs = (
            idxs[:, None] + np.arange(sequence_length)[None, :]
        )  # (batch_size, sequence_length)
        all_idxs = all_idxs.flatten()

        # Batch fetch data to avoid loops
        batch_observations = self["observations"][all_idxs].reshape(
            batch_size, sequence_length, *self["observations"].shape[1:]
        )
        batch_next_observations = self["next_observations"][all_idxs].reshape(
            batch_size, sequence_length, *self["next_observations"].shape[1:]
        )
        batch_actions = self["actions"][all_idxs].reshape(
            batch_size, sequence_length, *self["actions"].shape[1:]
        )
        batch_rewards = self["rewards"][all_idxs].reshape(
            batch_size, sequence_length, *self["rewards"].shape[1:]
        )
        batch_masks = self["masks"][all_idxs].reshape(
            batch_size, sequence_length, *self["masks"].shape[1:]
        )
        batch_terminals = self["terminals"][all_idxs].reshape(
            batch_size, sequence_length, *self["terminals"].shape[1:]
        )

        # Calculate next_actions
        next_action_idxs = np.minimum(all_idxs + 1, self.size - 1)
        batch_next_actions = self["actions"][next_action_idxs].reshape(
            batch_size, sequence_length, *self["actions"].shape[1:]
        )

        # Use vectorized operations to calculate cumulative rewards and masks
        rewards = np.zeros((batch_size, sequence_length), dtype=float)
        masks = np.ones((batch_size, sequence_length), dtype=float)
        terminals = np.zeros((batch_size, sequence_length), dtype=float)
        valid = np.ones((batch_size, sequence_length), dtype=float)

        # Vectorized calculation
        rewards[:, 0] = batch_rewards[:, 0].squeeze()
        masks[:, 0] = batch_masks[:, 0].squeeze()
        terminals[:, 0] = batch_terminals[:, 0].squeeze()

        discount_powers = discount ** np.arange(sequence_length)
        for i in range(1, sequence_length):
            rewards[:, i] = (
                rewards[:, i - 1] + batch_rewards[:, i].squeeze() * discount_powers[i]
            )
            masks[:, i] = np.minimum(masks[:, i - 1], batch_masks[:, i].squeeze())
            terminals[:, i] = np.maximum(
                terminals[:, i - 1], batch_terminals[:, i].squeeze()
            )
            valid[:, i] = 1.0 - terminals[:, i - 1]

        # Reorganize observations data format - maintain the exact same shape as the original function
        if len(batch_observations.shape) == 5:  # Visual data: (batch, seq, h, w, c)
            # Transpose to (batch, h, w, seq, c) format, consistent with the original function
            observations = batch_observations.transpose(
                0, 2, 3, 1, 4
            )  # (batch_size, h, w, sequence_length, c)
            next_observations = batch_next_observations.transpose(
                0, 2, 3, 1, 4
            )  # (batch_size, h, w, sequence_length, c)
        else:  # State data: maintain (batch, seq, state_dim) shape
            observations = (
                batch_observations  # (batch_size, sequence_length, state_dim)
            )
            next_observations = (
                batch_next_observations  # (batch_size, sequence_length, state_dim)
            )

        # Maintain the 3D shape of actions and next_actions, consistent with the original function
        actions = batch_actions  # (batch_size, sequence_length, action_dim)
        next_actions = batch_next_actions  # (batch_size, sequence_length, action_dim)

        return dict(
            observations=data["observations"].copy(),
            full_observations=observations,
            actions=actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
            next_actions=next_actions,
        )

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result["next_actions"] = self._dict["actions"][
                np.minimum(idxs + 1, self.size - 1)
            ]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate(
            [crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1
        )
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: (
                    np.array(batched_random_crop(arr, crop_froms, padding))
                    if len(arr.shape) == 4
                    else arr
                ),
                batch[key],
            )


class MultistepDataset(FrozenDict):
    """Dataset class with optional multi-step action sampling."""

    @classmethod
    def create(cls, freeze=True, **fields):
        assert "observations" in fields and "actions" in fields
        assert "episode_ends" in fields or "terminals" in fields

        N = len(fields["observations"])
        if "episode_ends" in fields:
            terminals = np.zeros(N, dtype=bool)
            terminals[fields["episode_ends"]] = True
        else:
            terminals = np.asarray(fields["terminals"], dtype=bool)

        data = {
            "observations": fields["observations"],
            "actions": fields["actions"],
            "terminals": terminals,
        }
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)

        self.frame_stack = None
        self.p_aug = None
        self.return_next_actions = False

        self.pred_horizon = None

        self.terminal_locs = np.nonzero(self["terminals"] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        return np.random.randint(self.size, size=num_idxs)

    def _compute_episode_ends(self, idxs):
        """Return episode end index for each idx."""
        ep_indices = np.searchsorted(self.terminal_locs, idxs, side="left")
        ep_ends = np.where(
            ep_indices < len(self.terminal_locs),
            self.terminal_locs[ep_indices],
            self.size - 1,
        )
        return ep_ends

    def _sample_multistep_actions(self, idxs):
        """
        Return (B, H, act_dim) action chunks with terminal padding.
        """
        B = len(idxs)
        H = self.pred_horizon

        ep_ends = self._compute_episode_ends(idxs)
        offsets = np.arange(H)  # (H,)
        candidate_idxs = idxs[:, None] + offsets[None, :]  # (B, H)
        action_idxs = np.minimum(candidate_idxs, ep_ends[:, None])

        return self._dict["actions"][action_idxs]

    def sample(self, batch_size: int, idxs=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)

        batch = self.get_subset(idxs)

        if self.pred_horizon is not None:
            batch["actions"] = self._sample_multistep_actions(idxs)

        if self.p_aug is not None:
            if np.random.rand() < self.p_aug:
                self.augment(batch, ["observations"])

        return batch

    def get_subset(self, idxs):
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            result["next_actions"] = self._dict["actions"][
                np.minimum(idxs + 1, self.size - 1)
            ]
        return result

    def augment(self, batch, keys):
        padding = 4
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate(
            [crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1
        )

        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: (
                    np.array(batched_random_crop(arr, crop_froms, padding))
                    if len(arr.shape) == 4
                    else arr
                ),
                batch[key],
            )



class Normalizer:
    """Min/max actions; standardize state scalars while preserving quaternions."""

    def __init__(self, stats):
        self.stats = stats

    @classmethod
    def fit(cls, observations, actions, observation_clip=5.0, clip_actions=True):
        dim = observations.shape[-1]
        if dim not in (6, 36):
            raise ValueError(f"Expected SO101 state dimension 6 or 36, got {dim}.")
        mean = observations.astype(np.float64).mean(axis=0)
        std = observations.astype(np.float64).std(axis=0)
        # Do not amplify numerical noise in nearly constant features.
        std = np.where(std < 1e-3, 1.0, std)
        if dim == 36:
            for start in (0, 7, 20):
                mean[start:start + 4], std[start:start + 4] = 0.0, 1.0
        return cls(dict(
            version=1,
            observation_mean=mean.tolist(), observation_std=std.tolist(),
            action_min=actions.min(axis=0).tolist(),
            action_max=actions.max(axis=0).tolist(), action_source="train_dataset",
            observation_clip=observation_clip, clip_actions=clip_actions,
        ))

    def normalize_observations(self, observations, dtype=np.float32):
        if isinstance(observations, dict):
            observations = observations["state"]
        observations = ((np.asarray(observations, dtype=dtype)
                 - np.asarray(self.stats["observation_mean"], dtype=dtype))
                / np.asarray(self.stats["observation_std"], dtype=dtype))
        clip = self.stats.get("observation_clip")
        if clip is not None:
            observations = np.clip(observations, -clip, clip)
        return observations.astype(np.float32)

    def normalize_actions(self, actions, dtype=np.float32):
        actions = np.asarray(actions, dtype=dtype)
        low = np.asarray(self.stats["action_min"], dtype=dtype)
        span = np.asarray(self.stats["action_max"], dtype=dtype) - low
        if not self.stats.get("clip_actions", True):
            return np.where(
                span < 1e-8, 0, 2 * (actions - low) / np.where(span < 1e-8, 1, span) - 1
            ).astype(np.float32)
        # A constant action dimension maps to zero and always decodes to low.
        return np.where(span > 1e-6, 2 * (actions - low) / np.where(span > 1e-6, span, 1) - 1, 0).astype(np.float32)

    def unnormalize_actions(self, actions, clip=None):
        low = np.asarray(self.stats["action_min"], dtype=np.float32)
        span = np.asarray(self.stats["action_max"], dtype=np.float32) - low
        if self.stats.get("clip_actions", True):
            span = np.where(span > 1e-6, span, 0)
        else:
            span = np.where(np.abs(span) < 1e-8, 1, span)
        actions = np.asarray(actions)
        if clip is None:
            clip = self.stats.get("clip_actions", True)
        if clip:
            actions = np.clip(actions, -1, 1)
        return low + (actions + 1) * span / 2

    def save(self, path):
        Path(path).write_text(json.dumps(self.stats, indent=2))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))
