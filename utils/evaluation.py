"""SO101 rollout evaluation at the teleoperation control cadence."""

import random

import jax
import numpy as np
from tqdm import tqdm


def evaluate(
    agent, env, normalizer, config, video_envs=0, video_frame_skip=3, seed=None
):
    """Run one complete episode per environment and report its metrics.

    Seeding the environment and torch makes the episode a function of `seed`
    alone, but `torch.manual_seed` is global, so the rollout runs inside a
    forked RNG: an evaluation must not decide what randomness the training that
    follows it sees.
    """
    import torch

    seed = 1042 if seed is None else int(seed)
    num_envs = env.unwrapped.num_envs
    episode_steps = env.unwrapped.max_episode_length
    act_steps = config["inference_steps"]
    # The policy may predict further ahead than it executes -- the reference's
    # furniture UNet configuration predicts 16 actions and runs the first 8 --
    # so only the executed prefix has to fit, and it has to tile the episode so
    # that no chunk straddles the horizon.
    if config["horizon_steps"] < act_steps or episode_steps % act_steps:
        raise ValueError(
            "inference_steps must not exceed horizon_steps and must divide the "
            "episode length."
        )

    rng = jax.random.PRNGKey(seed + 100000)
    success = np.zeros(num_envs, dtype=bool)
    stacked = success.copy()
    first = np.full(num_envs, -1.0)
    totals = {
        key: np.zeros(num_envs)
        for key in (
            "dense_return",
            "success_steps",
            "reward_distance",
            "reward_lift",
            "reward_align",
            "reward_success",
        )
    }
    renders = [[] for _ in range(min(video_envs, num_envs))]

    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            observation, _ = env.reset(seed=seed)
            torch.manual_seed(seed + 100000)
            if renders:
                # The first render attaches the annotator and comes back empty.
                env.render()
            for chunk in tqdm(range(episode_steps // act_steps), desc="evaluation"):
                state = normalizer.normalize_observations(
                    observation["state"].cpu().numpy()
                )
                rng, sample_rng = jax.random.split(rng)
                actions = torch.as_tensor(
                    normalizer.unnormalize_actions(
                        np.asarray(agent.sample_actions(state, rng=sample_rng))
                    ),
                    dtype=torch.float32,
                    device=env.unwrapped.device,
                )
                dense = torch.zeros(num_envs, device=env.unwrapped.device)
                solved_steps = torch.zeros_like(dense)
                components = torch.zeros((4, num_envs), device=env.unwrapped.device)
                chunk_success = torch.zeros_like(dense, dtype=torch.bool)
                chunk_stacked = torch.zeros_like(chunk_success)
                chunk_first = torch.full_like(dense, -1)
                for step in range(act_steps):
                    observation, reward, terminated, truncated, info = env.step(
                        actions[:, step]
                    )
                    # Every environment runs the full horizon, so a reset here
                    # would sum rewards across an episode boundary.
                    if (terminated | truncated).any().item():
                        raise RuntimeError("An environment reset inside the rollout.")
                    solved = info["success"].bool()
                    dense += reward
                    solved_steps += solved.float()
                    chunk_first = torch.where(
                        solved & (chunk_first < 0),
                        chunk * act_steps + step + 1,
                        chunk_first,
                    )
                    # Success replaces the shaping terms rather than adding to
                    # them, so the four components sum to the dense return.
                    shaping = (~solved).float()
                    components[0] += 0.1 * info["reward_distance"] * shaping
                    components[1] += 1.5 * info["reward_lift"] * shaping
                    components[2] += 2.0 * info["reward_align"] * shaping
                    components[3] += 16.0 * solved.float()
                    chunk_success |= solved
                    chunk_stacked |= info["stacked"].bool()
                    if renders and (chunk * act_steps + step) % video_frame_skip == 0:
                        # One viewport, so a frame per environment means
                        # rendering the same physics state once for each.
                        camera = env.unwrapped.viewport_camera_controller
                        for env_index, frames in enumerate(renders):
                            camera.set_view_env_index(env_index)
                            frames.append(env.render().copy())
                # One transfer for everything the chunk produced.
                (
                    dense,
                    solved_steps,
                    chunk_first,
                    chunk_success,
                    chunk_stacked,
                    *components,
                ) = (
                    torch.cat(
                        (
                            dense[None],
                            solved_steps[None],
                            chunk_first[None],
                            chunk_success[None],
                            chunk_stacked[None],
                            components,
                        )
                    )
                    .cpu()
                    .numpy()
                )
                success |= chunk_success.astype(bool)
                stacked |= chunk_stacked.astype(bool)
                hit = (first < 0) & (chunk_first >= 0)
                first[hit] = chunk_first[hit]
                totals["dense_return"] += dense
                totals["success_steps"] += solved_steps
                for key, value in zip(
                    (
                        "reward_distance",
                        "reward_lift",
                        "reward_align",
                        "reward_success",
                    ),
                    components,
                ):
                    totals[key] += value
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

    metrics = {key: float(value.mean()) for key, value in totals.items()}
    metrics.update(
        success=float(success.mean()),
        stacked=float(stacked.mean()),
        success_step_fraction=float(totals["success_steps"].mean() / episode_steps),
        first_success_step=float(first[success].mean()) if success.any() else -1.0,
    )
    return metrics, [np.asarray(frames) for frames in renders]
