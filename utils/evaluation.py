"""SO101 rollout evaluation at the teleoperation control cadence."""

import time

import jax
import numpy as np
from tqdm import tqdm


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped


def evaluate(agent, env, normalizer, config, video_envs=0, video_frame_skip=3):
    import torch

    # Called positionally below: passing `observations` by keyword would add a
    # second jit cache entry for the same computation.
    actor_fn = supply_rng(
        agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))
    )
    max_episode_steps = env.unwrapped.max_episode_length
    action_steps = min(config["inference_steps"], config["horizon_steps"])
    num_envs = env.unwrapped.num_envs
    # Only one viewport exists, so a frame per environment means re-rendering
    # the same physics state once per recorded environment.
    video_envs = min(video_envs, num_envs)
    camera = env.unwrapped.viewport_camera_controller
    if video_envs > 0 and camera is None:
        raise RuntimeError(
            "Video recording needs a rendering-enabled app; create the "
            "environment with record_video=True."
        )
    renders = [[] for _ in range(video_envs)]

    # One episode per environment, all stepped together. Episodes finish at
    # different steps, so `active` freezes an environment's accounting once it
    # succeeds or terminates while the remaining ones keep running.
    active = np.ones(num_envs, dtype=bool)
    returns = np.zeros(num_envs, dtype=np.float64)
    lengths = np.zeros(num_envs, dtype=np.int64)
    successes = np.zeros(num_envs, dtype=bool)
    # Stacking without the gripper-away requirement: latched the same way as
    # success, so it reports whether the cube was ever placed on the target.
    stackeds = np.zeros(num_envs, dtype=bool)
    step_times = []

    with torch.inference_mode():
        observation, _ = env.reset()
        # Compile and synchronize the policy before measuring control timing.
        state = observation["state"].detach().cpu().numpy()
        np.asarray(actor_fn(normalizer.normalize_observations(state)))
        if video_envs > 0:
            # The first render attaches the rgb annotator and comes back empty.
            env.render()

    action_chunk = None
    for step in tqdm(range(max_episode_steps), desc="evaluation", leave=False):
        step_started = time.perf_counter()
        with torch.inference_mode():
            if step % action_steps == 0:
                state = observation["state"].detach().cpu().numpy()
                observations = normalizer.normalize_observations(state)
                actions = actor_fn(observations)
                action_chunk = normalizer.unnormalize_actions(np.asarray(actions))

            actions = torch.tensor(
                action_chunk[:, step % action_steps],
                dtype=torch.float32,
                device=env.unwrapped.device,
            )
            # Exactly one env.step per target, using the environment control period.
            observation, reward, terminated, truncated, info = env.step(actions)
            step_reward = reward.detach().cpu().numpy()
            step_success = info["success"].detach().cpu().numpy().astype(bool)
            step_stacked = info["stacked"].detach().cpu().numpy().astype(bool)
            step_done = (
                torch.logical_or(terminated, truncated).detach().cpu().numpy()
            )

        # The step that ends an episode still counts towards its return and
        # length, matching the single-environment rollout this replaces.
        returns += step_reward * active
        lengths += active
        successes |= step_success & active
        stackeds |= step_stacked & active
        active &= ~(step_success | step_done)

        if video_envs > 0 and (step % video_frame_skip == 0 or not active.any()):
            with torch.inference_mode():
                for env_index in range(video_envs):
                    camera.set_view_env_index(env_index)
                    renders[env_index].append(env.render().copy())

        # No wall-clock pacing here. The simulator advances by a fixed dt per
        # step with no real-time coupling, so sleeping to the control period
        # would slow evaluation down without changing the rollout.
        # (teleop_task.py does pace, because a human and the leader arm are in
        # the loop there.) step_time measures how long one batched control step
        # costs; it is a throughput number, not the single-robot control
        # latency, which only a num_envs=1 rollout can report.
        step_times.append(time.perf_counter() - step_started)
        if not active.any():
            break

    metrics = {
        "return": float(returns.mean()),
        "length": float(lengths.mean()),
        "success": float(successes.mean()),
        "stacked": float(stackeds.mean()),
        "step_time": float(np.mean(step_times)),
    }
    return metrics, [np.asarray(render) for render in renders]
