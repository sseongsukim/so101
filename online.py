"""Fine-tune a pretrained SO101 diffusion policy in Isaac Lab."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_LOGGING_LEVEL", "WARNING")
import contextlib
import json
import pickle
import random
import time

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from utils.datasets import Normalizer
from utils.env_utils import create_env
from utils.flax_utils import save_agent
from utils.reward_scaling import RunningMeanStd, RunningRewardScaler
from utils.log_utils import (
    CsvLogger,
    get_exp_name,
    get_flag_dict,
    get_wandb_group,
    get_wandb_video,
    setup_wandb,
)


@contextlib.contextmanager
def isolated_rng():
    """Run a block without leaving the global RNGs where it found them.

    Evaluation draws from the same Python/NumPy/torch generators the rollout
    does, so without this an evaluation would shift every later training
    rollout and two runs with the same seed would diverge the moment their
    evaluation schedules differed.
    """
    import torch

    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def collect_rollout(
    agent,
    env,
    normalizer,
    config,
    seed,
    reward_mode,
    training=False,
    deterministic=True,
    video_envs=0,
    video_frame_skip=2,
):
    """Collect one complete episode per environment.

    `training` also keeps the denoising chains the PPO update needs; evaluation
    only wants the metrics. The dense reward decomposition is accumulated either
    way so that `success_only` runs stay comparable to dense ones.
    """
    import torch

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
    # ResiP corrects every environment step, so its transitions are steps rather
    # than chunks: the base policy proposes a chunk, and the residual is redrawn
    # against the state the environment is actually in at each step of it.
    resip = config["agent_name"] == "resip"
    observations, chains, rewards = [], [], []
    residuals, log_probs, values = [], [], []
    success = np.zeros(num_envs, dtype=bool)
    stacked = success.copy()
    first = np.full(num_envs, -1.0)
    totals = {
        key: np.zeros(num_envs)
        for key in (
            "dense_return",
            "return",
            "success_steps",
            "reward_distance",
            "reward_lift",
            "reward_align",
            "reward_success",
        )
    }
    renders = [[] for _ in range(min(video_envs, num_envs))]
    with torch.inference_mode():
        observation, _ = env.reset(seed=int(seed))
        torch.manual_seed(int(seed) + 100000)
        if renders:
            # The first render attaches the annotator and comes back empty.
            env.render()
        for chunk in tqdm.tqdm(range(episode_steps // act_steps), desc="rollout"):
            state = normalizer.normalize_observations(
                observation["state"].cpu().numpy()
            )
            rng, sample_rng = jax.random.split(rng)
            if resip:
                # The residual, not the chunk, is the action here.
                base_actions = np.asarray(
                    agent.sample_base_actions(state, rng=sample_rng)
                )
                actions = None
            elif training:
                actions, chain = agent.sample_chains(
                    state, rng=sample_rng, deterministic=deterministic
                )
                observations.append(state.copy())
                chains.append(np.asarray(chain))
            else:
                actions = agent.sample_actions(state, rng=sample_rng)
            if actions is not None:
                actions = torch.as_tensor(
                    normalizer.unnormalize_actions(np.asarray(actions)),
                    dtype=torch.float32,
                    device=env.unwrapped.device,
                )
            chunk_rewards = torch.zeros(
                (act_steps if resip else 0, num_envs), device=env.unwrapped.device
            )
            dense = torch.zeros(num_envs, device=env.unwrapped.device)
            solved_steps = torch.zeros_like(dense)
            components = torch.zeros((4, num_envs), device=env.unwrapped.device)
            chunk_success = torch.zeros_like(dense, dtype=torch.bool)
            chunk_stacked = torch.zeros_like(chunk_success)
            chunk_first = torch.full_like(dense, -1)
            for step in range(act_steps):
                if resip:
                    rng, step_rng = jax.random.split(rng)
                    residual_observation = np.concatenate(
                        (
                            normalizer.normalize_observations(
                                observation["state"].cpu().numpy()
                            ),
                            base_actions[:, step],
                        ),
                        axis=-1,
                    )
                    residual, log_prob, value = agent.sample_residuals(
                        residual_observation,
                        rng=step_rng,
                        deterministic=deterministic,
                    )
                    residual = np.asarray(residual)
                    action = torch.as_tensor(
                        normalizer.unnormalize_actions(
                            base_actions[:, step] + config["action_scale"] * residual
                        ),
                        dtype=torch.float32,
                        device=env.unwrapped.device,
                    )
                    if training:
                        observations.append(residual_observation)
                        residuals.append(residual)
                        log_probs.append(np.asarray(log_prob))
                        values.append(np.asarray(value))
                else:
                    action = actions[:, step]
                observation, reward, terminated, truncated, info = env.step(action)
                # Every environment runs the full horizon; a reset here would
                # sum rewards across an episode boundary and leave that
                # environment out of phase with the shared chunk grid.
                if (terminated | truncated).any().item():
                    raise RuntimeError("An environment reset inside the rollout.")
                solved = info["success"].bool()
                dense += reward
                solved_steps += solved.float()
                if resip:
                    chunk_rewards[step] = (
                        reward if reward_mode == "dense" else solved.float()
                    )
                chunk_first = torch.where(
                    solved & (chunk_first < 0),
                    chunk * act_steps + step + 1,
                    chunk_first,
                )
                # Success replaces the shaping terms rather than adding to them.
                shaping = (~solved).float()
                components[0] += 0.1 * info["reward_distance"] * shaping
                components[1] += 1.5 * info["reward_lift"] * shaping
                components[2] += 2.0 * info["reward_align"] * shaping
                components[3] += 16.0 * solved.float()
                chunk_success |= solved
                chunk_stacked |= info["stacked"].bool()
                if renders and (chunk * act_steps + step) % video_frame_skip == 0:
                    # One viewport, so a frame per environment means rendering
                    # the same physics state once for each recorded one.
                    camera = env.unwrapped.viewport_camera_controller
                    for env_index, frames in enumerate(renders):
                        camera.set_view_env_index(env_index)
                        frames.append(env.render().copy())
            # One transfer for everything the chunk produced.
            transferred = (
                torch.cat(
                    (
                        dense[None],
                        solved_steps[None],
                        chunk_first[None],
                        chunk_success[None],
                        chunk_stacked[None],
                        components,
                        chunk_rewards,
                    )
                )
                .cpu()
                .numpy()
            )
            dense, solved_steps, chunk_first = transferred[:3]
            chunk_success, chunk_stacked = transferred[3:5]
            components, step_rewards = transferred[5:9], transferred[9:]
            reward = dense if reward_mode == "dense" else solved_steps
            if training:
                rewards.extend(step_rewards if resip else [reward.copy()])
            success |= chunk_success.astype(bool)
            stacked |= chunk_stacked.astype(bool)
            hit = (first < 0) & (chunk_first >= 0)
            first[hit] = chunk_first[hit]
            totals["dense_return"] += dense
            totals["return"] += reward
            totals["success_steps"] += solved_steps
            for key, value in zip(
                ("reward_distance", "reward_lift", "reward_align", "reward_success"),
                components,
            ):
                totals[key] += value
    metrics = {key: float(value.mean()) for key, value in totals.items()}
    metrics.update(
        success=float(success.mean()),
        stacked=float(stacked.mean()),
        success_step_fraction=float(totals["success_steps"].mean() / episode_steps),
        first_success_step=float(first[success].mean()) if success.any() else -1.0,
    )
    rollout = None
    if training:
        # The horizon ends the episode, so only its last transition is terminal.
        terminated = np.zeros((len(rewards), num_envs), dtype=bool)
        terminated[-1] = True
        rollout = dict(
            observations=np.stack(observations),
            rewards=np.stack(rewards),
            terminated=terminated,
        )
        if resip:
            rollout.update(
                actions=np.stack(residuals),
                log_probs=np.stack(log_probs),
                values=np.stack(values),
            )
        else:
            rollout["chains"] = np.stack(chains)
    return metrics, rollout, [np.asarray(frames) for frames in renders]


FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "debug", "Run group.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("env_name", "so101-StackCube-v0", "Environment name.")
flags.DEFINE_string("save_dir", "exp/", "Save directory.")
flags.DEFINE_string("restore_path", None, "Pretrained DBC run directory.")
flags.DEFINE_integer("restore_epoch", None, "Pretrained checkpoint epoch.")
flags.DEFINE_string("wandb_mode", "offline", "Wandb mode.")
flags.DEFINE_string("wandb_group", None, "Wandb group override.")
flags.DEFINE_string(
    "wandb_group_format", "{run_group}/{env_name}/{agent_name}", "Wandb group format."
)
flags.DEFINE_integer("online_iters", 3000, "Collect-and-update iterations.")
flags.DEFINE_integer("num_envs", 1024, "Parallel environments.")
flags.DEFINE_enum(
    "reward_mode", "success_only", ["dense", "success_only"], "Training reward."
)
flags.DEFINE_integer("log_interval", 1, "Logging interval.")
flags.DEFINE_integer("save_interval", 5, "Checkpoint interval.")
flags.DEFINE_integer("eval_interval", 25, "Evaluation interval; 0 disables.")
flags.DEFINE_list(
    "eval_seeds",
    ["1042", "2043"],
    "Base seeds for evaluation. Each one advances by the iteration count, "
    "so an evaluation never repeats the layouts an earlier one used while "
    "two runs with the same --seed still evaluate on the same episodes.",
)
flags.DEFINE_integer("video_envs", 4, "Environments to record during evaluation.")
flags.DEFINE_integer("video_frame_skip", 2, "Steps between video frames.")
flags.DEFINE_integer("video_fps", 15, "Video frame rate.")
flags.DEFINE_string("device", "cuda:0", "Isaac Lab device.")
flags.DEFINE_bool("headless", True, "Run Isaac Sim without a GUI.")
config_flags.DEFINE_config_file("agent", "agents/dppo.py", lock_config=False)


def main(_):
    assert FLAGS.restore_path is not None, "Pass --restore_path"
    assert FLAGS.restore_epoch is not None, "Pass --restore_epoch"
    config = FLAGS.agent
    with open(os.path.join(FLAGS.restore_path, "agent_config.json")) as f:
        pretrain_config = json.load(f)
    if (
        pretrain_config.get("architecture") != "diffusion_mlp"
        or pretrain_config["network_type"] != "mlp"
    ):
        raise ValueError(
            "Retrain DBC with the DiffusionMLP architecture before fine-tuning."
        )
    for key, value in pretrain_config.items():
        if key not in config:
            config[key] = tuple(value) if isinstance(value, list) else value
    resip = config["agent_name"] == "resip"
    # A cosine cycle that outlives the run would stop the learning rate before
    # it ever anneals, so an unset cycle length takes the length of the run.
    config["lr_cycle_iters"] = config["lr_cycle_iters"] or FLAGS.online_iters
    # Both agents run only the first `inference_steps` actions of the chunk and
    # re-plan from the state that leaves them in, so the policy may predict
    # further ahead than it executes -- the reference fine-tunes a 16-step
    # prediction with an 8-step execution that way. DPPO scores only the
    # executed prefix (`reward_horizon`) and ResiP corrects it step by step.
    if config["inference_steps"] > config["horizon_steps"]:
        raise ValueError("The execution horizon must not exceed the prediction one.")
    if config["update_epochs"] < 1:
        raise ValueError("Use update_epochs >= 1.")
    if resip and config["num_minibatches"] < 1:
        raise ValueError("Use num_minibatches >= 1.")
    if not resip and config["batch_size"] < 2:
        raise ValueError("Use batch_size >= 2.")
    eval_seeds = [int(seed) for seed in FLAGS.eval_seeds]
    if FLAGS.eval_interval and not eval_seeds:
        raise ValueError("Provide at least one evaluation seed.")

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    exp_name = get_exp_name(config["agent_name"], seed=FLAGS.seed)
    FLAGS.save_dir = os.path.join(
        FLAGS.save_dir, FLAGS.env_name, config["agent_name"], exp_name
    )
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
        json.dump(get_flag_dict(), f)
    with open(os.path.join(FLAGS.save_dir, "agent_config.json"), "w") as f:
        json.dump(config.to_dict(), f)
    setup_wandb(
        project="so101",
        name=exp_name,
        mode=FLAGS.wandb_mode,
        group=FLAGS.wandb_group
        or get_wandb_group(
            FLAGS.run_group,
            FLAGS.env_name,
            config["agent_name"],
            FLAGS.wandb_group_format,
        ),
        config={**get_flag_dict(), **config.to_dict()},
    )
    normalizer = Normalizer.load(os.path.join(FLAGS.restore_path, "normalization.json"))
    # Preserve the pretrained observation transform, including legacy runs
    # trained without clipping. New pretraining runs save observation_clip=5.
    normalizer.stats["clip_actions"] = False
    normalizer.save(os.path.join(FLAGS.save_dir, "normalization.json"))
    ob_dim = len(normalizer.stats["observation_mean"])
    action_dim = len(normalizer.stats["action_min"])
    with open(
        os.path.join(FLAGS.restore_path, f"params_{FLAGS.restore_epoch}.pkl"), "rb"
    ) as f:
        checkpoint = pickle.load(f)["agent"]
    pretrain_params = checkpoint["network"]["params"]["modules_actor"]
    agent = agents[config["agent_name"]].create(
        seed=FLAGS.seed,
        config=config,
        ex_transition={
            "observations": np.zeros((2, ob_dim), dtype=np.float32),
            "actions": np.zeros(
                (2, config["horizon_steps"], action_dim), dtype=np.float32
            ),
        },
        pretrain_params=pretrain_params,
    )
    env, simulation_app = create_env(
        env_name=FLAGS.env_name,
        num_envs=FLAGS.num_envs,
        device=FLAGS.device,
        headless=FLAGS.headless,
        seed=FLAGS.seed,
        record_video=FLAGS.video_envs > 0,
        terminate_on_success=False,
        truncate_on_timeout=False,
    )
    episode_steps = env.unwrapped.max_episode_length
    act_steps = config["inference_steps"]
    chain_steps = None if resip else config["ft_denoising_steps"]
    if episode_steps % act_steps:
        raise ValueError("The execution horizon must divide the episode length.")
    # ResiP divides the reward by the running standard deviation of the reward
    # itself, as its reference does; DPPO divides by the std of a rolling
    # discounted sum instead, following reference/dppo.
    if not config["reward_scale_running"]:
        reward_scaler = None
    elif resip:
        reward_scaler = RunningMeanStd()
    else:
        reward_scaler = RunningRewardScaler(
            num_envs=FLAGS.num_envs,
            clip_reward=config["reward_scale_clip"],
            gamma=config["reward_scale_gamma"],
        )
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, "eval.csv"))
    save_agent(agent, FLAGS.save_dir, "initial")
    first_time = time.time()
    env_steps = 0

    for i in tqdm.tqdm(range(1, FLAGS.online_iters + 1), desc="online training"):
        started = time.time()
        rollout_metrics, rollout, _ = collect_rollout(
            agent,
            env,
            normalizer,
            config,
            seed=FLAGS.seed + i,
            training=True,
            deterministic=False,
            reward_mode=FLAGS.reward_mode,
        )
        env_steps += FLAGS.num_envs * episode_steps
        if resip:
            # The residual is Gaussian and on-policy per step, so the rollout
            # already carries its log-probabilities and values.
            observations = rollout["observations"].reshape(-1, ob_dim + action_dim)
            actions = rollout["actions"].reshape(-1, action_dim)
            log_probs = rollout["log_probs"].reshape(-1)
            values = rollout["values"].reshape(-1)
            n = len(observations)
        else:
            observations = rollout["observations"].reshape(-1, ob_dim)
            chains = rollout["chains"].reshape(
                -1, chain_steps + 1, config["horizon_steps"], action_dim
            )
            n = len(observations)
            values = np.empty(n, dtype=np.float32)
            log_probs = np.empty(
                (n, chain_steps, config["horizon_steps"], action_dim), dtype=np.float32
            )
            for start in range(0, n, config["logprob_batch_size"]):
                end = min(start + config["logprob_batch_size"], n)
                states = observations[start:end]
                values[start:end] = np.asarray(agent.compute_values(states))
                log_probs[start:end] = np.asarray(
                    agent.log_probs(
                        np.repeat(states, chain_steps, axis=0),
                        chains[start:end, :-1].reshape(
                            -1, config["horizon_steps"], action_dim
                        ),
                        chains[start:end, 1:].reshape(
                            -1, config["horizon_steps"], action_dim
                        ),
                        np.tile(np.arange(chain_steps), end - start),
                    )
                ).reshape(end - start, chain_steps, config["horizon_steps"], action_dim)

        rewards = rollout["rewards"].copy()
        reward_scale = 1.0
        if reward_scaler is None:
            pass
        elif resip:
            # The reference rescales each step as the rollout runs; folding the
            # whole rollout in at once keeps one scale within an iteration.
            reward_scaler.update(rewards.reshape(-1))
            reward_scale = float(np.sqrt(reward_scaler.var + 1e-8))
            clip = config["reward_scale_clip"]
            rewards = np.clip(rewards / reward_scale, -clip, clip)
        else:
            first = np.zeros_like(rewards)
            first[0] = 1
            rewards = reward_scaler(rewards.T, first.T).T
            reward_scale = float(reward_scaler.scale)
        value_trajs = values.reshape(rewards.shape)
        advantages = np.zeros_like(rewards)
        tail = np.zeros(FLAGS.num_envs, dtype=np.float32)
        for step in reversed(range(len(rewards))):
            next_value = 0.0 if step == len(rewards) - 1 else value_trajs[step + 1]
            alive = 1.0 - rollout["terminated"][step]
            delta = (
                rewards[step] * config["reward_scale_const"]
                + config["gamma"] * next_value * alive
                - value_trajs[step]
            )
            tail = delta + config["gamma"] * config["gae_lambda"] * alive * tail
            advantages[step] = tail
        returns = (advantages + value_trajs).reshape(-1)
        advantages = advantages.reshape(-1)
        variance = np.var(returns)
        explained_variance = (
            float(1 - np.var(returns - values) / variance) if variance > 1e-12 else 0.0
        )

        infos = []
        stop = False
        # ResiP cuts the rollout into `num_minibatches` pieces the way its
        # reference does, so the minibatch tracks --num_envs; DPPO takes the
        # fixed minibatch size reference/dppo is configured with.
        samples = n if resip else n * chain_steps
        minibatch = (
            -(-samples // config["num_minibatches"]) if resip else config["batch_size"]
        )
        for _ in range(config["update_epochs"]):
            indices = np.random.permutation(samples)
            for start in range(0, samples, minibatch):
                inds = indices[start : start + minibatch]
                if len(inds) < 2:
                    continue
                if resip:
                    batch = dict(
                        observations=observations[inds],
                        actions=actions[inds],
                        log_probs=log_probs[inds],
                        advantages=advantages[inds],
                        returns=returns[inds],
                        values=values[inds],
                    )
                else:
                    transitions, denoisings = inds // chain_steps, inds % chain_steps
                    batch = dict(
                        observations=observations[transitions],
                        chains_prev=chains[transitions, denoisings],
                        chains_next=chains[transitions, denoisings + 1],
                        denoising_inds=denoisings,
                        log_probs=log_probs[transitions, denoisings],
                        advantages=advantages[transitions],
                        returns=returns[transitions],
                        values=values[transitions],
                    )
                updated_agent, info = agent.update(batch, iteration=i)
                if not bool(info["finite"]):
                    raise FloatingPointError("Non-finite loss or actor gradient.")
                agent = updated_agent
                infos.append({key: float(value) for key, value in info.items()})
                if (
                    i > config["critic_warmup_iters"]
                    and config["target_kl"] is not None
                    and infos[-1]["actor/approx_kl"] > config["target_kl"]
                ):
                    stop = True
                    break
            if stop:
                break
        if not infos:
            raise ValueError("No valid minibatches in the rollout.")
        if i % FLAGS.log_interval == 0:
            metrics = {
                f"training/{key}": float(np.mean([info[key] for info in infos]))
                for key in infos[0]
                if key != "finite"
            }
            metrics.update(
                {f"rollout/{key}": value for key, value in rollout_metrics.items()}
            )
            metrics.update(
                {
                    "training/num_minibatches": len(infos),
                    "training/explained_variance": explained_variance,
                    "training/return_target_mean": float(returns.mean()),
                    "rollout/reward_scale": reward_scale,
                    "time/epoch_time": time.time() - started,
                    "time/total_time": time.time() - first_time,
                    "env_step": env_steps,
                }
            )
            wandb.log(metrics, step=i)
            train_logger.log(metrics, step=i)

        if FLAGS.eval_interval > 0 and i % FLAGS.eval_interval == 0:
            results, renders = [], []
            with isolated_rng():
                # Every evaluation draws fresh cube placements: a fixed seed
                # would reset the same 512 layouts each time and measure the
                # policy on one sample of the task rather than on the task. The
                # offset keeps it reproducible across runs with the same --seed.
                for seed_index, base_seed in enumerate(eval_seeds):
                    seed = base_seed + i * len(eval_seeds)
                    metrics, _, frames = collect_rollout(
                        agent,
                        env,
                        normalizer,
                        config,
                        seed=seed,
                        reward_mode=FLAGS.reward_mode,
                        video_envs=FLAGS.video_envs if seed_index == 0 else 0,
                        video_frame_skip=FLAGS.video_frame_skip,
                    )
                    results.append(metrics)
                    renders.extend(frames)
            metrics = {
                key: float(np.mean([result[key] for result in results]))
                for key in results[0]
            }
            weights = np.array([result["success"] for result in results])
            metrics["first_success_step"] = (
                float(
                    np.average(
                        [max(result["first_success_step"], 0) for result in results],
                        weights=weights,
                    )
                )
                if weights.sum()
                else -1.0
            )
            eval_metrics = {
                f"evaluation/{key}": value for key, value in metrics.items()
            }
            if renders:
                eval_metrics["evaluation/video"] = get_wandb_video(
                    renders=renders, fps=FLAGS.video_fps
                )
            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()
    wandb.finish()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    app.run(main)
