"""Evaluate a trained SO101 policy checkpoint and log the rollouts to wandb."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Isaac Lab sets the root logger to DEBUG, which uncovers JAX's compilation
# debug logs; pin the jax/jaxlib loggers before jax is imported. WARNING keeps
# TF_CPP_MIN_LOG_LEVEL at its default, unlike INFO.
os.environ.setdefault("JAX_LOGGING_LEVEL", "WARNING")
import json
import random

import numpy as np
import wandb
from absl import app, flags
from ml_collections import ConfigDict

from agents import agents
from utils.datasets import Normalizer
from utils.env_utils import create_env
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent
from utils.log_utils import (
    CsvLogger,
    get_exp_name,
    get_flag_dict,
    get_wandb_group,
    get_wandb_video,
    setup_wandb,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("restore_path", None, "Trained experiment directory to evaluate.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")
flags.DEFINE_string(
    "env_name", None, "Environment name; taken from the run's flags.json when unset."
)
flags.DEFINE_integer("seed", 42, "Random seed; each run derives its own seed from it.")
flags.DEFINE_integer(
    "num_envs", 50, "Parallel evaluation environments; each runs one episode."
)
flags.DEFINE_integer("num_runs", 1, "Evaluation runs, for variance estimation.")
flags.DEFINE_integer(
    "video_envs", 4, "Environments to record during evaluation; 0 disables video."
)
flags.DEFINE_integer("video_frame_skip", 2, "Environment steps between frames.")
flags.DEFINE_integer("video_fps", 15, "Frame rate of the logged video.")
flags.DEFINE_string("device", "cuda:0", "Isaac Lab device.")
flags.DEFINE_bool("headless", True, "Run Isaac Sim without a GUI.")
flags.DEFINE_string("save_dir", "exp/", "Save directory.")
flags.DEFINE_string("wandb_mode", "offline", "Wandb mode.")
flags.DEFINE_string("run_group", "eval", "Run group.")
flags.DEFINE_string("wandb_group", None, "Wandb group override.")
flags.DEFINE_string(
    "wandb_group_format",
    "{run_group}/{env_name}/{agent_name}",
    "Wandb group format. Available fields: run_group, env_name, agent_name.",
)


def main(_):
    assert FLAGS.restore_path is not None, "Pass --restore_path"
    assert FLAGS.restore_epoch is not None, "Pass --restore_epoch"

    with open(os.path.join(FLAGS.restore_path, "agent_config.json")) as f:
        config = ConfigDict(json.load(f))

    env_name = FLAGS.env_name
    if env_name is None:
        with open(os.path.join(FLAGS.restore_path, "flags.json")) as f:
            env_name = json.load(f)["env_name"]

    exp_name = get_exp_name(config["agent_name"], seed=FLAGS.seed)
    save_dir = os.path.join(
        FLAGS.save_dir, "eval", env_name, config["agent_name"], exp_name
    )
    os.makedirs(save_dir, exist_ok=True)

    # Save parameters
    flag_dict = get_flag_dict()
    flag_dict["env_name"] = env_name
    with open(os.path.join(save_dir, "flags.json"), "w") as f:
        json.dump(flag_dict, f)

    # Seed
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Wandb
    setup_wandb(
        project="so101",
        group=FLAGS.wandb_group
        or get_wandb_group(
            FLAGS.run_group,
            env_name,
            config["agent_name"],
            FLAGS.wandb_group_format,
        ),
        name=exp_name,
        config={
            **flag_dict,
            **config.to_dict(),
        },
        mode=FLAGS.wandb_mode,
    )

    # Env
    eval_env, simulation_app = create_env(
        env_name=env_name,
        num_envs=FLAGS.num_envs,
        device=FLAGS.device,
        headless=FLAGS.headless,
        seed=FLAGS.seed,
        record_video=FLAGS.video_envs > 0,
    )

    # The training normalization statistics belong to the checkpoint, not to the
    # dataset directory: reloading them keeps the policy's input/output scaling
    # identical to training without touching the dataset.
    normalizer = Normalizer.load(
        os.path.join(FLAGS.restore_path, "normalization.json")
    )

    # `create` only reads the shapes off the example transition, so the training
    # dataset does not have to be loaded to rebuild the agent.
    ob_dim = len(normalizer.stats["observation_mean"])
    action_dim = len(normalizer.stats["action_min"])
    ex_transition = {
        "observations": np.zeros((2, ob_dim), dtype=np.float32),
        "actions": np.zeros((2, config["horizon_steps"], action_dim), dtype=np.float32),
    }

    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(
        seed=FLAGS.seed, ex_transition=ex_transition, config=config
    )
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    eval_logger = CsvLogger(os.path.join(save_dir, "eval.csv"))
    all_stats = {}

    for run_idx in range(FLAGS.num_runs):
        # Each run gets a reproducibly different random state, so the initial
        # states and the policy sampling noise differ across runs.
        run_seed = np.random.randint(0, 2**31)
        random.seed(run_seed)
        np.random.seed(run_seed)

        eval_info, renders = evaluate(
            agent=agent,
            env=eval_env,
            normalizer=normalizer,
            config=config,
            video_envs=FLAGS.video_envs,
            video_frame_skip=FLAGS.video_frame_skip,
        )
        eval_metrics = {f"evaluation/{k}": v for k, v in eval_info.items()}
        eval_metrics["run_seed"] = run_seed
        if len(renders) > 0:
            # CsvLogger drops wandb media types, so this only goes to wandb.
            eval_metrics["evaluation/video"] = get_wandb_video(
                renders=renders, fps=FLAGS.video_fps
            )
        wandb.log(eval_metrics, step=run_idx)
        eval_logger.log(eval_metrics, step=run_idx)

        for k, v in eval_info.items():
            all_stats.setdefault(k, []).append(v)

        print(
            f"\n=== Evaluation run {run_idx + 1}/{FLAGS.num_runs} "
            f"({env_name}, {config['agent_name']}, epoch {FLAGS.restore_epoch}) ==="
        )
        for k, v in sorted(eval_info.items()):
            print(f"  {k}: {v:.4f}")

    if FLAGS.num_runs > 1:
        summary_metrics = {}
        for k, v in all_stats.items():
            summary_metrics[f"evaluation_mean/{k}"] = float(np.mean(v))
            summary_metrics[f"evaluation_std/{k}"] = float(np.std(v))
        wandb.log(summary_metrics, step=FLAGS.num_runs)
        # Not through CsvLogger: its header is fixed by the first per-run row,
        # so these keys would be dropped.
        with open(os.path.join(save_dir, "eval_summary.json"), "w") as f:
            json.dump(summary_metrics, f, indent=2)
        print(f"\n=== Aggregate over {FLAGS.num_runs} runs ===")
        for k, v in sorted(summary_metrics.items()):
            print(f"  {k}: {v:.4f}")

    eval_logger.close()
    wandb.finish()
    eval_env.close()
    simulation_app.close()


if __name__ == "__main__":
    app.run(main)
