import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Isaac Lab sets the root logger to DEBUG, which uncovers JAX's compilation
# debug logs; pin the jax/jaxlib loggers before jax is imported. WARNING keeps
# TF_CPP_MIN_LOG_LEVEL at its default, unlike INFO.
os.environ.setdefault("JAX_LOGGING_LEVEL", "WARNING")
import random
import time
import json
import pickle

import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from utils.datasets import Dataset, MultistepDataset, Normalizer
from utils.env_utils import create_env
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import (
    get_exp_name,
    get_wandb_video,
    setup_wandb,
    get_flag_dict,
    get_wandb_group,
    CsvLogger,
)

from agents import agents

FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "debug", "Run group.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("env_name", "so101-StackCube-v0", "Environment (dataset) name.")
flags.DEFINE_string("save_dir", "exp/", "Save directory.")
flags.DEFINE_string("dataset_dir", "data/", "Dataset directory.")
flags.DEFINE_string("restore_path", None, "Restore path.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")
flags.DEFINE_string("wandb_mode", "offline", "Wandb mode.")
flags.DEFINE_string("wandb_group", None, "Wandb group override.")
flags.DEFINE_string(
    "wandb_group_format",
    "{run_group}/{env_name}/{agent_name}",
    "Wandb group format. Available fields: run_group, env_name, agent_name.",
)

flags.DEFINE_integer("offline_steps", 2000000, "Number of offline steps.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("save_interval", 1000000, "Save interval.")
flags.DEFINE_integer("eval_interval", 250000, "Evaluation interval; 0 disables.")
flags.DEFINE_integer(
    "num_envs", 50, "Parallel evaluation environments; each runs one episode."
)
flags.DEFINE_integer(
    "video_envs", 4, "Environments to record during evaluation; 0 disables video."
)
flags.DEFINE_integer("video_frame_skip", 2, "Environment steps between frames.")
flags.DEFINE_integer("video_fps", 15, "Frame rate of the logged video.")
flags.DEFINE_string("device", "cuda:0", "Isaac Lab device.")
flags.DEFINE_bool("headless", True, "Run Isaac Sim without a GUI.")
config_flags.DEFINE_config_file("agent", "agents/dbc.py", lock_config=False)


def main(_):
    config = FLAGS.agent
    config["train_steps"] = FLAGS.offline_steps

    exp_name = get_exp_name(config["agent_name"], seed=FLAGS.seed)

    # Save dir
    FLAGS.save_dir = os.path.join(
        FLAGS.save_dir, FLAGS.env_name, config["agent_name"], exp_name
    )
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    # Save parameters
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
        json.dump(flag_dict, f)

    config_dict = config.to_dict()
    with open(os.path.join(FLAGS.save_dir, "agent_config.json"), "w") as f:
        json.dump(config_dict, f)

    # Seed
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Wandb
    setup_wandb(
        project="so101",
        group=FLAGS.wandb_group
        or get_wandb_group(
            FLAGS.run_group,
            FLAGS.env_name,
            config["agent_name"],
            FLAGS.wandb_group_format,
        ),
        name=exp_name,
        config={
            **get_flag_dict(),
            **config.to_dict(),
        },
        mode=FLAGS.wandb_mode,
    )

    # Env
    if FLAGS.eval_interval > 0:
        # One environment per evaluation episode: every episode runs in a single
        # batched rollout, and Isaac Lab's per-step cost barely grows with the
        # environment count.
        eval_env, simulation_app = create_env(
            env_name=FLAGS.env_name,
            num_envs=FLAGS.num_envs,
            device=FLAGS.device,
            headless=FLAGS.headless,
            seed=FLAGS.seed,
            record_video=FLAGS.video_envs > 0,
        )

    # Dataset
    with open(os.path.join(FLAGS.dataset_dir, f"{FLAGS.env_name}.pkl"), "rb") as f:
        train_dataset = pickle.load(f)
    train_dataset["terminals"][-1] = True

    # Normalize before freezing the dataset.
    if FLAGS.restore_path is not None:
        normalizer = Normalizer.load(
            os.path.join(FLAGS.restore_path, "normalization.json")
        )
    else:
        normalizer = Normalizer.fit(
            train_dataset["observations"],
            train_dataset["actions"],
        )
    train_dataset["observations"] = normalizer.normalize_observations(
        train_dataset["observations"]
    )
    train_dataset["actions"] = normalizer.normalize_actions(train_dataset["actions"])
    if "next_observations" in train_dataset:
        train_dataset["next_observations"] = normalizer.normalize_observations(
            train_dataset["next_observations"]
        )
    normalizer.save(os.path.join(FLAGS.save_dir, "normalization.json"))

    dataset_class = {
        "Dataset": Dataset,
        "MultistepDataset": MultistepDataset,
    }
    train_dataset = dataset_class[config["dataset_class"]].create(**train_dataset)

    if hasattr(train_dataset, "pred_horizon"):
        train_dataset.pred_horizon = config["horizon_steps"]

    ex_transition = train_dataset.sample(2)

    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(
        seed=FLAGS.seed, ex_transition=ex_transition, config=config
    )

    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, "eval.csv"))
    first_time = time.time()
    last_time = time.time()

    for i in tqdm.tqdm(
        range(1, config["train_steps"] + 1), desc="training", smoothing=0.1
    ):
        batch = train_dataset.sample(config["batch_size"])
        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": float(v) for k, v in update_info.items()}
            train_metrics["time/epoch_time"] = (
                time.time() - last_time
            ) / FLAGS.log_interval
            train_metrics["time/total_time"] = time.time() - first_time
            train_metrics["train_step"] = i
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        if FLAGS.eval_interval > 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            eval_info, renders = evaluate(
                agent=agent,
                env=eval_env,
                normalizer=normalizer,
                config=config,
                video_envs=FLAGS.video_envs,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            eval_metrics = {f"evaluation/{k}": v for k, v in eval_info.items()}
            if len(renders) > 0:
                # CsvLogger drops wandb media types, so this only goes to wandb.
                eval_metrics["evaluation/video"] = get_wandb_video(
                    renders=renders, fps=FLAGS.video_fps
                )
            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()
    wandb.finish()
    if FLAGS.eval_interval > 0:
        eval_env.close()
        simulation_app.close()


if __name__ == "__main__":
    app.run(main)
