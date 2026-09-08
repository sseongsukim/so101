"""Create the same single-environment simulation used for teleoperation."""


import logging


def create_env(
    env_name,
    num_envs=1,
    device="cuda:0",
    headless=True,
    seed=0,
    record_video=False,
    video_resolution=(480, 270),
):
    # Isaac Lab's SimulationContext sets the root logger to DEBUG, which would
    # otherwise surface JAX's compilation logs (main.py also pins
    # JAX_LOGGING_LEVEL before jax is imported).
    logging.getLogger("jax").setLevel(logging.INFO)
    logging.getLogger("jaxlib").setLevel(logging.INFO)

    from isaaclab.app import AppLauncher

    # Rendering is off unless video is requested: headless only hides the
    # window, while enable_cameras is what loads the RTX pipeline that makes
    # offscreen rendering possible at all.
    simulation_app = AppLauncher(
        device=device, headless=headless, enable_cameras=record_video
    ).app

    # Isaac Lab task imports must follow AppLauncher, as in teleop_task.py.
    import gymnasium as gym
    import so101.tasks  # noqa: F401
    from so101.configs import make_env_cfg

    env_cfg = make_env_cfg(env_name, num_envs=num_envs, device=device)
    env_cfg.seed = seed
    env_cfg.terminate_on_success = False
    env_cfg.truncate_on_timeout = False
    if record_video:
        # The viewport render product is sized from the viewer config, and the
        # 1280x720 default costs ~1 GB of frames for a full rollout.
        env_cfg.viewer.resolution = tuple(video_resolution)
    env = gym.make(
        env_name,
        cfg=env_cfg,
        render_mode="rgb_array" if record_video else None,
    )
    return env, simulation_app
