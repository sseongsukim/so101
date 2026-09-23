from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
import optax

from utils.networks import (
    DiffusionMLP,
    ResidualActor,
    ResidualCritic,
    UNetActorVectorField,
)
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.encoders import encoder_modules
from utils.diffusion_utils import ddim_schedule, ddpm_schedule, cosine_schedule


class ResiPAgent(flax.struct.PyTreeNode):
    """Residual Policy (ResiP) agent.

    The pretrained diffusion policy stays frozen and proposes a chunk of
    normalized actions. At every environment step a small Gaussian MLP sees the
    current observation together with the base action for that step and corrects
    it by `action_scale * residual`; PPO trains only that residual and its
    critic, so the RL problem is a low-dimensional Gaussian one rather than the
    denoising MDP DPPO optimizes.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    actor_opt_state: Any
    critic_opt_state: Any
    actor_tx: Any = nonpytree_field()
    critic_tx: Any = nonpytree_field()

    def _coef(self, name, t, ndim):
        """Gather a schedule coefficient at timestep `t`, broadcast to `ndim`."""
        coefs = jnp.asarray(self.config[name])
        return coefs[t].reshape(*t.shape, *((1,) * (ndim - t.ndim)))

    def _predict_x_start(self, x, t, eps, alpha=None, sqrt_one_minus_alpha=None):
        """Recover x₀ from the network output at timestep `t`."""
        if not self.config["predict_epsilon"]:
            return eps  # the network directly predicts x₀

        if self.config["use_ddim"]:
            return (x - sqrt_one_minus_alpha * eps) / jnp.sqrt(alpha)

        return (
            self._coef("sqrt_recip_alphas_cumprod", t, x.ndim) * x
            - self._coef("sqrt_recipm1_alphas_cumprod", t, x.ndim) * eps
        )

    @jax.jit
    def sample_base_actions(self, observations, rng=None):
        """Sample a chunk of normalized actions from the frozen base policy.

        Args:
            observations: (batch_size, *ob_dims)
            rng: PRNGKey

        Returns:
            base_actions: (batch_size, horizon_steps, action_dim)
        """
        seed = self.rng if rng is None else rng

        if len(observations.shape) == 1:
            observations = jnp.expand_dims(observations, axis=0)

        batch_dims = observations.shape[: -len(self.config["ob_dims"])]
        noise_seed, denoise_seed = jax.random.split(seed)
        x = jax.random.normal(
            noise_seed,
            (*batch_dims, self.config["horizon_steps"], self.config["action_dim"]),
        )

        denoised_clip_value = self.config["denoised_clip_value"]
        randn_clip_value = self.config["randn_clip_value"]
        eps_clip_value = self.config["eps_clip_value"]

        def ddpm_iter(i, carry):
            x, iter_rng = carry
            iter_rng, step_rng = jax.random.split(iter_rng)

            t_index = self.config["denoising_steps"] - 1 - i
            t = jnp.full(batch_dims, t_index, dtype=jnp.int32)

            eps = self.network.select("base")(observations, x, t.astype(jnp.float32))
            x_recon = self._predict_x_start(x, t, eps)
            if denoised_clip_value is not None:
                x_recon = jnp.clip(x_recon, -denoised_clip_value, denoised_clip_value)

            mu = (
                self._coef("ddpm_mu_coef1", t, x.ndim) * x_recon
                + self._coef("ddpm_mu_coef2", t, x.ndim) * x
            )
            logvar = self._coef("ddpm_logvar_clipped", t, x.ndim)

            std = jnp.where(t_index == 0, 0.0, jnp.clip(jnp.exp(0.5 * logvar), 1e-3))
            noise = jnp.clip(
                jax.random.normal(step_rng, x.shape),
                -randn_clip_value,
                randn_clip_value,
            )
            return mu + std * noise, iter_rng

        def ddim_iter(i, carry):
            x, iter_rng = carry

            index = jnp.full((*batch_dims,), i, dtype=jnp.int32)
            t = jnp.asarray(self.config["ddim_t"])[index]
            alpha = self._coef("ddim_alphas", index, x.ndim)
            alpha_prev = self._coef("ddim_alphas_prev", index, x.ndim)
            sqrt_one_minus_alpha = self._coef(
                "ddim_sqrt_one_minus_alphas", index, x.ndim
            )
            sigma = self._coef("ddim_sigmas", index, x.ndim)

            eps = self.network.select("base")(observations, x, t.astype(jnp.float32))
            x_recon = self._predict_x_start(
                x, t, eps, alpha=alpha, sqrt_one_minus_alpha=sqrt_one_minus_alpha
            )
            if denoised_clip_value is not None:
                x_recon = jnp.clip(x_recon, -denoised_clip_value, denoised_clip_value)
                eps = (x - jnp.sqrt(alpha) * x_recon) / sqrt_one_minus_alpha
            if eps_clip_value is not None:
                eps = jnp.clip(eps, -eps_clip_value, eps_clip_value)

            dir_xt = jnp.sqrt(jnp.clip(1.0 - alpha_prev - sigma**2, 0.0)) * eps
            return jnp.sqrt(alpha_prev) * x_recon + dir_xt, iter_rng

        if self.config["use_ddim"]:
            num_steps, denoise_iter = self.config["ddim_steps"], ddim_iter
        else:
            num_steps, denoise_iter = self.config["denoising_steps"], ddpm_iter

        x, _ = jax.lax.fori_loop(0, num_steps, denoise_iter, (x, denoise_seed))

        final_action_clip_value = self.config["final_action_clip_value"]
        if final_action_clip_value is not None:
            x = jnp.clip(x, -final_action_clip_value, final_action_clip_value)

        return x

    @partial(jax.jit, static_argnames=("deterministic",))
    def sample_residuals(self, observations, rng=None, deterministic=False):
        """Sample the residual correction for one environment step.

        Args:
            observations: (batch_size, ob_dim + action_dim), the observation
                concatenated with the base action of the current step.
            rng: PRNGKey
            deterministic: Return the mean instead of a sample (evaluation).

        Returns:
            residuals: (batch_size, action_dim), *unscaled* -- the environment
                action is `base_action + action_scale * residual`.
            log_probs: (batch_size,)
            values: (batch_size,)
        """
        seed = self.rng if rng is None else rng

        mean, log_std = self.network.select("actor")(observations)
        std = jnp.exp(log_std)
        if deterministic:
            residuals = mean
        else:
            residuals = mean + std * jax.random.normal(seed, mean.shape)
        log_probs = (
            -0.5 * ((residuals - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)
        ).sum(axis=-1)

        return residuals, log_probs, self.network.select("critic")(observations)

    @jax.jit
    def compute_values(self, observations):
        """Predict the state values used for advantage estimation."""
        return self.network.select("critic")(observations)

    def actor_loss(self, batch, grad_params):
        """Compute the clipped PPO policy loss over the residual actions."""
        mean, log_std = self.network.select("actor")(
            batch["observations"], params=grad_params
        )
        std = jnp.exp(log_std)
        log_probs = (
            -0.5 * ((batch["actions"] - mean) / std) ** 2
            - log_std
            - 0.5 * jnp.log(2 * jnp.pi)
        ).sum(axis=-1)
        entropy = (log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum(axis=-1).mean()

        advantages = batch["advantages"]
        if self.config["norm_adv"]:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(ddof=1) + 1e-8
            )

        log_ratio = log_probs - batch["log_probs"]
        ratio = jnp.exp(log_ratio)
        clip_coef = self.config["clip_coef"]
        pg_loss = jnp.maximum(
            -advantages * ratio,
            -advantages * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
        ).mean()

        # Keeping the correction small is what makes the residual stay a
        # correction rather than a second policy.
        l1_loss = jnp.abs(mean).mean()
        l2_loss = jnp.square(mean).mean()
        actor_loss = (
            pg_loss
            - self.config["ent_coef"] * entropy
            + self.config["residual_l1"] * l1_loss
            + self.config["residual_l2"] * l2_loss
        )

        return actor_loss, {
            "actor_loss": actor_loss,
            "pg_loss": pg_loss,
            "entropy": entropy,
            "approx_kl": ((ratio - 1.0) - log_ratio).mean(),
            "ratio": ratio.mean(),
            "clip_frac": (jnp.abs(ratio - 1.0) > clip_coef).mean(),
            "adv_mean": advantages.mean(),
            "residual_l1": l1_loss,
            "residual_l2": l2_loss,
            "std_mean": std.mean(),
        }

    def critic_loss(self, batch, grad_params):
        """Compute the value loss."""
        values = self.network.select("critic")(
            batch["observations"], params=grad_params
        )
        clip_vloss_coef = self.config["clip_vloss_coef"]
        if clip_vloss_coef is not None:
            values_clipped = batch["values"] + jnp.clip(
                values - batch["values"], -clip_vloss_coef, clip_vloss_coef
            )
            critic_loss = (
                0.5
                * jnp.maximum(
                    (values - batch["returns"]) ** 2,
                    (values_clipped - batch["returns"]) ** 2,
                ).mean()
            )
        else:
            critic_loss = 0.5 * ((values - batch["returns"]) ** 2).mean()

        return critic_loss, {"critic_loss": critic_loss, "v_mean": values.mean()}

    @jax.jit
    def total_loss(self, batch, grad_params):
        """Compute the total loss."""
        info = {}

        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        loss = actor_loss + self.config["vf_coef"] * critic_loss
        return loss, info

    @jax.jit
    def update(self, batch, iteration):
        """Update the critic and, after warmup, the actor."""
        (loss, info), grads = jax.value_and_grad(
            self.total_loss, argnums=1, has_aux=True
        )(batch, self.network.params)
        actor_enabled = iteration > self.config["critic_warmup_iters"]
        actor_step = jnp.maximum(iteration - self.config["critic_warmup_iters"] - 1, 0)
        actor_lr = cosine_schedule(
            actor_step,
            self.config["lr"],
            self.config["min_lr"],
            self.config["lr_warmup_iters"],
            self.config["lr_cycle_iters"],
        )
        critic_lr = cosine_schedule(
            iteration - 1,
            self.config["critic_lr"],
            self.config["critic_min_lr"],
            self.config["critic_lr_warmup_iters"],
            self.config["lr_cycle_iters"],
        )
        params = self.network.params
        # The reference clips the residual policy as a whole, so the actor and
        # the critic share one gradient norm.
        actor_grads, critic_grads = grads["modules_actor"], grads["modules_critic"]
        grad_norm = optax.global_norm((actor_grads, critic_grads))
        # PyTorch clip_grad_norm_ adds 1e-6 to the norm.
        scale = jnp.minimum(1.0, self.config["max_grad_norm"] / (grad_norm + 1e-6))
        actor_grads = jax.tree.map(lambda g: g * scale, actor_grads)
        critic_grads = jax.tree.map(lambda g: g * scale, critic_grads)

        def update_actor(_):
            updates, state = self.actor_tx.update(
                actor_grads, self.actor_opt_state, params["modules_actor"]
            )
            updates = jax.tree.map(lambda u: actor_lr * u, updates)
            return optax.apply_updates(params["modules_actor"], updates), state

        actor_params, actor_opt_state = jax.lax.cond(
            actor_enabled,
            update_actor,
            lambda _: (params["modules_actor"], self.actor_opt_state),
            None,
        )
        updates, critic_opt_state = self.critic_tx.update(
            critic_grads, self.critic_opt_state, params["modules_critic"]
        )
        updates = jax.tree.map(lambda u: critic_lr * u, updates)
        params = {
            **params,
            "modules_actor": actor_params,
            "modules_critic": optax.apply_updates(params["modules_critic"], updates),
        }
        info["lr"] = actor_lr
        info["critic_lr"] = critic_lr
        info["actor_enabled"] = actor_enabled
        info["finite"] = jnp.isfinite(loss) & jnp.isfinite(grad_norm)
        return (
            self.replace(
                network=self.network.replace(params=params, step=self.network.step + 1),
                actor_opt_state=actor_opt_state,
                critic_opt_state=critic_opt_state,
            ),
            info,
        )

    @jax.jit
    def sample_actions(self, observations, rng=None, temperature=None):
        """
        Sample a corrected action chunk, keeping the BC agents' evaluation API.

        The residual is evaluated once per chunk on the observation the chunk
        starts from, so the whole chunk is corrected open-loop. `online.py`
        instead re-evaluates it at every environment step, which is how ResiP is
        trained and how it should be evaluated; this path exists so that the
        chunked evaluation helpers keep working.

        Args:
            observations: (batch_size, *ob_dims)
            rng: PRNGKey
            temperature: Unused (kept for API compatibility)

        Returns:
            actions: (batch_size, horizon_steps, action_dim)
        """
        base_actions = self.sample_base_actions(observations, rng=rng)
        if len(observations.shape) == 1:
            observations = jnp.expand_dims(observations, axis=0)
        residual_observations = jnp.concatenate(
            (
                jnp.repeat(
                    observations[..., None, :], self.config["horizon_steps"], axis=-2
                ),
                base_actions,
            ),
            axis=-1,
        )
        mean, _ = self.network.select("actor")(residual_observations)
        return base_actions + self.config["action_scale"] * mean

    @classmethod
    def create(
        cls,
        seed,
        ex_transition,
        config,
        pretrain_params=None,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_transition: Example batch of observations and actions.
            config: Configuration dictionary.
            pretrain_params: Actor parameters of the pretrained BC checkpoint,
                which the frozen base policy is loaded from.
        """
        assert 0 <= config["lr_warmup_iters"] < config["lr_cycle_iters"], (
            f"lr_warmup_iters ({config['lr_warmup_iters']}) must be below "
            f"lr_cycle_iters ({config['lr_cycle_iters']}), which follows "
            "--online_iters unless it is set. Lower the warm-up or lengthen "
            "the run."
        )
        assert (
            config["encoder"] is None
        ), "ResiP conditions the residual on flat state observations."

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = ex_transition["observations"]
        ex_actions = ex_transition["actions"]

        ob_dims = ex_observations.shape[1:]

        batch_size, horizon_steps, action_dim = ex_actions.shape
        ex_times = np.zeros(shape=(batch_size,), dtype=np.float32)
        # The residual sees one base action at a time, next to the observation.
        ex_residual_observations = np.zeros(
            shape=(batch_size, ob_dims[0] + action_dim), dtype=np.float32
        )

        if config["network_type"] == "mlp":
            base_def = DiffusionMLP(
                hidden_dims=config["hidden_dims"],
                time_step_embed_dim=config["time_step_embed_dim"],
                horizon_steps=config["horizon_steps"],
                action_dim=action_dim,
                layer_norm=config["layer_norm"],
            )
        elif config["network_type"] == "unet":
            base_def = UNetActorVectorField(
                action_dim=action_dim,
                cond_dim=ob_dims[0],
                time_step_embed_dim=config["time_step_embed_dim"],
                dim=config["dim"],
                dim_mults=config["dim_mults"],
                kernel_size=config["kernel_size"],
                n_groups=config["n_groups"],
            )

        actor_def = ResidualActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            init_log_std=config["init_log_std"],
            learn_std=config["learn_std"],
            action_head_std=config["action_head_std"],
            layer_norm=config["actor_layer_norm"],
        )

        critic_def = ResidualCritic(
            hidden_dims=config["critic_hidden_dims"],
            last_layer_std=config["critic_last_layer_std"],
            last_layer_bias=config["critic_last_layer_bias"],
            layer_norm=config["critic_layer_norm"],
        )

        network_info = dict(
            base=(base_def, (ex_observations, ex_actions, ex_times)),
            actor=(actor_def, (ex_residual_observations,)),
            critic=(critic_def, (ex_residual_observations,)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)

        actor_tx = optax.adamw(learning_rate=1.0, weight_decay=config["weight_decay"])
        critic_tx = optax.adamw(
            learning_rate=1.0, weight_decay=config["critic_weight_decay"]
        )

        network_variables = network_def.init(init_rng, **network_args)
        network_params = flax.core.unfreeze(network_variables["params"])

        if pretrain_params is not None:
            expected = jax.tree.map(lambda p: p.shape, network_params["modules_base"])
            actual = jax.tree.map(lambda p: p.shape, pretrain_params)
            if actual != expected:
                raise ValueError(
                    "Pretrained actor shapes do not match the base architecture."
                )
            network_params["modules_base"] = flax.serialization.from_state_dict(
                network_params["modules_base"], pretrain_params
            )

        network = TrainState.create(network_def, network_params)

        config["action_dim"] = action_dim
        config["ob_dims"] = ob_dims

        ddpm_params = ddpm_schedule(
            config["denoising_steps"],
            beta_schedule=config["beta_schedule"],
            cosine_s=config["cosine_s"],
        )
        for key, value in ddpm_params.items():
            config[key] = value

        if config["use_ddim"]:
            assert config[
                "predict_epsilon"
            ], "DDIM requires predicting epsilon for now."
            ddim_steps = config["ddim_steps"]
            assert (
                ddim_steps is not None and 0 < ddim_steps <= config["denoising_steps"]
            ), "ddim_steps must be in (0, denoising_steps] when use_ddim is set."
            ddim_params = ddim_schedule(
                ddpm_params["alphas_cumprod"],
                denoising_steps=config["denoising_steps"],
                ddim_steps=ddim_steps,
                ddim_discretize=config["ddim_discretize"],
                ddim_eta=config["ddim_eta"],
            )
            for key, value in ddim_params.items():
                config[key] = value

        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            actor_opt_state=actor_tx.init(network_params["modules_actor"]),
            critic_opt_state=critic_tx.init(network_params["modules_critic"]),
            actor_tx=actor_tx,
            critic_tx=critic_tx,
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="resip",
            # Actor learning rate, on the reference's cosine schedule.
            lr=3e-4,
            min_lr=3e-5,
            lr_warmup_iters=5,
            # 0 follows --online_iters, so the cosine cycle spans the run.
            lr_cycle_iters=0,
            critic_lr=5e-3,
            # The reference anneals the critic to zero with no warm-up.
            critic_min_lr=0.0,
            critic_lr_warmup_iters=0,
            weight_decay=1e-6,
            critic_weight_decay=1e-6,
            # The reference splits each rollout into this many minibatches, so
            # the minibatch follows --num_envs instead of being pinned to it.
            num_minibatches=1,
            actor_hidden_dims=(256, 256),
            actor_layer_norm=False,
            critic_hidden_dims=(256, 256),
            critic_layer_norm=False,
            critic_last_layer_std=0.25,
            critic_last_layer_bias=0.25,
            # The correction is a fraction of the normalized action range.
            action_scale=0.1,
            # The reference uses -1.0, but so101's actions are absolute joint
            # targets: resampling that much noise every control step turns the
            # correction into jitter and drives the rollout success to zero.
            init_log_std=-4.0,
            learn_std=False,
            # 0 starts fine-tuning from the unmodified base policy.
            action_head_std=0.0,
            max_grad_norm=1.0,
            critic_warmup_iters=0,
            gamma=0.999,
            gae_lambda=0.95,
            update_epochs=50,
            vf_coef=1.0,
            ent_coef=0.0,
            clip_coef=0.2,
            clip_vloss_coef=ml_collections.config_dict.placeholder(float),
            norm_adv=True,
            target_kl=0.1,
            residual_l1=0.0,
            residual_l2=0.0,
            reward_scale_const=1.0,
            reward_scale_running=True,
            reward_scale_clip=5.0,
        )
    )
    return config
