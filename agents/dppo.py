import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
import optax

from utils.networks import DiffusionMLP, UNetActorVectorField, Value
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.encoders import encoder_modules
from utils.diffusion_utils import ddpm_schedule, cosine_schedule


class DPPOAgent(flax.struct.PyTreeNode):
    """Diffusion Policy Policy Optimization (DPPO) agent."""

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

    def _predict_x_start(self, x, t, eps):
        """Recover x₀ from the network output at timestep `t`."""
        if not self.config["predict_epsilon"]:
            return eps  # the network directly predicts x₀

        return (
            self._coef("sqrt_recip_alphas_cumprod", t, x.ndim) * x
            - self._coef("sqrt_recipm1_alphas_cumprod", t, x.ndim) * eps
        )

    def _p_mean_logvar(self, observations, x, t, name, grad_params=None):
        """Mean and log-variance of the DDPM reverse step at timestep `t`."""
        eps = self.network.select(name)(
            observations, x, t.astype(jnp.float32), params=grad_params
        )
        x_recon = self._predict_x_start(x, t, eps)
        denoised_clip_value = self.config["denoised_clip_value"]
        if denoised_clip_value is not None:
            x_recon = jnp.clip(x_recon, -denoised_clip_value, denoised_clip_value)

        mu = (
            self._coef("ddpm_mu_coef1", t, x.ndim) * x_recon
            + self._coef("ddpm_mu_coef2", t, x.ndim) * x
        )
        return mu, self._coef("ddpm_logvar_clipped", t, x.ndim)

    @jax.jit
    def log_probs(
        self, observations, chains_prev, chains_next, denoising_inds, grad_params=None
    ):
        """Log-density of one fine-tuned reverse step of the denoising chain.

        Args:
            observations: (batch_size, *ob_dims)
            chains_prev: (batch_size, horizon_steps, action_dim), step input.
            chains_next: (batch_size, horizon_steps, action_dim), step output.
            denoising_inds: (batch_size,), index into the fine-tuning window.
            grad_params: Actor parameters to flow gradients through.

        Returns:
            log_probs: (batch_size, horizon_steps, action_dim)
        """
        t = self.config["ft_denoising_steps"] - 1 - denoising_inds
        mu, logvar = self._p_mean_logvar(
            observations, chains_prev, t, "actor", grad_params=grad_params
        )
        std = jnp.clip(jnp.exp(0.5 * logvar), self.config["min_logprob_denoising_std"])

        return (
            -0.5 * ((chains_next - mu) / std) ** 2
            - jnp.log(std)
            - 0.5 * jnp.log(2 * jnp.pi)
        )

    @jax.jit
    def compute_values(self, observations):
        """Predict the state values used for advantage estimation."""
        return self.network.select("critic")(observations)

    def actor_loss(self, batch, grad_params):
        """Compute the clipped PPO policy loss over the denoising MDP."""
        ft_denoising_steps = self.config["ft_denoising_steps"]
        denoising_inds = batch["denoising_inds"]

        log_probs = self.log_probs(
            batch["observations"],
            batch["chains_prev"],
            batch["chains_next"],
            denoising_inds,
            grad_params=grad_params,
        )
        reward_horizon = min(
            self.config["inference_steps"], self.config["horizon_steps"]
        )
        log_probs = jnp.clip(log_probs[:, :reward_horizon], -5.0, 2.0).mean(
            axis=(-2, -1)
        )
        old_log_probs = jnp.clip(
            batch["log_probs"][:, :reward_horizon], -5.0, 2.0
        ).mean(axis=(-2, -1))

        advantages = batch["advantages"]
        if self.config["norm_adv"]:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(ddof=1) + 1e-8
            )
        advantages = jnp.clip(
            advantages,
            jnp.quantile(advantages, self.config["clip_advantage_lower_quantile"]),
            jnp.quantile(advantages, self.config["clip_advantage_upper_quantile"]),
        )
        advantages = advantages * self.config["gamma_denoising"] ** (
            ft_denoising_steps - 1 - denoising_inds
        )

        log_ratio = log_probs - old_log_probs
        ratio = jnp.exp(log_ratio)

        if ft_denoising_steps > 1:
            clip_ploss_coef_base = self.config["clip_ploss_coef_base"]
            clip_ploss_coef_rate = self.config["clip_ploss_coef_rate"]
            clip_coef = clip_ploss_coef_base + (
                self.config["clip_ploss_coef"] - clip_ploss_coef_base
            ) * jnp.expm1(
                clip_ploss_coef_rate * denoising_inds / (ft_denoising_steps - 1)
            ) / np.expm1(
                clip_ploss_coef_rate
            )
        else:
            clip_coef = self.config["clip_ploss_coef"]

        actor_loss = jnp.maximum(
            -advantages * ratio,
            -advantages * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
        ).mean()

        return actor_loss, {
            "actor_loss": actor_loss,
            "approx_kl": ((ratio - 1.0) - log_ratio).mean(),
            "ratio": ratio.mean(),
            "clip_frac": (jnp.abs(ratio - 1.0) > clip_coef).mean(),
            "adv_mean": advantages.mean(),
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
        params = self.network.params
        actor_grads = grads["modules_actor"]
        grad_norm = optax.global_norm(actor_grads)
        # PyTorch clip_grad_norm_ adds 1e-6 to the norm.
        actor_grads = jax.tree.map(
            lambda g: g
            * jnp.minimum(1.0, self.config["max_grad_norm"] / (grad_norm + 1e-6)),
            actor_grads,
        )

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
            grads["modules_critic"], self.critic_opt_state, params["modules_critic"]
        )
        params = {
            **params,
            "modules_actor": actor_params,
            "modules_critic": optax.apply_updates(params["modules_critic"], updates),
        }
        info["lr"] = actor_lr
        info["actor_enabled"] = actor_enabled
        info["finite"] = jnp.isfinite(loss) & (~actor_enabled | jnp.isfinite(grad_norm))
        return (
            self.replace(
                network=self.network.replace(params=params, step=self.network.step + 1),
                actor_opt_state=actor_opt_state,
                critic_opt_state=critic_opt_state,
            ),
            info,
        )

    @partial(jax.jit, static_argnames=("deterministic",))
    def sample_chains(self, observations, rng=None, deterministic=False):
        """
        Sample multi-step actions together with the denoising chain behind them.

        Args:
            observations: Dict[str, jnp.ndarray], batched observations
            rng: PRNGKey
            deterministic: Keep the pretrained noise schedule (evaluation)
                instead of the raised exploration noise used to collect rollouts

        Returns:
            actions: (batch_size, horizon_steps, action_dim)
            chains: (batch_size, ft_denoising_steps + 1, horizon_steps, action_dim)
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

        denoising_steps = self.config["denoising_steps"]
        ft_denoising_steps = self.config["ft_denoising_steps"]
        randn_clip_value = self.config["randn_clip_value"]
        chains = jnp.zeros((ft_denoising_steps + 1, *x.shape))

        def ddpm_iter(i, carry):
            x, chains, iter_rng = carry
            iter_rng, step_rng = jax.random.split(iter_rng)

            t_index = denoising_steps - 1 - i
            t = jnp.full(batch_dims, t_index, dtype=jnp.int32)
            chains = chains.at[
                jnp.clip(ft_denoising_steps - 1 - t_index, 0, ft_denoising_steps)
            ].set(x)

            mu, logvar = jax.lax.cond(
                t_index < ft_denoising_steps,
                lambda: self._p_mean_logvar(observations, x, t, "actor"),
                lambda: self._p_mean_logvar(observations, x, t, "actor_bc"),
            )

            if deterministic:
                std = jnp.where(
                    t_index == 0, 0.0, jnp.clip(jnp.exp(0.5 * logvar), 1e-3)
                )
            else:
                std = jnp.clip(
                    jnp.exp(0.5 * logvar), self.config["min_sampling_denoising_std"]
                )

            noise = jnp.clip(
                jax.random.normal(step_rng, x.shape),
                -randn_clip_value,
                randn_clip_value,
            )
            return mu + std * noise, chains, iter_rng

        x, chains, _ = jax.lax.fori_loop(
            0, denoising_steps, ddpm_iter, (x, chains, denoise_seed)
        )

        final_action_clip_value = self.config["final_action_clip_value"]
        if final_action_clip_value is not None:
            x = jnp.clip(x, -final_action_clip_value, final_action_clip_value)
        chains = chains.at[ft_denoising_steps].set(x)

        return x, jnp.moveaxis(chains, 0, len(batch_dims))

    @jax.jit
    def sample_actions(self, observations, rng=None, temperature=None):
        """
        Sample multi-step actions, keeping the BC agents' evaluation API.

        Args:
            observations: Dict[str, jnp.ndarray], batched observations
            rng: PRNGKey
            temperature: Unused (kept for API compatibility)

        Returns:
            actions: (batch_size, horizon_steps, action_dim)
        """
        actions, _ = self.sample_chains(observations, rng=rng, deterministic=True)
        return actions

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
            pretrain_params: Actor parameters of the pretrained BC checkpoint.
                Both the fine-tuned and the frozen actor start from them.
        """
        assert 0 <= config["lr_warmup_iters"] < config["lr_cycle_iters"], (
            f"lr_warmup_iters ({config['lr_warmup_iters']}) must be below "
            f"lr_cycle_iters ({config['lr_cycle_iters']}). Lower the "
            "warm-up or increase --agent.lr_cycle_iters."
        )
        assert not config[
            "use_ddim"
        ], "DPPO fine-tunes the DDPM chain; pretrain or reload with use_ddim=False."
        assert (
            0 < config["ft_denoising_steps"] <= config["denoising_steps"]
        ), "ft_denoising_steps must be in (0, denoising_steps]."

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = ex_transition["observations"]
        ex_actions = ex_transition["actions"]

        ob_dims = ex_observations.shape[1:]

        batch_size, horizon_steps, action_dim = ex_actions.shape
        ex_times = np.zeros(shape=(batch_size,), dtype=np.float32)

        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["actor"] = encoder_module()
            encoders["critic"] = encoder_module()

        if config["network_type"] == "mlp":
            actor_def = DiffusionMLP(
                hidden_dims=config["hidden_dims"],
                time_step_embed_dim=config["time_step_embed_dim"],
                horizon_steps=config["horizon_steps"],
                action_dim=action_dim,
                layer_norm=config["layer_norm"],
                encoder=encoders.get("actor"),
            )
        elif config["network_type"] == "unet":
            actor_def = UNetActorVectorField(
                action_dim=action_dim,
                cond_dim=ob_dims[0],
                time_step_embed_dim=config["time_step_embed_dim"],
                dim=config["dim"],
                dim_mults=config["dim_mults"],
                kernel_size=config["kernel_size"],
                n_groups=config["n_groups"],
                encoder=encoders.get("actor"),
            )

        critic_def = Value(
            hidden_dims=config["critic_hidden_dims"],
            layer_norm=config["critic_layer_norm"],
            encoder=encoders.get("critic"),
        )

        network_info = dict(
            actor=(actor_def, (ex_observations, ex_actions, ex_times)),
            actor_bc=(
                copy.deepcopy(actor_def),
                (ex_observations, ex_actions, ex_times),
            ),
            critic=(critic_def, (ex_observations,)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)

        actor_tx = optax.adamw(learning_rate=1.0, weight_decay=config["weight_decay"])
        critic_tx = optax.adamw(
            learning_rate=config["critic_lr"],
            weight_decay=config["critic_weight_decay"],
        )

        network_variables = network_def.init(init_rng, **network_args)
        network_params = flax.core.unfreeze(network_variables["params"])

        if pretrain_params is not None:
            expected = jax.tree.map(lambda p: p.shape, network_params["modules_actor"])
            actual = jax.tree.map(lambda p: p.shape, pretrain_params)
            if actual != expected:
                raise ValueError(
                    "Pretrained actor shapes do not match the DPPO architecture."
                )
            network_params["modules_actor"] = flax.serialization.from_state_dict(
                network_params["modules_actor"], pretrain_params
            )
            network_params["modules_actor_bc"] = network_params["modules_actor"]

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
            agent_name="dppo",
            # StackCube trial: slower actor updates with slightly more exploration.
            lr=3e-6,
            min_lr=3e-7,
            lr_warmup_iters=10,
            # Match the reference stable run's 1000-iteration cosine cycle.
            lr_cycle_iters=3000,
            critic_lr=1e-3,
            weight_decay=0.0,
            critic_weight_decay=0.0,
            batch_size=7500,
            logprob_batch_size=10240,
            critic_hidden_dims=(256, 256, 256),
            critic_layer_norm=False,
            max_grad_norm=1.0,
            critic_warmup_iters=10,
            randn_clip_value=3.0,
            gamma=0.999,
            gae_lambda=0.95,
            update_epochs=1,
            vf_coef=0.5,
            target_kl=1.0,
            norm_adv=True,
            clip_advantage_lower_quantile=0.0,
            clip_advantage_upper_quantile=1.0,
            clip_vloss_coef=ml_collections.config_dict.placeholder(float),
            reward_scale_const=1.0,
            reward_scale_running=True,
            reward_scale_gamma=0.99,
            reward_scale_clip=10.0,
            ft_denoising_steps=10,
            gamma_denoising=0.99,
            clip_ploss_coef=0.001,
            clip_ploss_coef_base=0.001,
            clip_ploss_coef_rate=3.0,
            min_sampling_denoising_std=0.015,
            min_logprob_denoising_std=0.1,
        )
    )
    return config
