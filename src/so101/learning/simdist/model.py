"""SO101 visual SimDist network adapted from the adjacent native torch port.

History o[t-H:t+1], sent a[t-H:t], future a[t:t+T] -> z[t+1:t+T+1],
r[t:t+T], V[t+1:t+T+1], policy a[t:t+T]. Reward/value attention is causal
so prefix predictions cannot see future actions beyond their own transition.
All forward scalar outputs are scaled; inference returns raw reward/value/radians.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .config import WorldModelConfig
from .encoder import VisualEncoder
from .modules import MLP, Embedding, TransformerDecoder, TransformerEncoder
from .scaler import Scaler, Stats

InputDict = Mapping[str, torch.Tensor]


class WorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig, stats: Stats):
        super().__init__()
        self.config = config

        latent_dim = config.latent_dim
        action_dim = config.action_dim
        T = config.prediction_length
        emb_hidden = latent_dim * config.emb_mlp_hidden_size_factor

        # --- input processing ---
        self.scaler = Scaler(stats)
        self.encoder = VisualEncoder(config)
        self.fut_acts_embed = Embedding(
            seq_len=T,
            input_dim=action_dim,
            hidden_dims=[emb_hidden] * config.emb_future_acts_layers,
            embed_dim=latent_dim,
        )

        self.future_time = MLP(1, [emb_hidden], latent_dim)

        # --- latent dynamics ---
        self.dynamics = TransformerDecoder(
            num_layers=config.dyn_layers,
            embed_dim=latent_dim,
            mlp_hidden_dim=latent_dim * config.dyn_mlp_hidden_size_factor,
            num_heads=config.dyn_heads,
            attention_dropout_rate=config.attention_dropout,
            mlp_dropout_rate=config.mlp_dropout,
            causal=True,
        )
        self.dynamics_residual = nn.Linear(latent_dim, latent_dim) if config.residual_dynamics else None
        if self.dynamics_residual is not None:
            nn.init.zeros_(self.dynamics_residual.weight)
            nn.init.zeros_(self.dynamics_residual.bias)
        self.joint_residual = nn.Linear(latent_dim, config.proprio_obs_dim) if config.joint_dynamics else None
        if self.joint_residual is not None:
            nn.init.zeros_(self.joint_residual.weight)
            nn.init.zeros_(self.joint_residual.bias)
            self.register_buffer("joint_residual_scale", torch.ones(T))
        self.visual_residual = nn.Linear(latent_dim, 2 * latent_dim) if config.visual_dynamics else None
        if self.visual_residual is not None:
            nn.init.zeros_(self.visual_residual.weight)
            nn.init.zeros_(self.visual_residual.bias)
            self.register_buffer("visual_residual_scale", torch.ones(T, 2))
        self._freeze_visual_forecaster = False

        # --- reward head: sees causal (latent, action) pairs ---
        self.reward_emb = Embedding(
            seq_len=T,
            input_dim=latent_dim + action_dim,
            hidden_dims=[emb_hidden] * config.emb_reward_layers,
            embed_dim=latent_dim,
        )
        self.reward = TransformerEncoder(
            num_layers=config.reward_attn_layers,
            embed_dim=latent_dim,
            mlp_hidden_dim=latent_dim * config.reward_attn_mlp_hidden_size_factor,
            num_heads=config.reward_attn_heads,
            attention_dropout_rate=config.attention_dropout,
            mlp_dropout_rate=config.mlp_dropout,
            causal=True,
        )
        self.reward_dec = MLP(
            latent_dim,
            [latent_dim * config.reward_dec_mlp_hidden_size_factor] * config.reward_dec_layers,
            1,
        )
        self.register_buffer("reward_calibration_scale", torch.ones(T))
        self.register_buffer("reward_calibration_bias", torch.zeros(T))

        # --- value head ---
        self.value_emb = Embedding(
            seq_len=T,
            input_dim=latent_dim,
            hidden_dims=[emb_hidden] * config.emb_value_layers,
            embed_dim=latent_dim,
        )
        self.value = TransformerEncoder(
            num_layers=config.value_attn_layers,
            embed_dim=latent_dim,
            mlp_hidden_dim=latent_dim * config.value_attn_mlp_hidden_size_factor,
            num_heads=config.value_attn_heads,
            attention_dropout_rate=config.attention_dropout,
            mlp_dropout_rate=config.mlp_dropout,
            causal=True,
        )
        self.value_dec = MLP(
            latent_dim,
            [latent_dim * config.value_dec_mlp_hidden_size_factor] * config.value_dec_layers,
            1,
        )
        self.register_buffer("value_calibration_scale", torch.ones(T))
        self.register_buffer("value_calibration_bias", torch.zeros(T))

        # --- policy (imitation) head ---
        self.policy_query = nn.Parameter(torch.randn(T, latent_dim))
        self.policy = TransformerDecoder(
            num_layers=config.policy_attn_layers,
            embed_dim=latent_dim,
            mlp_hidden_dim=latent_dim * config.policy_attn_mlp_hidden_size_factor,
            num_heads=config.policy_attn_heads,
            attention_dropout_rate=config.attention_dropout,
            mlp_dropout_rate=config.mlp_dropout,
            causal=True,
        )
        self.policy_dec = MLP(
            latent_dim,
            [latent_dim * config.policy_dec_mlp_hidden_size_factor] * config.policy_dec_layers,
            action_dim,
        )

    def context(self, x: InputDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode RGB once per replan; candidate actions share this context."""
        x = self.scaler.scale(x)
        history, latent = self.encoder(x["proprio_obs_hist"], x["acts_hist"],
                                       x["front_hist"], x["wrist_hist"], x["remaining_ticks_hist"])
        return torch.cat([history, latent.unsqueeze(1)], dim=1), latent

    def persistence_latents(self, x: InputDict) -> torch.Tensor:
        """Time-aware persistence: current observable content, future task clock."""
        scaled = self.scaler.scale(x)
        content = self.encoder.content(scaled["proprio_obs_hist"][:, -1],
                                        x["front_hist"][:, -1], x["wrist_hist"][:, -1])
        remaining = x["remaining_ticks_hist"][:, -1, 0]
        steps = torch.arange(1, self.config.prediction_length + 1, device=remaining.device)
        future_remaining = (remaining[:, None] - steps).clamp_min(0)
        repeated = content[:, None, :].expand(-1, self.config.prediction_length, -1)
        return self.encoder.latent_from_content(repeated, future_remaining.unsqueeze(-1))

    def current_visual_features(self, x: InputDict) -> dict[str, torch.Tensor]:
        """Encode only the current RGB frames for future joint-conditioned latents."""
        return {
            "front": self.encoder.front(x["front_hist"][:, -1]),
            "wrist": self.encoder.wrist(x["wrist_hist"][:, -1]),
        }

    def freeze_visual_forecaster(self):
        """Freeze target representation and calibrated joint/action forecaster."""
        if self.visual_residual is None:
            raise ValueError("visual dynamics head is not configured")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.visual_residual.parameters():
            parameter.requires_grad_(True)
        self._freeze_visual_forecaster = True
        self.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._freeze_visual_forecaster:
            # Keep target encoder and action/joint forecast deterministic while
            # gradients flow through frozen modules into the visual delta head.
            for module in (self.encoder, self.fut_acts_embed, self.future_time,
                           self.dynamics, self.joint_residual, self.reward_emb,
                           self.reward, self.reward_dec, self.value_emb, self.value,
                           self.value_dec, self.policy, self.policy_dec):
                if module is not None:
                    module.eval()
        return self

    def predict(self, context, latent, fut_acts, remaining_ticks, persistence_latents=None,
                current_joints=None, current_visual_features=None):
        """Inputs/actions scaled; outputs scaled except latents."""
        remaining_ticks = remaining_ticks.reshape(-1, 1).to(fut_acts.device)
        if remaining_ticks.shape[0] == 1 and fut_acts.shape[0] > 1:
            remaining_ticks = remaining_ticks.expand(fut_acts.shape[0], -1)
        if remaining_ticks.shape[0] != fut_acts.shape[0]:
            raise ValueError("remaining_ticks batch must match candidate batch")
        steps = torch.arange(1, self.config.prediction_length + 1, device=fut_acts.device)
        future_remaining = (remaining_ticks - steps).clamp_min(0)
        time = self.future_time(future_remaining.unsqueeze(-1).float() / self.config.task_horizon_ticks)
        dynamics_features = self.dynamics(self.fut_acts_embed(fut_acts) + time, context)
        if self.dynamics_residual is None:
            latents = dynamics_features
        else:
            if persistence_latents is None:
                raise ValueError("residual dynamics requires time-aware persistence_latents")
            expected_shape = (fut_acts.shape[0], self.config.prediction_length, self.config.latent_dim)
            if tuple(persistence_latents.shape) != expected_shape:
                raise ValueError(f"persistence_latents must have shape {expected_shape}")
            latents = persistence_latents + self.dynamics_residual(dynamics_features)
        proprio_predictions = None
        visual_predictions = None
        if self.joint_residual is not None:
            if current_joints is None:
                raise ValueError("joint dynamics requires current_joints in normalized radians")
            expected_joints = (fut_acts.shape[0], self.config.proprio_obs_dim)
            if tuple(current_joints.shape) != expected_joints:
                raise ValueError(f"current_joints must have shape {expected_joints}")
            proprio_delta = self.joint_residual(dynamics_features) * self.joint_residual_scale[None, :, None]
            proprio_predictions = current_joints.unsqueeze(1) + proprio_delta
        if self.config.joint_conditioned_latents:
            if proprio_predictions is None:
                raise ValueError("joint-conditioned latents require joint dynamics")
            if current_visual_features is None or set(current_visual_features) != {"front", "wrist"}:
                raise ValueError("joint-conditioned latents require current front/wrist features")
            expected_feature = (fut_acts.shape[0], self.config.latent_dim)
            for name in ("front", "wrist"):
                if tuple(current_visual_features[name].shape) != expected_feature:
                    raise ValueError(f"current {name} features must have shape {expected_feature}")
            # Forecast q from the current observation and candidate actions; pair it
            # with current visual content and the exact successor task clock.
            joints = self.encoder.joints(proprio_predictions)
            front = current_visual_features["front"][:, None].expand_as(joints)
            wrist = current_visual_features["wrist"][:, None].expand_as(joints)
            if self.visual_residual is not None:
                visual_delta = self.visual_residual(dynamics_features).reshape(
                    fut_acts.shape[0], self.config.prediction_length, 2, self.config.latent_dim)
                visual_delta = visual_delta * self.visual_residual_scale[None, :, :, None]
                front = front + visual_delta[:, :, 0]
                wrist = wrist + visual_delta[:, :, 1]
                visual_predictions = {"front": front, "wrist": wrist}
            content = torch.cat((joints, front, wrist), dim=-1)
            latents = self.encoder.latent_from_content(content, future_remaining.unsqueeze(-1))
        previous = torch.cat([latent.unsqueeze(1), latents[:, :-1]], dim=1)
        rew_in = self.reward_emb(torch.cat([previous, fut_acts], dim=-1))
        rewards = self.reward_dec(self.reward(rew_in)).squeeze(-1)
        values = self.value_dec(self.value(self.value_emb(latents))).squeeze(-1)
        rewards = rewards * self.reward_calibration_scale[None, :] + self.reward_calibration_bias[None, :]
        values = values * self.value_calibration_scale[None, :] + self.value_calibration_bias[None, :]
        # Exact objective boundary in scaled units (inference unscales to zero).
        zero_reward = -self.scaler.rewards_mean / self.scaler.rewards_std
        zero_value = -self.scaler.values_mean / self.scaler.values_std
        rewards = torch.where(remaining_ticks > steps - 1, rewards, zero_reward)
        values = torch.where(future_remaining > 0, values, zero_value)
        query = self.policy_query.unsqueeze(0).expand(fut_acts.shape[0], -1, -1)
        actions = self.policy_dec(self.policy(query, context))
        outputs = {"latents": latents, "rewards": rewards, "values": values, "actions": actions}
        if proprio_predictions is not None:
            outputs["proprio_predictions"] = proprio_predictions
        if visual_predictions is not None:
            outputs["visual_predictions"] = visual_predictions
        return outputs

    def forward(self, x: InputDict) -> dict[str, torch.Tensor]:
        context, latent = self.context(x)
        persistence = self.persistence_latents(x) if self.config.residual_dynamics else None
        current_joints = (self.scaler.scale(x)["proprio_obs_hist"][:, -1]
                          if self.config.joint_dynamics else None)
        current_visual = self.current_visual_features(x) if self.config.joint_conditioned_latents else None
        return self.predict(context, latent, self.scaler.scale({"fut_acts": x["fut_acts"]})["fut_acts"],
                            x["remaining_ticks_hist"][:, -1], persistence, current_joints, current_visual)

    @torch.no_grad()
    def inference(self, x: InputDict) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError("call model.eval() before deterministic planning")
        return self.scaler.unscale(self.forward(x))

    def encode_latent(self, joints, front, wrist, remaining_ticks):
        """Scaled measured joints + uint8 RGB target for latent consistency."""
        return self.encoder.encode_latent(joints, front, wrist, remaining_ticks)
