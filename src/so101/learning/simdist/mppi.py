"""Constrained native-policy-proposal MPPI in raw absolute joint radians.

Every candidate is projected before model rollout. History receives the actual
sent action that produced each observation. No hardware library is imported.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch


@dataclass
class MppiConfig:
    num_samples: int = 64
    iterations: int = 3
    num_elites: int = 16
    mixture_coef: float = 0.5
    init_std_rad: float = 0.10
    min_std_rad: float = 0.01
    policy_std_rad: float = 0.03
    max_policy_deviation_rad: float = 0.25
    temperature: float = 1.0
    value_weight: float = 1.0
    seed: int = 42

    def __post_init__(self):
        if not 1 <= self.num_elites <= self.num_samples or self.iterations < 1:
            raise ValueError("invalid samples/elites/iterations")
        if not 0 < self.mixture_coef <= 1 or self.temperature <= 0:
            raise ValueError("positive policy proposal fraction and temperature required")
        if min(self.init_std_rad, self.min_std_rad, self.policy_std_rad, self.max_policy_deviation_rad) <= 0:
            raise ValueError("noise scales must be positive radians")


def project_actions(actions, previous, lower, upper, max_step):
    """Sequential clipping in candidate rollouts, from actual previous command."""
    if not all(torch.isfinite(v).all() for v in (actions, previous, lower, upper, max_step)):
        raise ValueError("actions and constraints must be finite")
    if (lower >= upper).any() or (max_step <= 0).any():
        raise ValueError("invalid limits or per-tick rate limits")
    if ((previous < lower) | (previous > upper)).any():
        raise ValueError("previous sent command outside limits")
    result, last = [], previous
    for t in range(actions.shape[-2]):
        candidate = actions[..., t, :].clamp(lower, upper)
        candidate = torch.maximum(torch.minimum(candidate, last + max_step), last - max_step)
        result.append(candidate)
        last = candidate
    return torch.stack(result, dim=-2)


def discounted_return(rewards, values, gamma, value_weight):
    T = rewards.shape[-1]
    discount = gamma ** torch.arange(T, device=rewards.device, dtype=rewards.dtype)
    # values[-1] = V(s[t+T]), hence exponent T, not T+1.
    return (rewards * discount).sum(-1) + value_weight * gamma ** T * values[..., -1]


def shift_plan(previous, policy, consumed):
    """Preserve unconsumed plan prefix and refill its tail from current policy."""
    shift = min(max(int(consumed), 0), policy.shape[0])
    return torch.cat((previous[shift:], policy[policy.shape[0]-shift:]), 0) if shift else previous.clone()


def project_policy_region(actions, previous, lower, upper, max_step, policy, radius):
    """Intersect candidate limits/rates with a bounded region of policy proposals."""
    result, last = [], previous
    for t in range(actions.shape[-2]):
        low = torch.maximum(torch.maximum(lower, policy[t] - radius), last - max_step)
        high = torch.minimum(torch.minimum(upper, policy[t] + radius), last + max_step)
        if (low > high + 1e-6).any():
            raise RuntimeError("empty policy/limit/rate intersection")
        candidate = torch.maximum(torch.minimum(actions[..., t, :], high), low)
        result.append(candidate); last = candidate
    return torch.stack(result, dim=-2)


class MppiController:
    def __init__(self, model, lower, upper, max_velocity_rad_s, config=None):
        self.model = model.eval()
        self.config = config or MppiConfig()
        self.device = next(model.parameters()).device
        self.H, self.T = model.config.history_length, model.config.prediction_length
        self.lower = torch.as_tensor(lower, dtype=torch.float32, device=self.device)
        self.upper = torch.as_tensor(upper, dtype=torch.float32, device=self.device)
        self.step = torch.as_tensor(max_velocity_rad_s, dtype=torch.float32, device=self.device) * model.config.tick_seconds
        if any(v.shape != (6,) for v in (self.lower, self.upper, self.step)):
            raise ValueError("six ordered radians limits/velocities required")
        self.rng = torch.Generator(device=self.device).manual_seed(self.config.seed)
        self._obs = deque(maxlen=self.H + 1)
        self._acts = deque(maxlen=self.H)
        self._mean = None
        self._since_plan = 0
        self.last_report = {}

    @property
    def ready(self):
        return len(self._obs) == self.H + 1 and len(self._acts) == self.H

    def reset(self, observation, previous_sent_action):
        self._obs.clear(); self._acts.clear()
        self._mean = None; self._since_plan = 0
        self.previous = self._action(previous_sent_action)
        obs = self._observation(observation)
        if obs["remaining_ticks"] > self.model.config.task_horizon_ticks:
            raise ValueError("observation task clock exceeds model horizon")
        self._obs.append(obs)

    def update(self, observation, sent_action):
        obs = self._observation(observation)
        if obs["remaining_ticks"] > self.model.config.task_horizon_ticks:
            raise ValueError("observation task clock exceeds model horizon")
        if not self._obs:
            raise RuntimeError("reset controller first")
        dt = obs["timestamp_seconds"] - self._obs[-1]["timestamp_seconds"]
        if not np.isclose(dt, self.model.config.tick_seconds, rtol=0.01, atol=0.0005):
            raise ValueError(f"history timestamp gap {dt:.6f}s; resample timing or reset")
        if obs["remaining_ticks"] != self._obs[-1]["remaining_ticks"] - 1:
            raise ValueError("remaining task clock must decrement by one real tick")
        self.previous = self._action(sent_action)
        self._acts.append(self.previous.cpu().numpy().copy())
        self._obs.append(obs)
        self._since_plan += 1

    def _action(self, value):
        action = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if action.shape != (6,) or not torch.isfinite(action).all():
            raise ValueError("sent action must be finite six radians")
        project_actions(action.unsqueeze(0), action, self.lower, self.upper, self.step)
        return action.clone()

    @staticmethod
    def _observation(value):
        result = {k: np.asarray(value[k]).copy() for k in ("joints", "front", "wrist")}
        if result["joints"].shape != (6,) or not np.isfinite(result["joints"]).all():
            raise ValueError("measured joints must be six finite radians")
        for key in ("front", "wrist"):
            if result[key].dtype != np.uint8 or result[key].ndim != 3 or result[key].shape[-1] != 3:
                raise ValueError(f"{key} must be uint8 HWC RGB")
        remaining = value["remaining_ticks"]
        if not np.isfinite(remaining) or int(remaining) != remaining or remaining < 0:
            raise ValueError("remaining_ticks must be a nonnegative integer")
        result["remaining_ticks"] = int(remaining)
        result["timestamp_seconds"] = float(value["timestamp_seconds"])
        if not np.isfinite(result["timestamp_seconds"]):
            raise ValueError("finite observation timestamp required")
        return result

    def _inputs(self):
        if not self.ready:
            raise RuntimeError("collect H real hold-action ticks after reset; history is never fabricated")
        def stack(key):
            return torch.as_tensor(np.stack([o[key] for o in self._obs]), device=self.device).unsqueeze(0)
        return {"remaining_ticks_hist": stack("remaining_ticks").unsqueeze(-1),
                "proprio_obs_hist": stack("joints").float(), "front_hist": stack("front"),
                "wrist_hist": stack("wrist"),
                "acts_hist": torch.as_tensor(np.stack(self._acts), device=self.device).unsqueeze(0).float()}

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def plan(self, policy_only=False, proposal_chunk=None):
        self._sync(); started = perf_counter()
        cfg = self.config
        if self._obs[-1]["remaining_ticks"] < 1:
            raise RuntimeError("finite task horizon finished; reset before planning")
        inputs = self._inputs()
        remaining = inputs["remaining_ticks_hist"][:, -1]
        context, latent = self.model.context(inputs)
        persistence = self.model.persistence_latents(inputs) if self.model.config.residual_dynamics else None
        current_joints = (self.model.scaler.scale({"proprio_obs_hist": inputs["proprio_obs_hist"]})[
            "proprio_obs_hist"][:, -1] if self.model.config.joint_dynamics else None)
        current_visual = (self.model.current_visual_features(inputs)
                          if self.model.config.joint_conditioned_latents else None)
        zeros = torch.zeros(1, self.T, 6, device=self.device)
        proposal_source = "world_model_policy"
        plan_score_diagnostics = None
        if proposal_chunk is not None:
            proposal = torch.as_tensor(proposal_chunk, dtype=torch.float32, device=self.device)
            expected = (self.T, self.model.config.action_dim)
            if tuple(proposal.shape) != expected or not torch.isfinite(proposal).all():
                raise ValueError(f"proposal_chunk must be finite with shape {expected}")
            policy = proposal
            proposal_source = "external_action_chunk"
        else:
            if persistence is None and current_joints is None and current_visual is None:
                policy_outputs = self.model.predict(context, latent, zeros, remaining)
            else:
                policy_outputs = self.model.predict(context, latent, zeros, remaining, persistence,
                                                    current_joints, current_visual)
            policy = self.model.scaler.unscale(policy_outputs)["actions"][0]
        project = lambda a: project_actions(a, self.previous, self.lower, self.upper, self.step)
        policy = project(policy)
        trust_project = lambda a: project_policy_region(a, self.previous, self.lower, self.upper,
                                                        self.step, policy, cfg.max_policy_deviation_rad)
        if policy_only:
            mean = policy
        else:
            if self._mean is None:
                mean = policy.clone()
            else:
                mean = trust_project(shift_plan(self._mean, policy, self._since_plan))
            std = torch.full_like(mean, cfg.init_std_rad)
            K = cfg.num_samples
            base_count = max(1, int(K * cfg.mixture_coef))
            ctx = context.expand(K, -1, -1)
            z = latent.expand(K, -1)
            for _ in range(cfg.iterations):
                noise = torch.randn(K, self.T, 6, generator=self.rng, device=self.device)
                candidates = mean + noise * std
                candidates[:base_count] = policy + noise[:base_count] * cfg.policy_std_rad
                candidates[0] = policy  # exact policy proposal always included
                candidates = trust_project(candidates)
                scaled = self.model.scaler.scale({"fut_acts": candidates})["fut_acts"]
                candidate_persistence = persistence.expand(K, -1, -1) if persistence is not None else None
                candidate_joints = current_joints.expand(K, -1) if current_joints is not None else None
                candidate_visual = ({name: feature.expand(K, -1) for name, feature in current_visual.items()}
                                    if current_visual is not None else None)
                if candidate_persistence is None and candidate_joints is None and candidate_visual is None:
                    predicted = self.model.predict(ctx, z, scaled, remaining)
                else:
                    predicted = self.model.predict(ctx, z, scaled, remaining,
                                                   candidate_persistence, candidate_joints, candidate_visual)
                outputs = self.model.scaler.unscale(predicted)
                scores = discounted_return(outputs["rewards"], outputs["values"],
                                           self.model.config.gamma, cfg.value_weight)
                if not torch.isfinite(scores).all():
                    raise RuntimeError("non-finite predicted returns")
                elite_scores, indices = scores.topk(cfg.num_elites)
                best_index = scores.argmax()
                plan_score_diagnostics = {
                    "proposal_score": scores[0].detach(),
                    "best_candidate_score": scores[best_index].detach(),
                    "candidate_score_gain": (scores[best_index] - scores[0]).detach(),
                    "best_candidate_index": best_index.detach(),
                    "best_candidate_deviation_rad": (candidates[best_index] - policy).abs().max().detach(),
                }
                elites = candidates[indices]
                weights = torch.softmax((elite_scores - elite_scores.max()) / cfg.temperature, 0)[:, None, None]
                mean = trust_project((weights * elites).sum(0))
                std = ((weights * (elites - mean).square()).sum(0)).sqrt().clamp(cfg.min_std_rad, cfg.init_std_rad)
        self._mean = mean.clone(); self._since_plan = 0
        self._sync()
        if plan_score_diagnostics is not None:
            plan_score_diagnostics = {key: int(value) if key.endswith("index") else float(value)
                                      if torch.is_tensor(value) else value
                                      for key, value in plan_score_diagnostics.items()}
        duration = perf_counter() - started
        self.last_report = {"latency_seconds": duration, "tick_seconds": self.model.config.tick_seconds,
                            "within_tick_budget": duration <= self.model.config.tick_seconds,
                            "source_timestamp_seconds": self._obs[-1]["timestamp_seconds"],
                            "history_count": len(self._obs), "value_weight": cfg.value_weight,
                            "remaining_ticks": self._obs[-1]["remaining_ticks"],
                            "valid_plan_ticks": min(self.T, self._obs[-1]["remaining_ticks"]),
                            "max_policy_deviation_rad": cfg.max_policy_deviation_rad,
                            "proposal_source": proposal_source,
                            "model_score_diagnostics": plan_score_diagnostics,
                            "policy_only": policy_only}
        return mean.cpu().numpy()
