"""ACT policy wrapper (action queue, temporal ensembling, training loss).

Ported from LeRobot's `modeling_act.py`, decoupled from the
PreTrainedPolicy/HubMixin machinery so it is a plain `nn.Module`. The only
changes versus upstream:

    PreTrainedPolicy base        ->  plain nn.Module
    super().__init__(config)     ->  super().__init__() + self.config = config
    config.image_features (dict) ->  config.image_keys (tuple)

`select_action`, `predict_action_chunk`, `forward` (L1 + kl_weight * KL), and
`ACTTemporalEnsembler` are copied verbatim so behavior matches upstream.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from so101.learning.act.checkpoint import load_checkpoint
from so101.learning.act.config import ACTION, OBS_IMAGES, ACTConfig
from so101.learning.act.model import ACT


class ACTPolicy(nn.Module):
    """Action Chunking Transformer policy: wraps the `ACT` network for training
    (loss computation) and inference (action queue / temporal ensembling)."""

    def __init__(self, config: ACTConfig):
        super().__init__()
        config.validate_architecture()
        config.validate_features()
        self.config = config

        self.model = ACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None) -> "ACTPolicy":
        """Load a policy saved via `so101.learning.act.checkpoint.save_checkpoint` (the
        `{"model", "config", "step"}` torch.save format `scripts/train_act.py`
        writes). Returns a ready-to-use policy in eval mode on `device`."""
        saved = load_checkpoint(checkpoint_path, device=device)
        policy = cls(ACTConfig(**saved["config"]))
        if device is not None:
            policy = policy.to(device)
        policy.load_state_dict(saved["model"])
        policy.eval()
        return policy

    def get_optim_params(self) -> list[dict]:
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith(("model.backbone", "model.backbones.")) and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith(("model.backbone", "model.backbones.")) and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `predict_action_chunk` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `predict_action_chunk` when
        the queue is empty.
        """
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.has_images:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_keys]

        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.has_images:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_keys]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae and log_sigma_x2_hat is not None:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://huggingface.co/papers/1312.6114 for more details).
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[: i + 1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action
