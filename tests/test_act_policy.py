"""Component 3 test: ACTPolicy wrapper — loss, action queue, temporal ensembling.

Runs on CPU with tiny inputs and no pretrained weights (offline-safe).
Standalone-runnable: `python tests/test_policy.py` (or via pytest).
"""

import torch

from so101.learning.act.config import ACTION, OBS_IMAGES, OBS_STATE, ACTConfig
from so101.learning.act.policy import ACTPolicy

B, CHUNK, STATE, ACT_DIM = 2, 8, 6, 6
H = W = 64
CAMS = ("observation.images.wrist", "observation.images.external")


def _cfg(**kw):
    base = dict(
        action_dim=ACT_DIM,
        robot_state_dim=STATE,
        image_keys=CAMS,
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        pretrained_backbone_weights=None,
    )
    base.update(kw)
    return ACTConfig(**base)


def _batch(cfg, with_action):
    """Batch with per-camera image keys (as the policy receives them)."""
    b = {OBS_STATE: torch.randn(B, STATE)}
    for key in cfg.image_keys:
        b[key] = torch.rand(B, 3, H, W)
    if with_action:
        b[ACTION] = torch.randn(B, CHUNK, ACT_DIM)
        b["action_is_pad"] = torch.zeros(B, CHUNK, dtype=torch.bool)
    return b


def test_forward_returns_loss_and_kld():
    cfg = _cfg()
    policy = ACTPolicy(cfg).train()
    loss, loss_dict = policy(_batch(cfg, with_action=True))
    assert loss.ndim == 0 and loss.requires_grad, loss
    assert "l1_loss" in loss_dict and "kld_loss" in loss_dict, loss_dict


def test_forward_no_vae_only_l1():
    cfg = _cfg(use_vae=False)
    policy = ACTPolicy(cfg).train()
    loss, loss_dict = policy(_batch(cfg, with_action=True))
    assert "l1_loss" in loss_dict and "kld_loss" not in loss_dict, loss_dict


def test_select_action_queue_shape_and_requery():
    cfg = _cfg(n_action_steps=4)
    policy = ACTPolicy(cfg)
    # First call fills the queue; each call returns one action.
    a0 = policy.select_action(_batch(cfg, with_action=False))
    assert a0.shape == (B, ACT_DIM), a0.shape
    # Consume the rest of the queued chunk without error.
    for _ in range(cfg.n_action_steps - 1):
        assert policy.select_action(_batch(cfg, with_action=False)).shape == (B, ACT_DIM)


def test_temporal_ensembler_select_action():
    cfg = _cfg(n_action_steps=1, temporal_ensemble_coeff=0.01)
    policy = ACTPolicy(cfg)
    assert hasattr(policy, "temporal_ensembler")
    action = policy.select_action(_batch(cfg, with_action=False))
    assert action.shape == (B, ACT_DIM), action.shape


def test_reset_clears_queue():
    cfg = _cfg(n_action_steps=4)
    policy = ACTPolicy(cfg)
    policy.select_action(_batch(cfg, with_action=False))
    assert len(policy._action_queue) == cfg.n_action_steps - 1
    policy.reset()
    assert len(policy._action_queue) == 0


def test_get_optim_params_separates_backbone():
    cfg = _cfg(optimizer_lr_backbone=1e-6)
    policy = ACTPolicy(cfg)
    groups = policy.get_optim_params()
    assert len(groups) == 2, groups
    assert groups[1]["lr"] == cfg.optimizer_lr_backbone


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    import sys

    sys.exit(_run())
