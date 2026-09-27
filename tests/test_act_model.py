"""Component 2 test: ACT network forward-pass shapes, VAE branches, gradients.

Runs on CPU with tiny inputs and no pretrained weights (offline-safe).
Standalone-runnable: `python tests/test_model.py` (or via pytest).
"""

import torch

from so101.learning.act.config import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE, ACTConfig
from so101.learning.act.model import ACT

# Small, offline config: tiny chunk + no pretrained download.
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
    b = {OBS_STATE: torch.randn(B, STATE)}
    if cfg.has_images:
        b[OBS_IMAGES] = [torch.rand(B, 3, H, W) for _ in cfg.image_keys]
    if cfg.has_env_state:
        b[OBS_ENV_STATE] = torch.randn(B, cfg.env_state_dim)
    if with_action:
        b[ACTION] = torch.randn(B, CHUNK, ACT_DIM)
        b["action_is_pad"] = torch.zeros(B, CHUNK, dtype=torch.bool)
    return b


def test_train_forward_shapes_and_latent():
    cfg = _cfg()
    model = ACT(cfg).train()
    actions, (mu, log_sigma_x2) = model(_batch(cfg, with_action=True))
    assert actions.shape == (B, CHUNK, ACT_DIM), actions.shape
    # VAE active in training => latent params returned with shape (B, latent_dim).
    assert mu.shape == (B, cfg.latent_dim), mu.shape
    assert log_sigma_x2.shape == (B, cfg.latent_dim), log_sigma_x2.shape


def test_eval_forward_no_latent():
    cfg = _cfg()
    model = ACT(cfg).eval()
    actions, (mu, log_sigma_x2) = model(_batch(cfg, with_action=False))
    assert actions.shape == (B, CHUNK, ACT_DIM), actions.shape
    # Eval => latent forced to zeros, no PDF params.
    assert mu is None and log_sigma_x2 is None


def test_use_vae_false_never_returns_latent():
    cfg = _cfg(use_vae=False)
    model = ACT(cfg).train()
    actions, (mu, log_sigma_x2) = model(_batch(cfg, with_action=True))
    assert actions.shape == (B, CHUNK, ACT_DIM)
    assert mu is None and log_sigma_x2 is None
    # No VAE submodules built when disabled.
    assert not hasattr(model, "vae_encoder")


def test_env_state_only_no_images():
    cfg = _cfg(image_keys=(), env_state_dim=10)
    model = ACT(cfg).train()
    actions, (mu, _) = model(_batch(cfg, with_action=True))
    assert actions.shape == (B, CHUNK, ACT_DIM)
    assert mu.shape == (B, cfg.latent_dim)
    assert not hasattr(model, "backbone")


def test_gradients_flow():
    cfg = _cfg()
    model = ACT(cfg).train()
    actions, _ = model(_batch(cfg, with_action=True))
    actions.abs().mean().backward()
    # A representative trainable param on each major branch got a gradient.
    checked = ["action_head.weight", "encoder_latent_input_proj.weight", "backbone.conv1.weight"]
    grads = dict(model.named_parameters())
    for name in checked:
        assert grads[name].grad is not None, f"no grad for {name}"
        assert torch.isfinite(grads[name].grad).all(), f"non-finite grad for {name}"


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
