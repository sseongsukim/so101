"""Component 1 test: ACTConfig defaults, derived flags, and validation.

Standalone-runnable: `python tests/test_config.py` (or via pytest).
"""

from so101.learning.act.config import ACTConfig


def test_defaults_match_upstream():
    c = ACTConfig()
    assert c.chunk_size == 100
    assert c.n_action_steps == 100
    assert c.dim_model == 512
    assert c.n_heads == 8
    assert c.n_encoder_layers == 4
    assert c.n_decoder_layers == 1
    assert c.use_vae is True
    assert c.latent_dim == 32
    assert c.kl_weight == 10.0


def test_presence_flags():
    c = ACTConfig(action_dim=6, robot_state_dim=6, image_keys=("observation.images.wrist",))
    assert c.has_images is True
    assert c.has_robot_state is True
    assert c.has_env_state is False

    c2 = ACTConfig(action_dim=4, robot_state_dim=None, env_state_dim=8, image_keys=())
    assert c2.has_images is False
    assert c2.has_robot_state is False
    assert c2.has_env_state is True


def test_requires_image_or_env_state():
    # bare config is valid (architecture defaults ok); feature presence is
    # only enforced by validate_features(), matching upstream.
    ACTConfig()
    try:
        ACTConfig(image_keys=(), env_state_dim=None).validate_features()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when no images and no env_state")


def test_n_action_steps_le_chunk():
    try:
        ACTConfig(chunk_size=10, n_action_steps=20, image_keys=("cam",))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when n_action_steps > chunk_size")


def test_temporal_ensemble_requires_single_step():
    try:
        ACTConfig(temporal_ensemble_coeff=0.01, n_action_steps=100, image_keys=("cam",))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError for ensemble + n_action_steps>1")

    # allowed when n_action_steps == 1
    ACTConfig(temporal_ensemble_coeff=0.01, n_action_steps=1, image_keys=("cam",))


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
