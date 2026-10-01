"""DP model pieces: UNet, encoders, schedulers, loss, sampling, action queue, checkpoints (CPU)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from so101.learning.dp.config import ACTION, OBS_STATE, DPConfig
from so101.learning.dp.policy import (
    DDIMNoiseScheduler,
    DDPMNoiseScheduler,
    DiffusionPolicy,
    SwitchEMA,
    build_optimizers,
    cosine_with_warmup,
    load_checkpoint,
    save_checkpoint,
)
from so101.learning.dp.unet import ConditionalUnet1D
from so101.learning.dp.vision import FrontCameraTransform, ResnetEncoder, WristCameraTransform
from test_act_data import H, W
from test_dp_data import _limit_threads, tiny_config  # noqa: F401  (autouse fixture)

STATS = {
    OBS_STATE: {"min": [-2.0] * 6, "max": [2.0] * 6},
    ACTION: {"min": [-1.0] * 6, "max": [3.0] * 6},
}


def _batch(cfg, B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    batch = {
        OBS_STATE: torch.rand(B, cfg.obs_horizon, 6, generator=g) * 4 - 2,
        ACTION: torch.rand(B, cfg.pred_horizon, 6, generator=g) * 4 - 1,
    }
    for key in cfg.image_keys:
        batch[key] = torch.randint(0, 256, (B, cfg.obs_horizon, 3, H, W), generator=g, dtype=torch.uint8)
    return batch


def test_default_config_is_the_chosen_paper_configuration():
    cfg = DPConfig()
    cfg.validate()
    assert (cfg.obs_horizon, cfg.pred_horizon, cfg.action_horizon) == (1, 32, 8)
    assert cfg.down_dims == (256, 512, 1024) and cfg.diffusion_step_embed_dim == 256
    assert (cfg.kernel_size, cfg.n_groups) == (5, 8)
    assert (cfg.num_diffusion_iters, cfg.beta_schedule, cfg.prediction_type, cfg.clip_sample) == (100, "squaredcos_cap_v2", "epsilon", True)
    assert cfg.inference_steps == 16 and cfg.warmstart_timestep == 50 and cfg.ddim_eta == 0.0
    assert (cfg.feature_layernorm, cfg.front_camera_dropout, cfg.weight_decay) == (True, 0.1, 1e-3)
    assert (cfg.actor_lr, cfg.encoder_lr, cfg.warmup_steps, cfg.encoder_warmup_steps) == (1e-4, 1e-5, 2000, 50000)
    assert cfg.batch_size == 256 and not cfg.ema_use and cfg.projection_dim == 128
    assert cfg.cond_dim == 6 + 2 * 128
    assert DPConfig.from_dict(cfg.to_dict()) == cfg


def test_unet_shapes():
    net = ConditionalUnet1D(input_dim=6, global_cond_dim=10, diffusion_step_embed_dim=16, down_dims=(16, 32, 64))
    x = torch.randn(3, 8, 6)
    assert net(x, torch.tensor([1, 5, 9]), global_cond=torch.randn(3, 10)).shape == (3, 8, 6)
    assert net(x, 7, global_cond=torch.randn(3, 10)).shape == (3, 8, 6)
    assert net(x, torch.tensor(7), global_cond=torch.randn(3, 10)).shape == (3, 8, 6)


def test_resnet_groupnorm_replaces_batchnorm_and_outputs_512():
    enc = ResnetEncoder("resnet18", pretrained=False)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in enc.modules())
    groups = {m.num_channels: m.num_groups for m in enc.modules() if isinstance(m, nn.GroupNorm)}
    assert groups == {64: 4, 128: 8, 256: 16, 512: 32}  # num_features // 16
    assert enc(torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)).shape == (2, 512)


@pytest.mark.skipif(
    not (Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth").is_file(),
    reason="ImageNet resnet18 weights not cached",
)
def test_pretrained_keeps_imagenet_convs_but_groupnorm_is_fresh():
    import torchvision

    ref = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    enc = ResnetEncoder("resnet18", pretrained=True)
    torch.testing.assert_close(enc.model.conv1.weight, ref.conv1.weight)
    torch.testing.assert_close(enc.model.layer3[1].conv2.weight, ref.layer3[1].conv2.weight)
    gn = enc.model.bn1
    assert isinstance(gn, nn.GroupNorm)
    assert torch.all(gn.weight == 1) and torch.all(gn.bias == 0)


def test_camera_transforms_train_and_eval():
    front = FrontCameraTransform((240, 320), 224, 20)
    wrist = WristCameraTransform(224)
    x = torch.randint(0, 256, (5, 3, 240, 320), dtype=torch.uint8)
    for t in (front, wrist):
        for mode in (True, False):
            t.train(mode)
            y = t(x)
            assert y.shape == (5, 3, 224, 224) and y.dtype == torch.uint8
    front.eval()
    torch.testing.assert_close(front(x), x[:, :, 8:232, 48:272])  # CenterCrop 224
    with pytest.raises(AssertionError):
        front(torch.zeros(1, 3, 100, 100, dtype=torch.uint8))


def test_policy_train_eval_switches_augmentation():
    policy = DiffusionPolicy(tiny_config(), STATS)
    policy.train()
    assert all(t.training for t in policy.transforms)
    policy.eval()
    assert not any(m.training for m in policy.modules())
    no_aug = DiffusionPolicy(tiny_config(augment_image=False), STATS).train()
    assert not any(t.training for t in no_aug.transforms)


def test_parameter_groups_split_encoders_from_actor():
    policy = DiffusionPolicy(tiny_config(), STATS)
    enc = {id(p) for p in policy.encoder_parameters()}
    act = {id(p) for p in policy.actor_parameters()}
    assert enc and act and not enc & act
    assert len(enc) + len(act) == len(list(policy.parameters()))
    # projection + LayerNorm train at the actor LR, as in the paper.
    assert all(id(p) in act for p in policy.projections.parameters())
    assert all(id(p) in act for p in policy.layernorms.parameters())


def test_schedulers_match_diffusers():
    pytest.importorskip("diffusers")
    from diffusers import DDIMScheduler, DDPMScheduler

    kw = dict(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
    ref_ddpm, ref_ddim = DDPMScheduler(**kw), DDIMScheduler(**kw)
    ddpm, ddim = DDPMNoiseScheduler(100), DDIMNoiseScheduler(100, clip_sample=True)
    x, n = torch.randn(4, 8, 6), torch.randn(4, 8, 6)
    t = torch.tensor([0, 17, 50, 99])
    torch.testing.assert_close(ddpm.add_noise(x, n, t), ref_ddpm.add_noise(x, n, t))
    for steps in (4, 8, 16):
        ref_ddim.set_timesteps(steps)
        ddim.set_timesteps(steps)
        assert ddim.timesteps.tolist() == ref_ddim.timesteps.tolist()
        sample = torch.randn(4, 8, 6)
        for k in ddim.timesteps:
            eps = torch.randn(4, 8, 6)
            expected = ref_ddim.step(eps, k, sample, eta=0.0).prev_sample
            torch.testing.assert_close(ddim.step(eps, k, sample, eta=0.0), expected)
            sample = expected


def test_front_camera_dropout_zeroes_front_features_before_layernorm():
    cfg = tiny_config(front_camera_dropout=1.0 - 1e-9, feature_layernorm=True)
    policy = DiffusionPolicy(cfg, STATS).train()
    with torch.no_grad():
        policy.layernorms[1].bias.fill_(0.25)
        nobs = policy.encode_obs(torch.zeros(3, 1, 6), {k: v[:3] for k, v in _batch(cfg).items() if k in cfg.image_keys})
    e = cfg.encoding_dim
    front = nobs[:, 6 + e : 6 + 2 * e]  # image_keys order: wrist, front
    torch.testing.assert_close(front, torch.full_like(front, 0.25))  # LayerNorm(0) = bias
    assert not torch.allclose(nobs[:, 6 : 6 + e], torch.full_like(front, 0.25))


def test_loss_decreases_when_overfitting_a_tiny_batch():
    torch.manual_seed(0)
    cfg = tiny_config(augment_image=False, front_camera_dropout=0.0, warmup_steps=0, encoder_warmup_steps=0,
                      actor_lr=1e-3, encoder_lr=1e-3, weight_decay=0.0, lr_scheduler="constant")
    policy = DiffusionPolicy(cfg, STATS)
    # ResNet weight gradients dominate CPU time and are not what this checks;
    # the projections, LayerNorms and UNet still train.
    for p in policy.encoder_parameters():
        p.requires_grad_(False)
    batch = _batch(cfg, B=4)

    def fixed_loss():
        policy.eval()
        torch.manual_seed(123)
        with torch.no_grad():
            losses = [policy.compute_loss(batch)[0].item() for _ in range(8)]
        policy.train()
        return float(np.mean(losses))

    before = fixed_loss()
    optimizers = build_optimizers(policy, total_steps=150)
    assert [name for name, _, _ in optimizers] == ["actor"]
    policy.train()
    for _ in range(150):
        loss, _ = policy.compute_loss(batch)
        for _, opt, _ in optimizers:
            opt.zero_grad()
        loss.backward()
        for _, opt, sched in optimizers:
            opt.step()
            sched.step()
    after = fixed_loss()
    assert after < 0.85 * before, (before, after)


def test_cosine_schedule_matches_diffusers():
    pytest.importorskip("diffusers")
    from diffusers.optimization import get_scheduler

    opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    ref = get_scheduler("cosine", optimizer=opt, num_warmup_steps=10, num_training_steps=50)
    ours = cosine_with_warmup(10, 50)
    for step in range(60):
        assert ref.lr_lambdas[0](step) == pytest.approx(ours(step))


def test_ddim_sampling_shapes_and_warm_start_state():
    cfg = tiny_config(inference_steps=4)
    policy = DiffusionPolicy(cfg, STATS).eval()
    b = _batch(cfg, B=3)
    chunk = policy.predict_action_chunk(b[OBS_STATE], {k: b[k] for k in cfg.image_keys})
    assert chunk.shape == (3, cfg.pred_horizon, 6)
    # clip_sample keeps the normalized chunk in [-1, 1] -> within the action stats.
    assert chunk.min() >= -1 - 1e-5 and chunk.max() <= 3 + 1e-5
    assert policy.prev_naction.shape == (3, cfg.pred_horizon, 6)
    policy.reset()
    assert policy.prev_naction is None


def test_warm_start_uses_the_previous_plan_remainder():
    cfg = tiny_config(inference_steps=4, warmstart_timestep=50)
    policy = DiffusionPolicy(cfg, STATS).eval()
    seen = []
    original = policy.inference_noise_scheduler.add_noise
    policy.inference_noise_scheduler.add_noise = lambda x, n, t: seen.append((x.clone(), t.clone())) or original(x, n, t)
    nobs = torch.randn(2, cfg.cond_dim)
    first = policy._normalized_action(nobs)
    policy._normalized_action(nobs)
    assert torch.all(seen[0][0] == 0) and seen[0][1].tolist() == [50, 50]
    h = cfg.pred_horizon - cfg.action_horizon
    torch.testing.assert_close(seen[1][0][:, :h], first[:, cfg.action_horizon :])

    pure = DiffusionPolicy(tiny_config(inference_steps=4, warmstart_timestep=None), STATS).eval()
    pure.inference_noise_scheduler.add_noise = lambda *a: pytest.fail("pure-noise DDIM must not add_noise")
    assert pure._normalized_action(nobs).shape == (2, cfg.pred_horizon, 6)


def test_action_queue_replans_every_action_horizon_steps():
    cfg = tiny_config(inference_steps=2, action_horizon=3)
    policy = DiffusionPolicy(cfg, STATS).eval()
    calls = []
    original = policy.predict_action_chunk
    chunks = []

    def spy(state, images):
        calls.append(state.shape)
        chunks.append(original(state, images))
        return chunks[-1]

    policy.predict_action_chunk = spy
    b = _batch(cfg, B=2)
    images = {k: b[k][:, 0] for k in cfg.image_keys}
    outs = [policy.select_action(b[OBS_STATE][:, 0], images) for _ in range(7)]
    assert len(calls) == 3 and calls[0] == (2, 1, 6)
    for i, out in enumerate(outs):
        torch.testing.assert_close(out, chunks[i // 3][:, i % 3])
    policy.reset()
    policy.select_action(b[OBS_STATE][:, 0], images)
    assert len(calls) == 4


def test_obs_history_is_filled_by_repeating_the_first_observation():
    cfg = tiny_config(obs_horizon=2, inference_steps=2)
    policy = DiffusionPolicy(cfg, STATS).eval()
    seen = []
    policy.predict_action_chunk = lambda s, im: seen.append(s) or torch.zeros(1, cfg.pred_horizon, 6)
    b = _batch(cfg, B=1)
    policy.select_action(b[OBS_STATE][:, 0], {k: b[k][:, 0] for k in cfg.image_keys})
    torch.testing.assert_close(seen[0][:, 0], seen[0][:, 1])


def test_checkpoint_round_trip(tmp_path):
    cfg = tiny_config(ema_use=True)
    policy = DiffusionPolicy(cfg, STATS)
    ema = SwitchEMA(policy, 0.5)
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(1.0)
    ema.update()
    path = save_checkpoint(tmp_path / "ckpt.pt", policy, 7, ema_model=ema.state_dict())
    ckpt = load_checkpoint(path)
    assert ckpt["step"] == 7 and ckpt["stats"] == policy.normalizer.stats()
    raw = DiffusionPolicy.from_checkpoint(path, use_ema=False)
    for (n, a), b in zip(policy.state_dict().items(), raw.state_dict().values()):
        torch.testing.assert_close(a, b, msg=n)
    smoothed = DiffusionPolicy.from_checkpoint(path)
    name = "model.final_conv.1.bias"
    torch.testing.assert_close(smoothed.state_dict()[name], ema.shadow[name])
    assert not torch.allclose(smoothed.state_dict()[name], policy.state_dict()[name])
    torch.testing.assert_close(smoothed.normalizer.action_max, torch.full((6,), 3.0))
    assert raw.config.pred_horizon == cfg.pred_horizon and not raw.training


def test_random_gaussian_noise_is_per_image_and_keeps_dtype():
    from so101.learning.dp.vision import RandomGaussianNoise

    torch.manual_seed(0)
    x = torch.full((64, 3, 16, 16), 128, dtype=torch.uint8)
    y = RandomGaussianNoise(4.0)(x)
    assert y.dtype == torch.uint8 and y.shape == x.shape
    per_image = (y.float() - 128).flatten(1).std(dim=1)
    assert per_image.max() < 5.0 and per_image.max() > 2.0   # std drawn from U(0, 4)
    assert per_image.std() > 0.5                              # differs across images
    assert torch.equal(RandomGaussianNoise(0.0)(x), x)
