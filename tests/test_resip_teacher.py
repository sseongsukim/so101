"""The PyTorch ResiP teacher reproduces the Flax policy it was exported from."""

from pathlib import Path

import numpy as np
import pytest
import torch

from so101.learning.resip_teacher import ResiPTeacher

EXPORTS = sorted((Path(__file__).resolve().parents[1] / "outputs/teachers").glob("*/reference.npz"))
pytestmark = pytest.mark.skipif(not EXPORTS, reason="no teacher export (run scripts/export_resip_teacher.py)")


@pytest.fixture(scope="module", params=[p.parent for p in EXPORTS], ids=lambda p: p.name)
def export(request):
    teacher = ResiPTeacher(request.param, device="cpu")
    ref = {k: torch.from_numpy(v) for k, v in np.load(request.param / "reference.npz").items()}
    return teacher, ref


def test_base_network_matches_flax(export):
    teacher, ref = export
    t = torch.full((len(ref["obs"]),), float(teacher.steps - 1))
    eps = teacher.base(ref["obs"], ref["x_init"], t)
    torch.testing.assert_close(eps, ref["first_eps"], atol=1e-4, rtol=1e-4)


def test_ddpm_sampling_with_replayed_noise_matches_flax(export):
    teacher, ref = export
    base = teacher.sample_base(ref["obs"], x_init=ref["x_init"], noises=ref["noises"])
    torch.testing.assert_close(base, ref["base_actions"], atol=1e-4, rtol=1e-4)


def test_residual_mean_matches_flax(export):
    teacher, ref = export
    residual = teacher.residual(ref["obs"], ref["base_actions"][:, 0])
    torch.testing.assert_close(residual, ref["residual_mean"], atol=1e-4, rtol=1e-4)


def test_act_replans_every_inference_steps(export):
    teacher, _ = export
    calls = []
    original = teacher.sample_base
    teacher.sample_base = lambda *a, **k: calls.append(1) or original(*a, **k)
    teacher.reset()
    obs = torch.zeros(2, teacher.ob_dim)
    for _ in range(teacher.act_steps * 2 + 1):
        assert teacher.act(obs).shape == (2, teacher.action_dim)
    assert len(calls) == 3
    teacher.sample_base = original
