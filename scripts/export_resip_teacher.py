"""Export a trained ResiP checkpoint for the PyTorch teacher (so101.learning.resip_teacher).

The Isaac Lab environments here have no JAX, and JAX's CUDA wheels clash with
the torch CUDA libraries Isaac Lab ships, so the teacher is run through a
PyTorch port of its inference path instead. This script is the bridge, and is
run once, in any environment that has jax + flax (CPU is enough):

    <run>/params_<epoch>.pkl  ->  <out>/params.npz          base + residual actor weights
    <run>/agent_config.json   ->  <out>/agent_config.json
    <run>/normalization.json  ->  <out>/normalization.json
                                  <out>/schedule.npz        DDPM coefficients
                                  <out>/reference.npz       Flax outputs on fixed inputs
                                                            and fixed noise, which the
                                                            torch port must reproduce

Example:
    python scripts/export_resip_teacher.py \\
        --run sweep/learn_std/so101-StackCube-v0/resip/resip_sd042_20260925_142902 --epoch 2000
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from utils.diffusion_utils import ddpm_schedule  # noqa: E402
from utils.networks import DiffusionMLP, ResidualActor  # noqa: E402


def flatten(tree: dict, prefix: str = "") -> dict[str, np.ndarray]:
    out = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out.update(flatten(value, f"{prefix}{key}/"))
        else:
            out[f"{prefix}{key}"] = np.asarray(value, dtype=np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epoch", default="2000")
    parser.add_argument("--out", type=Path, default=None, help="default: outputs/teachers/<run name>_e<epoch>")
    parser.add_argument("--reference-batch", type=int, default=8)
    args = parser.parse_args()

    out = args.out or REPO_ROOT / "outputs/teachers" / f"{args.run.name}_e{args.epoch}"
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.run / "agent_config.json").read_text())
    if config["agent_name"] != "resip" or config["network_type"] != "mlp" or config["use_ddim"]:
        raise SystemExit("only ResiP with the DiffusionMLP base and DDPM sampling is supported")
    if config.get("encoder") is not None or config["layer_norm"] or config["actor_layer_norm"]:
        raise SystemExit("encoder / layer_norm variants are not ported")

    with (args.run / f"params_{args.epoch}.pkl").open("rb") as f:
        params = pickle.load(f)["agent"]["network"]["params"]
    base_params, actor_params = params["modules_base"], params["modules_actor"]
    np.savez(out / "params.npz", **{f"base/{k}": v for k, v in flatten(base_params).items()},
             **{f"actor/{k}": v for k, v in flatten(actor_params).items()})
    for name in ("agent_config.json", "normalization.json"):
        shutil.copy(args.run / name, out / name)

    # ---- reference outputs from the original Flax modules -------------------
    normalization = json.loads((args.run / "normalization.json").read_text())
    ob_dim = len(normalization["observation_mean"])
    action_dim = len(normalization["action_min"])
    horizon = config["horizon_steps"]
    steps = config["denoising_steps"]
    base = DiffusionMLP(
        hidden_dims=tuple(config["hidden_dims"]),
        time_step_embed_dim=config["time_step_embed_dim"],
        horizon_steps=horizon,
        action_dim=action_dim,
    )
    actor = ResidualActor(
        hidden_dims=tuple(config["actor_hidden_dims"]),
        action_dim=action_dim,
        init_log_std=config["init_log_std"],
        learn_std=config["learn_std"],
        action_head_std=config["action_head_std"],
    )
    schedule = {k: np.asarray(v, dtype=np.float32) for k, v in ddpm_schedule(
        steps, beta_schedule=config["beta_schedule"], cosine_s=config["cosine_s"]).items()}

    np.savez(out / "schedule.npz", **schedule)

    rng = np.random.default_rng(0)
    batch = args.reference_batch
    obs = rng.normal(size=(batch, ob_dim)).astype(np.float32)          # already normalized
    x_init = rng.normal(size=(batch, horizon, action_dim)).astype(np.float32)
    noises = np.clip(rng.normal(size=(steps, batch, horizon, action_dim)), -config["randn_clip_value"],
                     config["randn_clip_value"]).astype(np.float32)

    # Same recursion as ResiPAgent.sample_base_actions (DDPM branch), with the
    # per-step noise supplied instead of drawn so torch can replay it.
    x = jnp.asarray(x_init)
    first_eps = None
    for i in range(steps):
        t_index = steps - 1 - i
        t = jnp.full((batch,), t_index, dtype=jnp.float32)
        eps = base.apply({"params": base_params}, obs, x, t)
        if first_eps is None:
            first_eps = np.asarray(eps)
        x_recon = schedule["sqrt_recip_alphas_cumprod"][t_index] * x - schedule["sqrt_recipm1_alphas_cumprod"][t_index] * eps
        x_recon = jnp.clip(x_recon, -config["denoised_clip_value"], config["denoised_clip_value"])
        mu = schedule["ddpm_mu_coef1"][t_index] * x_recon + schedule["ddpm_mu_coef2"][t_index] * x
        std = 0.0 if t_index == 0 else max(float(np.exp(0.5 * schedule["ddpm_logvar_clipped"][t_index])), 1e-3)
        x = mu + std * noises[i]
    base_actions = np.asarray(x)
    residual_mean, _ = actor.apply({"params": actor_params},
                                   jnp.concatenate([obs, base_actions[:, 0]], axis=-1))
    np.savez(
        out / "reference.npz",
        obs=obs, x_init=x_init, noises=noises, first_eps=first_eps,
        base_actions=base_actions, residual_mean=np.asarray(residual_mean),
    )
    (out / "meta.json").write_text(json.dumps(
        {"source_run": str(args.run.resolve()), "epoch": args.epoch, "jax": jax.__version__}, indent=2))
    print(f"exported {args.run} epoch {args.epoch} -> {out}")


if __name__ == "__main__":
    main()
