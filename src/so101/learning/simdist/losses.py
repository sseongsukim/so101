"""Native latent/reward/value/expert-policy losses with censored-value masks."""
import torch


def world_model_loss(model, x, y):
    pred = model(x)
    scaled = model.scaler.scale(y)
    with torch.no_grad():
        target = model.encode_latent(scaled["proprio_obs"], y["front"], y["wrist"], y["next_remaining_ticks"])
    dynamics = (pred["latents"] - target).square().mean()
    reward_mask = y["reward_valid"].float()
    reward = ((pred["rewards"] - scaled["rewards"]).square() * reward_mask).sum() / reward_mask.sum().clamp_min(1)
    valid = y["value_valid"].float()
    value = ((pred["values"] - scaled["values"]).square() * valid).sum() / valid.sum().clamp_min(1)
    expert = torch.cumprod(y["exp_pol_flags"].float(), dim=-1).unsqueeze(-1)
    action = ((pred["actions"] - scaled["actions"]).square() * expert).sum() / (expert.sum() * 6).clamp_min(1)
    raw = {"latent_dynamics": dynamics, "reward": reward, "value": value, "action": action}
    if model.config.joint_dynamics:
        raw["joint_dynamics"] = (pred["proprio_predictions"] - scaled["proprio_obs"]).square().mean()
    if model.config.visual_dynamics:
        with torch.no_grad():
            target_front = model.encoder.front(y["front"])
            target_wrist = model.encoder.wrist(y["wrist"])
        raw["visual_dynamics"] = 0.5 * (
            (pred["visual_predictions"]["front"] - target_front).square().mean()
            + (pred["visual_predictions"]["wrist"] - target_wrist).square().mean()
        )
    terms = {k: v * model.config.loss_weights[k] for k, v in raw.items()}
    return sum(terms.values()), terms
