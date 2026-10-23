"""CPU semantic regressions: actual timing, RGB, causality, masking, constraints."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from so101.learning.simdist.config import WorldModelConfig
from so101.learning.simdist.dataset import VisualWindowDataset, split_episodes, split_episode_groups
from so101.learning.simdist.losses import world_model_loss
from so101.learning.simdist.model import WorldModel
from so101.learning.simdist.mppi import MppiController, MppiConfig, discounted_return, project_actions, shift_plan, project_policy_region
from so101.learning.simdist.recording import TransitionRecorder


def small_config():
    return WorldModelConfig(history_length=2, prediction_length=3, image_size=16, stored_image_shape=(16,16), latent_dim=16, task_horizon_ticks=8,
                            mlp_dropout=0, attention_dropout=0, dyn_layers=1, policy_attn_layers=1)


def synthetic_data():
    N, L = 24, 8
    joints = np.arange(N*6, dtype=np.float32).reshape(N, 6) * 0.001
    rgb = np.arange(N*16*16*3, dtype=np.uint8).reshape(N, 16, 16, 3)
    rgb = (rgb.astype(np.uint16) + np.arange(N)[:,None,None,None] * 7).astype(np.uint8)
    term = np.zeros(N, bool); term[L-1::L] = True
    rewards = np.linspace(0, 1, N, dtype=np.float32)
    values = np.zeros(N, dtype=np.float32)
    for start in range(0, N, L):
        carry = 0.0
        for i in range(start+L-1, start-1, -1):
            values[i] = carry
            carry = float(rewards[i]) + 0.99 * carry
    return {"observations": joints, "next_observations": np.roll(joints, -1, axis=0), "actions": joints + 0.001,
            "front_images": rgb, "wrist_images": rgb[::-1].copy(), "next_front_images": np.roll(rgb, -1, axis=0),
            "next_wrist_images": np.roll(rgb[::-1].copy(), -1, axis=0), "rewards": rewards,
            "values": values, "value_valid": np.ones(N, bool),
            "expert_policy_flag": np.ones(N, bool), "terminated": term, "truncated": np.zeros(N, bool),
            "reward_valid": np.ones(N, bool), "reward_semantics": np.full(N, "dense_environment_reward"),
            "episode_starts": np.arange(0, N, L), "episode_lengths": np.full(3, L),
            "gamma": np.array(0.99), "tick_seconds": np.full(N, 1/30),
            "ticks": np.tile(np.arange(L),3), "remaining_ticks": np.tile(L-np.arange(L),3),
            "next_remaining_ticks": np.tile(L-np.arange(L)-1,3),
            "task_horizon_ticks": np.full(N,L), "objective_terminal": term.copy(),
            "value_objective": np.array("finite_horizon_dense_mc_v2"),
            "episode_seeds": np.array([10, 10, 11]), "episode_env_indices": np.array([0, 1, 0]),
            "timestamps_seconds": np.tile(np.arange(L, dtype=np.float32)/30, 3)}


class SimdistSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.cfg = small_config()
        self.data = synthetic_data()
        self.ds = VisualWindowDataset(self.data, self.cfg, [0, 1])
        self.model = WorldModel(self.cfg, self.ds.stats()).eval()
        batch = self.ds[0]
        self.x = {k: v.unsqueeze(0) for k,v in batch["model_in"].items()}
        self.y = {k: v.unsqueeze(0) for k,v in batch["labels"].items()}

    def test_future_targets_are_actual_successors(self):
        np.testing.assert_allclose(self.y["proprio_obs"][0], self.data["next_observations"][2:5])
        np.testing.assert_array_equal(self.y["front"][0], self.data["next_front_images"][2:5])
        train, val = split_episodes(self.data, seed=3)
        self.assertFalse(set(train) & set(val))
        self.assertFalse(set(self.data["episode_seeds"][train]) & set(self.data["episode_seeds"][val]))
        three_seed_data = dict(self.data, episode_seeds=np.array([10, 11, 12]))
        train3, val3, calibration3 = split_episode_groups(
            three_seed_data, seed=3, validation_fraction=0.2, calibration_fraction=0.2)
        parts = [set(three_seed_data["episode_seeds"][part]) for part in (train3, val3, calibration3)]
        self.assertTrue(all(parts[i].isdisjoint(parts[j]) for i in range(3) for j in range(i+1,3)))
        for t in self.ds.windows:
            self.assertEqual((t-self.cfg.history_length)//8, (t+self.cfg.prediction_length-1)//8)

    def test_pixels_and_actual_sent_history_affect_context(self):
        c, _ = self.model.context(self.x)
        altered = dict(self.x); altered["front_hist"] = torch.zeros_like(self.x["front_hist"])
        changed, _ = self.model.context(altered)
        self.assertGreater(float((c-changed).abs().max()), 1e-5)

        altered = dict(self.x); altered["acts_hist"] = self.x["acts_hist"] + 0.1
        changed, _ = self.model.context(altered)
        self.assertGreater(float((c-changed).abs().max()), 1e-5)

    def test_identical_pixels_with_different_remaining_time_change_context(self):
        c, _ = self.model.context(self.x)
        altered = dict(self.x); altered["remaining_ticks_hist"] = self.x["remaining_ticks_hist"] - 1
        changed, _ = self.model.context(altered)
        self.assertGreater(float((c-changed).abs().max()), 1e-5)

    def test_residual_variant_initializes_exactly_at_time_aware_persistence(self):
        from so101.learning.simdist.dynamics_audit import time_aware_persistence
        cfg = small_config()
        cfg.residual_dynamics = True
        model = WorldModel(cfg, self.ds.stats()).eval()
        expected = model.persistence_latents(self.x)
        audit_baseline = time_aware_persistence(model, self.x, self.y["next_remaining_ticks"])
        torch.testing.assert_close(expected, audit_baseline, atol=1e-6, rtol=0)
        actual = model.inference(self.x)["latents"]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
        self.assertEqual(tuple(actual.shape), (1, cfg.prediction_length, cfg.latent_dim))
        context, latent = model.context(self.x)
        actions = model.scaler.scale({"fut_acts": self.x["fut_acts"]})["fut_acts"]
        remaining = self.x["remaining_ticks_hist"][:, -1]
        with self.assertRaisesRegex(ValueError, "requires time-aware persistence"):
            model.predict(context, latent, actions, remaining)
        with self.assertRaisesRegex(ValueError, "must have shape"):
            model.predict(context, latent, actions, remaining, expected[:, :-1])

    def test_residual_head_has_finite_first_step_gradient(self):
        cfg = small_config()
        cfg.residual_dynamics = True
        model = WorldModel(cfg, self.ds.stats()).train()
        loss, _ = world_model_loss(model, self.x, self.y)
        loss.backward()
        grad = model.dynamics_residual.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.norm()), 0.0)

    def test_joint_dynamics_head_starts_at_current_joint_persistence(self):
        cfg = small_config()
        cfg.joint_dynamics = True
        model = WorldModel(cfg, self.ds.stats()).eval()
        current = model.scaler.scale(self.x)["proprio_obs_hist"][:, -1]
        predicted = model.inference(self.x)["proprio_predictions"]
        expected = model.scaler.unscale({
            "proprio_predictions": current[:, None, :].expand(-1, cfg.prediction_length, -1)
        })["proprio_predictions"]
        torch.testing.assert_close(predicted, expected, atol=1e-6, rtol=0)
        context, latent = model.context(self.x)
        actions = model.scaler.scale({"fut_acts": self.x["fut_acts"]})["fut_acts"]
        remaining = self.x["remaining_ticks_hist"][:, -1]
        with self.assertRaisesRegex(ValueError, "requires current_joints"):
            model.predict(context, latent, actions, remaining)
        with self.assertRaisesRegex(ValueError, "current_joints must have shape"):
            model.predict(context, latent, actions, remaining, None, current[:, :3])

    def test_joint_head_has_finite_first_step_gradient_and_mppi_path(self):
        cfg = small_config()
        cfg.joint_dynamics = True
        model = WorldModel(cfg, self.ds.stats()).train()
        loss, _ = world_model_loss(model, self.x, self.y)
        loss.backward()
        grad = model.joint_residual.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.norm()), 0.0)
        model.eval()
        def obs(i):
            return {"joints": self.data["observations"][i], "front": self.data["front_images"][i],
                    "wrist": self.data["wrist_images"][i], "timestamp_seconds": i/30,
                    "remaining_ticks": 8-i}
        ctrl = MppiController(model, [-1]*6, [1]*6, [3]*6,
                              MppiConfig(num_samples=4, num_elites=2, iterations=1))
        ctrl.reset(obs(0), [0]*6)
        ctrl.update(obs(1), self.data["actions"][1])
        ctrl.update(obs(2), self.data["actions"][2])
        plan = ctrl.plan()
        self.assertEqual(plan.shape, (cfg.prediction_length, cfg.action_dim))
        self.assertTrue(np.isfinite(plan).all())

    def test_joint_conditioned_latents_match_persistence_at_zero_correction(self):
        cfg = small_config()
        cfg.joint_dynamics = True
        cfg.joint_conditioned_latents = True
        model = WorldModel(cfg, self.ds.stats()).eval()
        torch.testing.assert_close(model.inference(self.x)["latents"], model.persistence_latents(self.x),
                                   atol=1e-6, rtol=0)
        context, latent = model.context(self.x)
        acts = model.scaler.scale({"fut_acts": self.x["fut_acts"]})["fut_acts"]
        joints = model.scaler.scale(self.x)["proprio_obs_hist"][:, -1]
        features = model.current_visual_features(self.x)
        with self.assertRaisesRegex(ValueError, "current front/wrist features"):
            model.predict(context, latent, acts, self.x["remaining_ticks_hist"][:, -1],
                          current_joints=joints)
        with self.assertRaisesRegex(ValueError, "current front features"):
            model.predict(context, latent, acts, self.x["remaining_ticks_hist"][:, -1],
                          current_joints=joints,
                          current_visual_features={"front": features["front"][:, :3], "wrist": features["wrist"]})

    def test_joint_conditioned_latents_use_only_current_visuals_and_expand_in_mppi(self):
        cfg = small_config()
        cfg.joint_dynamics = True
        cfg.joint_conditioned_latents = True
        model = WorldModel(cfg, self.ds.stats()).eval()
        # Prediction API receives only current RGB features; future label tensors
        # are outside this path and can be varied without changing the forecast.
        baseline = model.inference(self.x)["latents"]
        changed_labels = {k: v.clone() for k, v in self.y.items()}
        changed_labels["front"].fill_(255)
        changed_labels["wrist"].zero_()
        changed_labels["proprio_obs"].add_(99)
        self.assertFalse(torch.equal(changed_labels["front"], self.y["front"]))
        self.assertFalse(torch.equal(changed_labels["proprio_obs"], self.y["proprio_obs"]))
        torch.testing.assert_close(model.inference(self.x)["latents"], baseline)
        context, latent = model.context(self.x)
        acts = model.scaler.scale({"fut_acts": self.x["fut_acts"]})["fut_acts"]
        joints = model.scaler.scale(self.x)["proprio_obs_hist"][:, -1]
        features = model.current_visual_features(self.x)
        pred = model.predict(context, latent, acts, self.x["remaining_ticks_hist"][:, -1],
                             current_joints=joints, current_visual_features=features)["latents"]
        K = 4
        candidate_features = {k: v.expand(K, -1) for k, v in features.items()}
        batched = model.predict(context.expand(K, -1, -1), latent.expand(K, -1), acts.expand(K, -1, -1),
                                self.x["remaining_ticks_hist"][:, -1].expand(K, -1),
                                current_joints=joints.expand(K, -1), current_visual_features=candidate_features)["latents"]
        torch.testing.assert_close(batched, pred.expand(K, -1, -1))
        ctrl = MppiController(model, [-1]*6, [1]*6, [3]*6,
                              MppiConfig(num_samples=4, num_elites=2, iterations=1))
        def obs(i):
            return {"joints": self.data["observations"][i], "front": self.data["front_images"][i],
                    "wrist": self.data["wrist_images"][i], "timestamp_seconds": i/30,
                    "remaining_ticks": 8-i}
        ctrl.reset(obs(0), [0]*6)
        ctrl.update(obs(1), self.data["actions"][1]); ctrl.update(obs(2), self.data["actions"][2])
        self.assertTrue(np.isfinite(ctrl.plan()).all())

    def test_zero_visual_delta_is_identity_and_frozen_target_has_visual_gradient(self):
        cfg = small_config()
        cfg.joint_dynamics = True
        cfg.joint_conditioned_latents = True
        baseline = WorldModel(cfg, self.ds.stats()).eval()
        cfg.visual_dynamics = True
        model = WorldModel(cfg, self.ds.stats()).eval()
        model.load_state_dict({k: v for k, v in baseline.state_dict().items()}, strict=False)
        model.visual_residual.load_state_dict({"weight": torch.zeros_like(model.visual_residual.weight),
                                               "bias": torch.zeros_like(model.visual_residual.bias)})
        torch.testing.assert_close(model(self.x)["latents"], baseline(self.x)["latents"], atol=1e-6, rtol=0)
        model.freeze_visual_forecaster()
        fixed = {name: p.detach().clone() for name, p in model.named_parameters()
                 if not name.startswith("visual_residual.")}
        model.train()
        self.assertFalse(model.dynamics.training)
        loss, _ = world_model_loss(model, self.x, self.y)
        loss.backward()
        grad = model.visual_residual.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.norm()), 0.0)
        for name, parameter in model.named_parameters():
            if name in fixed:
                torch.testing.assert_close(parameter, fixed[name], atol=0, rtol=0)

    def test_collector_frontend_resize_is_required_for_live_rgb(self):
        from so101.learning.act.data import to_uint8_rgb
        source = torch.arange(32*40*3, dtype=torch.int64).remainder(256).byte().reshape(32,40,3)
        recorded = to_uint8_rgb(source, self.cfg.stored_image_shape)
        live = to_uint8_rgb(source.float()/255, self.cfg.stored_image_shape)
        np.testing.assert_array_equal(recorded, live)
        # Live front-end and recorded frames must produce identical features.
        encoded_recorded = self.model.encoder.front(torch.tensor(recorded)[None])
        encoded_live = self.model.encoder.front(torch.tensor(live)[None])
        torch.testing.assert_close(encoded_recorded, encoded_live)
        with self.assertRaisesRegex(ValueError, "collector preprocessing"):
            self.model.encoder.front(source[None])

    def test_horizon_successor_and_extra_rollout_have_zero_raw_value_reward(self):
        x = dict(self.x); x["remaining_ticks_hist"] = torch.tensor([[[3], [2], [1]]])
        raw = self.model.inference(x)
        torch.testing.assert_close(raw["values"], torch.zeros_like(raw["values"]), atol=1e-6, rtol=0)
        torch.testing.assert_close(raw["rewards"][:,1:], torch.zeros_like(raw["rewards"][:,1:]), atol=1e-6, rtol=0)
        predicted = discounted_return(raw["rewards"], raw["values"], self.cfg.gamma, 1)
        torch.testing.assert_close(predicted, raw["rewards"][:,0], atol=1e-6, rtol=0)

    def test_mislabeled_time_and_value_are_rejected(self):
        d = {k: v.copy() for k,v in self.data.items()}
        d["next_remaining_ticks"][0] += 1
        with self.assertRaisesRegex(ValueError, "successor remaining"):
            VisualWindowDataset(d, self.cfg, [0])
        d = {k: v.copy() for k,v in self.data.items()}
        d["values"][2] += 1
        with self.assertRaisesRegex(ValueError, "successor MC return"):
            VisualWindowDataset(d, self.cfg, [0])

    def test_heldout_episode_never_changes_normalization(self):
        expected = self.ds.stats()
        d = {k: v.copy() for k,v in self.data.items()}
        d["observations"][16:] += 100
        d["next_observations"][16:] += 100
        d["rewards"][16:] += 100
        d["values"][16:] += 100
        actual = VisualWindowDataset(d, self.cfg, [0,1]).stats()
        self.assertEqual(expected, actual)

    def test_future_actions_cannot_change_prefix_predictions(self):
        before = self.model(self.x)
        altered = dict(self.x); altered["fut_acts"] = self.x["fut_acts"].clone()
        altered["fut_acts"][:, 1:] += 0.7
        after = self.model(altered)
        for key in ("latents", "rewards", "values"):
            torch.testing.assert_close(before[key][:, :1], after[key][:, :1], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(before["actions"], after["actions"])

    def test_timeout_value_and_nonexpert_action_masks(self):
        masked = dict(self.y)
        masked["value_valid"] = torch.zeros_like(self.y["value_valid"])
        masked["exp_pol_flags"] = torch.zeros_like(self.y["exp_pol_flags"])
        loss, terms = world_model_loss(self.model, self.x, masked)
        self.assertEqual(float(terms["value"]), 0)
        self.assertEqual(float(terms["action"]), 0)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.encoder.front.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_gamma_horizon_algebra(self):
        r = torch.tensor([[2., 4., 8.]])
        v = torch.tensor([[100., 100., 16.]])
        torch.testing.assert_close(discounted_return(r, v, 0.5, 1), torch.tensor([8.]))
        torch.testing.assert_close(discounted_return(r, v, 0.5, 0), torch.tensor([6.]))

    def test_warmstart_keeps_unconsumed_steps(self):
        previous = torch.tensor([[1.], [2.], [3.], [4.]])
        policy = torch.tensor([[10.], [20.], [30.], [40.]])
        torch.testing.assert_close(shift_plan(previous, policy, 2), torch.tensor([[3.], [4.], [30.], [40.]]))
        torch.testing.assert_close(shift_plan(previous, policy, 7), policy)

    def test_recorder_preserves_pause_and_actual_successor(self):
        import tempfile
        import pickle
        recorder = TransitionRecorder(operator_label="failure", task_horizon_ticks=300)
        def observation(t, remaining):
            return {"joints": np.full(6,t,np.float32), "front": self.data["front_images"][0],
                    "wrist": self.data["wrist_images"][0], "timestamp_seconds": t,
                    "front_timestamp_seconds": t-0.001, "wrist_timestamp_seconds": t-0.002,
                    "remaining_ticks": remaining}
        for i,(a,b) in enumerate(((1.,1.033), (1.033,2.033))):
            recorder.append(observation(a,300-i), [0.1]*6, observation(b,299-i))
        with tempfile.TemporaryDirectory() as directory:
            path = recorder.save(Path(directory)/"episode.pkl")
            with path.open("rb") as f:
                saved = pickle.load(f)
        self.assertFalse(saved["recording_trimmed"])
        np.testing.assert_allclose(saved["next_timestamps_seconds"], [1.033,2.033])
        self.assertFalse(saved["reward_valid"].any())
        self.assertFalse(saved["value_valid"].any())

    def test_candidate_projection_limits_each_tick(self):
        raw = torch.tensor([[[100.]*6, [-100.]*6, [100.]*6]])
        low, high, rate = torch.full((6,), -1.), torch.ones(6), torch.full((6,), 0.1)
        previous = torch.tensor([0.8]*6)
        constrained = project_actions(raw, previous, low, high, rate)
        self.assertTrue((constrained.abs() <= 1).all())
        steps = torch.diff(torch.cat((previous[None,None], constrained), 1), dim=1)
        self.assertTrue((steps.abs() <= 0.100001).all())
        policy = project_actions(torch.zeros_like(raw[0]), previous, low, high, rate)
        trust = project_policy_region(raw, previous, low, high, rate, policy, 0.05)
        self.assertTrue(((trust-policy).abs() <= 0.050001).all())
        steps = torch.diff(torch.cat((previous[None,None], trust), 1), dim=1)
        self.assertTrue((steps.abs() <= 0.100001).all())

    def test_mppi_uses_sent_history_and_real_startup_frames(self):
        def obs(i):
            return {"joints": self.data["observations"][i], "front": self.data["front_images"][i],
                    "wrist": self.data["wrist_images"][i], "timestamp_seconds": i/30, "remaining_ticks": 8-i}
        ctrl = MppiController(self.model, [-1]*6, [1]*6, [3]*6,
                              MppiConfig(num_samples=8, num_elites=3, iterations=2))
        ctrl.reset(obs(0), [0]*6)
        self.assertFalse(ctrl.ready)
        with self.assertRaises(RuntimeError):
            ctrl.plan()
        sent = np.full(6, 0.07, np.float32)
        ctrl.update(obs(1), sent); ctrl.update(obs(2), sent)
        self.assertTrue(ctrl.ready)
        torch.testing.assert_close(ctrl._inputs()["acts_hist"], torch.tensor(np.stack((sent,sent))[None]))
        candidate_actions = []
        predict_batches = []
        original = self.model.predict
        def spy(context, latent, acts, remaining, *args, **kwargs):
            predict_batches.append(acts.shape[0])
            if acts.shape[0] > 1:
                candidate_actions.append(self.model.scaler.unscale({"fut_acts": acts})["fut_acts"].detach())
            return original(context, latent, acts, remaining, *args, **kwargs)
        self.model.predict = spy
        plan = ctrl.plan()
        for actions in candidate_actions:
            self.assertTrue((actions.abs() <= 1.000001).all())
            first = torch.tensor(sent).expand(actions.shape[0],1,-1)
            self.assertTrue((torch.diff(torch.cat((first,actions),1),dim=1).abs() <= 0.100001).all())
        self.assertEqual(plan.shape, (3,6))
        self.assertIn("latency_seconds", ctrl.last_report)
        proposal = torch.full((self.cfg.prediction_length, self.cfg.action_dim), 0.2)
        proposed_plan = ctrl.plan(proposal_chunk=proposal)
        self.assertEqual(proposed_plan.shape, (3,6))
        self.assertEqual(ctrl.last_report["proposal_source"], "external_action_chunk")
        self.assertGreaterEqual(ctrl.last_report["model_score_diagnostics"]["candidate_score_gain"], -1e-6)
        # An externally supplied DP chunk bypasses the unused native policy
        # forward; only the batched candidate rollout is evaluated.
        self.assertEqual(predict_batches[-1], 8)
        first = torch.tensor(sent)
        self.assertTrue((torch.diff(torch.cat((first[None,None], torch.as_tensor(proposed_plan)[None]),1).squeeze(0),dim=0).abs() <= 0.100001).all())
        with self.assertRaisesRegex(ValueError, "proposal_chunk must be finite"):
            ctrl.plan(proposal_chunk=torch.zeros(2,6))
        exact_ctrl = MppiController(self.model, [-1]*6, [1]*6, [3]*6,
                                    MppiConfig(num_samples=1, num_elites=1, iterations=1))
        exact_ctrl.reset(obs(0), sent)
        exact_ctrl.update(obs(1), sent); exact_ctrl.update(obs(2), sent)
        exact_plan = exact_ctrl.plan(proposal_chunk=proposal)
        expected = project_actions(proposal.unsqueeze(0), torch.tensor(sent),
                                   exact_ctrl.lower, exact_ctrl.upper, exact_ctrl.step)[0]
        np.testing.assert_allclose(exact_plan, expected.numpy(), atol=1e-6)
        with self.assertRaises(ValueError):
            ctrl.update(obs(5), sent)


if __name__ == "__main__":
    unittest.main()
