"""CPU contracts; tiny frozen camera fixture is not a learned performance claim."""
from __future__ import annotations

import hashlib
import json
import pickle
import sys

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from so101.learning.simdist.cube_pose_pixel import (
    MASK_SEMANTICS, POSE_SEMANTICS, FrozenEarlyDPEncoder, PixelCubePose,
    PixelPoseOutput, front_centre_targets, pixel_pose_loss, resize_pixel_centres,
    unpack_resize_masks,
)
from so101.learning.simdist.cube_pose_pixel_training import load_pixel_records
from so101.learning.simdist.cube_pose_region import CubeTaskRegion, region_pose_metrics


class TinyEarlyEncoder(nn.Module):
    channels = (4, 8, 16)
    dp_camera_indices = (1, 0)

    def __init__(self, dp=None):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(3, c, 1) for c in self.channels])
        for p in self.parameters():
            p.requires_grad_(False)
        self.calls = 0

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, front, wrist):
        self.calls += 1
        with torch.no_grad():
            return [[layer(F.avg_pool2d(x.float()/255, stride)) for layer, stride in zip(self.layers, (4, 8, 16))]
                    for x in (front, wrist)]


def stats():
    return {"q_mean": np.zeros(6), "q_std": np.ones(6),
            "pose_mean": np.zeros(6), "pose_std": np.ones(6)}


def episode(seed=601, n=3):
    poses = np.zeros((n, 2, 7), np.float32); poses[..., 3] = 1
    k = np.broadcast_to(np.array([[400., 0, 319.5], [0, 400, 239.5], [0, 0, 1]]), (n, 2, 3, 3)).copy()
    xyz = np.broadcast_to(np.array([[.3, 0, .05], [.4, -.02, .05]], np.float32), (n, 2, 3)).copy()
    xyz[:, 0, 0] += np.arange(n)*.002
    images = np.full((n, 240, 320, 3), seed % 200, np.uint8)
    masks = np.zeros((n, 2, 2, 480, 640), np.uint8)
    masks[:, :, 0, 170:180, 310:320] = 1
    masks[:, :, 1, 185:200, 335:345] = 1
    masks[1, :, 0] = 0  # Fully occluded held object remains GT-inside XYZ target.
    return {"seed": seed, "env_index": 0, "task_horizon_ticks": n, "tick": np.arange(n),
        "observations": np.arange(n*6, dtype=np.float32).reshape(n, 6)*.001,
        "front_images": images, "wrist_images": images+1,
        "cube_positions_robot_relative_m": xyz,
        "next_cube_positions_robot_relative_m": np.concatenate((xyz[1:], xyz[-1:])),
        "cube_position_order": ["held_cube", "fixed_cube"],
        "cube_position_label_semantics": "simulator_xyz_robot_root_relative_training_only",
        "table_height_robot_relative_m": np.full(n, .0300814467),
        "camera_poses_robot_relative_m_wxyz": poses, "camera_intrinsics_source_px": k,
        "camera_order": ["front", "wrist"], "camera_pose_semantics": POSE_SEMANTICS,
        "camera_intrinsics_semantics": "pinhole_3x3_source_sensor_pixel_coordinates",
        "camera_source_image_shape_hw": [[480, 640], [480, 640]],
        "cube_instance_masks_packed": np.packbits(masks, axis=-1, bitorder="big"),
        "cube_instance_mask_camera_order": ["front", "wrist"],
        "cube_instance_mask_object_order": ["held_cube", "fixed_cube"],
        "cube_instance_mask_semantics": MASK_SEMANTICS,
        "cube_instance_mask_storage": "np.packbits_axis_last_bitorder_big",
        "cube_instance_mask_source_shape_hw": [[480, 640], [480, 640]],
        "camera_sensor_frame_counter_semantics": "same_current_render_used_for_rgb_and_instance_masks",
        "camera_sensor_frame_counters": np.repeat(np.arange(n)[:, None]+2, 2, 1),
        "cube_instance_id_to_usd_path_by_camera": {camera: {"11": "/World/envs/env_0/HeldAsset/geometry",
                                                        "21": "/World/envs/env_0/FixedAsset/geometry"}
                                                   for camera in ("front", "wrist")},
        # Preservation markers; this trainer must not reinterpret these fields.
        "rewards": np.arange(n, dtype=np.float32), "values": np.ones(n),
        "terminated": np.array([False]*(n-1)+[True]), "truncated": np.zeros(n, bool)}


def test_unpack_padding_nearest_matches_pil_and_full_rgb_alignment():
    masks = np.zeros((2, 2, 2, 6, 13), np.uint8)
    masks[:, 0, 0, 1:5, 2:9] = 1
    packed = np.packbits(masks, axis=-1, bitorder="big")
    packed[..., -1] |= 7  # Padding bits must be excluded, not interpreted as foreground.
    result = unpack_resize_masks(packed, (6, 13), (3, 7)).numpy()
    for b in range(2):
        for c in range(2):
            for o in range(2):
                expected = np.asarray(Image.fromarray(masks[b,c,o]).resize((7,3), Image.Resampling.NEAREST)) > 0
                np.testing.assert_array_equal(result[b,c,o], expected)
    assert not result[:, 1].any()
    with pytest.raises(ValueError, match="packed masks"):
        unpack_resize_masks(packed.astype(np.int32), (6, 13))


def test_pixel_centres_follow_resize_half_pixel_rule():
    source = np.array([[0., 0.], [639., 479.], [319.5, 239.5]])
    result = resize_pixel_centres(source, (480, 640))
    np.testing.assert_allclose(result, [[-.25,-.25], [319.25,239.25], [159.5,119.5]])
    np.testing.assert_allclose((source+.5)/[640,480], (result+.5)/[320,240])


def targets(p):
    return front_centre_targets(p["cube_positions_robot_relative_m"], p["camera_poses_robot_relative_m_wxyz"],
        p["camera_intrinsics_source_px"], p["cube_instance_masks_packed"], (480,640),
        CubeTaskRegion().labels(p["cube_positions_robot_relative_m"]))


def test_front_world_optical_projection_and_source_visibility_not_centroid():
    p = episode()
    t = targets(p)
    # WORLD identity: forward X, image right -Y and down -Z.
    expected = (np.array([319.5, 239.5-400*.05/.3])+.5)/[640,480]
    np.testing.assert_allclose(t["centre_targets"][0,0], expected, atol=1e-7)
    assert t["centre_eligible"].tolist() == [[True,True], [False,True], [True,True]]
    assert t["source_mask_counts"][0,0,0] == 100
    # Different (even unverified) finite wrist geometry cannot alter front targets.
    p["camera_poses_robot_relative_m_wxyz"][:, 1, :3] = 999
    np.testing.assert_array_equal(targets(p)["centre_targets"], t["centre_targets"])
    p["camera_poses_robot_relative_m_wxyz"][:, 0, 3] = 2
    with pytest.raises(ValueError, match="unit"):
        targets(p)


@pytest.mark.parametrize("case", ["behind", "frustum", "seven_pixels", "outside"])
def test_centre_eligibility_edge_cases_keep_safe_targets(case):
    p = episode()
    if case == "behind":
        p["cube_positions_robot_relative_m"][:, 0, 0] = -.3
    elif case == "frustum":
        p["cube_positions_robot_relative_m"][:, 0, 1] = .9
    elif case == "outside":
        p["cube_positions_robot_relative_m"][:, 0, 0] = 2
    else:
        masks = np.zeros((3,2,2,480,640), np.uint8); masks[:, :, 0, 10, :7] = 1
        p["cube_instance_masks_packed"] = np.packbits(masks, axis=-1)
    t = targets(p)
    assert not t["centre_eligible"][:,0].any()
    assert np.isfinite(t["centre_targets"]).all()


def test_model_rgb_q_only_frozen_trunk_gradient_and_checkpoint_roundtrip():
    torch.set_num_threads(1)
    model = PixelCubePose(TinyEarlyEncoder(), stats(), width=4).train()
    front = torch.zeros(1,3,240,320,dtype=torch.uint8)
    pred = model(front, front, torch.zeros(1,6))
    assert pred.mask_logits.shape == (1,2,2,240,320)
    assert pred.centres.shape == (1,2,2,2) and pred.normalized_xyz.shape == (1,2,3)
    assert not model.encoder.training
    pred.normalized_xyz.sum().backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert model.laterals[0][0].weight.grad is not None
    clone = PixelCubePose(TinyEarlyEncoder(), stats(), width=4)
    clone.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(clone(front,front,torch.zeros(1,6)).normalized_xyz, pred.normalized_xyz)
    with pytest.raises(TypeError):
        model(front, front, torch.zeros(1,6), GT_mask=torch.ones(1))
    with pytest.raises(ValueError, match="full"):
        model(front[:,:,8:232,48:272],front[:,:,8:232,48:272],torch.zeros(1,6))


def test_actual_resnet_early_wrapper_excludes_actor_layer4_and_keeps_full_image_shape():
    from types import SimpleNamespace
    from so101.learning.dp.vision import ResnetEncoder
    torch.set_num_threads(1)
    source = SimpleNamespace(config=SimpleNamespace(vision_backbone="resnet18", image_keys=["wrist", "front"], front_key="front"),
                             encoders=nn.ModuleList([ResnetEncoder(pretrained=False), ResnetEncoder(pretrained=False)]),
                             actor=nn.Linear(1,100))
    encoder = FrozenEarlyDPEncoder(source).train()
    assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())
    assert not any("layer4" in k or "actor" in k or ".fc" in k for k in encoder.state_dict())
    assert encoder.encoders[0].model["layer1"] is source.encoders[1].model.layer1
    assert sum(p.numel() for p in encoder.parameters()) < sum(p.numel() for p in source.encoders.parameters())
    with torch.no_grad():
        maps = encoder(torch.zeros(1,3,240,320,dtype=torch.uint8),torch.zeros(1,3,240,320,dtype=torch.uint8))
    assert [tuple(x.shape[1:]) for x in maps[0]] == [(64,60,80),(128,30,40),(256,15,20)]


def test_empty_occluded_masks_have_bg_loss_zero_centre_grad_but_all_inside_pose():
    xyz = torch.ones(2,2,3,requires_grad=True)
    logits = torch.zeros(2,2,requires_grad=True)
    masks = torch.zeros(2,2,2,4,6,dtype=torch.bool)
    mask_logits = torch.zeros_like(masks,dtype=torch.float32,requires_grad=True)
    centres = torch.ones(2,2,2,2,requires_grad=True)
    pred = PixelPoseOutput(xyz,logits,mask_logits,centres)
    loss = pixel_pose_loss(pred,torch.zeros(2,6),torch.ones(2,2,dtype=torch.bool),
        torch.tensor([[0.,1.],[0.,1.]]),masks,torch.full((2,2,2),float("nan")),torch.zeros(2,2,dtype=torch.bool))
    torch.testing.assert_close(loss["mask"],torch.tensor(np.log(2),dtype=torch.float32))
    assert loss["centre"].item() == 0 and loss["xyz"].item() == 1
    loss["loss"].backward()
    assert centres.grad.count_nonzero() == 0 and xyz.grad.count_nonzero() == xyz.numel()
    assert torch.isfinite(mask_logits.grad).all() and mask_logits.grad.count_nonzero() == mask_logits.numel()
    gt = np.zeros((2,2,3)); estimate = gt.copy(); estimate[1,0,2] = .1
    report = region_pose_metrics(estimate, gt, np.ones((2,2),bool), np.full((2,2),-999.))
    assert report["denominators"]["held_height_frames"] == 2 and not report["pose_gate"]["passed"]


def write_archive(path, episodes):
    first = episodes[0]
    mask_fields = ("cube_instance_mask_camera_order", "cube_instance_mask_object_order", "cube_instance_mask_semantics",
        "cube_instance_mask_storage", "cube_instance_mask_source_shape_hw", "camera_sensor_frame_counter_semantics")
    camera_fields = ("camera_order","camera_pose_semantics","camera_intrinsics_semantics","camera_source_image_shape_hw")
    keys = ("front_images","wrist_images","observations","cube_positions_robot_relative_m",
        "next_cube_positions_robot_relative_m","table_height_robot_relative_m","camera_poses_robot_relative_m_wxyz",
        "camera_intrinsics_source_px","cube_instance_masks_packed","camera_sensor_frame_counters")
    n = len(first["observations"])
    np.savez_compressed(path, **{k: np.concatenate([p[k] for p in episodes]) for k in keys},
        ticks=np.tile(np.arange(n),len(episodes)), task_horizon_ticks=np.full(n*len(episodes),n),
        episode_starts=np.arange(len(episodes))*n, episode_lengths=np.full(len(episodes),n),
        episode_seeds=np.array([p["seed"] for p in episodes]),episode_env_indices=np.zeros(len(episodes),np.int64),
        cube_position_label_semantics=first["cube_position_label_semantics"],cube_position_order=first["cube_position_order"],
        camera_geometry_contract_json=json.dumps({k:first[k] for k in camera_fields}),
        cube_instance_mask_contract_json=json.dumps({k:first[k] for k in mask_fields}),
        episode_cube_instance_id_to_usd_path_json=np.array([json.dumps(p["cube_instance_id_to_usd_path_by_camera"]) for p in episodes]))


@pytest.mark.parametrize("archive", [False, True])
def test_cpu_one_epoch_trainer_fixture_keeps_archives_and_full_pose_denominators(tmp_path, monkeypatch, archive):
    import train_simdist_cube_pose as trainer
    import so101.learning.simdist.cube_pose_pixel_training as training
    from so101.learning.dp.config import DPConfig
    from test_simdist_cube_pose_spatial_contract import StubDP
    torch.set_num_threads(1)
    config = DPConfig(pretrained_backbone=False,image_shape=(32,32),crop_size=32,front_crop_margin=0)
    dp = tmp_path/"dp.pt"; torch.save({"config":config.to_dict(),"model":StubDP(config).state_dict()},dp)
    episodes = [episode(seed) for seed in range(601,605)]
    if archive:
        source = tmp_path/"aligned.npz"; write_archive(source,episodes)
        source_args = ["--data-archive",str(source)]
        immutable = [source]
    else:
        immutable = []
        for p in episodes:
            source = tmp_path/f"trajectory_{p['seed']}.pkl"
            with source.open("wb") as stream: pickle.dump(p,stream)
            immutable.append(source)
        source_args = ["--data-dir",str(tmp_path)]
    digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in immutable]
    out = tmp_path/"pixel.pt"
    monkeypatch.setattr(trainer,"DiffusionPolicy",StubDP)
    monkeypatch.setattr(training,"FrozenEarlyDPEncoder",TinyEarlyEncoder)
    monkeypatch.setattr(sys,"argv",["train_simdist_cube_pose.py",*source_args,"--dp-checkpoint",str(dp),
        "--out",str(out),"--device","cpu","--epochs","1","--representation","pixel",
        "--region-conditioned","--feature-batch-size","2","--train-batch-size","9"])
    trainer.main()
    report = json.loads(out.with_suffix(".json").read_text())
    saved = torch.load(out,weights_only=True)
    assert report["stream"]["feature_cache_bytes"] == 0 and report["effective_batch_size"] == 2
    assert report["fit"]["denominators"]["held_height_frames"] == 6
    assert report["selection"]["denominators"]["relative_xy_both_inside_frames"] == 3
    assert report["selection"]["pixel_auxiliary"]["front_eligible_object_frames"] == 5
    assert saved["region_class_weighting"]["both_classes_observed_per_object"] == [False,False]
    assert saved["centre_auxiliary_cameras"] == ["front"]
    assert saved["termination_value_semantics"].startswith("unchanged")
    clone = PixelCubePose(TinyEarlyEncoder(),saved["normalization"])
    clone.load_state_dict(saved["model"],strict=True)
    assert digests == [hashlib.sha256(p.read_bytes()).hexdigest() for p in immutable]


@pytest.mark.parametrize("field", ["cube_instance_mask_storage","camera_pose_semantics","camera_sensor_frame_counters"])
def test_loader_fails_closed_bad_annotation_contract(tmp_path,field):
    p = episode()
    p[field] = "bad" if field != "camera_sensor_frame_counters" else np.zeros((3,2))
    with (tmp_path/"trajectory_1.pkl").open("wb") as stream: pickle.dump(p,stream)
    with pytest.raises(ValueError):
        load_pixel_records([tmp_path],[],CubeTaskRegion())
