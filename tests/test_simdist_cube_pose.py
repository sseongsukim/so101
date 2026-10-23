from __future__ import annotations

import pytest
import torch

from so101.learning.simdist.cube_pose import CubePoseHead, CubeSpatialPoseHead, dp_spatial_features


def test_cube_pose_head_maps_front_wrist_and_joints_to_two_xyz_poses():
    model = CubePoseHead()
    out = model(torch.randn(5, 128), torch.randn(5, 128), torch.randn(5, 6))
    assert out.shape == (5, 6)


def test_cube_pose_head_rejects_mismatched_camera_and_joint_shapes():
    model = CubePoseHead()
    with pytest.raises(ValueError, match="matching 128-D"):
        model(torch.randn(2, 128), torch.randn(2, 64), torch.randn(2, 6))
    with pytest.raises(ValueError, match="six values"):
        model(torch.randn(2, 128), torch.randn(2, 128), torch.randn(2, 5))


def test_spatial_attention_starts_exactly_at_matched_mean_control():
    model = CubeSpatialPoseHead(channels=4)
    control = CubeSpatialPoseHead(channels=4, attention=False)
    control.load_state_dict(model.state_dict())
    front, wrist, q = torch.randn(3, 4, 5, 7), torch.randn(3, 4, 5, 7), torch.randn(3, 6)
    assert torch.equal(model(front, wrist, q), control(front, wrist, q))
    assert not control.front_attention.weight.requires_grad
    model(front, wrist, q).square().mean().backward()
    assert model.front_attention.weight.grad.abs().sum() > 0
    assert model.wrist_attention.weight.grad.abs().sum() > 0


def test_mean_control_ignores_spatial_permutation_but_attention_can_use_location():
    model = CubeSpatialPoseHead(channels=4)
    control = CubeSpatialPoseHead(channels=4, attention=False)
    control.load_state_dict(model.state_dict())
    with torch.no_grad():
        model.front_attention.weight[:, 0, 0, 0] = 2.0
        model.wrist_attention.weight[:, 0, 0, 0] = 2.0
    # Flipping maps preserves average content, while nonuniform attention's
    # soft image coordinates flip. No object pose labels are available here.
    front, wrist, q = torch.randn(2, 4, 5, 7), torch.randn(2, 4, 5, 7), torch.randn(2, 6)
    flipped_front, flipped_wrist = front.flip(-1), wrist.flip(-1)
    assert torch.allclose(control(front, wrist, q), control(flipped_front, flipped_wrist, q), atol=1e-7)
    assert not torch.allclose(model(front, wrist, q), model(flipped_front, flipped_wrist, q), atol=1e-7)


def test_spatial_head_accepts_half_cache_and_rejects_wrong_inputs():
    model = CubeSpatialPoseHead(channels=4)
    front, wrist, q = torch.randn(2, 4, 5, 7).half(), torch.randn(2, 4, 5, 7).half(), torch.randn(2, 6)
    assert model(front, wrist, q).shape == (2, 6)
    with pytest.raises(ValueError, match="spatial_channels"):
        model(front, wrist[..., :6], q)
    with pytest.raises(ValueError, match="six values"):
        model(front, wrist, q[:, :5])


def test_spatial_extraction_preserves_frozen_dp_prepool_content():
    from so101.learning.dp.vision import ResnetEncoder

    torch.set_num_threads(1)
    encoder = ResnetEncoder(pretrained=False, use_groupnorm=True, freeze=True).eval()
    images = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    with torch.no_grad():
        maps = dp_spatial_features(encoder, images)
        pooled = encoder(images)
    assert maps.shape == (2, 512, 2, 2)
    assert torch.equal(maps.mean((-2, -1)), pooled)
    assert all(not p.requires_grad for p in encoder.parameters())
