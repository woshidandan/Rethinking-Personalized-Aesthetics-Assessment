# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from pam_mmpose.models import build_loss


def test_rle_loss():
    # test RLELoss without target weight(default None)
    loss_cfg = dict(type='RLELoss')
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 4))
    fake_label = torch.zeros((1, 3, 2))
    loss(fake_pred, fake_label)

    # test RLELoss with Q(error) changed to "Gaussian"(default "Laplace")
    loss_cfg = dict(type='RLELoss', q_dis='gaussian')
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 4))
    fake_label = torch.zeros((1, 3, 2))
    loss(fake_pred, fake_label)

    # test RLELoss._apply(fn)
    loss_cfg = dict(type='RLELoss', size_average=False)
    loss = build_loss(loss_cfg)
    loss.cpu()

    fake_pred = torch.zeros((1, 3, 4))
    fake_label = torch.zeros((1, 3, 2))
    loss(fake_pred, fake_label)

    # test RLELoss with size_average(default True) changed to False
    loss_cfg = dict(type='RLELoss', size_average=False)
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 4))
    fake_label = torch.zeros((1, 3, 2))
    loss(fake_pred, fake_label)

    # test RLELoss with residual(default True) changed to False
    loss_cfg = dict(type='RLELoss', residual=False)
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 4))
    fake_label = torch.zeros((1, 3, 2))
    loss(fake_pred, fake_label)

    # test RLELoss with target weight
    loss_cfg = dict(type='RLELoss', use_target_weight=True)
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 4))
    fake_label = torch.zeros((1, 3, 2))
    loss(fake_pred, fake_label, torch.ones_like(fake_label))

    fake_pred = torch.ones((1, 3, 4))
    fake_label = torch.zeros((1, 3, 2))
    loss(fake_pred, fake_label, torch.ones_like(fake_label))


def test_smooth_l1_loss():
    # test SmoothL1Loss without target weight(default None)
    loss_cfg = dict(type='SmoothL1Loss')
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(loss(fake_pred, fake_label), torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(loss(fake_pred, fake_label), torch.tensor(.5))

    # test SmoothL1Loss with target weight
    loss_cfg = dict(type='SmoothL1Loss', use_target_weight=True)
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(
        loss(fake_pred, fake_label, torch.ones_like(fake_label)),
        torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(
        loss(fake_pred, fake_label, torch.ones_like(fake_label)),
        torch.tensor(.5))


def test_wing_loss():
    # test WingLoss without target weight(default None)
    loss_cfg = dict(type='WingLoss')
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(loss(fake_pred, fake_label), torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.gt(loss(fake_pred, fake_label), torch.tensor(.5))

    # test WingLoss with target weight
    loss_cfg = dict(type='WingLoss', use_target_weight=True)
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(
        loss(fake_pred, fake_label, torch.ones_like(fake_label)),
        torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.gt(
        loss(fake_pred, fake_label, torch.ones_like(fake_label)),
        torch.tensor(.5))


def test_soft_wing_loss():
    # test SoftWingLoss without target weight(default None)
    loss_cfg = dict(type='SoftWingLoss')
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(loss(fake_pred, fake_label), torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.gt(loss(fake_pred, fake_label), torch.tensor(.5))

    # test SoftWingLoss with target weight
    loss_cfg = dict(type='SoftWingLoss', use_target_weight=True)
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(
        loss(fake_pred, fake_label, torch.ones_like(fake_label)),
        torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.gt(
        loss(fake_pred, fake_label, torch.ones_like(fake_label)),
        torch.tensor(.5))


def test_mse_regression_loss():
    # w/o target weight(default None)
    loss_cfg = dict(type='MSELoss')
    loss = build_loss(loss_cfg)
    fake_pred = torch.zeros((1, 3, 3))
    fake_label = torch.zeros((1, 3, 3))
    assert torch.allclose(loss(fake_pred, fake_label), torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 3))
    fake_label = torch.zeros((1, 3, 3))
    assert torch.allclose(loss(fake_pred, fake_label), torch.tensor(1.))

    # w/ target weight
    loss_cfg = dict(type='MSELoss', use_target_weight=True)
    loss = build_loss(loss_cfg)
    fake_pred = torch.zeros((1, 3, 3))
    fake_label = torch.zeros((1, 3, 3))
    assert torch.allclose(
        loss(fake_pred, fake_label, torch.ones_like(fake_label)),
        torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 3))
    fake_label = torch.zeros((1, 3, 3))
    assert torch.allclose(
        loss(fake_pred, fake_label, torch.ones_like(fake_label)),
        torch.tensor(1.))


def test_bone_loss():
    # w/o target weight(default None)
    loss_cfg = dict(type='BoneLoss', joint_parents=[0, 0, 1])
    loss = build_loss(loss_cfg)
    fake_pred = torch.zeros((1, 3, 3))
    fake_label = torch.zeros((1, 3, 3))
    assert torch.allclose(loss(fake_pred, fake_label), torch.tensor(0.))

    fake_pred = torch.tensor([[[0, 0, 0], [1, 1, 1], [2, 2, 2]]],
                             dtype=torch.float32)
    fake_label = fake_pred * 2
    assert torch.allclose(loss(fake_pred, fake_label), torch.tensor(3**0.5))

    # w/ target weight
    loss_cfg = dict(
        type='BoneLoss', joint_parents=[0, 0, 1], use_target_weight=True)
    loss = build_loss(loss_cfg)
    fake_pred = torch.zeros((1, 3, 3))
    fake_label = torch.zeros((1, 3, 3))
    fake_weight = torch.ones((1, 2))
    assert torch.allclose(
        loss(fake_pred, fake_label, fake_weight), torch.tensor(0.))

    fake_pred = torch.tensor([[[0, 0, 0], [1, 1, 1], [2, 2, 2]]],
                             dtype=torch.float32)
    fake_label = fake_pred * 2
    fake_weight = torch.ones((1, 2))
    assert torch.allclose(
        loss(fake_pred, fake_label, fake_weight), torch.tensor(3**0.5))


def test_semi_supervision_loss():
    loss_cfg = dict(
        type='SemiSupervisionLoss',
        joint_parents=[0, 0, 1],
        warmup_iterations=1)
    loss = build_loss(loss_cfg)

    unlabeled_pose = torch.rand((1, 3, 3))
    unlabeled_traj = torch.ones((1, 1, 3))
    labeled_pose = unlabeled_pose.clone()
    fake_pred = dict(
        labeled_pose=labeled_pose,
        unlabeled_pose=unlabeled_pose,
        unlabeled_traj=unlabeled_traj)

    intrinsics = torch.tensor([[1, 1, 1, 1, 0.1, 0.1, 0.1, 0, 0]],
                              dtype=torch.float32)
    unlabled_target_2d = loss.project_joints(unlabeled_pose + unlabeled_traj,
                                             intrinsics)
    fake_label = dict(
        unlabeled_target_2d=unlabled_target_2d, intrinsics=intrinsics)

    # test warmup
    losses = loss(fake_pred, fake_label)
    assert not losses

    # test semi-supervised loss
    losses = loss(fake_pred, fake_label)
    assert torch.allclose(losses['proj_loss'], torch.tensor(0.))
    assert torch.allclose(losses['bone_loss'], torch.tensor(0.))


def test_soft_weight_smooth_l1_loss():
    loss_cfg = dict(
        type='SoftWeightSmoothL1Loss', use_target_weight=False, beta=0.5)
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(loss(fake_pred, fake_label), torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(loss(fake_pred, fake_label), torch.tensor(.75))

    loss_cfg = dict(
        type='SoftWeightSmoothL1Loss',
        use_target_weight=True,
        supervise_empty=True)
    loss = build_loss(loss_cfg)

    fake_pred = torch.ones((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    fake_weight = torch.arange(6).reshape(1, 3, 2).float()
    assert torch.allclose(
        loss(fake_pred, fake_label, fake_weight), torch.tensor(1.25))

    loss_cfg = dict(
        type='SoftWeightSmoothL1Loss',
        use_target_weight=True,
        supervise_empty=False)
    loss = build_loss(loss_cfg)
    assert torch.allclose(
        loss(fake_pred, fake_label, fake_weight), torch.tensor(1.5))

    with pytest.raises(ValueError):
        _ = loss.smooth_l1_loss(fake_pred, fake_label, reduction='fake')

    output = loss.smooth_l1_loss(fake_pred, fake_label, reduction='sum')
    assert torch.allclose(output, torch.tensor(3.0))
