"""Meaningful axis and backward-compatibility checks for image silhouette guidance."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from image_projection_loss import ImageProjectionLoss


class ImageProjectionLossTest(unittest.TestCase):
    def test_front_and_right_silhouettes_reward_the_same_visible_voxel(self):
        target_front = np.zeros((64, 64), np.float32)
        target_right = np.zeros_like(target_front)
        weight_front = np.zeros_like(target_front)
        weight_right = np.zeros_like(target_front)
        target_front[5, 7] = weight_front[5, 7] = 1
        target_right[9, 7] = weight_right[9, 7] = 1
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'multi.npz'
            np.savez(path, target_front=target_front, weight_front=weight_front,
                     target_right=target_right, weight_right=weight_right,
                     active_threshold=np.float32(.1))
            loss = ImageProjectionLoss(str(path), torch.device('cpu'), .3)
            empty = torch.full((64, 64, 64), -5.0)
            filled = empty.clone()
            filled[5, 9, 7] = 5.0
            envelope = torch.ones_like(empty)
            self.assertLess(loss.loss(filled, envelope), loss.loss(empty, envelope))
            self.assertEqual([row[1] for row in loss.projections], [1, 0])

    def test_legacy_top_target_still_projects_along_z(self):
        target = np.zeros((64, 64), np.float32)
        weight = np.zeros_like(target)
        target[5, 9] = weight[5, 9] = 1
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'top.npz'
            np.savez(path, target=target, weight=weight)
            loss = ImageProjectionLoss(str(path), torch.device('cpu'), .3)
            empty = torch.full((64, 64, 64), -5.0)
            filled = empty.clone()
            filled[5, 9, 7] = 5.0
            envelope = torch.ones_like(empty)
            self.assertLess(loss.loss(filled, envelope), loss.loss(empty, envelope))
            self.assertEqual(loss.projections[0][1], 2)

    def test_camera_grid_rewards_the_visible_voxel_and_has_gradient(self):
        target = np.zeros((64, 64), np.float32)
        weight = np.zeros_like(target)
        target[11, 13] = weight[11, 13] = 1
        # Every ray except the supervised pixel samples outside the volume.
        camera = np.full((1, 64, 64, 3), 2.0, np.float32)
        # grid_sample coordinates are Z,Y,X; the target voxel is X=5,Y=9,Z=7.
        camera[0, 11, 13] = [2 * 7 / 63 - 1, 2 * 9 / 63 - 1, 2 * 5 / 63 - 1]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'camera.npz'
            np.savez(path, target_front=target, weight_front=weight,
                     camera_grid_front=camera)
            loss = ImageProjectionLoss(str(path), torch.device('cpu'), .3)
            empty = torch.full((64, 64, 64), -5.0)
            filled = empty.clone()
            filled[5, 9, 7] = 5.0
            envelope = torch.ones_like(empty)
            self.assertLess(loss.loss(filled, envelope), loss.loss(empty, envelope))
            differentiable = empty.clone().requires_grad_()
            loss.loss(differentiable, envelope).backward()
            self.assertLess(differentiable.grad[5, 9, 7].item(), 0)


if __name__ == '__main__':
    unittest.main()
