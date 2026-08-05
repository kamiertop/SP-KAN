import unittest

import torch

from util.losses import TargetAwareBoundaryLoss, deep_supervision_loss


class TargetAwareLossTests(unittest.TestCase):
    def test_loss_is_finite_for_one_pixel_target(self):
        criterion = TargetAwareBoundaryLoss()
        prediction = torch.full((1, 1, 8, 8), 0.05)
        target = torch.zeros_like(prediction)
        target[..., 3, 4] = 1
        self.assertTrue(torch.isfinite(criterion(prediction, target)))

    def test_deep_supervision_normalizes_head_count(self):
        criterion = TargetAwareBoundaryLoss(dice_weight=0)
        target = torch.zeros(1, 1, 8, 8)
        target[..., 3:5, 3:5] = 1
        one = deep_supervision_loss(criterion, torch.full_like(target, 0.1), target)
        many = deep_supervision_loss(criterion, (torch.full_like(target, 0.1),) * 6, target)
        self.assertAlmostEqual(float(one), float(many), places=5)

    def test_deep_supervision_supports_backward(self):
        criterion = TargetAwareBoundaryLoss()
        target = torch.zeros(1, 1, 8, 8)
        target[..., 2, 2] = 1
        predictions = tuple(torch.full_like(target, 0.2, requires_grad=True) for _ in range(6))
        deep_supervision_loss(criterion, predictions, target).backward()
        self.assertTrue(all(pred.grad is not None for pred in predictions))

    def test_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError):
            TargetAwareBoundaryLoss(max_pos_weight=0.5)


if __name__ == '__main__':
    unittest.main()
