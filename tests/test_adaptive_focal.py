import unittest

import torch

from util.losses import AdaptiveThresholdFocalResidual, TargetAwareBoundaryLoss


class AdaptiveFocalTests(unittest.TestCase):
    def test_focal_residual_is_finite_and_differentiable(self):
        prediction = torch.full((2, 1, 8, 8), 0.2, requires_grad=True)
        target = torch.zeros_like(prediction)
        target[..., 3, 4] = 1
        loss = AdaptiveThresholdFocalResidual()(prediction, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(prediction.grad)

    def test_zero_focal_weight_matches_tabds_without_focal_term(self):
        prediction = torch.full((1, 1, 8, 8), 0.2)
        target = torch.zeros_like(prediction)
        target[..., 3, 4] = 1
        tabds = TargetAwareBoundaryLoss(focal_weight=0.0)
        adaptive = TargetAwareBoundaryLoss(focal_weight=0.0)
        self.assertAlmostEqual(float(tabds(prediction, target)), float(adaptive(prediction, target)), places=6)


if __name__ == '__main__':
    unittest.main()
