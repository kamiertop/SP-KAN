import unittest

import torch

from model.SP_KAN import SP_KAN
from util.noise import add_infrared_noise, prediction_consistency_loss


class NoiseBranchTest(unittest.TestCase):
    def test_noise_preserves_shape_and_is_finite(self):
        image = torch.zeros(2, 1, 16, 20)
        noisy = add_infrared_noise(image, gaussian_std=0.04, stripe_prob=1.0,
                                   dead_pixel_prob=0.05)
        self.assertEqual(noisy.shape, image.shape)
        self.assertTrue(torch.isfinite(noisy).all())

    def test_consistency_loss_backpropagates(self):
        clean = torch.sigmoid(torch.randn(2, 1, 8, 8, requires_grad=True))
        noisy = torch.sigmoid(torch.randn(2, 1, 8, 8, requires_grad=True))
        loss = prediction_consistency_loss(clean, noisy)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(clean.grad_fn)

    def test_restoration_toggle_keeps_output_contract(self):
        x = torch.randn(2, 1, 64, 64)
        model_on = SP_KAN(restoration_branch=True, deepsuper=True).eval()
        model_off = SP_KAN(restoration_branch=False, deepsuper=True).eval()
        with torch.no_grad():
            on = model_on(x)
            off = model_off(x)
        self.assertEqual(len(on), len(off))
        self.assertEqual(on[-1].shape, off[-1].shape)


if __name__ == '__main__':
    unittest.main()
