import unittest
import torch

from model.SP_KAN import MiMISTDBlock, CGSSMBlock, SP_KAN
from util.losses import TargetAwareBoundaryLoss
from util.train_helpers import postprocess_masks


class MiMISTDTests(unittest.TestCase):
    def test_block_shape_and_grad(self):
        block = MiMISTDBlock(8)
        x = torch.randn(2, 8, 5, 7, requires_grad=True)
        y = block(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())
        y.mean().backward()
        self.assertIsNotNone(x.grad)

    def test_model_toggle_shape(self):
        plain_net = SP_KAN(1, 1, deepsuper=False, mamba_branch=False).eval()
        mamba_net = SP_KAN(1, 1, deepsuper=False, mamba_branch=True).eval()
        with torch.no_grad():
            sample = torch.randn(1, 1, 64, 64)
            plain = plain_net(sample)
            mamba = mamba_net(sample)
        self.assertEqual(plain.shape, mamba.shape)

    def test_cg_ssm_shape_and_grad(self):
        block = CGSSMBlock(4)
        x = torch.randn(1, 4, 5, 7, requires_grad=True)
        y = block(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())
        y.mean().backward()
        self.assertIsNotNone(x.grad)

    def test_validation_postprocess_and_loss_shape(self):
        net = SP_KAN(1, 1, mode='train', deepsuper=True, mamba_branch=True).eval()
        criterion = TargetAwareBoundaryLoss()
        with torch.no_grad():
            sample = torch.randn(1, 1, 64, 64)
            target = torch.zeros(1, 1, 31, 47)
            pred = net(sample)
            if isinstance(pred, (tuple, list)):
                pred = pred[-1]
            pred = postprocess_masks(pred, [64, 64], [31, 47])
            loss = criterion(pred, target)
        self.assertEqual(pred.shape, target.shape)
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
