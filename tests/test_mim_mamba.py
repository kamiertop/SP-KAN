import unittest
import torch

from model.SP_KAN import MiMISTDBlock, SP_KAN


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


if __name__ == '__main__':
    unittest.main()
