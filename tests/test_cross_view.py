import unittest
import torch

from model.SP_KAN import CrossViewBackgroundFusion, SP_KAN


class CrossViewTests(unittest.TestCase):
    def test_topk_fusion_shape_and_grad(self):
        block = CrossViewBackgroundFusion(8, topk=0.25)
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        y = block(x)
        self.assertEqual(y.shape, x.shape)
        y.mean().backward()
        self.assertIsNotNone(x.grad)

    def test_model_cross_view_shape(self):
        model = SP_KAN(1, 1, mode='test', deepsuper=True,
                       cross_view_branch=True)
        with torch.no_grad():
            y = model(torch.randn(1, 1, 64, 64))
        self.assertEqual(y.shape, (1, 1, 64, 64))


if __name__ == '__main__':
    unittest.main()
