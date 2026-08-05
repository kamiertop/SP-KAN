import unittest

import torch

from model.SP_KAN import FocalContextBlock, SP_KAN


class FocalContextTests(unittest.TestCase):
    def test_context_block_preserves_feature_shape_and_gradients(self):
        block = FocalContextBlock(16)
        features = torch.randn(2, 16, 15, 17, requires_grad=True)
        output = block(features)
        self.assertEqual(output.shape, features.shape)
        output.mean().backward()
        self.assertIsNotNone(features.grad)

    def test_focal_and_cvit_model_contracts_match(self):
        image = torch.randn(1, 1, 64, 64)
        for context in ('focal', 'cvit'):
            model = SP_KAN(1, 1, mode='train', deepsuper=True, context=context)
            outputs = model(image)
            self.assertEqual(len(outputs), 6)
            self.assertEqual(outputs[-1].shape, image.shape)


if __name__ == '__main__':
    unittest.main()
