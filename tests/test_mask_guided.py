import unittest

import torch

from model.SP_KAN import MaskGuidedSkipGate, SP_KAN


class MaskGuidedTests(unittest.TestCase):
    def test_gate_preserves_shape_and_floor(self):
        gate = MaskGuidedSkipGate(16, 16, floor=0.25)
        decoder = torch.randn(2, 16, 8, 8)
        skip = torch.randn(2, 16, 16, 16)
        filtered, mask = gate(decoder, skip)
        self.assertEqual(filtered.shape, skip.shape)
        self.assertEqual(mask.shape, (2, 1, 16, 16))
        self.assertGreaterEqual(float(mask.min()), 0.0)
        self.assertLessEqual(float(mask.max()), 1.0)

    def test_model_contract_with_and_without_gate(self):
        image = torch.randn(1, 1, 64, 64)
        for enabled in (True, False):
            model = SP_KAN(1, 1, mode='train', deepsuper=True,
                           projection='interp', mask_guided=enabled)
            outputs = model(image)
            self.assertEqual(len(outputs), 6)
            self.assertEqual(outputs[-1].shape, image.shape)


if __name__ == '__main__':
    unittest.main()
