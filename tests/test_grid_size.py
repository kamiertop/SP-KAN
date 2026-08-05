import unittest

import torch

from model.SP_KAN import SP_KAN


class GridSizeTests(unittest.TestCase):
    def test_grid_size_changes_basis_capacity_without_shape_change(self):
        image = torch.randn(1, 1, 64, 64)
        outputs = []
        for grid_size in (5, 12):
            model = SP_KAN(1, 1, mode='train', deepsuper=True, kan_grid_size=grid_size)
            outputs.append(model(image))
        self.assertEqual(outputs[0][-1].shape, outputs[1][-1].shape)

    def test_invalid_grid_size_rejected(self):
        with self.assertRaises(ValueError):
            SP_KAN(1, 1, kan_grid_size=0)


if __name__ == '__main__':
    unittest.main()
