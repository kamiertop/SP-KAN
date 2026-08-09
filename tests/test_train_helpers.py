import unittest

import torch

from util.train_helpers import align_prediction_to_target


class TrainHelperTests(unittest.TestCase):
    def test_alignment_falls_back_to_ground_truth_shape(self):
        prediction = torch.rand(1, 1, 512, 512)
        ground_truth = torch.zeros(1, 1, 400, 592)

        aligned = align_prediction_to_target(prediction, ground_truth, (220, 325), (220, 325))

        self.assertEqual(aligned.shape, ground_truth.shape)


if __name__ == '__main__':
    unittest.main()
