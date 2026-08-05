import unittest

import torch

from test import SamplewiseSigmoidMetric, align_prediction_to_target, evaluation_jobs
from util.metrics import F1


class MetricTests(unittest.TestCase):
    def test_f1_accumulates_across_updates(self):
        metric = F1(threshold=0.7)
        metric.update(torch.tensor([[[[0.9]]]]), torch.ones(1, 1, 1, 1))
        metric.update(torch.tensor([[[[0.6]]]]), torch.ones(1, 1, 1, 1))
        self.assertAlmostEqual(metric.get(), 2 / 3, places=3)

    def test_samplewise_iou_uses_configured_threshold(self):
        output = torch.tensor([[[[0.6]]]])
        target = torch.ones(1, 1, 1, 1)
        low_threshold = SamplewiseSigmoidMetric(nclass=1, score_thresh=0.5)
        high_threshold = SamplewiseSigmoidMetric(nclass=1, score_thresh=0.7)
        low_threshold.update(output, target)
        high_threshold.update(output, target)
        self.assertAlmostEqual(low_threshold.get(), 1.0)
        self.assertAlmostEqual(high_threshold.get(), 0.0)

    def test_evaluation_jobs_broadcast_or_reject_mismatch(self):
        jobs = list(evaluation_jobs(['SP_KAN'], ['SIRST3', 'NUDT-SIRST'], ['a.pth', 'b.pth']))
        self.assertEqual(jobs, [('SP_KAN', 'SIRST3', 'a.pth'), ('SP_KAN', 'NUDT-SIRST', 'b.pth')])
        with self.assertRaises(ValueError):
            list(evaluation_jobs(['a', 'b'], ['x', 'y', 'z'], ['only.pth']))
        with self.assertRaises(ValueError):
            list(evaluation_jobs(['OtherModel'], ['SIRST3'], ['only.pth']))

    def test_prediction_is_aligned_to_ground_truth_shape(self):
        pred = torch.rand(1, 1, 8, 8)
        target = torch.zeros(1, 1, 7, 9)
        aligned = align_prediction_to_target(pred, target, (8, 8), (7, 9))
        self.assertEqual(aligned.shape, target.shape)

    def test_test_script_supports_explicit_train_split(self):
        from pathlib import Path
        source = Path(__file__).parents[1].joinpath("test.py").read_text()
        self.assertIn("--eval_split", source)
        self.assertIn("f'{opt.eval_split}_{opt.test_dataset_name}.txt'", source)


if __name__ == '__main__':
    unittest.main()
