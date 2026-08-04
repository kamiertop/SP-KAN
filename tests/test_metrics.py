import unittest

import torch

from test import SamplewiseSigmoidMetric, evaluation_jobs
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


if __name__ == '__main__':
    unittest.main()
