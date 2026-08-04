import numpy as np
import torch
from skimage import measure


def _binary_target(target):
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.ndim != 4:
        raise ValueError(f'Expected a 3D or 4D target, got {target.ndim}D')
    return target.float()


def cal_tp_pos_fp_neg(output, target, score_thresh):
    target = _binary_target(target)
    predict = (output > score_thresh).float()
    tp = (predict * target).sum()
    fp = (predict * (1 - target)).sum()
    fn = ((1 - predict) * target).sum()
    tn = ((1 - predict) * (1 - target)).sum()
    return tp, tp + fn, fp, fp + tn, tp + fp


class F1:
    """Dataset-level F1 for binary segmentation."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def update(self, preds, labels):
        tp, pos, fp, neg, class_pos = cal_tp_pos_fp_neg(preds, labels, self.threshold)
        self.tp += float(tp)
        self.pos += float(pos)
        self.fp += float(fp)
        self.neg += float(neg)
        self.class_pos += float(class_pos)

    def get(self):
        recall = self.tp / (self.pos + 1e-3)
        precision = self.tp / (self.class_pos + 1e-3)
        return (2.0 * recall * precision) / (recall + precision + 1e-5)

    def reset(self):
        self.tp = self.pos = self.fp = self.neg = self.class_pos = 0.0


class mIoU:
    def __init__(self):
        self.reset()

    def update(self, preds, labels):
        target = _binary_target(labels)
        predict = (preds > 0).float()
        self.total_correct += float((predict * target).sum())
        self.total_label += float(target.sum())
        self.total_inter += float((predict * target).sum())
        self.total_union += float((predict + target - predict * target).sum())

    def get(self):
        pix_acc = self.total_correct / (self.total_label + np.spacing(1))
        miou = self.total_inter / (self.total_union + np.spacing(1))
        return pix_acc, miou

    def reset(self):
        self.total_inter = self.total_union = 0.0
        self.total_correct = self.total_label = 0.0


class PD_FA:
    def __init__(self):
        self.reset()

    def update(self, preds, labels, size):
        predicted_regions = list(measure.regionprops(measure.label(preds.detach().cpu().numpy(), connectivity=2)))
        target_regions = measure.regionprops(measure.label(labels.detach().cpu().numpy(), connectivity=2))
        self.target += len(target_regions)
        matched_count = 0
        for target_region in target_regions:
            target_centroid = np.asarray(target_region.centroid)
            for index, predicted_region in enumerate(predicted_regions):
                if np.linalg.norm(np.asarray(predicted_region.centroid) - target_centroid) < 3:
                    matched_count += 1
                    del predicted_regions[index]
                    break
        self.dismatch_pixel += sum(region.area for region in predicted_regions)
        self.all_pixel += int(size[0]) * int(size[1])
        self.detected += matched_count

    def get(self):
        pd = self.detected / self.target if self.target else 0.0
        fa = self.dismatch_pixel / self.all_pixel if self.all_pixel else 0.0
        return pd, fa

    def reset(self):
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.detected = 0
        self.target = 0
