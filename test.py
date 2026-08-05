from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from torch.utils.data import DataLoader
from tqdm import tqdm
import threading
from util.dataset_resize import *
from util.metrics import *
from util.utils import *
import time
from collections import OrderedDict
from model.SP_KAN import SP_KAN as SP_KAN
import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument("--model_names", default=['SP_KAN'], nargs='+', choices=['SP_KAN'],
                    help="Model architecture to evaluate")
parser.add_argument("--pth_dirs", default=['SIRST3/SP-KAN-best.pth.tar'], nargs='+')
parser.add_argument("--dataset_dir", default=r'./datasets', type=str, help="train_dataset_dir")
parser.add_argument("--dataset_names", default=['SIRST3'], nargs='+',
                    help="dataset_name: 'SIRST3','NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K'")
parser.add_argument("--patchSize_eva", type=int, default=512, help="Evaluation patch size")
parser.add_argument("--threads", type=int, default=0, help="Number of data loader workers")
parser.add_argument("--img_norm_cfg", default=None,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--save_img", "--save-img", default=True, action=argparse.BooleanOptionalAction,
                    help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default=r'./Result/',
                    help="path of saved image")
parser.add_argument("--save_log", type=str, default=r'./log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--context", choices=['focal', 'cvit'], default='focal',
                    help="Context encoder used by the checkpoint")
parser.add_argument("--max_test_steps", type=int, default=None,
                    help="Optional cap on evaluated batches (useful for smoke tests)")

opt = None


def test() -> None:
    test_set = TestSetLoader_Re_Pad(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.patchSize_eva,
                                    opt.img_norm_cfg)
    worker_options = {'persistent_workers': True} if opt.threads else {}
    test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False,
                             **worker_options)
    # *************************固定阈值**********************
    # 计算mIOU
    IOU = mIoU()
    # 计算nIOU
    nIoU_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=opt.threshold)
    # 计算PD_FA
    eval_05 = PD_FA()
    metric_f1 = F1(opt.threshold)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SP_KAN(1, 1, mode='test', deepsuper=True, context=opt.context).to(device)
    if not os.path.isfile(opt.pth_dir):
        raise FileNotFoundError(
            f'Checkpoint not found: {opt.pth_dir}. Set --pth_dirs to a checkpoint relative to --save_log.'
        )
    state_dict = torch.load(opt.pth_dir, map_location=device, weights_only=False)
    new_state_dict = OrderedDict()
    #
    for k, v in state_dict.get('state_dict', state_dict).items():
        name = k.removeprefix('module.').removeprefix('model.')
        new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
    net.load_state_dict(new_state_dict)
    net.eval()
    tbar = tqdm(test_loader)
    with torch.no_grad():
        for idx_iter, (img, gt_mask, target_size, org_size, img_dir) in enumerate(tbar):
            # img = Variable(img)
            pred = net(img.to(device))
            # pred = pred[:, :, :size[0], :size[1]]

            pred = align_prediction_to_target(pred, gt_mask, target_size, org_size)

            gt_mask = gt_mask.to(device)

            pred = torch.clamp(pred, min=0.0, max=1.0)
            # Fix  threshold ##########################################################
            # IOU
            IOU.update((pred > opt.threshold), gt_mask)  # 像素
            # nIOU
            nIoU_metric.update(pred, gt_mask)  # 像素
            eval_05.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], org_size)  # 目标
            metric_f1.update(pred, gt_mask)
            # ROC_05.update
            # save img
            if opt.save_img == True:
                # A  二值图:
                predA = pred
                predB = pred

                predict = (predA[0, 0, :, :] > opt.threshold).float().cpu()
                img_save_b = transforms.ToPILImage()(predict)

                # B  显著图:
                img_save_s = transforms.ToPILImage()((predB[0, 0, :, :]).cpu())

                binary_dir = os.path.join(opt.save_img_dir, opt.test_dataset_name, opt.model_name, 'Binary')
                silence_dir = os.path.join(opt.save_img_dir, opt.test_dataset_name, opt.model_name, 'Silance')
                os.makedirs(binary_dir, exist_ok=True)
                os.makedirs(silence_dir, exist_ok=True)
                img_save_b.save(os.path.join(binary_dir, img_dir[0] + '.png'))
                img_save_s.save(os.path.join(silence_dir, img_dir[0] + '.png'))
            if opt.max_test_steps is not None and idx_iter + 1 >= opt.max_test_steps:
                break

        # 0.5

        pixAcc, mIOU = IOU.get()

        nIoU = nIoU_metric.get()

        results2 = eval_05.get()

        F1_score = metric_f1.get()

        print('pixAcc: %.4f| mIoU: %.4f | nIoU: %.4f | Pd: %.4f| Fa: %.4f |F1: %.4f'
              % (pixAcc * 100, mIOU * 100, nIoU * 100, results2[0] * 100, results2[1] * 1e+6, F1_score * 100))
        with open(opt.metrics_path, 'a') as metrics_file:
            metrics_file.write(json.dumps({
                'run_id': f'{opt.test_dataset_name}_{opt.model_name}',
                'dataset': opt.test_dataset_name,
                'model': opt.model_name,
                'checkpoint': opt.pth_dir,
                'threshold': opt.threshold,
                'pixacc': float(pixAcc),
                'miou': float(mIOU),
                'niou': float(nIoU),
                'pd': float(results2[0]),
                'fa': float(results2[1]),
                'f1': float(F1_score),
            }, ensure_ascii=True) + '\n')

def postprocess_masks(
    pred: torch.Tensor,
    input_size: Sequence[int | torch.Tensor],
    original_size: Sequence[int | torch.Tensor],
) -> torch.Tensor:
    input_h, input_w = (int(value.item()) if isinstance(value, torch.Tensor) else int(value)
                        for value in input_size)
    original_h, original_w = (int(value.item()) if isinstance(value, torch.Tensor) else int(value)
                              for value in original_size)
    preds = pred[..., :input_h, :input_w]
    preds = F.interpolate(preds, (original_h, original_w), mode="bicubic", align_corners=False)

    return preds


def align_prediction_to_target(
    pred: torch.Tensor,
    gt_mask: torch.Tensor,
    input_size: Sequence[int | torch.Tensor],
    original_size: Sequence[int | torch.Tensor],
) -> torch.Tensor:
    """Postprocess and, if necessary, force prediction to GT spatial size."""
    if pred.shape[-2:] != gt_mask.shape[-2:]:
        pred = postprocess_masks(pred, input_size, original_size)
    if pred.shape[-2:] != gt_mask.shape[-2:]:
        pred = F.interpolate(pred, size=gt_mask.shape[-2:], mode='bilinear', align_corners=False)
    return pred


def cal_tp_pos_fp_neg(
    output: torch.Tensor,
    target: torch.Tensor,
    nclass: int,
    score_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    predict = (output > score_thresh).float()
    if len(target.shape) == 3:
        print('？？？？')  # 加一个维度 使得target与 output的size一致
        target = target.unsqueeze(dim=0)
        target.to('cuda', torch.float)

    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    # 现在predict中高于阈值的部分为全1矩阵   target是GT

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()  # 对的预测为对的
    fp = (predict * ((predict != target).float())).sum()  # 错的预测为对的 虚警像素数
    tn = ((1 - predict) * ((predict == target).float())).sum()  # 错的预测为错的
    fn = (((predict != target).float()) * (1 - predict)).sum()  # 对的预测为错的
    pos = tp + fn  # 标签中 阳性的个数
    neg = fp + tn  # 标签中 阴性的个数
    class_pos = tp + fp  # 检测出的个数

    return tp, pos, fp, neg, class_pos


class SamplewiseSigmoidMetric(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass, score_thresh=0.5):
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.lock = threading.Lock()
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NDArray' or list of `NDArray`
            The labels of the data.

        preds : 'NDArray' or list of `NDArray`
            Predicted values.
        """

        def evaluate_worker(self, label, pred):
            inter_arr, union_arr = batch_intersection_union_n(
                pred, label, self.nclass, self.score_thresh)
            with self.lock:
                self.total_inter = np.append(self.total_inter, inter_arr)
                self.total_union = np.append(self.total_union, union_arr)

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                        )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        nIoU = IoU.mean()
        return nIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = np.array([])
        self.total_union = np.array([])
        self.total_correct = np.array([])
        self.total_label = np.array([])


def batch_intersection_union_n(output, target, nclass, score_thresh):
    """nIoU"""
    mini = 1
    maxi = 1  # nclass
    nbins = 1  # nclass
    outputnp = output.detach().cpu().numpy()
    # outputsig = F.sigmoid(output).detach().cpu().numpy()
    # outputsig = nd.sigmoid(output).asnumpy()
    predict = (outputnp > score_thresh).astype('int64')
    # predict = predict.detach().cpu().numpy()
    # predict = (output.asnumpy() > 0).astype('int64') # P
    if len(target.shape) == 3:
        target = target.unsqueeze(1).cpu().numpy().astype('int64')  # T
    elif len(target.shape) == 4:
        target = target.cpu().numpy().astype('int64')  # T
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * (predict == target)  # TP  交集

    num_sample = intersection.shape[0]
    area_inter_arr = np.zeros(num_sample)
    area_pred_arr = np.zeros(num_sample)
    area_lab_arr = np.zeros(num_sample)
    area_union_arr = np.zeros(num_sample)
    for b in range(num_sample):
        # areas of intersection and union
        area_inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
        area_inter_arr[b] = area_inter.item()

        area_pred, _ = np.histogram(predict[b], bins=nbins, range=(mini, maxi))
        area_pred_arr[b] = area_pred.item()

        area_lab, _ = np.histogram(target[b], bins=nbins, range=(mini, maxi))
        area_lab_arr[b] = area_lab.item()

        area_union = area_pred + area_lab - area_inter
        area_union_arr[b] = area_union.item()

        assert (area_inter <= area_union).all(), \
            "Intersection area should be smaller than Union area"

    return area_inter_arr, area_union_arr


class ROCMetric05():
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass, bins):
        # bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        # nclass :有几个类别 红外弱小目标检测只有一个类别
        super(ROCMetric05, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)
        # self.reset()

    # 网络输入的结果和标签 计算两者之前的东西
    def update(self, preds, labels):
        for iBin in range(self.bins + 1):
            # score_thresh = (iBin + 0.0) / self.bins
            score_thresh = (0.0 + iBin) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg, i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass, score_thresh)
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp  # 虚警像素数
            self.neg_arr[iBin] += i_neg
            self.class_pos[iBin] += i_class_pos

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)  # tp_rates = recall = TP/(TP+FN)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)  # fp_rates =  FP/(FP+TN)
        FP = self.fp_arr / (self.neg_arr + self.pos_arr)
        recall = self.tp_arr / (self.pos_arr + 0.001)  # recall = TP/(TP+FN)
        precision = self.tp_arr / (self.class_pos + 0.001)  # precision = TP/(TP+FP)
        f1_score = (2.0 * recall[5] * precision[5]) / (recall[5] + precision[5] + 0.00001)

        return tp_rates, fp_rates, recall, precision, FP, f1_score

    def reset(self):
        self.tp_arr = np.zeros([11])
        self.pos_arr = np.zeros([11])
        self.fp_arr = np.zeros([11])
        self.neg_arr = np.zeros([11])
        self.class_pos = np.zeros([11])


class mIoU():

    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        correct, labeled = batch_pix_accuracy(preds, labels)  # labeled: GT中目标的像素数目   correct:预测正确的像素数
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return float(pixAcc), mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


class PDFA():
    def __init__(self, ):
        super(PDFA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0

    def update(self, preds, labels, size):
        predits = preds.detach().cpu().numpy().astype('int64')
        labelss = labels.detach().cpu().numpy().astype('int64')

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match = []
        self.dismatch = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)

                    del coord_image[m]
                    break

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
        self.dismatch_pixel += np.sum(self.dismatch)
        self.all_pixel += size[0] * size[1]
        self.PD += len(self.distance_match)

    def get(self):
        if self.all_pixel == 0:
            return 0.0, 0.0
        final_fa = float(self.dismatch_pixel) / float(self.all_pixel)
        final_pd = float(self.PD) / float(self.target) if self.target else 0.0
        return final_pd, final_fa

    def reset(self):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])


def batch_pix_accuracy(output, target):
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float()) * ((target > 0)).float()).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _ = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union


class PD_FA():
    def __init__(self, ):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0

    def update(self, preds, labels, size):
        predits = preds.detach().cpu().numpy().astype('int64')
        labelss = labels.detach().cpu().numpy().astype('int64')

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)  # 目标总数  直接就搞GT的连通域个数
        self.image_area_total = []  # 图像中预测的区域列表
        self.image_area_match = []
        self.distance_match = []
        self.dismatch = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        for i in range(len(coord_label)):  # image 与 label 之间 根据中心点 进行连通域的确定
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)

                    del coord_image[m]  # 匹配上一个之后就 清除一个
                    break

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]  # 在image里面 但是不在label里面

        self.dismatch_pixel += np.sum(self.dismatch)  # Fa 虚警个数 像素的虚警
        # print(self.dismatch_pixel)
        self.all_pixel += int(size[0]) * int(size[1])
        self.PD += len(self.distance_match)  # 如果中心点之间距离在3一下 就算Pd  所以Pd 是匹配上了的目标的个数

    def get(self):
        if self.all_pixel == 0:
            return 0.0, 0.0
        final_fa = float(self.dismatch_pixel) / float(self.all_pixel)
        final_pd = float(self.PD) / float(self.target) if self.target else 0.0
        return final_pd, final_fa

    def reset(self):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])


def _broadcast(values, count, name):
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f'{name} must contain one value or exactly {count} values.')
    return values


def evaluation_jobs(model_names, dataset_names, checkpoint_paths):
    unsupported = set(model_names) - {'SP_KAN'}
    if unsupported:
        raise ValueError(f'Unsupported model names: {sorted(unsupported)}')
    count = max(len(model_names), len(dataset_names), len(checkpoint_paths))
    return zip(
        _broadcast(model_names, count, '--model_names'),
        _broadcast(dataset_names, count, '--dataset_names'),
        _broadcast(checkpoint_paths, count, '--pth_dirs'),
    )


def resolve_checkpoint_path(checkpoint_path, save_log):
    """Resolve absolute, existing relative, or save-log-relative checkpoints."""
    checkpoint_path = os.path.expanduser(checkpoint_path)
    if os.path.isabs(checkpoint_path) or os.path.isfile(checkpoint_path):
        return os.path.abspath(checkpoint_path)
    return os.path.abspath(os.path.join(save_log, checkpoint_path))


def main():
    global opt
    opt = parser.parse_args()
    os.makedirs(opt.save_log, exist_ok=True)
    opt.metrics_path = os.path.join(opt.save_log, 'test_metrics.jsonl')
    log_path = os.path.join(opt.save_log, 'test_' + time.strftime('%Y%m%d_%H%M%S') + '.txt')
    with open(log_path, 'w') as opt.f:
        for model_name, dataset_name, pth_dir in evaluation_jobs(
                opt.model_names, opt.dataset_names, opt.pth_dirs):
            opt.model_name = model_name
            opt.train_dataset_name = dataset_name
            opt.test_dataset_name = dataset_name
            opt.pth_dir = resolve_checkpoint_path(pth_dir, opt.save_log)
            print(pth_dir)
            print(dataset_name)
            opt.f.write(f'{pth_dir}\n{dataset_name}\n')
            test()
            print()
            opt.f.write('\n')


if __name__ == '__main__':
    main()
