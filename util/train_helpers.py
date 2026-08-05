import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from model.SP_KAN import SP_KAN
from util.losses import TargetAwareBoundaryLoss, deep_supervision_loss


class Net(nn.Module):
    def __init__(self, model_name, mode, loss_name='target_aware', boundary_weight=2.0,
                 dice_weight=1.0, max_pos_weight=20.0, restoration_branch=True):
        super().__init__()
        self.model_name = model_name
        if loss_name == 'bce':
            self.cal_loss = nn.BCELoss()
        elif loss_name == 'target_aware':
            self.cal_loss = TargetAwareBoundaryLoss(
                boundary_weight=boundary_weight,
                dice_weight=dice_weight,
                max_pos_weight=max_pos_weight,
            )
        else:
            raise ValueError(f'Unsupported loss: {loss_name}')
        self.loss_name = loss_name
        if model_name != 'SP_KAN':
            raise ValueError(f'Unsupported model: {model_name}')
        self.model = SP_KAN(1, 1, mode=mode, deepsuper=True, restoration_branch=restoration_branch)
        print('input channels: 1')
        print('---------------------------------------------------------------')

    def forward(self, img):
        return self.model(img)

    def loss(self, preds, gt_masks):
        if isinstance(self.cal_loss, TargetAwareBoundaryLoss):
            return deep_supervision_loss(self.cal_loss, preds, gt_masks)
        if isinstance(preds, list):
            return sum(self.cal_loss(pred, gt_masks[i]) for i, pred in enumerate(preds)) / len(preds)
        if isinstance(preds, tuple):
            return sum(self.cal_loss(pred, gt_masks) for pred in preds)
        return self.cal_loss(preds, gt_masks)


def weights_init_kaiming(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(module.weight.data, 1.0, 0.02)
        init.constant_(module.bias.data, 0.0)


def save_checkpoint(state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)
    return save_path


def postprocess_masks(pred, input_size, original_size):
    input_size = tuple(int(v.item()) if isinstance(v, torch.Tensor) else int(v) for v in input_size)
    original_size = tuple(int(v.item()) if isinstance(v, torch.Tensor) else int(v) for v in original_size)
    pred = pred[..., :input_size[0], :input_size[1]]
    return F.interpolate(pred, original_size, mode='bilinear', align_corners=False)
