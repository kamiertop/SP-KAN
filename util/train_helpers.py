import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from model.SP_KAN import SP_KAN


class Net(nn.Module):
    def __init__(self, model_name, mode, loss_name='bce', cross_view=False,
                 cross_view_noise_std=0.03, cross_view_topk=0.2,
                 cross_view_consistency_weight=0.1, mamba_branch=False,
                 central_contrast=False):
        super().__init__()
        self.model_name = model_name
        if loss_name != 'bce':
            raise ValueError(f'Unsupported loss: {loss_name}')
        self.cal_loss = nn.BCELoss()
        self.loss_name = loss_name
        if model_name != 'SP_KAN':
            raise ValueError(f'Unsupported model: {model_name}')
        self.cross_view = cross_view
        self.cross_view_noise_std = cross_view_noise_std
        self.cross_view_consistency_weight = float(cross_view_consistency_weight)
        if self.cross_view_consistency_weight < 0:
            raise ValueError('cross_view_consistency_weight must be non-negative')
        self.mamba_branch = mamba_branch
        self.model = SP_KAN(1, 1, mode=mode, deepsuper=True,
                            cross_view_branch=cross_view, cross_view_topk=cross_view_topk,
                            mamba_branch=mamba_branch, central_contrast=central_contrast)
        self._consistency = None
        print('input channels: 1')
        print('---------------------------------------------------------------')

    def forward(self, img):
        clean = self.model(img)
        self._consistency = None
        if self.cross_view and self.training:
            noisy_img = (img + torch.randn_like(img) * self.cross_view_noise_std).clamp(0, 1)
            noisy = self.model(noisy_img)
            cp = clean[-1] if isinstance(clean, (list, tuple)) else clean
            np = noisy[-1] if isinstance(noisy, (list, tuple)) else noisy
            self._consistency = (cp - np).pow(2).mean()
        return clean

    def loss(self, preds, gt_masks):
        # Every deep-supervision head uses the same batched ground-truth mask.
        if isinstance(preds, (list, tuple)):
            loss = sum(self.cal_loss(pred, gt_masks) for pred in preds) / len(preds)
        else:
            loss = self.cal_loss(preds, gt_masks)
        # Cross-view alignment is orthogonal to the pixel loss and should also
        # regularize the BCE baseline when that branch is enabled.
        return loss + (self.cross_view_consistency_weight * self._consistency
                       if self._consistency is not None else 0.0)


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


def align_prediction_to_target(pred, gt_mask, input_size, original_size):
    """Restore an evaluation prediction and guarantee its shape matches its label."""
    pred = postprocess_masks(pred, input_size, original_size)
    if pred.shape[-2:] != gt_mask.shape[-2:]:
        pred = F.interpolate(pred, size=gt_mask.shape[-2:], mode='bilinear', align_corners=False)
    return pred
