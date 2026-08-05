"""Losses for highly imbalanced infrared small-target segmentation.

The target-aware loss is intentionally implemented on probabilities because the
SP-KAN forward pass already applies ``sigmoid``.  It can therefore be used for
both the final prediction and the deep-supervision predictions without changing
checkpoint compatibility.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


class AdaptiveThresholdFocalResidual(torch.nn.Module):
    """Focal residual whose hard-example threshold follows target sparsity."""

    def __init__(self, gamma=2.0, threshold_scale=0.2, temperature=0.1):
        super().__init__()
        if gamma < 0 or threshold_scale < 0 or temperature <= 0:
            raise ValueError('invalid adaptive focal hyperparameters')
        self.gamma = float(gamma)
        self.threshold_scale = float(threshold_scale)
        self.temperature = float(temperature)

    def forward(self, prediction, target):
        target = target.float().clamp(0.0, 1.0)
        if prediction.shape != target.shape:
            target = F.interpolate(target, size=prediction.shape[-2:], mode='nearest')
        prediction = prediction.float().clamp(1e-6, 1.0 - 1e-6)
        positive_ratio = target.mean().detach()
        threshold = (0.5 + self.threshold_scale * (0.1 - positive_ratio)).clamp(0.5, 0.9)
        pt = prediction * target + (1.0 - prediction) * (1.0 - target)
        hard_weight = torch.sigmoid((threshold - pt) / self.temperature)
        focal_weight = (1.0 - pt).pow(self.gamma) * (0.5 + hard_weight)
        return (focal_weight * F.binary_cross_entropy(prediction, target, reduction='none')).mean()


class TargetAwareBoundaryLoss(torch.nn.Module):
    """Foreground-ratio adaptive BCE + boundary Dice loss.

    Small targets occupy very few pixels.  The positive term is reweighted by
    the foreground/background ratio of each batch, while a cheap morphological
    gradient gives extra weight to target contours.  The Dice term keeps the
    loss useful when a target contains only a handful of pixels.
    """

    def __init__(
        self,
        boundary_weight: float = 2.0,
        dice_weight: float = 1.0,
        max_pos_weight: float = 20.0,
        smooth: float = 1.0,
        focal_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_threshold_scale: float = 0.2,
        focal_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if boundary_weight < 0 or dice_weight < 0 or focal_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if max_pos_weight < 1:
            raise ValueError("max_pos_weight must be at least 1")
        self.boundary_weight = float(boundary_weight)
        self.dice_weight = float(dice_weight)
        self.max_pos_weight = float(max_pos_weight)
        self.smooth = float(smooth)
        self.focal_weight = float(focal_weight)
        self.adaptive_focal = AdaptiveThresholdFocalResidual(
            gamma=focal_gamma, threshold_scale=focal_threshold_scale,
            temperature=focal_temperature,
        )

    @staticmethod
    def _boundary_map(target: torch.Tensor) -> torch.Tensor:
        """Return a one-pixel morphological boundary for a binary target."""
        dilated = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-target, kernel_size=3, stride=1, padding=1)
        return (dilated - eroded).clamp_(0.0, 1.0)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            target = F.interpolate(target.float(), size=prediction.shape[-2:], mode="nearest")
        # Do not mutate the shared GT tensor: deep supervision evaluates this
        # criterion repeatedly and autograd may retain it for earlier heads.
        target = target.float().clamp(0.0, 1.0)
        prediction = prediction.float().clamp(1e-6, 1.0 - 1e-6)

        positive = target.sum()
        negative = target.numel() - positive
        ratio = negative / (positive + self.smooth)
        pos_weight = ratio.clamp(min=1.0, max=self.max_pos_weight)

        boundary = self._boundary_map(target)
        pixel_weight = 1.0 + self.boundary_weight * boundary
        pixel_weight = pixel_weight * (1.0 + (pos_weight - 1.0) * target)
        bce = (F.binary_cross_entropy(prediction, target, reduction="none") * pixel_weight).mean()

        intersection = (prediction * target).sum()
        dice = 1.0 - (2.0 * intersection + self.smooth) / (
            prediction.sum() + target.sum() + self.smooth
        )
        focal = self.adaptive_focal(prediction, target) if self.focal_weight else prediction.new_zeros(())
        return bce + self.dice_weight * dice + self.focal_weight * focal


def deep_supervision_loss(
    criterion: torch.nn.Module,
    predictions: torch.Tensor | Sequence[torch.Tensor],
    target: torch.Tensor,
) -> torch.Tensor:
    """Apply a criterion to SP-KAN's deep-supervision tuple.

    Earlier decoder outputs receive a smaller weight and the weighted sum is
    normalized, so changing the number of supervision heads does not change the
    loss scale.
    """
    if isinstance(predictions, (tuple, list)):
        weights = (0.5, 0.5, 0.75, 1.0, 1.0, 1.0)
        active = weights[: len(predictions)]
        weighted = [weight * criterion(prediction, target) for weight, prediction in zip(active, predictions)]
        return sum(weighted) / sum(active)
    return criterion(predictions, target)
