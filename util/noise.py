"""Synthetic infrared corruption and prediction-consistency helpers."""

from __future__ import annotations

import torch


def add_infrared_noise(image, gaussian_std=0.04, stripe_prob=0.25, dead_pixel_prob=0.01):
    """Create a differentiable noisy view without changing the target mask."""
    noisy = image + torch.randn_like(image) * gaussian_std
    if stripe_prob > 0:
        batch, _, height, width = image.shape
        row_mask = (torch.rand(batch, 1, height, 1, device=image.device) < stripe_prob).float()
        col_mask = (torch.rand(batch, 1, 1, width, device=image.device) < stripe_prob).float()
        stripe = (row_mask * torch.randn(batch, 1, height, 1, device=image.device) * gaussian_std * 2.0
                  + col_mask * torch.randn(batch, 1, 1, width, device=image.device) * gaussian_std * 2.0)
        noisy = noisy + stripe
    if dead_pixel_prob > 0:
        dead = torch.rand_like(noisy) < dead_pixel_prob
        noisy = torch.where(dead, torch.zeros_like(noisy), noisy)
    return noisy


def prediction_consistency_loss(clean, noisy):
    """Symmetric probability consistency for a clean/noisy prediction pair."""
    if isinstance(clean, (tuple, list)):
        clean = clean[-1]
    if isinstance(noisy, (tuple, list)):
        noisy = noisy[-1]
    clean = clean.clamp(1e-5, 1.0 - 1e-5)
    noisy = noisy.clamp(1e-5, 1.0 - 1e-5)
    kl_clean = clean * (clean.log() - noisy.log()) + (1 - clean) * ((1 - clean).log() - (1 - noisy).log())
    kl_noisy = noisy * (noisy.log() - clean.log()) + (1 - noisy) * ((1 - noisy).log() - (1 - clean).log())
    return 0.5 * (kl_clean + kl_noisy).mean()
