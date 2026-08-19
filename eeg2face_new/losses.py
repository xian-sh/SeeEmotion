from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    l1 = F.l1_loss(predicted, target)
    mse = F.mse_loss(predicted, target)
    return l1 + 0.5 * mse


def latent_loss(predicted_norm: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predicted_norm, target_norm)


def total_variation_loss(images: torch.Tensor) -> torch.Tensor:
    dy = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]).mean()
    dx = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]).mean()
    return dx + dy
