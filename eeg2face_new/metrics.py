from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def image_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = predicted.detach().cpu().clamp(0.0, 1.0).numpy()
    tgt = target.detach().cpu().clamp(0.0, 1.0).numpy()
    mse = float(np.mean((pred - tgt) ** 2))
    mae = float(np.mean(np.abs(pred - tgt)))
    psnr = 100.0 if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    foreground = tgt > 0.05
    foreground_error = (pred - tgt)[foreground]
    foreground_mse = float(np.mean(foreground_error**2)) if foreground_error.size else float("nan")

    pred_binary = pred > 0.10
    tgt_binary = tgt > 0.10
    intersection = float(np.logical_and(pred_binary, tgt_binary).sum())
    union = float(np.logical_or(pred_binary, tgt_binary).sum())
    pred_area = float(pred_binary.sum())
    tgt_area = float(tgt_binary.sum())
    dice = 2.0 * intersection / max(pred_area + tgt_area, 1.0)
    iou = intersection / max(union, 1.0)

    ssim_values = []
    crop_ssim_values = []
    try:
        from skimage.metrics import structural_similarity

        for pred_img, tgt_img in zip(pred, tgt):
            target_image = tgt_img[0]
            predicted_image = pred_img[0]
            ssim_values.append(structural_similarity(target_image, predicted_image, data_range=1.0))
            target_crop, predicted_crop = crop_pair_to_foreground(target_image, predicted_image)
            crop_ssim_values.append(structural_similarity(target_crop, predicted_crop, data_range=1.0))
        ssim = float(np.mean(ssim_values))
        crop_ssim = float(np.mean(crop_ssim_values))
    except Exception:
        ssim = float("nan")
        crop_ssim = float("nan")
    return {
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
        "ssim": ssim,
        "foreground_mse": foreground_mse,
        "crop_ssim": crop_ssim,
        "dice": float(dice),
        "iou": float(iou),
    }


def crop_pair_to_foreground(target: np.ndarray, predicted: np.ndarray, margin: int = 4) -> tuple[np.ndarray, np.ndarray]:
    mask = target > 0.05
    if not mask.any():
        return target, predicted
    rows, cols = np.where(mask)
    height, width = target.shape
    row_start = max(0, int(rows.min()) - margin)
    row_end = min(height, int(rows.max()) + margin + 1)
    col_start = max(0, int(cols.min()) - margin)
    col_end = min(width, int(cols.max()) + margin + 1)
    target_crop = target[row_start:row_end, col_start:col_end]
    predicted_crop = predicted[row_start:row_end, col_start:col_end]
    if min(target_crop.shape) < 7:
        return target, predicted
    return target_crop, predicted_crop


def merge_metric_sums(metric_sums: dict[str, float], metrics: dict[str, float], batch_size: int) -> None:
    for key, value in metrics.items():
        if not math.isnan(value):
            metric_sums[key] = metric_sums.get(key, 0.0) + value * batch_size


def finalize_metric_sums(metric_sums: dict[str, float], total: int) -> dict[str, float]:
    return {key: value / max(1, total) for key, value in metric_sums.items()}


def save_reconstruction_grid(targets: torch.Tensor, predictions: torch.Tensor, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets = targets.detach().cpu().clamp(0.0, 1.0)
    predictions = predictions.detach().cpu().clamp(0.0, 1.0)
    n_images = min(8, targets.shape[0])
    if n_images == 0:
        return

    figure, axes = plt.subplots(2, n_images, figsize=(2.0 * n_images, 4.0), squeeze=False)
    for index in range(n_images):
        axes[0, index].imshow(targets[index, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, index].imshow(predictions[index, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    axes[0, 0].set_title("Target")
    axes[1, 0].set_title("Predicted")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
