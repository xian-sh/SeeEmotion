from __future__ import annotations

import json
import logging
import math
import random
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from autoencoder import FrozenAutoencoderKL
from config import Config, build_config, parse_args
from data import create_dataloaders
from losses import latent_loss, reconstruction_loss
from metrics import finalize_metric_sums, image_metrics, merge_metric_sums, save_reconstruction_grid
from model import EEGToFaceLatent


LOGGER = logging.getLogger(__name__)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log", encoding="utf-8"),
        ],
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def experiment_dir(config: Config) -> Path:
    return config.train.results_dir / config.data.dataset.lower() / config.train.protocol


def train_main(argv: list[str] | None = None, protocol: str | None = None) -> dict[str, Any]:
    args = parse_args()
    config = build_config(args)
    if protocol is not None:
        config.train.protocol = protocol
        if protocol == "single" and config.data.subject_ids is None:
            config.data.subject_ids = "1"
        if protocol == "cross" and config.data.split_mode == "trial":
            config.data.split_mode = "subject"
    return run_training(config)


def lower_is_better(metric: str) -> bool:
    return metric == "loss" or metric.endswith("mse") or metric.endswith("mae")


def is_better(current: float, best: float, metric: str) -> bool:
    if math.isnan(current):
        return False
    return current < best if lower_is_better(metric) else current > best


def run_training(config: Config) -> dict[str, Any]:
    output_dir = experiment_dir(config)
    setup_logging(output_dir)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Configuration:\n%s", json.dumps(_jsonable(config), indent=2))

    train_loader, val_loader, test_loader, split_info = create_dataloaders(config)
    LOGGER.info("Split info: %s", split_info)

    autoencoder = FrozenAutoencoderKL(
        checkpoint=config.autoencoder.checkpoint,
        model_type=config.autoencoder.model_type,
        emoji_size=config.data.emoji_size,
        latent_dim=config.autoencoder.latent_dim,
        device=device,
        sample_latent=config.autoencoder.sample_latent,
    )
    LOGGER.info("Autoencoder latent shape: %s", autoencoder.latent_shape)

    model = EEGToFaceLatent(
        data_config=config.data,
        model_config=config.model,
        latent_shape=autoencoder.latent_shape,
        num_sample_identities=1,
    ).to(device)
    LOGGER.info("EEG-to-latent trainable parameters: %d", sum(param.numel() for param in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.train.epochs),
        eta_min=config.train.min_lr,
    )

    best_metric = config.train.best_metric
    best_score = float("inf") if lower_is_better(best_metric) else -float("inf")
    best_epoch = 0
    best_state = None
    history: list[dict[str, float]] = []
    started_at = time.time()

    for epoch in range(1, config.train.epochs + 1):
        train_metrics = run_epoch(model, autoencoder, train_loader, device, config, optimizer, desc=f"train {epoch:03d}")
        val_metrics = run_epoch(model, autoencoder, val_loader, device, config, None, desc=f"val {epoch:03d}")
        scheduler.step()
        history.append({"epoch": epoch, **_prefix("train", train_metrics), **_prefix("val", val_metrics)})
        LOGGER.info(
            "Epoch %03d/%03d train_loss=%.6f train_mse=%.6f train_psnr=%.3f train_ssim=%.6f "
            "val_loss=%.6f val_mse=%.6f val_psnr=%.3f val_ssim=%.6f",
            epoch,
            config.train.epochs,
            train_metrics["loss"],
            train_metrics.get("mse", float("nan")),
            train_metrics.get("psnr", float("nan")),
            train_metrics.get("ssim", float("nan")),
            val_metrics["loss"],
            val_metrics.get("mse", float("nan")),
            val_metrics.get("psnr", float("nan")),
            val_metrics.get("ssim", float("nan")),
        )

        current_score = val_metrics.get(best_metric)
        if current_score is None:
            raise KeyError(f"Unknown best metric: {best_metric}. Available metrics: {sorted(val_metrics)}")
        if is_better(float(current_score), best_score, best_metric):
            best_score = float(current_score)
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            save_checkpoint(output_dir / "best_eeg2face_clean.pt", model, config, split_info, best_epoch, best_score)
            LOGGER.info("Saved best_eeg2face_clean.pt by val_%s=%.6f at epoch %03d.", best_metric, best_score, best_epoch)

        if config.train.save_every and epoch % config.train.save_every == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", model, config, split_info, epoch, float(current_score))

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train = run_epoch(
        model,
        autoencoder,
        train_loader,
        device,
        config,
        None,
        desc="final train",
        save_examples=output_dir / "train_reconstructions.png",
    )
    final_val = run_epoch(
        model,
        autoencoder,
        val_loader,
        device,
        config,
        None,
        desc="final val",
        save_examples=output_dir / "val_reconstructions.png",
    )
    final_test = run_epoch(
        model,
        autoencoder,
        test_loader,
        device,
        config,
        None,
        desc="final test",
        save_examples=output_dir / "test_reconstructions.png",
    )
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    results = {
        **split_info,
        "best_epoch": best_epoch,
        "best_metric": f"val_{best_metric}",
        "best_score": best_score,
        "training_seconds": time.time() - started_at,
        **_prefix("train", final_train),
        **_prefix("val", final_val),
        **_prefix("test", final_test),
        "output_dir": str(output_dir),
    }
    pd.DataFrame([results]).to_csv(output_dir / "results.csv", index=False)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(results), handle, indent=2)
    LOGGER.info("Final metrics: %s", results)
    return results


def run_epoch(
    model: EEGToFaceLatent,
    autoencoder: FrozenAutoencoderKL,
    loader,
    device: str,
    config: Config,
    optimizer: torch.optim.Optimizer | None,
    desc: str,
    save_examples: Path | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    autoencoder.eval()
    loss_sums: dict[str, float] = {"loss": 0.0, "latent_loss": 0.0, "recon_loss": 0.0}
    metric_sums: dict[str, float] = {}
    total = 0
    saved_targets = None
    saved_predictions = None

    for batch in tqdm(loader, desc=desc, leave=False):
        eeg = batch["eeg"].to(device, non_blocking=True).float()
        faces = batch["face"].to(device, non_blocking=True).float()
        subject_id = batch["subject_id"].to(device, non_blocking=True)
        trial_key = batch["trial_key"].to(device, non_blocking=True)
        frame_id = batch["frame_id"].to(device, non_blocking=True)
        identity_id = batch["identity_id"].to(device, non_blocking=True)
        with torch.no_grad():
            target_latent = autoencoder.encode(faces)

        if training:
            predicted_latent = model(eeg, subject_id, trial_key, frame_id, identity_id)
            predicted_faces = autoencoder.decode(predicted_latent)
            l_latent = latent_loss(predicted_latent, target_latent)
            l_recon = reconstruction_loss(predicted_faces, faces)
            loss = config.train.latent_loss_weight * l_latent + config.train.recon_loss_weight * l_recon
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()
        else:
            with torch.no_grad():
                predicted_latent = model(eeg, subject_id, trial_key, frame_id, identity_id)
                predicted_faces = autoencoder.decode(predicted_latent)
                l_latent = latent_loss(predicted_latent, target_latent)
                l_recon = reconstruction_loss(predicted_faces, faces)
                loss = config.train.latent_loss_weight * l_latent + config.train.recon_loss_weight * l_recon

        batch_size = eeg.shape[0]
        loss_sums["loss"] += float(loss.item()) * batch_size
        loss_sums["latent_loss"] += float(l_latent.item()) * batch_size
        loss_sums["recon_loss"] += float(l_recon.item()) * batch_size
        merge_metric_sums(metric_sums, image_metrics(predicted_faces, faces), batch_size)
        total += batch_size
        if save_examples is not None and saved_targets is None:
            saved_targets = faces[:8].detach().cpu()
            saved_predictions = predicted_faces[:8].detach().cpu()

    metrics = {key: value / max(1, total) for key, value in loss_sums.items()}
    metrics.update(finalize_metric_sums(metric_sums, total))
    if save_examples is not None and saved_targets is not None and saved_predictions is not None:
        save_reconstruction_grid(
            saved_targets,
            saved_predictions,
            save_examples,
            f"EEG-to-face clean, SSIM={metrics.get('ssim', float('nan')):.4f}",
        )
    return metrics


def save_checkpoint(path: Path, model: EEGToFaceLatent, config: Config, split_info: dict, epoch: int, score: float) -> None:
    payload = {
        "model": model.state_dict(),
        "config": _jsonable(config),
        "split_info": split_info,
        "epoch": epoch,
        "score": score,
    }
    torch.save(payload, path)


def _prefix(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _jsonable(config: Config) -> dict[str, Any]:
    payload = asdict(config)
    for section in payload.values():
        for key, value in list(section.items()):
            if isinstance(value, Path):
                section[key] = str(value)
    return payload


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value
