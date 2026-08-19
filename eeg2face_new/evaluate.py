from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from autoencoder import FrozenAutoencoderKL
from config import build_config, parse_args
from data import create_dataloaders
from metrics import finalize_metric_sums, image_metrics, merge_metric_sums, save_reconstruction_grid
from model import EEGToFaceLatent
from train_utils import resolve_device, set_seed, setup_logging

def main() -> None:
    args = parse_args()
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for evaluate.py.")
    config = build_config(args)
    output_dir = config.train.results_dir / config.data.dataset.lower() / "eval"
    setup_logging(output_dir)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)

    _, _, test_loader, split_info = create_dataloaders(config)
    autoencoder = FrozenAutoencoderKL(
        checkpoint=config.autoencoder.checkpoint,
        model_type=config.autoencoder.model_type,
        emoji_size=config.data.emoji_size,
        latent_dim=config.autoencoder.latent_dim,
        device=device,
        sample_latent=config.autoencoder.sample_latent,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_config = checkpoint.get("config", {})
    model_config = config.model
    if isinstance(saved_config, dict) and isinstance(saved_config.get("model"), dict):
        saved_model = saved_config["model"]
        model_config = replace(
            model_config,
            backbone_hidden_dim=int(saved_model.get("backbone_hidden_dim", model_config.backbone_hidden_dim)),
            backbone_depth=int(saved_model.get("backbone_depth", model_config.backbone_depth)),
            dropout=float(saved_model.get("dropout", model_config.dropout)),
            hidden_dims=tuple(saved_model.get("hidden_dims", model_config.hidden_dims)),
            use_subject_embedding=bool(saved_model.get("use_subject_embedding", model_config.use_subject_embedding)),
            subject_embedding_dim=int(saved_model.get("subject_embedding_dim", model_config.subject_embedding_dim)),
            use_trial_frame_embedding=bool(saved_model.get("use_trial_frame_embedding", model_config.use_trial_frame_embedding)),
            trial_embedding_dim=int(saved_model.get("trial_embedding_dim", model_config.trial_embedding_dim)),
            frame_embedding_dim=int(saved_model.get("frame_embedding_dim", model_config.frame_embedding_dim)),
            use_sample_embedding=False,
            sample_embedding_dim=0,
        )
    model = EEGToFaceLatent(
        data_config=config.data,
        model_config=model_config,
        latent_shape=autoencoder.latent_shape,
        num_sample_identities=1,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    metric_sums = {}
    total = 0
    saved_targets = None
    saved_predictions = None
    with torch.no_grad():
        for batch in test_loader:
            eeg = batch["eeg"].to(device).float()
            faces = batch["face"].to(device).float()
            subject_id = batch["subject_id"].to(device)
            trial_key = batch["trial_key"].to(device)
            frame_id = batch["frame_id"].to(device)
            identity_id = batch["identity_id"].to(device)
            predicted_latent = model(eeg, subject_id, trial_key, frame_id, identity_id)
            predicted = autoencoder.decode(predicted_latent)
            merge_metric_sums(metric_sums, image_metrics(predicted, faces), eeg.shape[0])
            total += eeg.shape[0]
            if saved_targets is None:
                saved_targets = faces[:8].detach().cpu()
                saved_predictions = predicted[:8].detach().cpu()

    metrics = finalize_metric_sums(metric_sums, total)
    payload = {**split_info, "test_samples": total, **metrics}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "eval_results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    pd.DataFrame([payload]).to_csv(output_dir / "eval_results.csv", index=False)
    if saved_targets is not None and saved_predictions is not None:
        save_reconstruction_grid(
            saved_targets,
            saved_predictions,
            output_dir / "eval_reconstructions.png",
            f"EEG-to-face evaluation, SSIM={metrics.get('ssim', float('nan')):.4f}",
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
