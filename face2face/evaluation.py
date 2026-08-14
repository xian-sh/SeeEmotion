"""Evaluate a trained AutoencoderKL on landmark-face reconstruction."""

import argparse
import json
import math
import random
from collections.abc import Mapping
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Face2FaceConfig
from data_preparation import collect_all_faces_data, get_valid_subject_ids


def set_seed(seed):
    """Set random seeds for reproducible evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_subject_ids(value):
    """Parse comma-separated IDs and ranges, for example: 1,3,5-8."""
    if value is None:
        return None

    subject_ids = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise argparse.ArgumentTypeError(f"Invalid subject range: {item}")
            subject_ids.update(range(start, end + 1))
        else:
            subject_ids.add(int(item))

    if not subject_ids:
        raise argparse.ArgumentTypeError("No valid subject IDs were provided.")
    return sorted(subject_ids)


def load_checkpoint(path, device):
    """Load a tensor-only checkpoint on both old and new PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_autoencoder_state(checkpoint):
    """Extract AutoencoderKL weights from supported checkpoint layouts."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("The checkpoint must contain a state-dict mapping.")

    if isinstance(checkpoint.get("face_autoencoder"), Mapping):
        state_dict = dict(checkpoint["face_autoencoder"])
    elif isinstance(checkpoint.get("state_dict"), Mapping):
        state_dict = dict(checkpoint["state_dict"])
    elif checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        state_dict = dict(checkpoint)
    else:
        raise KeyError(
            "Cannot find AutoencoderKL weights. Expected 'face_autoencoder', "
            "'state_dict', or a raw tensor state dict."
        )

    for prefix in ("module.face_autoencoder.", "face_autoencoder.", "module."):
        if state_dict and all(key.startswith(prefix) for key in state_dict):
            state_dict = {key[len(prefix) :]: value for key, value in state_dict.items()}
            break
    return state_dict


def create_autoencoder_kl(config):
    """Create the same AutoencoderKL architecture used by the training script."""
    try:
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
    except ImportError as error:
        raise ImportError(
            "AutoencoderKL requires diffusers. Install it before evaluation."
        ) from error

    model_configs = {
        "light": {
            "block_out_channels": (64, 128, 256),
            "layers_per_block": 1,
        },
        "medium": {
            "block_out_channels": (64, 128, 256, 512),
            "layers_per_block": 1,
        },
        "heavy": {
            "block_out_channels": (128, 256, 512, 512),
            "layers_per_block": 2,
        },
    }
    architecture = model_configs[config.model_type]
    n_blocks = len(architecture["block_out_channels"])
    model = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        down_block_types=("DownEncoderBlock2D",) * n_blocks,
        up_block_types=("UpDecoderBlock2D",) * n_blocks,
        block_out_channels=architecture["block_out_channels"],
        layers_per_block=architecture["layers_per_block"],
        act_fn="silu",
        latent_channels=config.latent_dim // 2,
        sample_size=config.emoji_size,
    )
    return model.to(config.device)


def load_model(model_path, config):
    """Build the configured AutoencoderKL and restore its trained weights."""
    checkpoint = load_checkpoint(model_path, config.device)
    checkpoint_class = checkpoint.get("model_class", checkpoint.get("autoencoder"))
    if checkpoint_class is not None and checkpoint_class != "AutoencoderKL":
        raise ValueError(
            f"Checkpoint reports model class '{checkpoint_class}', not AutoencoderKL."
        )

    model = create_autoencoder_kl(config)
    state_dict = extract_autoencoder_state(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(config.device)
    model.eval()
    return model, checkpoint


def reconstruct(model, faces, sample_latent=False):
    """Reconstruct faces using the posterior mean or a posterior sample."""
    posterior = model.encode(faces).latent_dist
    latent = posterior.sample() if sample_latent else posterior.mode()
    reconstructed = model.decode(latent).sample
    return reconstructed, latent


def calculate_batch_metrics(original, reconstructed):
    """Return per-image MSE, PSNR, and SSIM values."""
    original_np = np.clip(original.detach().cpu().numpy(), 0.0, 1.0)
    reconstructed_np = np.clip(reconstructed.detach().cpu().numpy(), 0.0, 1.0)

    mse_values = []
    psnr_values = []
    ssim_values = []

    for original_image, reconstructed_image in zip(original_np, reconstructed_np):
        original_image = original_image[0]
        reconstructed_image = reconstructed_image[0]
        mse_value = float(np.mean((original_image - reconstructed_image) ** 2))
        psnr_value = 100.0 if mse_value == 0.0 else 10.0 * math.log10(1.0 / mse_value)
        ssim_value = structural_similarity(
            original_image,
            reconstructed_image,
            data_range=1.0,
        )
        mse_values.append(mse_value)
        psnr_values.append(psnr_value)
        ssim_values.append(float(ssim_value))

    return {
        "MSE": np.asarray(mse_values, dtype=np.float64),
        "PSNR": np.asarray(psnr_values, dtype=np.float64),
        "SSIM": np.asarray(ssim_values, dtype=np.float64),
    }


def save_reconstruction_plot(originals, reconstructions, metrics, output_path):
    """Save side-by-side original and reconstructed examples."""
    n_images = len(originals)
    if n_images == 0:
        return

    figure, axes = plt.subplots(2, n_images, figsize=(2.0 * n_images, 4.2), squeeze=False)
    for index, (original, reconstruction) in enumerate(zip(originals, reconstructions)):
        axes[0, index].imshow(original[0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, index].imshow(reconstruction[0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, index].axis("off")
        axes[1, index].axis("off")

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstructed")
    figure.suptitle(
        "AutoencoderKL Reconstruction "
        f"(MSE={metrics['MSE']:.4f}, PSNR={metrics['PSNR']:.2f}, "
        f"SSIM={metrics['SSIM']:.4f})"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_latent_plots(latent_values, output_dir):
    """Save a latent histogram and a normal Q-Q diagnostic plot."""
    if latent_values.size == 0:
        return {}

    latent_stats = {
        "num_values": int(latent_values.size),
        "mean": float(latent_values.mean()),
        "std": float(latent_values.std()),
        "min": float(latent_values.min()),
        "max": float(latent_values.max()),
    }

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(latent_values, bins=50, density=True, alpha=0.7, color="#3973AC")
    axis.axvline(latent_stats["mean"], color="#C43C39", linestyle="--", label="Mean")
    axis.set_xlabel("Latent value")
    axis.set_ylabel("Density")
    axis.set_title("AutoencoderKL Latent Distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "kl_latent_distribution.png", dpi=200)
    plt.close(figure)

    rng = np.random.default_rng(42)
    qq_values = latent_values
    if qq_values.size > 5000:
        qq_values = rng.choice(qq_values, size=5000, replace=False)
    qq_values = np.sort(qq_values)
    probabilities = (np.arange(1, qq_values.size + 1) - 0.5) / qq_values.size
    normal = NormalDist()
    theoretical = np.fromiter(
        (normal.inv_cdf(float(probability)) for probability in probabilities),
        dtype=np.float64,
        count=qq_values.size,
    )

    slope, intercept = np.polyfit(theoretical, qq_values, deg=1)
    fitted = slope * theoretical + intercept

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(latent_values, bins=50, density=True, alpha=0.65, color="#3973AC")
    if latent_stats["std"] > 0:
        x_values = np.linspace(
            latent_stats["mean"] - 3 * latent_stats["std"],
            latent_stats["mean"] + 3 * latent_stats["std"],
            200,
        )
        density = np.exp(
            -0.5 * ((x_values - latent_stats["mean"]) / latent_stats["std"]) ** 2
        ) / (latent_stats["std"] * math.sqrt(2 * math.pi))
        axes[0].plot(x_values, density, color="#C43C39", linewidth=2)
    axes[0].set_title("Latent Distribution and Normal Fit")
    axes[0].set_xlabel("Latent value")
    axes[0].set_ylabel("Density")

    axes[1].scatter(theoretical, qq_values, s=8, alpha=0.45, color="#3973AC")
    axes[1].plot(theoretical, fitted, color="#C43C39", linewidth=2)
    axes[1].set_title("Normal Q-Q Plot")
    axes[1].set_xlabel("Theoretical quantile")
    axes[1].set_ylabel("Observed quantile")
    figure.tight_layout()
    figure.savefig(output_dir / "kl_normality_check.png", dpi=200)
    plt.close(figure)
    return latent_stats


def evaluate(config, model_path, subject_ids=None, num_visualizations=8, sample_latent=False,
             max_samples=None, num_workers=0):
    """Evaluate the checkpoint on all available test landmark faces."""
    set_seed(config.seed)
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    if subject_ids is None:
        if not Path(config.eeg_dir).is_dir():
            raise FileNotFoundError(
                f"EEG directory not found: {config.eeg_dir}. "
                "Set --eeg-dir or provide --subject-ids."
            )
        subject_ids = get_valid_subject_ids(config)
    if not subject_ids:
        raise RuntimeError("No subject IDs are available for evaluation.")

    print(f"Loading checkpoint: {model_path}")
    model, checkpoint = load_model(model_path, config)

    print(f"Collecting test faces for subjects: {subject_ids}")
    _, test_faces = collect_all_faces_data(subject_ids, config)
    if test_faces is None or len(test_faces) == 0:
        raise RuntimeError("No test landmark faces were collected.")
    if max_samples is not None:
        test_faces = test_faces[:max_samples]

    test_tensor = torch.as_tensor(test_faces, dtype=torch.float32)
    test_loader = DataLoader(
        test_tensor,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(config.device).startswith("cuda"),
    )

    output_dir = Path(config.results_dir) / "face_autoencoder" / "evaluation_kl"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_visualizations = min(num_visualizations, len(test_tensor))
    visualization_indices = set(
        np.linspace(0, len(test_tensor) - 1, num=n_visualizations, dtype=int).tolist()
    )
    originals = []
    reconstructions = []
    visualization_latents = []
    metric_values = {"MSE": [], "PSNR": [], "SSIM": []}
    sample_offset = 0

    with torch.inference_mode():
        for faces in tqdm(test_loader, desc="Evaluating AutoencoderKL"):
            faces = faces.to(config.device, non_blocking=True)
            reconstructed, latent = reconstruct(model, faces, sample_latent=sample_latent)
            batch_metrics = calculate_batch_metrics(faces, reconstructed)
            for name, values in batch_metrics.items():
                metric_values[name].append(values)

            for local_index in range(faces.size(0)):
                global_index = sample_offset + local_index
                if global_index in visualization_indices:
                    originals.append(faces[local_index].detach().cpu())
                    reconstructions.append(reconstructed[local_index].detach().cpu().clamp(0, 1))
                    visualization_latents.append(latent[local_index].detach().cpu())
            sample_offset += faces.size(0)

    metrics = {
        name: float(np.concatenate(values).mean())
        for name, values in metric_values.items()
    }
    latent_values = (
        torch.stack(visualization_latents).numpy().reshape(-1)
        if visualization_latents
        else np.empty(0, dtype=np.float32)
    )
    latent_stats = save_latent_plots(latent_values, output_dir)
    save_reconstruction_plot(
        originals,
        reconstructions,
        metrics,
        output_dir / "kl_reconstruction_samples.png",
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    results = {
        "model_path": str(model_path),
        "model_type": config.model_type,
        "emoji_size": config.emoji_size,
        "latent_dim": config.latent_dim,
        "latent_mode": "sample" if sample_latent else "posterior_mean",
        "subject_ids": subject_ids,
        "num_test_samples": len(test_tensor),
        "num_parameters": parameter_count,
        "checkpoint_model_class": checkpoint.get(
            "model_class", checkpoint.get("autoencoder", "unknown")
        ),
        "metrics": metrics,
        "visualization_latent_statistics": latent_stats,
    }
    with open(output_dir / "kl_evaluation_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nEvaluation results")
    print(f"  Test samples: {len(test_tensor)}")
    print(f"  MSE:  {metrics['MSE']:.6f}")
    print(f"  PSNR: {metrics['PSNR']:.6f}")
    print(f"  SSIM: {metrics['SSIM']:.6f}")
    print(f"  Results directory: {output_dir}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Face2Face AutoencoderKL.")
    parser.add_argument("--model", required=True, help="Path to the trained .pth checkpoint.")
    parser.add_argument("--model-type", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--emoji-size", type=int, default=56)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--results-dir", default="./face2face_results")
    parser.add_argument("--eeg-dir", default=None)
    parser.add_argument("--emoji-root-template", default=None)
    parser.add_argument(
        "--subject-ids",
        type=parse_subject_ids,
        default=None,
        help="Optional IDs such as 1,3,5-8. Otherwise IDs are discovered from --eeg-dir.",
    )
    parser.add_argument("--num-visualizations", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--sample-latent",
        action="store_true",
        help="Sample the KL posterior instead of using its deterministic mean/mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_visualizations < 0:
        raise ValueError("--num-visualizations must be non-negative.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")

    config = Face2FaceConfig(
        seed=args.seed,
        emoji_size=args.emoji_size,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        model_type=args.model_type,
        model_class="AutoencoderKL",
        results_dir=Path(args.results_dir),
    )
    if args.eeg_dir is not None:
        config.eeg_dir = Path(args.eeg_dir)
    if args.emoji_root_template is not None:
        config.emoji_root_template = args.emoji_root_template
    if args.device is not None:
        config.device = args.device

    evaluate(
        config=config,
        model_path=args.model,
        subject_ids=args.subject_ids,
        num_visualizations=args.num_visualizations,
        sample_latent=args.sample_latent,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
