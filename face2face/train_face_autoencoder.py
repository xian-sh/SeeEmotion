import argparse
import random
import time
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

from autoencoders import count_parameters_mb, create_face_autoencoder
from config import Face2FaceConfig
from data_preparation import collect_all_faces_data, get_valid_subject_ids


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def calculate_metrics(original_images, generated_images):
    """Calculate MSE, PSNR, and SSIM for reconstructed images."""
    original = original_images.detach().cpu().numpy()
    generated = generated_images.detach().cpu().numpy()

    mse_values = []
    psnr_values = []
    ssim_values = []

    for i in range(original.shape[0]):
        orig_img = np.clip(original[i, 0], 0, 1)
        gen_img = np.clip(generated[i, 0], 0, 1)

        mse_val = mse(orig_img, gen_img)
        mse_values.append(mse_val)
        psnr_values.append(100 if mse_val == 0 else psnr(orig_img, gen_img, data_range=1.0))
        ssim_values.append(ssim(orig_img, gen_img, data_range=1.0))

    return {
        "MSE": np.mean(mse_values),
        "PSNR": np.mean(psnr_values),
        "SSIM": np.mean(ssim_values),
    }


def train_face_autoencoder(model, train_loader, val_loader, config: Face2FaceConfig, use_kl_model=False):
    """Train the face autoencoder and save the best checkpoint."""
    print("Start face autoencoder training...")
    start_time = time.time()

    if use_kl_model or hasattr(model, "face_autoencoder"):
        optimizer = optim.Adam(model.face_autoencoder.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.Adam(
            list(model.face_encoder.parameters()) + list(model.face_decoder.parameters()),
            lr=config.learning_rate,
        )

    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.5)

    best_loss = float("inf")
    best_model_state = None

    output_dir = Path(config.results_dir) / "face_autoencoder"
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_train_losses = []
    epoch_val_losses = []
    epoch_times = []

    for epoch in range(config.epochs):
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0

        for faces in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs} train"):
            faces = faces.to(config.device)

            optimizer.zero_grad()
            reconstructed_faces, _ = model.face2face(faces)
            loss = criterion(reconstructed_faces, faces)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * faces.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for faces in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.epochs} val"):
                faces = faces.to(config.device)
                reconstructed_faces, _ = model.face2face(faces)
                loss = criterion(reconstructed_faces, faces)
                val_loss += loss.item() * faces.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_losses.append(avg_val_loss)

        epoch_duration = time.time() - epoch_start_time
        epoch_times.append(epoch_duration)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = config.epochs - (epoch + 1)
        estimated_remaining_time = avg_epoch_time * remaining_epochs

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1}/{config.epochs}:")
        print(f"  Train loss: {avg_train_loss:.6f}, val loss: {avg_val_loss:.6f}")
        print(
            f"  Epoch time: {timedelta(seconds=int(epoch_duration))}, "
            f"elapsed: {timedelta(seconds=int(time.time() - start_time))}, "
            f"remaining: {timedelta(seconds=int(estimated_remaining_time))}"
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss

            best_model_state = {
                "face_autoencoder": model.face_autoencoder.state_dict(),
                "is_kl_model": True,
                "autoencoder": "AutoencoderKL",
            }

            print(f"  New best model found. Val loss: {best_loss:.6f}")
            save_path = output_dir / "best_face_autoencoder.pth"
            torch.save(best_model_state, save_path)
            print(f"  Saved model to: {save_path}")

            if epoch > 0 and (epoch % 5 == 0 or epoch == config.epochs - 1):
                with torch.no_grad():
                    sample_faces = next(iter(val_loader))[:8].to(config.device)
                    reconstructed, _ = model.face2face(sample_faces)
                    metrics = calculate_metrics(sample_faces, reconstructed)

                    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
                    for i in range(8):
                        axes[0, i].imshow(sample_faces[i, 0].cpu().numpy(), cmap="gray")
                        axes[0, i].axis("off")
                        axes[1, i].imshow(reconstructed[i, 0].cpu().numpy(), cmap="gray")
                        axes[1, i].axis("off")

                    plt.suptitle(
                        "Face Reconstruction - "
                        f"Epoch {epoch + 1} "
                        f"(MSE: {metrics['MSE']:.4f}, "
                        f"PSNR: {metrics['PSNR']:.2f}, "
                        f"SSIM: {metrics['SSIM']:.4f})"
                    )
                    plt.tight_layout()
                    plt.savefig(output_dir / f"reconstruction_epoch{epoch + 1}.png")
                    plt.close()

    total_time = time.time() - start_time
    print(
        f"Face autoencoder training finished. Time: {timedelta(seconds=int(total_time))}, "
        f"best val loss: {best_loss:.6f}"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, config.epochs + 1), epoch_train_losses, "b-", label="Train loss")
    plt.plot(range(1, config.epochs + 1), epoch_val_losses, "r-", label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()

    if best_model_state:
        model.face_autoencoder.load_state_dict(best_model_state["face_autoencoder"])

    return model, best_model_state


def pretrain_with_all_subjects(config: Face2FaceConfig):
    """Pretrain the face autoencoder with face data from all valid subjects."""
    total_start_time = time.time()
    set_seed(config.seed)
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    subject_ids = get_valid_subject_ids(config)

    if not subject_ids:
        print("No valid subject IDs found. Stop.")
        return None

    print(f"\n{'=' * 60}")
    print(f"Start pretraining face autoencoder with {len(subject_ids)} subjects")
    print(f"Model class: AutoencoderKL, model type: {config.model_type}")
    print(f"{'=' * 60}")

    all_train_faces, all_test_faces = collect_all_faces_data(subject_ids, config)

    if all_train_faces is None or all_test_faces is None:
        print("Not enough face data for pretraining.")
        return None

    train_loader = DataLoader(
        torch.tensor(all_train_faces, dtype=torch.float32),
        batch_size=config.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        torch.tensor(all_test_faces, dtype=torch.float32),
        batch_size=config.batch_size,
        shuffle=False,
    )

    print(f"Data loaded. Train samples: {len(all_train_faces)}, test samples: {len(all_test_faces)}")

    model, is_kl = create_face_autoencoder(config)

    if hasattr(model, "face_autoencoder"):
        params, params_mb = count_parameters_mb(model.face_autoencoder)
    else:
        encoder_params, encoder_mb = count_parameters_mb(model.face_encoder)
        decoder_params, decoder_mb = count_parameters_mb(model.face_decoder)
        params = encoder_params + decoder_params
        params_mb = encoder_mb + decoder_mb

    print(f"Total model parameters: {params:,} ({params_mb:.2f} MB)")

    model, best_model_state = train_face_autoencoder(
        model,
        train_loader,
        val_loader,
        config=config,
        use_kl_model=is_kl,
    )

    if best_model_state:
        model_info = f"AutoencoderKL_{config.model_type}"
        final_save_path = (
            Path(config.results_dir)
            / "face_autoencoder"
            / f"pretrained_face_autoencoder_{model_info}.pth"
        )
        torch.save(best_model_state, final_save_path)

        total_time = time.time() - total_start_time
        print(f"All-subject pretraining finished. Total time: {timedelta(seconds=int(total_time))}")
        print(f"Model saved to: {final_save_path}")
        print(f"Model type: {model_info}, parameters: {params:,} ({params_mb:.2f} MB)")

    return best_model_state


def parse_args():
    parser = argparse.ArgumentParser(description="Train a face-to-face autoencoder.")
    parser.add_argument("--emoji-size", type=int, default=56)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--model-type", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--results-dir", default="./face2face_results")
    parser.add_argument("--eeg-dir", default=None)
    parser.add_argument("--emoji-root-template", default=None)
    parser.add_argument("--face-emotion-labels", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Face2FaceConfig(
        emoji_size=args.emoji_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        model_type=args.model_type,
        results_dir=Path(args.results_dir),
    )

    if args.eeg_dir is not None:
        config.eeg_dir = Path(args.eeg_dir)
    if args.emoji_root_template is not None:
        config.emoji_root_template = args.emoji_root_template
    if args.face_emotion_labels is not None:
        config.face_emotion_labels = Path(args.face_emotion_labels)
    if args.device is not None:
        config.device = args.device

    print("Training configuration:")
    print("  Model class: AutoencoderKL")
    print(f"  Model type: {config.model_type}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Results dir: {config.results_dir}")
    print(f"  Device: {config.device}")

    pretrain_with_all_subjects(config)


if __name__ == "__main__":
    main()