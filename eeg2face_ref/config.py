from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DatasetPreset:
    eeg_dir: Path
    face_root: Path
    eeg_channels: int
    sampling_rate: float
    trial_seconds: float
    n_frames: int
    emoji_size: int = 56


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "EAV": DatasetPreset(
        eeg_dir=Path("/home/devuser/hjj/seeemotion/data/EAV/EEG"),
        face_root=Path("/home/devuser/hjj/seeemotion/data/EAV/Vision_Landmarks_25x56x56"),
        eeg_channels=30,
        sampling_rate=100.0,
        trial_seconds=5.0,
        n_frames=25,
        emoji_size=56
    ),
    "MMER": DatasetPreset(
        eeg_dir=Path("/home/devuser/hjj/seeemotion/data/MMER/EEG"),
        face_root=Path("/home/devuser/hjj/seeemotion/data/MMER/Landmarks_4x64x64"),
        eeg_channels=18,
        sampling_rate=300.0,
        trial_seconds=20.0,
        n_frames=40,
        emoji_size=64
    ),
}


@dataclass
class DataConfig:
    dataset: str = "EAV"
    eeg_dir: Path = DATASET_PRESETS["EAV"].eeg_dir
    face_root: Path = DATASET_PRESETS["EAV"].face_root
    eeg_channels: int = DATASET_PRESETS["EAV"].eeg_channels
    sampling_rate: float = DATASET_PRESETS["EAV"].sampling_rate
    trial_seconds: float = DATASET_PRESETS["EAV"].trial_seconds
    n_frames: int = DATASET_PRESETS["EAV"].n_frames
    emoji_size: int = 56
    eeg_window_seconds: float = 1.0
    split_mode: str = "paired_reference"
    repeat: int = 4
    test_ratio: float = 0.3
    eeg_normalization: str = "zscore"
    batch_size: int = 128
    num_workers: int = 2
    subject_ids: Optional[str] = "1"
    train_subjects: Optional[str] = None
    test_subjects: Optional[str] = None
    max_subject_id: int = 128
    max_trials_per_subject: int = 2048


@dataclass
class AutoencoderConfig:
    checkpoint: Path = Path("/home/devuser/hjj/seeemotion/models/AutoencoderKL.pth")
    model_type: str = "light"
    latent_dim: int = 512
    sample_latent: bool = False


@dataclass
class ModelConfig:
    backbone_hidden_dim: int = 128
    backbone_depth: int = 5
    dropout: float = 0.0
    use_channel_merger: bool = True
    merger_channels: int = 8
    merger_pos_dim: int = 288
    hidden_dims: tuple[int, ...] = (4096, 4096, 4096, 2048)
    use_subject_embedding: bool = True
    subject_embedding_dim: int = 32
    use_trial_frame_embedding: bool = True
    trial_embedding_dim: int = 512
    frame_embedding_dim: int = 64
    use_sample_embedding: bool = True
    sample_embedding_dim: int = 256


@dataclass
class TrainConfig:
    protocol: str = "reference"
    seed: int = 42
    epochs: int = 300
    lr: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    latent_loss_weight: float = 1.0
    recon_loss_weight: float = 1.0
    best_metric: str = "ssim"
    device: str = "cuda:0"
    results_dir: Path = Path("./eeg2face_reference_results")
    save_every: int = 0


@dataclass
class Config:
    data: DataConfig
    autoencoder: AutoencoderConfig
    model: ModelConfig
    train: TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reference EEG-to-face training with a frozen Face2Face AutoencoderKL tail.")
    parser.add_argument("--dataset", choices=sorted(DATASET_PRESETS), default="MMER")
    parser.add_argument("--eeg-dir", type=Path, default=None)
    parser.add_argument("--face-root", type=Path, default=None)
    parser.add_argument("--subject-ids", default="1")
    parser.add_argument("--train-subjects", default=None)
    parser.add_argument("--test-subjects", default=None)
    parser.add_argument("--eeg-channels", type=int, default=None)
    parser.add_argument("--sampling-rate", type=float, default=None)
    parser.add_argument("--trial-seconds", type=float, default=None)
    parser.add_argument("--n-frames", type=int, default=None)
    parser.add_argument("--emoji-size", type=int, default=None, choices=[56, 64])
    parser.add_argument("--eeg-window-seconds", type=float, default=None)
    parser.add_argument("--split-mode", choices=["random", "paired_reference"], default="paired_reference")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--eeg-normalization", choices=["zscore", "max_abs", "none"], default="zscore")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-subject-id", type=int, default=128)
    parser.add_argument("--max-trials-per-subject", type=int, default=2048)

    parser.add_argument("--autoencoder", default="/home/devuser/hjj/seeemotion/models/AutoencoderKL.pth")
    parser.add_argument("--autoencoder-type", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--sample-latent", action="store_true")

    parser.add_argument("--backbone-hidden-dim", type=int, default=128)
    parser.add_argument("--backbone-depth", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-channel-merger", action="store_true")
    parser.add_argument("--merger-channels", type=int, default=8)
    parser.add_argument("--hidden-dims", default="4096,4096,4096,2048")
    parser.add_argument("--no-subject-embedding", action="store_true")
    parser.add_argument("--subject-embedding-dim", type=int, default=32)
    parser.add_argument("--no-trial-frame-embedding", action="store_true")
    parser.add_argument("--trial-embedding-dim", type=int, default=512)
    parser.add_argument("--frame-embedding-dim", type=int, default=64)
    parser.add_argument("--no-sample-embedding", action="store_true")
    parser.add_argument("--sample-embedding-dim", type=int, default=256)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--recon-loss-weight", type=float, default=1.0)
    parser.add_argument("--best-metric", default="ssim")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=Path("./eeg2face_reference_results"))
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Only used by evaluate.py.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    preset = DATASET_PRESETS[args.dataset]
    hidden_dims = tuple(int(item.strip()) for item in args.hidden_dims.split(",") if item.strip())
    return Config(
        data=DataConfig(
            dataset=args.dataset,
            eeg_dir=args.eeg_dir or preset.eeg_dir,
            face_root=args.face_root or preset.face_root,
            eeg_channels=args.eeg_channels or preset.eeg_channels,
            sampling_rate=args.sampling_rate or preset.sampling_rate,
            trial_seconds=args.trial_seconds or preset.trial_seconds,
            n_frames=args.n_frames or preset.n_frames,
            emoji_size=args.emoji_size or preset.emoji_size,
            eeg_window_seconds=args.eeg_window_seconds if args.eeg_window_seconds is not None else (0.5 if args.dataset == "MMER" else 1.0),
            split_mode=args.split_mode,
            repeat=args.repeat,
            test_ratio=args.test_ratio,
            eeg_normalization=args.eeg_normalization,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            subject_ids=args.subject_ids,
            train_subjects=args.train_subjects,
            test_subjects=args.test_subjects,
            max_subject_id=args.max_subject_id,
            max_trials_per_subject=args.max_trials_per_subject,
        ),
        autoencoder=AutoencoderConfig(
            checkpoint=args.autoencoder,
            model_type=args.autoencoder_type,
            latent_dim=args.latent_dim,
            sample_latent=args.sample_latent,
        ),
        model=ModelConfig(
            backbone_hidden_dim=args.backbone_hidden_dim,
            backbone_depth=args.backbone_depth,
            dropout=args.dropout,
            use_channel_merger=not args.no_channel_merger,
            merger_channels=args.merger_channels,
            hidden_dims=hidden_dims,
            use_subject_embedding=not args.no_subject_embedding,
            subject_embedding_dim=args.subject_embedding_dim,
            use_trial_frame_embedding=not args.no_trial_frame_embedding,
            trial_embedding_dim=args.trial_embedding_dim,
            frame_embedding_dim=args.frame_embedding_dim,
            use_sample_embedding=not args.no_sample_embedding,
            sample_embedding_dim=args.sample_embedding_dim,
        ),
        train=TrainConfig(
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            latent_loss_weight=args.latent_loss_weight,
            recon_loss_weight=args.recon_loss_weight,
            best_metric=args.best_metric,
            device=args.device,
            results_dir=args.results_dir,
            save_every=args.save_every,
        ),
    )


def parse_id_list(value: Optional[str]) -> Optional[list[int]]:
    if value is None or value.strip() == "":
        return None
    ids: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            ids.update(range(int(start), int(end) + 1))
        else:
            ids.add(int(item))
    return sorted(ids)
