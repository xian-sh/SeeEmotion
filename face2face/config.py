from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Face2FaceConfig:
    seed: int = 42
    results_dir: Path = Path("./face2face_results")
    emoji_size: int = 56
    batch_size: int = 32
    learning_rate: float = 1e-4
    latent_dim: int = 512
    eeg_emb_dim: int = 256
    n_frames: int = 25

    eeg_dir: Path = Path("/kaggle/input/datasets/jingjinghuhu/eva-feat/Input_images/eeg")
    emoji_root_template: str = (
        "/kaggle/input/notebooks/jingjinghuhu/eav-binary-{emoji_size}/"
        "Vision_Landmarks_sampled_25x{emoji_size}x{emoji_size}"
    )
    face_emotion_labels: Path = Path(
        "/kaggle/input/datasets/jingjinghuhu/eav-image-labels/face_emotions.json"
    )

    model_type: str = "light"
    model_class: str = "AutoencoderKL"
    epochs: int = 30
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


EVA_EMOTIONS = ["Neutral", "Sadness", "Anger", "Happiness", "Calm"]
FACE_EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EVA_NUM_CLASSES = len(EVA_EMOTIONS)
FACE_NUM_CLASSES = len(FACE_EMOTIONS)

