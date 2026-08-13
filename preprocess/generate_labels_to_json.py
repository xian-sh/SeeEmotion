import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor


# Configuration
SUBJECTS = list(range(1, 43))
VISION_BASE = Path("./EAV/data/Inputs/Vision")
LABELS_DIR = Path(
    "./labels_face"   # output
)
OUTPUT_JSON = Path(
    "./face_emotions.json"  # output
)
MODEL_NAME = Path(
    "./cache_dir/face-model"   # download from https://huggingface.co/dima806/facial_emotions_image_detection
)

BATCH_SIZE = 64
OVERWRITE = False
INCLUDE_TEST = True
LOCAL_FILES_ONLY = True


def resolve_device():
    """Use CUDA when available and otherwise fall back to the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    print("[WARN] CUDA is unavailable. Falling back to the CPU.")
    return torch.device("cpu")


def load_emotion_model(model_name, device, local_files_only=True):
    """Load the facial-emotion classifier and its image processor."""
    print(f"Loading the facial-emotion model from {model_name}...")

    model = ViTForImageClassification.from_pretrained(
        str(model_name),
        local_files_only=local_files_only,
    )
    processor = ViTImageProcessor.from_pretrained(
        str(model_name),
        local_files_only=local_files_only,
    )
    model = model.to(device).eval()
    id2label = getattr(model.config, "id2label", None)

    return model, processor, id2label


def flatten_video_frames(video_np):
    """Flatten an (N, T, H, W[, C]) video array into indexed frames."""
    num_samples, num_frames = video_np.shape[:2]
    frames = []

    for sample_idx in range(num_samples):
        for frame_idx in range(num_frames):
            frames.append(
                (sample_idx, frame_idx, video_np[sample_idx, frame_idx])
            )

    return frames


def frame_to_pil(frame):
    """Convert a grayscale, RGB, or RGBA frame into a three-channel PIL image."""
    array = np.asarray(frame)

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    elif array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Unsupported frame shape: {array.shape}")

    if array.dtype != np.uint8:
        array = array.astype(np.uint8)

    return Image.fromarray(array)


def predict_video_pseudo_labels(
    video_np,
    model,
    processor,
    device,
    batch_size,
):
    """
    Predict a class ID and confidence for every frame in a video array.

    Args:
        video_np: Video array with shape (N, T, H, W, C) or (N, T, H, W).
        model: Facial-emotion classification model.
        processor: Image processor associated with the model.
        device: Torch inference device.
        batch_size: Number of frames processed per inference batch.

    Returns:
        predictions: Integer array with shape (N, T).
        confidences: Float array with shape (N, T).
    """
    num_samples, num_frames = video_np.shape[:2]
    predictions = np.zeros((num_samples, num_frames), dtype=np.int64)
    confidences = np.zeros((num_samples, num_frames), dtype=np.float32)
    frames = flatten_video_frames(video_np)

    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]
            images = [frame_to_pil(frame) for _, _, frame in batch]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_ids = probabilities.argmax(dim=-1).cpu().numpy()
            predicted_confidences = probabilities.max(dim=-1).values.cpu().numpy()

            for batch_idx, (sample_idx, frame_idx, _) in enumerate(batch):
                predictions[sample_idx, frame_idx] = int(
                    predicted_ids[batch_idx]
                )
                confidences[sample_idx, frame_idx] = float(
                    predicted_confidences[batch_idx]
                )

    return predictions, confidences


def generate_subject_pseudo_labels(
    subjects,
    vision_base,
    labels_dir,
    model,
    processor,
    id2label,
    device,
    batch_size,
    overwrite=False,
):
    """Generate and save one pseudo-label pickle file for each subject."""
    labels_dir.mkdir(parents=True, exist_ok=True)

    for subject_id in tqdm(subjects, desc="Generating subject labels"):
        input_file = vision_base / f"subject_{subject_id:02d}_vis.pkl"
        output_file = labels_dir / f"pseudo_labels_subject_{subject_id:02d}.pkl"

        if not input_file.exists():
            print(
                f"[WARN] Input file not found for subject {subject_id:02d}: "
                f"{input_file}"
            )
            continue

        if output_file.exists() and not overwrite:
            print(
                f"[INFO] Output already exists for subject {subject_id:02d}; "
                f"skipping: {output_file}"
            )
            continue

        try:
            with input_file.open("rb") as file:
                vision_data = pickle.load(file)

            video_train = vision_data[0]
            video_test = vision_data[2]

            print(
                f"Processing subject {subject_id:02d}: "
                f"train_shape={getattr(video_train, 'shape', None)}, "
                f"test_shape={getattr(video_test, 'shape', None)}"
            )

            pseudo_train, pseudo_train_confidence = predict_video_pseudo_labels(
                video_train,
                model,
                processor,
                device,
                batch_size,
            )
            pseudo_test, pseudo_test_confidence = predict_video_pseudo_labels(
                video_test,
                model,
                processor,
                device,
                batch_size,
            )

            output_data = {
                "subject": subject_id,
                "pseudo_train": pseudo_train,
                "pseudo_train_conf": pseudo_train_confidence,
                "pseudo_test": pseudo_test,
                "pseudo_test_conf": pseudo_test_confidence,
                "model_id2label": id2label,
            }

            with output_file.open("wb") as file:
                pickle.dump(output_data, file)

            print(
                f"Saved {output_file} "
                f"(train={pseudo_train.shape}, test={pseudo_test.shape})"
            )

        except Exception as error:
            print(
                f"[ERROR] Failed to process subject {subject_id:02d}: {error}"
            )


def get_emotion_label(id2label, emotion_id):
    """Resolve an emotion ID for mappings that use integer or string keys."""
    if not id2label:
        return str(emotion_id)

    if emotion_id in id2label:
        return id2label[emotion_id]

    return id2label.get(str(emotion_id), str(emotion_id))


def add_split_records(
    all_emotions,
    subject_id,
    split_name,
    predictions,
    confidences,
    id2label,
):
    """Add all frame-level records from one dataset split to the JSON mapping."""
    if predictions.shape != confidences.shape:
        raise ValueError(
            f"Prediction/confidence shape mismatch for subject {subject_id:02d} "
            f"{split_name}: {predictions.shape} versus {confidences.shape}"
        )

    num_samples, num_frames = predictions.shape

    for sample_idx in range(num_samples):
        for frame_idx in range(num_frames):
            emotion_id = int(predictions[sample_idx, frame_idx])
            confidence = float(confidences[sample_idx, frame_idx])
            key = (
                f"sub{subject_id:02d}_{split_name}_sample{sample_idx}_"
                f"frame{frame_idx}"
            )

            all_emotions[key] = {
                "emotion_id": emotion_id,
                "emotion": get_emotion_label(id2label, emotion_id),
                "confidence": confidence,
            }


def convert_pseudo_labels_to_json(labels_dir, output_json, include_test=True):
    """Combine all subject pseudo-label pickle files into one JSON file."""
    print(f"Loading pseudo-label files from {labels_dir}...")
    label_files = sorted(labels_dir.glob("pseudo_labels_subject_*.pkl"))

    if not label_files:
        print(f"[WARN] No pseudo-label files were found in {labels_dir}.")
        return 0

    print(f"Found {len(label_files)} pseudo-label files.")
    all_emotions = {}

    for label_file in tqdm(label_files, desc="Converting label files"):
        try:
            subject_id = int(label_file.stem.split("_")[-1])

            with label_file.open("rb") as file:
                data = pickle.load(file)

            id2label = data.get("model_id2label")
            add_split_records(
                all_emotions,
                subject_id,
                "train",
                data["pseudo_train"],
                data["pseudo_train_conf"],
                id2label,
            )

            if include_test:
                add_split_records(
                    all_emotions,
                    subject_id,
                    "test",
                    data["pseudo_test"],
                    data["pseudo_test_conf"],
                    id2label,
                )

        except Exception as error:
            print(f"[ERROR] Failed to convert {label_file}: {error}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(all_emotions, file, indent=2, ensure_ascii=False)

    print(f"Converted {len(all_emotions)} records and saved them to {output_json}.")

    sample_keys = list(all_emotions)[:3]
    if sample_keys:
        print("\nSample records:")
        for key in sample_keys:
            print(f"{key}: {all_emotions[key]}")

        emotion_counts = {}
        for item in all_emotions.values():
            emotion = item["emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        print("\nEmotion distribution:")
        total_records = len(all_emotions)
        for emotion, count in sorted(emotion_counts.items()):
            percentage = count / total_records * 100
            print(f"{emotion}: {count} ({percentage:.2f}%)")

    return len(all_emotions)


def main():
    """Generate subject-level pseudo labels and export all records to JSON."""
    device = resolve_device()
    model, processor, id2label = load_emotion_model(
        MODEL_NAME,
        device,
        local_files_only=LOCAL_FILES_ONLY,
    )

    generate_subject_pseudo_labels(
        subjects=SUBJECTS,
        vision_base=VISION_BASE,
        labels_dir=LABELS_DIR,
        model=model,
        processor=processor,
        id2label=id2label,
        device=device,
        batch_size=BATCH_SIZE,
        overwrite=OVERWRITE,
    )

    convert_pseudo_labels_to_json(
        labels_dir=LABELS_DIR,
        output_json=OUTPUT_JSON,
        include_test=INCLUDE_TEST,
    )


if __name__ == "__main__":
    main()
