import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor


# Configuration
face_img_base_dir = "./Aligned_data/Images_0.5"  # Directory containing all face image folders
out_file = "face_emotion_labels_0.5.json"  # Output JSON filename

model_name = "/home/u2022111067/jupyterlab/face-eeg/cache_dir/face-model"  # "dima806/facial_emotions_image_detection"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64


# Load model
print("Loading facial emotion recognition model...")
emo_model = ViTForImageClassification.from_pretrained(model_name, local_files_only=True)
emo_proc = ViTImageProcessor.from_pretrained(model_name, local_files_only=True)
emo_model = emo_model.to(device).eval()
id2label = emo_model.config.id2label if hasattr(emo_model.config, "id2label") else None
print(f"Class mapping: {id2label}")


def get_all_face_folders(base_dir):
    """Get all face image folders, such as '1_face_imgs' and '2_face_imgs'."""
    face_folders = []
    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path) and item.endswith("_face_imgs"):
            face_folders.append(full_path)
    return face_folders


def get_all_image_files(folders):
    """Get all image files and their corresponding keys."""
    image_files = []
    image_keys = []

    for folder in folders:
        folder_name = os.path.basename(folder)
        subject_idx = folder_name.split("_")[0]

        for file in os.listdir(folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")) and file.startswith("video_"):
                img_path = os.path.join(folder, file)
                key = f"{subject_idx}_face_imgs/{file}"

                image_files.append(img_path)
                image_keys.append(key)

    return image_files, image_keys


def process_images_in_batches(image_files, image_keys):
    """Run batched emotion prediction for all images."""
    results = {}
    total = len(image_files)
    softmax = torch.nn.Softmax(dim=-1)

    with torch.no_grad():
        for start in tqdm(range(0, total, batch_size), desc="Processing images"):
            end = min(start + batch_size, total)
            batch_files = image_files[start:end]
            batch_keys = image_keys[start:end]
            imgs = []
            valid_indices = []

            for i, img_path in enumerate(batch_files):
                try:
                    img = Image.open(img_path).convert("RGB")
                    imgs.append(img)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

            if not imgs:
                continue

            inputs = emo_proc(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = emo_model(**inputs)
            probs = softmax(outputs.logits)
            pred_ids = probs.argmax(dim=-1).cpu().numpy()
            pred_conf = probs.max(dim=-1).values.cpu().numpy()

            for i, idx in enumerate(valid_indices):
                key = batch_keys[idx]
                results[key] = {
                    "emotion_id": int(pred_ids[i]),
                    "emotion": id2label[int(pred_ids[i])] if id2label else str(int(pred_ids[i])),
                    "confidence": float(pred_conf[i]),
                }

    return results


def main():
    print(f"Searching face image folders in {face_img_base_dir}...")
    face_folders = get_all_face_folders(face_img_base_dir)
    print(f"Found {len(face_folders)} face image folders")

    if not face_folders:
        print("No face image folders found. Exit.")
        return

    print("Collecting all image files...")
    image_files, image_keys = get_all_image_files(face_folders)
    print(f"Found {len(image_files)} image files")

    if not image_files:
        print("No image files found. Exit.")
        return

    results = process_images_in_batches(image_files, image_keys)

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Emotion labels saved to {out_file}. Total records: {len(results)}")

    print("\nSample data:")
    sample_keys = list(results.keys())[:3]
    for key in sample_keys:
        print(f"{key}: {results[key]}")


if __name__ == "__main__":
    main()