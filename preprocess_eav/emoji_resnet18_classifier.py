import copy
import json
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.models import ResNet18_Weights
from tqdm import tqdm


# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True


# Constants
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 30
NUM_CLASSES = 7
MODEL_TYPE = "ResNet18"


# Data directories
DATA_DIRS = {
    56: "/kaggle/input/notebooks/jingjinghuhu/eav-binary-56/Vision_Landmarks_sampled_25x56x56",
}

EMOTION_LABELS_PATH = "/kaggle/input/datasets/jingjinghuhu/eav-image-labels/face_emotions.json"
RESULTS_DIR = "./emotion_classifier_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# Emotion classes
EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_emotion_labels(json_path=EMOTION_LABELS_PATH):
    """Load pre-generated emotion labels."""
    print(f"Loading emotion labels: {json_path}")
    with open(json_path, "r") as f:
        emotion_data = json.load(f)
    return emotion_data


def get_available_subjects(emoji_size=56):
    """Get available subject IDs for a given emoji size."""
    data_dir = DATA_DIRS[emoji_size]
    subject_ids = []

    for file in os.listdir(data_dir):
        if file.startswith("emoji_vis_subject_") and file.endswith(".pkl"):
            subject_id = int(file.split("_")[-1].split(".")[0])
            subject_ids.append(subject_id)

    return sorted(subject_ids)


def load_emoji_data(subject_id, emoji_size=56):
    """Load emoji data for a specific subject."""
    data_dir = DATA_DIRS[emoji_size]
    emoji_path = f"{data_dir}/emoji_vis_subject_{subject_id:02d}.pkl"
    print(f"Loading emoji data ({emoji_size}x{emoji_size}): {emoji_path}")

    try:
        with open(emoji_path, "rb") as f:
            emoji_data = pickle.load(f)

        # Data format: [train_emojis, train_labels, test_emojis, test_labels]
        emoji_train = emoji_data[0]
        emoji_test = emoji_data[2]

        print(
            f"Subject {subject_id} - train shape: {emoji_train.shape}, "
            f"test shape: {emoji_test.shape}"
        )
        return emoji_train, emoji_test

    except Exception as e:
        print(f"Failed to load emoji data (subject {subject_id}, size {emoji_size}): {e}")
        return None, None


def get_sample_label(subject_id, split, sample_idx, frame_idx, emotion_data):
    """Get emotion label by subject ID, split, sample index, and frame index."""
    key = f"sub{subject_id:02d}_{split}_sample{sample_idx}_frame{frame_idx}"

    if key in emotion_data:
        return emotion_data[key]["emotion_id"]

    return 6


class EmojiEmotionDataset(Dataset):
    """Emoji emotion classification dataset."""

    def __init__(self, emoji_data, subject_id, split, emotion_data, emoji_size=56, transform=None):
        """
        Args:
            emoji_data: Emoji data with shape (n_videos, n_frames, h, w).
            subject_id: Subject ID.
            split: 'train' or 'test'.
            emotion_data: Emotion label dictionary.
            emoji_size: Emoji image size.
            transform: Data transform.
        """
        self.emoji_data = emoji_data
        self.subject_id = subject_id
        self.split = split
        self.emotion_data = emotion_data
        self.emoji_size = emoji_size
        self.transform = transform
        self.samples = []

        for sample_idx in range(len(emoji_data)):
            for frame_idx in range(emoji_data.shape[1]):
                emotion_id = get_sample_label(
                    subject_id, split, sample_idx, frame_idx, emotion_data
                )
                self.samples.append((sample_idx, frame_idx, emotion_id))

        print(f"Subject {subject_id} ({split}) - created {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_idx, frame_idx, emotion_id = self.samples[idx]
        emoji = self.emoji_data[sample_idx, frame_idx]

        if emoji.dtype == np.uint8:
            emoji = emoji.astype(np.float32) / 255.0
        elif emoji.max() > 1.0:
            emoji = emoji.astype(np.float32) / 255.0
        else:
            emoji = emoji.astype(np.float32)

        if self.transform:
            emoji = (emoji * 255).astype(np.uint8)
            emoji_tensor = self.transform(emoji)
        else:
            emoji_3ch = np.stack([emoji] * 3, axis=0)
            emoji_tensor = torch.FloatTensor(emoji_3ch)

        return emoji_tensor, emotion_id, self.subject_id


class ResNet18(nn.Module):
    def __init__(self, num_classes=7, pretrained=False):
        super().__init__()

        if pretrained:
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet18(weights=None)

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def get_transforms(emoji_size):
    """Get train and validation transforms for ResNet18."""
    to_three_channels = lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(emoji_size, scale=(0.9, 1.1), ratio=(1.0, 1.0)),
            transforms.ToTensor(),
            to_three_channels,
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((emoji_size, emoji_size)),
            transforms.ToTensor(),
            to_three_channels,
        ]
    )

    return train_transform, val_transform


def create_model(num_classes=7, pretrained=False):
    """Create a ResNet18 classifier."""
    return ResNet18(num_classes=num_classes, pretrained=pretrained)


def train_classifier(
    model,
    train_loader,
    val_loader,
    emoji_size,
    model_name,
    num_epochs=EPOCHS,
    device="cuda",
    outdir=RESULTS_DIR,
):
    """Train the emotion classifier and save the best model."""
    model_dir = f"{outdir}/emoji_size_{emoji_size}_{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    print(f"Start training ({emoji_size}x{emoji_size}, model: {model_name})")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=3, factor=0.5
    )

    model.to(device)

    best_acc = 0.0
    best_model_weights = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for emojis, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} train"):
            emojis, labels = emojis.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(emojis)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * emojis.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for emojis, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} val"):
                emojis, labels = emojis.to(device), labels.to(device)

                outputs = model(emojis)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * emojis.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train loss: {avg_train_loss:.4f}, train acc: {train_acc:.4f}")
        print(f"  Val loss: {avg_val_loss:.4f}, val acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"  New best model found. Val acc: {best_acc:.4f}")

    if best_model_weights:
        print(f"Saving best model. Best val acc: {best_acc:.4f}")
        torch.save(best_model_weights, f"{model_dir}/best_model.pth")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train")
    plt.plot(val_accs, label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{model_dir}/training_history.png")
    plt.close()

    print(f"Training finished. Best val acc: {best_acc:.4f}")

    if best_model_weights:
        model.load_state_dict(best_model_weights)

    return model, {"best_acc": best_acc}


def test_classifier(model, test_loader, emoji_size, model_name, device, outdir=RESULTS_DIR):
    """Evaluate the classifier on the test set."""
    model_dir = f"{outdir}/emoji_size_{emoji_size}_{model_name}"
    print(f"Evaluating on test set ({emoji_size}x{emoji_size}, model: {model_name})")

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    test_correct = 0
    test_total = 0
    subject_preds = {}
    subject_labels = {}

    with torch.no_grad():
        for emojis, labels, subjects in tqdm(test_loader, desc="Test"):
            emojis, labels = emojis.to(device), labels.to(device)

            outputs = model(emojis)
            _, predicted = torch.max(outputs, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            preds_cpu = predicted.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            subjects_cpu = subjects.cpu().numpy()

            all_preds.extend(preds_cpu)
            all_labels.extend(labels_cpu)

            for i, sub_id in enumerate(subjects_cpu):
                if sub_id not in subject_preds:
                    subject_preds[sub_id] = []
                    subject_labels[sub_id] = []

                subject_preds[sub_id].append(preds_cpu[i])
                subject_labels[sub_id].append(labels_cpu[i])

    test_acc = test_correct / test_total
    print(f"Test accuracy: {test_acc:.4f}")

    subject_accs = {}
    for sub_id in sorted(subject_preds.keys()):
        sub_acc = accuracy_score(subject_labels[sub_id], subject_preds[sub_id])
        subject_accs[sub_id] = sub_acc
        print(f"Subject {sub_id} test accuracy: {sub_acc:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Test Confusion Matrix (Acc={test_acc:.4f})")
    plt.colorbar()

    tick_marks = np.arange(len(EMOTION_CLASSES))
    plt.xticks(tick_marks, EMOTION_CLASSES, rotation=45)
    plt.yticks(tick_marks, EMOTION_CLASSES)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(f"{model_dir}/test_confusion_matrix.png")
    plt.close()

    report = classification_report(
        all_labels,
        all_preds,
        target_names=EMOTION_CLASSES,
        digits=4,
        zero_division=0,
    )
    with open(f"{model_dir}/test_classification_report.txt", "w") as f:
        f.write(report)

    return test_acc, subject_accs


def train_evaluate_model(emoji_size, pretrained=False):
    """Train and evaluate one ResNet18 model."""
    model_name = f"{MODEL_TYPE}_{'pretrain' if pretrained else 'no_pretrain'}"
    print(
        f"\n=== Start training {model_name} "
        f"({emoji_size}x{emoji_size}, pretrained: {pretrained}) ===\n"
    )

    emotion_data = load_emotion_labels()

    subject_ids = get_available_subjects(emoji_size)
    print(f"Available subjects for {emoji_size}x{emoji_size}: {subject_ids}")

    if not subject_ids:
        print(f"No data found for {emoji_size}x{emoji_size}. Skip.")
        return None, None

    train_transform, val_transform = get_transforms(emoji_size)
    train_datasets = []
    test_datasets = []

    for subject_id in subject_ids:
        emoji_train, emoji_test = load_emoji_data(subject_id, emoji_size)

        if emoji_train is None or emoji_test is None:
            print(f"Failed to load subject {subject_id}. Skip.")
            continue

        train_dataset = EmojiEmotionDataset(
            emoji_train,
            subject_id,
            "train",
            emotion_data,
            emoji_size,
            transform=train_transform,
        )

        test_dataset = EmojiEmotionDataset(
            emoji_test,
            subject_id,
            "test",
            emotion_data,
            emoji_size,
            transform=val_transform,
        )

        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    if not train_datasets or not test_datasets:
        print(f"No valid dataset found for {emoji_size}x{emoji_size}. Skip.")
        return None, None

    combined_train_dataset = ConcatDataset(train_datasets)
    combined_test_dataset = ConcatDataset(test_datasets)

    train_loader = DataLoader(
        combined_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        combined_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    print(f"Combined train size: {len(combined_train_dataset)}")
    print(f"Combined test size: {len(combined_test_dataset)}")

    model = create_model(num_classes=NUM_CLASSES, pretrained=pretrained)
    print(f"Created {model_name} classifier with {NUM_CLASSES} classes")

    trained_model, history = train_classifier(
        model,
        train_loader,
        test_loader,
        emoji_size,
        model_name,
        num_epochs=EPOCHS,
        device=device,
        outdir=RESULTS_DIR,
    )

    test_acc, subject_accs = test_classifier(
        trained_model, test_loader, emoji_size, model_name, device, outdir=RESULTS_DIR
    )

    results = {
        "test_acc": test_acc,
        "subject_accs": subject_accs,
        "best_acc": history["best_acc"],
        "model_type": MODEL_TYPE,
        "pretrained": pretrained,
    }

    print(f"{model_name} ({emoji_size}x{emoji_size}) finished. Test acc: {test_acc:.4f}")
    return trained_model, results


def train_all_models():
    """Train and evaluate ResNet18 only."""
    results = {}

    for emoji_size in [56]:
        size_results = {}

        _, no_pretrain_results = train_evaluate_model(emoji_size, pretrained=False)
        if no_pretrain_results:
            size_results["ResNet18_no_pretrain"] = no_pretrain_results

        _, pretrain_results = train_evaluate_model(emoji_size, pretrained=True)
        if pretrain_results:
            size_results["ResNet18_pretrain"] = pretrain_results

        results[emoji_size] = size_results

    with open(f"{RESULTS_DIR}/resnet18_summary.pkl", "wb") as f:
        pickle.dump(results, f)

    plt.figure(figsize=(8, 6))

    for emoji_size, size_results in results.items():
        model_names = []
        accuracies = []

        for model_name, model_results in size_results.items():
            model_names.append(model_name)
            accuracies.append(model_results["test_acc"])

        plt.bar(model_names, accuracies)

        for i, value in enumerate(accuracies):
            plt.text(i, value + 0.01, f"{value:.4f}", ha="center")

    plt.xlabel("Model")
    plt.ylabel("Test Accuracy")
    plt.title(f"ResNet18 Performance (Emoji Size: {emoji_size}x{emoji_size})")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/resnet18_comparison.png")
    plt.close()

    print("\n=== ResNet18 Performance ===")
    for emoji_size, size_results in results.items():
        print(f"\nSize {emoji_size}x{emoji_size}:")
        for model_name, model_results in sorted(
            size_results.items(), key=lambda x: x[1]["test_acc"], reverse=True
        ):
            print(f"  {model_name}:")
            print(f"    Test accuracy: {model_results['test_acc']:.4f}")
            print(f"    Best val accuracy: {model_results['best_acc']:.4f}")

    return results


def main():
    results = train_all_models()
    print("ResNet18 training and evaluation finished.")


if __name__ == "__main__":
    main()
