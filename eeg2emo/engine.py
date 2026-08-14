# engine.py
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from model_eeg import EEGClassifier
from util import EAV_EEG_CHANNEL_POSITIONS, MMER_EEG_CHANNEL_POSITIONS, SEED_EEG_CHANNEL_POSITIONS

try:
    from backbones import EEGNet, LMDA, DGCNN, TSLANet
except ImportError:
    EEGNet = LMDA = DGCNN = TSLANet = None


def create_model(dataset_name, model_config, dataset_config, device):
    """Create an EEG emotion recognition model from dataset and model configs."""
    model_name = model_config['model_name']
    n_classes = dataset_config['num_classes']
    n_chans = dataset_config['eeg_channels']
    n_times = dataset_config['sequence_length']
    sfreq = dataset_config['sampling_rate']
    window_seconds = dataset_config['window_seconds']

    if model_config.get('use_custom', False):
        positions = {
            'EAV': EAV_EEG_CHANNEL_POSITIONS,
            'MMER': MMER_EEG_CHANNEL_POSITIONS,
            'SEED': SEED_EEG_CHANNEL_POSITIONS,
        }.get(dataset_name)
        if positions is None:
            raise ValueError(f'Unsupported dataset: {dataset_name}')

        model = EEGClassifier(
            n_classes=n_classes,
            eeg_channels=n_chans,
            hidden_dim=dataset_config.get('hidden_dim', 64),
            depth=dataset_config.get('depth', 5),
            merger=dataset_config.get('merger', True),
            merger_pos_dim=dataset_config.get('merger_pos_dim', 288),
            merger_channels=dataset_config.get('merger_channels', 8),
            use_dual_path=dataset_config.get('use_dual_path', 0),
            positions=positions,
            dilation_period=dataset_config.get('dilation_period', 5),
        )
    elif model_name == 'EEGNet' and EEGNet is not None:
        model = EEGNet(n_classes=n_classes, Chans=n_chans, Samples=n_times)
    elif model_name == 'LMDA' and LMDA is not None:
        model = LMDA(chunk_size=n_times, num_electrodes=n_chans, num_classes=n_classes, depth=9, kernel=25)
    elif model_name == 'DGCNN' and DGCNN is not None:
        model = DGCNN(in_channels=n_times, num_electrodes=n_chans, hid_channels=32, num_layers=2, num_classes=n_classes)
    elif model_name == 'TSLANet' and TSLANet is not None:
        model = TSLANet(num_classes=n_classes, chunk_size=n_times, patch_size=100, num_electrodes=n_chans)
    elif 'model' in model_config:
        model = model_config['model'](
            n_chans=n_chans,
            n_outputs=n_classes,
            n_times=n_times,
            input_window_seconds=window_seconds,
            sfreq=sfreq,
        )
    else:
        raise ValueError(f'Model {model_name} is unavailable. Check backbones.py or choose --run 1.')

    return model.to(device)


def _classification_logits(outputs):
    return outputs[0] if isinstance(outputs, tuple) else outputs


def train_one_epoch(model, loader, optim, criterion, device, crf=None, epoch=None, total_epochs=None, **_):
    """Train the EEG classifier for one epoch."""
    model.train()
    running_loss = 0.0
    num_batches = 0
    epoch_start_time = time.time()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = _classification_logits(model(x))
        loss = criterion(logits, y)
        loss.backward()
        optim.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / max(1, num_batches)
    if crf:
        crf.training_stats['total_epochs'] = epoch if epoch else crf.training_stats['total_epochs'] + 1
        crf.training_stats.setdefault('epoch_times', []).append(time.time() - epoch_start_time)

    return avg_loss, avg_loss, 0.0


@torch.no_grad()
def evaluate(model, loader, device, **_):
    """Evaluate the EEG classifier."""
    model.eval()
    all_pred, all_true = [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    num_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = _classification_logits(model(x))
        loss = criterion(logits, y)
        pred = logits.argmax(1).cpu()

        all_pred.append(pred)
        all_true.append(y.cpu())
        total_loss += loss.item()
        num_batches += 1

    all_pred = torch.cat(all_pred).numpy()
    all_true = torch.cat(all_true).numpy()
    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average='weighted')
    cm = confusion_matrix(all_true, all_pred)
    avg_loss = total_loss / max(1, num_batches)
    return acc, f1, cm, avg_loss, avg_loss, 0.0


@torch.no_grad()
def extract_features(model, loader, device):
    """Extract classifier features or logits for downstream analysis."""
    model.eval()
    all_features = []
    all_labels = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        try:
            feats = _classification_logits(model(x))
            if feats.dim() > 2:
                feats = feats.mean(dim=-1)
            all_features.append(feats.cpu().numpy())
            all_labels.append(y.cpu().numpy())
        except Exception as exc:
            print(f'Warning: feature extraction failed: {exc}')
            all_features.append(np.zeros((x.shape[0], 64)))
            all_labels.append(y.cpu().numpy())

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)
