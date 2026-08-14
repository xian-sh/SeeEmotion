# dataset.py
import os
import pickle
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from scipy.signal import butter, filtfilt, iirnotch
from torch.utils.data import Dataset


class EAVDataset(Dataset):
    """EEG-only EAV dataset with trial-level sliding windows."""

    def __init__(
        self,
        subject_ids: List[int],
        eeg_data_root: str,
        window_seconds: float = 1.0,
        fs: int = 100,
        stride_seconds: float = None,
        overlap: float = 0.0,
    ):
        super().__init__()
        self.fs = fs
        self.window_seconds = window_seconds
        self.window_samples = int(fs * window_seconds)

        if stride_seconds is None:
            self.stride_samples = int(self.window_samples * (1 - overlap)) if overlap > 0 else self.window_samples
        else:
            self.stride_samples = int(fs * stride_seconds)
        self.stride_samples = max(1, self.stride_samples)

        print('EAV dataset configuration:')
        print(f'  Window length: {window_seconds}s ({self.window_samples} samples)')
        print(f'  Stride: {self.stride_samples} samples')
        print(f'  Overlap: {(self.window_samples - self.stride_samples) / self.window_samples * 100:.1f}%')

        self.all_trials = []
        self.all_labels = []
        self.trial_info = []

        for subject_id in subject_ids:
            self._load_subject_data(subject_id, eeg_data_root)

        if not self.all_trials:
            raise ValueError('No EEG data was loaded successfully.')

        print(f'Loaded {len(self.all_trials)} raw trials.')
        self._print_label_distribution()
        self._create_sliding_windows()

    def _load_subject_data(self, subject_id: int, eeg_data_root: str):
        """Load one subject from the EAV EEG pickle file."""
        eeg_path = os.path.join(eeg_data_root, f'subject_{subject_id:02d}_eeg.pkl')
        if not os.path.exists(eeg_path):
            print(f'Warning: EEG file not found for subject {subject_id}: {eeg_path}')
            return

        try:
            with open(eeg_path, 'rb') as f:
                tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = pickle.load(f)

            x_all = np.concatenate([tr_x_eeg, te_x_eeg], axis=0)
            y_all = np.concatenate([tr_y_eeg, te_y_eeg], axis=0).squeeze()

            for trial_idx, (trial_data, label) in enumerate(zip(x_all, y_all)):
                if trial_data.ndim == 2 and trial_data.shape[0] > trial_data.shape[1]:
                    trial_data = trial_data.T

                self.all_trials.append(trial_data.astype(np.float32))
                self.all_labels.append(int(label))
                self.trial_info.append({
                    'subject_id': subject_id,
                    'original_trial_idx': trial_idx,
                    'split': 'train' if trial_idx < len(tr_x_eeg) else 'test',
                })

            print(f'Subject {subject_id}: loaded {len(x_all)} trials (train: {len(tr_x_eeg)}, test: {len(te_x_eeg)}).')
        except Exception as exc:
            print(f'Error loading subject {subject_id}: {exc}')

    def _print_label_distribution(self):
        label_counts = {}
        for label in self.all_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        print('Raw label distribution:')
        emotion_names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad']
        for label, count in sorted(label_counts.items()):
            name = emotion_names[label] if label < len(emotion_names) else f'Unknown({label})'
            print(f'  {name} (label={label}): {count}')

    def _create_sliding_windows(self):
        """Create sliding-window indices for all EEG trials."""
        self.window_indices = []
        self.window_labels = []
        self.window_info = []

        for trial_idx, (trial_data, label) in enumerate(zip(self.all_trials, self.all_labels)):
            n_samples = trial_data.shape[1]
            if n_samples < self.window_samples:
                print(f'Warning: trial {trial_idx} has {n_samples} samples, smaller than window size {self.window_samples}.')
                continue

            n_windows = (n_samples - self.window_samples) // self.stride_samples + 1
            for win_idx in range(n_windows):
                start_idx = win_idx * self.stride_samples
                end_idx = start_idx + self.window_samples
                self.window_indices.append((trial_idx, start_idx, end_idx))
                self.window_labels.append(label)
                self.window_info.append({
                    'subject_id': self.trial_info[trial_idx]['subject_id'],
                    'original_trial_idx': self.trial_info[trial_idx]['original_trial_idx'],
                    'window_idx': win_idx,
                    'split': self.trial_info[trial_idx]['split'],
                })

        print(f'Generated {len(self.window_indices)} EEG window samples.')

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        trial_idx, start_idx, end_idx = self.window_indices[idx]
        window_data = self.all_trials[trial_idx][:, start_idx:end_idx]
        label = self.window_labels[idx]
        return torch.from_numpy(window_data.astype(np.float32)), torch.tensor(label, dtype=torch.long)

    def get_window_info(self, idx):
        if idx < 0 or idx >= len(self.window_info):
            raise IndexError(f'Index {idx} out of range for {len(self.window_info)} windows.')
        return self.window_info[idx]

    def get_subject_windows(self, subject_id):
        return [i for i, info in enumerate(self.window_info) if info['subject_id'] == subject_id]

    def get_split_windows(self, split):
        return [i for i, info in enumerate(self.window_info) if info['split'] == split]


class MMERDataset(Dataset):
    """EEG-only MMER dataset with video-level chunking."""

    def __init__(
        self,
        subject_ids: List[int],
        eeg_data_root: str,
        label_data_root: str = None,
        fs: int = 300,
        time_window: int = 20,
        chunk_len: int = None,
        stride: int = None,
    ):
        super().__init__()
        self.fs = fs
        self.time_window = time_window
        self.samples_per_window = fs * time_window
        self.chunk_len = chunk_len if chunk_len is not None else self.samples_per_window
        self.stride = stride if stride is not None else self.chunk_len

        if self.chunk_len > self.samples_per_window:
            raise ValueError(f'chunk_len ({self.chunk_len}) cannot exceed the time-window length ({self.samples_per_window}).')

        print('MMER dataset configuration:')
        print(f'  Time window: {time_window}s ({self.samples_per_window} samples)')
        print(f'  Chunk length: {self.chunk_len} samples')
        print(f'  Stride: {self.stride} samples')

        self.all_eeg_data = []
        self.all_labels = []
        self.video_info = []

        for subject_id in subject_ids:
            self._load_subject_data(subject_id, eeg_data_root, label_data_root)

        if not self.all_eeg_data:
            raise ValueError('No EEG data was loaded successfully.')

        print(f'Loaded {len(self.all_eeg_data)} video trials.')
        print(f'Label distribution: Positive={self.all_labels.count(0)}, Negative={self.all_labels.count(1)}, Mixed={self.all_labels.count(2)}')
        self._create_sliding_windows()

    def _load_subject_data(self, subject_id: int, eeg_data_root: str, label_data_root: str = None):
        eeg_path = os.path.join(eeg_data_root, f'{subject_id}_eeg_20s.pkl')
        if not os.path.exists(eeg_path):
            print(f'Warning: EEG file not found: {eeg_path}')
            return

        label_path = os.path.join(label_data_root, f'{subject_id}_Emotions.csv') if label_data_root else os.path.join(eeg_data_root, '..', 'Labels_emotion', f'{subject_id}_Emotions.csv')
        if not os.path.exists(label_path):
            print(f'Warning: label file not found: {label_path}')
            return

        try:
            with open(eeg_path, 'rb') as f:
                eeg_data = pickle.load(f)
            label_df = pd.read_csv(label_path)

            for idx, row in label_df.iterrows():
                video_id = int(row.iloc[0])
                amusement = float(row.iloc[1])
                disgust = float(row.iloc[2])

                if amusement > disgust:
                    emotion_label = 0
                elif amusement < disgust:
                    emotion_label = 1
                else:
                    emotion_label = 2

                if idx >= len(eeg_data):
                    continue
                video_eeg = eeg_data[idx] if isinstance(eeg_data, (list, np.ndarray)) else eeg_data
                if isinstance(video_eeg, np.ndarray):
                    self.all_eeg_data.append(video_eeg.astype(np.float32))
                    self.all_labels.append(emotion_label)
                    self.video_info.append({
                        'subject_id': subject_id,
                        'video_id': video_id,
                        'amusement': amusement,
                        'disgust': disgust,
                    })
        except Exception as exc:
            print(f'Error loading subject {subject_id}: {exc}')

    def _create_sliding_windows(self):
        self.index_map = []
        self.chunk_labels = []
        self.chunk_info = []

        for trial_idx, (eeg, label) in enumerate(zip(self.all_eeg_data, self.all_labels)):
            n_frames = eeg.shape[1] if eeg.ndim == 2 else eeg.shape[0]
            if n_frames < self.chunk_len:
                print(f'Warning: trial {trial_idx} has {n_frames} frames, smaller than chunk length {self.chunk_len}.')
                continue

            n_chunks = (n_frames - self.chunk_len) // self.stride + 1
            for chunk_idx in range(n_chunks):
                start = chunk_idx * self.stride
                end = start + self.chunk_len
                self.index_map.append((trial_idx, start, end))
                self.chunk_labels.append(label)
                self.chunk_info.append({
                    'subject_id': self.video_info[trial_idx]['subject_id'],
                    'video_id': self.video_info[trial_idx]['video_id'],
                    'chunk_idx': chunk_idx,
                    'amusement': self.video_info[trial_idx]['amusement'],
                    'disgust': self.video_info[trial_idx]['disgust'],
                })

        print(f'Generated {len(self.index_map)} EEG chunks.')

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        trial_idx, start, end = self.index_map[idx]
        eeg_data = self.all_eeg_data[trial_idx]
        label = self.chunk_labels[idx]

        if eeg_data.ndim == 2:
            x = eeg_data[:, start:end]
        else:
            x = eeg_data[start:end]
            if x.ndim == 1:
                x = x[np.newaxis, :]

        return torch.from_numpy(x.astype(np.float32)), torch.tensor(label, dtype=torch.long)

    def get_chunk_info(self, idx):
        return self.chunk_info[idx]


def bandpass_filter(data, lowcut=4.0, highcut=45.0, fs=200, order=5):
    """Apply a zero-phase band-pass filter channel by channel."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered[i, :] = data[i, :] if data.shape[1] <= order else filtfilt(b, a, data[i, :])
    return filtered


def notch_filter(data, freq=50.0, fs=200, Q=30):
    """Apply a notch filter to remove power-line noise."""
    b, a = iirnotch(freq / (fs / 2), Q)
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered[i, :] = data[i, :] if data.shape[1] <= 10 else filtfilt(b, a, data[i, :])
    return filtered


def standardize_per_channel(data):
    """Apply z-score normalization to each EEG channel."""
    data = data.copy()
    for i in range(data.shape[0]):
        mean = np.mean(data[i, :])
        std = np.std(data[i, :])
        data[i, :] = (data[i, :] - mean) / std if std > 1e-6 else data[i, :] - mean
    return data


class SEEDDataset(Dataset):
    """EEG-only SEED dataset with trial-level chunking."""

    def __init__(
        self,
        mat_list: List[str],
        label_path: str,
        chunk_len: int = 800,
        stride: int = None,
        apply_bandpass: bool = True,
        apply_notch: bool = True,
        apply_standardize: bool = False,
        fs: int = 200,
    ):
        super().__init__()
        self.chunk_len = chunk_len
        self.stride = stride if stride is not None else chunk_len
        self.fs = fs

        print('SEED dataset configuration:')
        print(f'  Chunk length: {chunk_len} samples')
        print(f'  Stride: {self.stride} samples')

        self.trials = []
        for mat_path in mat_list:
            mat = sio.loadmat(mat_path)
            pattern = re.compile(r'(\w*eeg(\d+))')
            hits = [(m.group(1), int(m.group(2))) for k in mat.keys() if (m := pattern.match(k))]
            if len(hits) < 15:
                raise ValueError(f'{mat_path} contains fewer than 15 EEG trial variables.')
            hits.sort(key=lambda x: x[1])
            for key, _ in hits:
                eeg = mat[key].astype(np.float32)
                if apply_notch:
                    eeg = notch_filter(eeg, freq=50.0, fs=fs, Q=30)
                if apply_bandpass:
                    eeg = bandpass_filter(eeg, lowcut=4.0, highcut=45.0, fs=fs, order=5)
                if apply_standardize:
                    eeg = standardize_per_channel(eeg)
                self.trials.append(eeg)

        label_mat = sio.loadmat(label_path)
        if 'label' not in label_mat:
            raise KeyError('label.mat does not contain the variable "label".')
        raw_labels = (label_mat['label'] + 1).astype(np.int64).ravel()
        n_trials = len(self.trials)
        if len(raw_labels) != 15 or n_trials % 15 != 0:
            raise ValueError('The number of trials must be a multiple of 15, and label.mat must contain 15 labels.')
        self.trial_labels = np.tile(raw_labels, n_trials // 15)

        self.index_map = []
        self.chunk_labels = []
        for trial_idx, (eeg, label) in enumerate(zip(self.trials, self.trial_labels)):
            n_frames = eeg.shape[1]
            if n_frames < self.chunk_len:
                continue
            n_chunks = (n_frames - self.chunk_len) // self.stride + 1
            for k in range(n_chunks):
                self.index_map.append((trial_idx, k * self.stride))
                self.chunk_labels.append(label)

        print(f'Generated {len(self.index_map)} EEG chunks.')

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        trial_idx, start = self.index_map[idx]
        end = start + self.chunk_len
        x = self.trials[trial_idx][:, start:end]
        y = self.chunk_labels[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def group_mat_files_by_subject(folder: str) -> Dict[str, List[str]]:
    """Group SEED .mat files by the leading subject id in each filename."""
    all_files = [f for f in os.listdir(folder) if f.endswith('.mat')]
    pattern = re.compile(r'^(\d+)_.*\.mat$')
    subject_dict: Dict[str, List[str]] = {}

    for filename in all_files:
        match = pattern.match(filename)
        if match:
            subject_id = match.group(1)
            subject_dict.setdefault(subject_id, []).append(os.path.join(folder, filename))

    for subject_id in subject_dict:
        subject_dict[subject_id].sort()
    return subject_dict
