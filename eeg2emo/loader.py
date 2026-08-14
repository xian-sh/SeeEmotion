# loader.py
import hashlib
import logging
import os
import pickle
import random

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from dataset import EAVDataset, MMERDataset, SEEDDataset, group_mat_files_by_subject


def trial_based_split_eav(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split EAV windows by original trial to avoid leakage."""
    trial_to_indices = {}
    for idx in range(len(dataset)):
        info = dataset.get_window_info(idx)
        key = (info['subject_id'], info['original_trial_idx'])
        trial_to_indices.setdefault(key, []).append(idx)

    unique_trials = list(trial_to_indices.keys())
    random.seed(seed)
    random.shuffle(unique_trials)

    n_trials = len(unique_trials)
    n_train = int(n_trials * train_ratio)
    n_val = int(n_trials * val_ratio)
    train_trials = unique_trials[:n_train]
    val_trials = unique_trials[n_train:n_train + n_val]
    test_trials = unique_trials[n_train + n_val:]

    train_indices, val_indices, test_indices = [], [], []
    for trial in train_trials:
        train_indices.extend(trial_to_indices[trial])
    for trial in val_trials:
        val_indices.extend(trial_to_indices[trial])
    for trial in test_trials:
        test_indices.extend(trial_to_indices[trial])

    print('EAV trial-level split:')
    print(f'  Train trials: {len(train_trials)}, train samples: {len(train_indices)}')
    print(f'  Val trials: {len(val_trials)}, val samples: {len(val_indices)}')
    print(f'  Test trials: {len(test_trials)}, test samples: {len(test_indices)}')

    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


def video_based_split_mmer(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split MMER chunks by video id to avoid leakage."""
    video_to_indices = {}
    for idx in range(len(dataset)):
        info = dataset.get_chunk_info(idx)
        key = (info['subject_id'], info['video_id'])
        video_to_indices.setdefault(key, []).append(idx)

    unique_videos = list(video_to_indices.keys())
    random.seed(seed)
    random.shuffle(unique_videos)

    n_videos = len(unique_videos)
    n_train = int(n_videos * train_ratio)
    n_val = int(n_videos * val_ratio)
    train_videos = unique_videos[:n_train]
    val_videos = unique_videos[n_train:n_train + n_val]
    test_videos = unique_videos[n_train + n_val:]

    train_indices, val_indices, test_indices = [], [], []
    for video in train_videos:
        train_indices.extend(video_to_indices[video])
    for video in val_videos:
        val_indices.extend(video_to_indices[video])
    for video in test_videos:
        test_indices.extend(video_to_indices[video])

    print('MMER video-level split:')
    print(f'  Train videos: {len(train_videos)}, train samples: {len(train_indices)}')
    print(f'  Val videos: {len(val_videos)}, val samples: {len(val_indices)}')
    print(f'  Test videos: {len(test_videos)}, test samples: {len(test_indices)}')

    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


def trial_based_split_seed(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split SEED chunks by trial to avoid leakage."""
    trial_to_indices = {}
    for idx in range(len(dataset)):
        trial_idx, _ = dataset.index_map[idx]
        trial_to_indices.setdefault(trial_idx, []).append(idx)

    unique_trials = list(trial_to_indices.keys())
    random.seed(seed)
    random.shuffle(unique_trials)

    n_trials = len(unique_trials)
    n_train = int(n_trials * train_ratio)
    n_val = int(n_trials * val_ratio)
    train_trials = unique_trials[:n_train]
    val_trials = unique_trials[n_train:n_train + n_val]
    test_trials = unique_trials[n_train + n_val:]

    train_indices, val_indices, test_indices = [], [], []
    for trial in train_trials:
        train_indices.extend(trial_to_indices[trial])
    for trial in val_trials:
        val_indices.extend(trial_to_indices[trial])
    for trial in test_trials:
        test_indices.extend(trial_to_indices[trial])

    print('SEED trial-level split:')
    print(f'  Train trials: {len(train_trials)}, train samples: {len(train_indices)}')
    print(f'  Val trials: {len(val_trials)}, val samples: {len(val_indices)}')
    print(f'  Test trials: {len(test_trials)}, test samples: {len(test_indices)}')

    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


class SingleDataLoader:
    def __init__(self, dataset_name, config):
        self.dataset_name = dataset_name
        self.config = config

    def get_subject_list(self):
        if self.dataset_name == 'MMER':
            return self.config['subject_list']
        return list(range(*self.config['subject_range']))

    def load_subject_data(self, subject_id):
        if self.dataset_name == 'EAV':
            return self._load_eav_subject(subject_id)
        if self.dataset_name == 'MMER':
            return self._load_mmer_subject(subject_id)
        if self.dataset_name == 'SEED':
            return self._load_seed_subject(subject_id)
        raise ValueError(f'Unsupported dataset: {self.dataset_name}')

    def _load_eav_subject(self, subject_id):
        if self.config['data_format'] == 'eav_dataset':
            dataset = EAVDataset(
                subject_ids=[subject_id],
                eeg_data_root=self.config['eeg_path'],
                window_seconds=self.config['window_seconds'],
                fs=self.config['sampling_rate'],
                overlap=self.config.get('overlap', 0.0),
            )
            train_set, val_set, test_set = trial_based_split_eav(
                dataset,
                train_ratio=self.config['train_ratio'],
                val_ratio=self.config['val_ratio'],
                seed=42,
            )
            combined_indices = list(train_set.indices) + list(val_set.indices)
            return Subset(dataset, combined_indices), test_set

        data_path = os.path.join(self.config['eeg_path'], f'subject_{subject_id:02d}_eeg.pkl')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'Subject data file not found: {data_path}')
        with open(data_path, 'rb') as f:
            tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = pickle.load(f)
        return (
            TensorDataset(torch.tensor(tr_x_eeg, dtype=torch.float32), torch.tensor(tr_y_eeg, dtype=torch.long).squeeze()),
            TensorDataset(torch.tensor(te_x_eeg, dtype=torch.float32), torch.tensor(te_y_eeg, dtype=torch.long).squeeze()),
        )

    def _load_mmer_subject(self, subject_id):
        dataset = MMERDataset(
            subject_ids=[subject_id],
            eeg_data_root=self.config['eeg_path'],
            label_data_root=self.config.get('label_path'),
            fs=self.config['fs'],
            time_window=self.config['time_window'],
            chunk_len=self.config['chunk_len'],
            stride=self.config['stride'],
        )
        train_set, val_set, test_set = video_based_split_mmer(
            dataset,
            train_ratio=self.config['train_ratio'],
            val_ratio=self.config['val_ratio'],
            seed=42,
        )
        combined_indices = list(train_set.indices) + list(val_set.indices)
        return Subset(dataset, combined_indices), test_set

    def _load_seed_subject(self, subject_id):
        names = group_mat_files_by_subject(self.config['data_path'])
        mat_list = names[f'{subject_id}']
        label_path = os.path.join(self.config['data_path'], 'label.mat')
        dataset = SEEDDataset(mat_list=mat_list, label_path=label_path, chunk_len=self.config['chunk_len'], stride=self.config['stride'])
        train_set, val_set, test_set = trial_based_split_seed(
            dataset,
            train_ratio=self.config['train_ratio'],
            val_ratio=self.config['val_ratio'],
            seed=42,
        )
        combined_indices = list(train_set.indices) + list(val_set.indices)
        return Subset(dataset, combined_indices), test_set


class CrossDataLoader:
    def __init__(self, config, dataset_type):
        self.config = config
        self.dataset_type = dataset_type

    def get_dataset_hash(self, subject_list, chunk_len, stride):
        unique_sorted = sorted(set(subject_list))
        subject_str = ','.join(map(str, unique_sorted))
        config_str = f'{subject_str}_chunk{chunk_len}_stride{stride}_{self.dataset_type}'
        return hashlib.md5(config_str.encode()).hexdigest()

    def create_dataset_from_subjects(self, subject_list):
        chunk_len = self.config.get('chunk_len', self.config['sequence_length'])
        stride = self.config.get('stride', chunk_len)

        if self.dataset_type == 'EAV':
            return EAVDataset(
                subject_ids=subject_list,
                eeg_data_root=self.config['eeg_path'],
                window_seconds=self.config['window_seconds'],
                fs=self.config['sampling_rate'],
                overlap=self.config.get('overlap', 0.0),
            )
        if self.dataset_type == 'MMER':
            return MMERDataset(
                subject_ids=subject_list,
                eeg_data_root=self.config['eeg_path'],
                label_data_root=self.config.get('label_path'),
                fs=300,
                time_window=20,
                chunk_len=chunk_len,
                stride=stride,
            )
        if self.dataset_type == 'SEED':
            names = group_mat_files_by_subject(self.config['eeg_path'])
            label_path = os.path.join(self.config['eeg_path'], 'label.mat')
            all_mat_files = []
            for subject_id in subject_list:
                mat_list = names.get(str(subject_id)) or names.get(subject_id) or names.get(f'{subject_id}')
                if mat_list is None:
                    logging.warning(f'No files found for subject {subject_id}; skipping.')
                    continue
                all_mat_files.extend(mat_list)
            if not all_mat_files:
                raise RuntimeError('No data found for the provided subject list.')
            return SEEDDataset(mat_list=all_mat_files, label_path=label_path, chunk_len=chunk_len, stride=stride)
        raise ValueError(f'Unsupported dataset type: {self.dataset_type}')

    def get_dataloaders(self):
        if self.dataset_type == 'MMER':
            all_subjects = self.config['subject_list'][:]
        else:
            all_subjects = list(range(1, self.config['num_subjects'] + 1))

        train_count = self.config['train_count']
        val_count = self.config['val_count']
        test_count = self.config['test_count']

        random.seed(42)
        random.shuffle(all_subjects)
        train_subjects = all_subjects[:train_count]
        val_subjects = all_subjects[train_count:train_count + val_count]
        test_subjects = all_subjects[train_count + val_count:train_count + val_count + test_count]

        logging.info(f'Train subjects: {train_subjects}')
        logging.info(f'Validation subjects: {val_subjects}')
        logging.info(f'Test subjects: {test_subjects}')

        if self.dataset_type == 'EAV':
            full_dataset = self.create_dataset_from_subjects(train_subjects + val_subjects + test_subjects)
            train_indices, val_indices, test_indices = [], [], []
            for idx in range(len(full_dataset)):
                subject_id = full_dataset.get_window_info(idx)['subject_id']
                if subject_id in train_subjects:
                    train_indices.append(idx)
                elif subject_id in val_subjects:
                    val_indices.append(idx)
                elif subject_id in test_subjects:
                    test_indices.append(idx)
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)
            test_dataset = Subset(full_dataset, test_indices)
        else:
            train_dataset = self.create_dataset_from_subjects(train_subjects)
            val_dataset = self.create_dataset_from_subjects(val_subjects)
            test_dataset = self.create_dataset_from_subjects(test_subjects)

        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['val_batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config['test_batch_size'], shuffle=False)
        logging.info(f'Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}')
        return train_loader, val_loader, test_loader, (train_subjects, val_subjects, test_subjects)
