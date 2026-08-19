from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config, parse_id_list


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleMeta:
    subject_id: int
    trial_id: int
    frame_id: int
    copy_id: int
    identity_id: int


def get_valid_subject_ids(eeg_dir: Path) -> list[int]:
    subject_ids: list[int] = []
    patterns = [
        re.compile(r"subject_(\d+)_eeg\.pkl"),
        re.compile(r"subject_(\d+).*\.pkl"),
        re.compile(r"sub(?:ject)?_?(\d+).*\.pkl"),
        re.compile(r"(\d+)_eeg_20s\.pkl"),
        re.compile(r"(\d+)_eeg\.pkl"),
        re.compile(r"s_?(\d+).*\.pkl"),
    ]
    for root in candidate_roots(eeg_dir, "eeg"):
        for path in sorted(root.glob("*.pkl")):
            for pattern in patterns:
                match = pattern.match(path.name.lower())
                if match:
                    subject_ids.append(int(match.group(1)))
                    break
    return sorted(set(subject_ids))


def candidate_roots(root: Path, kind: str) -> list[Path]:
    roots: list[Path] = [root]
    if kind == "eeg" and root.name.lower() == "eeg" and root.parent.name.lower() == "aligned_data":
        roots.append(root.parent.parent / "EEG")
    if kind == "face":
        if root.name.lower() == "landmarks" and root.parent.name.lower() == "aligned_data":
            roots.append(root.parent.parent / "Landmarks")
        roots.extend([
            root.parent / "Vision_Landmarks_25x56x56",
            root.parent / "Vision_Landmarks_2x56x56",
            root.parent / "Landmarks_64x64",
            root.parent / "Landmarks_4x64x64",
            root.parent / "Aligned_data" / "Landmarks",
        ])

    seen: set[Path] = set()
    existing: list[Path] = []
    for item in roots:
        if item not in seen and item.exists():
            existing.append(item)
            seen.add(item)
    return existing


def candidate_file_paths(root: Path, subject_id: int, kind: str) -> list[Path]:
    if kind == "eeg":
        names = [
            f"{subject_id}_eeg_20s.pkl",
            f"{subject_id:02d}_eeg_20s.pkl",
            f"{subject_id}_eeg.pkl",
            f"{subject_id:02d}_eeg.pkl",
            f"subject_{subject_id:02d}_eeg.pkl",
            f"subject_{subject_id}_eeg.pkl",
            f"subject{subject_id:02d}_eeg.pkl",
            f"subject{subject_id}_eeg.pkl",
            f"subject_{subject_id:02d}.pkl",
            f"subject_{subject_id}.pkl",
            f"sub{subject_id:02d}_eeg.pkl",
            f"sub{subject_id}_eeg.pkl",
            f"s{subject_id:02d}_eeg.pkl",
            f"s{subject_id}_eeg.pkl",
        ]
    else:
        names = [
            f"{subject_id}_landmarks.pkl",
            f"{subject_id:02d}_landmarks.pkl",
            f"{subject_id}_Landmarks.pkl",
            f"{subject_id:02d}_Landmarks.pkl",
            f"{subject_id}_face.pkl",
            f"{subject_id:02d}_face.pkl",
            f"{subject_id}.pkl",
            f"{subject_id:02d}.pkl",
            f"emoji_vis_subject_{subject_id:02d}.pkl",
            f"emoji_vis_subject_{subject_id}.pkl",
            f"emoji_vis_subject{subject_id:02d}.pkl",
            f"emoji_vis_subject{subject_id}.pkl",
            f"subject_{subject_id:02d}_landmarks.pkl",
            f"subject_{subject_id}_landmarks.pkl",
            f"subject{subject_id:02d}_landmarks.pkl",
            f"subject{subject_id}_landmarks.pkl",
            f"subject_{subject_id:02d}.pkl",
            f"subject_{subject_id}.pkl",
        ]
    return [candidate_root / name for candidate_root in candidate_roots(root, kind) for name in names]


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def subject_path_score(path: Path, subject_id: int, kind: str) -> int:
    stem = path.stem.lower()
    if not path.name.lower().endswith(".pkl"):
        return -1

    subject_pattern = rf"(?:^|[_-])(?:subject|sub|s)?[_-]?0*{subject_id}(?!\d)"
    leading_pattern = rf"^0*{subject_id}(?!\d)"
    if re.search(subject_pattern, stem):
        score = 50
    elif re.search(leading_pattern, stem):
        score = 45
    else:
        return -1

    preferred = ("eeg",) if kind == "eeg" else ("landmark", "emoji", "face", "vis")
    wrong = ("landmark", "emoji", "face", "vis") if kind == "eeg" else ("eeg", "label", "emotion")
    score += sum(10 for token in preferred if token in stem)
    score -= sum(20 for token in wrong if token in stem)
    score -= len(path.parts)
    return score


def find_subject_file(root: Path, subject_id: int, kind: str) -> Path | None:
    direct = first_existing(candidate_file_paths(root, subject_id, kind))
    if direct is not None:
        return direct

    scored: list[tuple[int, Path]] = []
    for candidate_root in candidate_roots(root, kind):
        for path in candidate_root.rglob("*.pkl"):
            score = subject_path_score(path, subject_id, kind)
            if score >= 0:
                scored.append((score, path))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], len(str(item[1])), str(item[1])))
    return scored[0][1]


def as_channel_first(eeg: np.ndarray) -> np.ndarray:
    eeg = np.asarray(eeg, dtype=np.float32)
    if eeg.ndim != 2:
        raise ValueError(f"Expected EEG trial with shape (C,T) or (T,C), got {eeg.shape}.")
    if eeg.shape[0] > eeg.shape[1]:
        eeg = eeg.T
    return eeg


def normalize_face(face: np.ndarray) -> np.ndarray:
    face = np.asarray(face, dtype=np.float32)
    if face.max(initial=0.0) > 1.0:
        face = face / 255.0
    return np.clip(face, 0.0, 1.0).astype(np.float32)


def trial_list_from_array(value: Any, kind: str) -> list[np.ndarray]:
    array = np.asarray(value)
    if array.size == 0:
        return []
    if kind == "eeg":
        if array.ndim == 2:
            return [array.astype(np.float32)]
        if array.ndim >= 3:
            return [np.asarray(item, dtype=np.float32) for item in array]
    else:
        if array.ndim == 2:
            return [array.astype(np.float32)]
        if array.ndim == 3:
            if array.shape[1] == array.shape[2] and array.shape[0] > 64:
                return [np.asarray(item, dtype=np.float32) for item in array]
            return [array.astype(np.float32)]
        if array.ndim >= 4:
            return [np.asarray(item, dtype=np.float32) for item in array]
    return []


def trial_list_from_payload(payload: Any, kind: str) -> list[np.ndarray]:
    if isinstance(payload, dict):
        paired_keys = [
            ("train", "test"),
            ("x_train", "x_test"),
            ("train_data", "test_data"),
            ("eeg_train", "eeg_test"),
            ("face_train", "face_test"),
            ("landmark_train", "landmark_test"),
            ("tr_x", "te_x"),
            ("tr_x_eeg", "te_x_eeg"),
        ]
        for train_key, test_key in paired_keys:
            if train_key in payload and test_key in payload:
                return trial_list_from_array(payload[train_key], kind) + trial_list_from_array(payload[test_key], kind)
        for key in ("data", "eeg", "face", "landmarks", "emoji", "images", "x"):
            if key in payload:
                return trial_list_from_array(payload[key], kind)
        trials: list[np.ndarray] = []
        for value in payload.values():
            if isinstance(value, (list, tuple, np.ndarray)):
                trials.extend(trial_list_from_array(value, kind))
        return trials

    if isinstance(payload, (list, tuple)):
        if len(payload) >= 4:
            first = np.asarray(payload[0])
            second = np.asarray(payload[1])
            third = np.asarray(payload[2])
            fourth = np.asarray(payload[3])
            looks_like_eav_tuple = first.ndim >= 3 and third.ndim >= 3 and second.ndim <= 2 and fourth.ndim <= 2
        else:
            looks_like_eav_tuple = False
        if looks_like_eav_tuple:
            merged = trial_list_from_array(payload[0], kind) + trial_list_from_array(payload[2], kind)
            if merged:
                return merged
        return [np.asarray(item, dtype=np.float32) for item in payload if isinstance(item, np.ndarray)]

    return trial_list_from_array(payload, kind)


def flatten_face_sequence(face_sequence: np.ndarray) -> np.ndarray:
    face_sequence = normalize_face(face_sequence)
    if face_sequence.ndim == 2:
        return face_sequence[None, :, :]
    if face_sequence.ndim == 3:
        return face_sequence
    if face_sequence.ndim == 4 and face_sequence.shape[-2] == face_sequence.shape[-1]:
        return face_sequence.reshape(-1, face_sequence.shape[-2], face_sequence.shape[-1])
    if face_sequence.ndim == 4 and face_sequence.shape[1] == 1:
        return face_sequence[:, 0]
    if face_sequence.ndim == 4 and face_sequence.shape[-1] == 1:
        return face_sequence[..., 0]
    raise ValueError(f"Expected face data with image axes at the end, got {face_sequence.shape}.")


def align_face_trials_to_eeg(eeg_trials: list[np.ndarray], face_trials: list[np.ndarray]) -> list[np.ndarray]:
    """Group flat MMER face frames back to EEG trials."""
    if len(face_trials) == len(eeg_trials):
        return [flatten_face_sequence(item) for item in face_trials]
    if len(eeg_trials) > 0 and len(face_trials) % len(eeg_trials) == 0:
        frames_per_trial = len(face_trials) // len(eeg_trials)
        aligned: list[np.ndarray] = []
        for trial_id in range(len(eeg_trials)):
            start = trial_id * frames_per_trial
            end = start + frames_per_trial
            aligned.append(flatten_face_sequence(np.asarray(face_trials[start:end], dtype=np.float32)))
        LOGGER.info(
            "Grouped %d flat face frames into %d EEG trials with %d frames per trial.",
            len(face_trials),
            len(eeg_trials),
            frames_per_trial,
        )
        return aligned
    return [flatten_face_sequence(item) for item in face_trials]


def normalize_eeg_trials(eeg_trials: list[np.ndarray], mode: str) -> dict[str, np.ndarray | float | str]:
    if mode == "none":
        return {"mode": mode, "scale": 1.0}
    if mode == "max_abs":
        max_abs = 0.0
        for trial in eeg_trials:
            max_abs = max(max_abs, float(np.max(np.abs(np.asarray(trial, dtype=np.float32)), initial=0.0)))
        return {"mode": mode, "scale": max(max_abs, 1e-6)}
    if mode == "zscore":
        stacked = np.concatenate([as_channel_first(trial) for trial in eeg_trials], axis=1)
        mean = stacked.mean(axis=1, keepdims=True).astype(np.float32)
        std = np.maximum(stacked.std(axis=1, keepdims=True).astype(np.float32), 1e-6)
        return {"mode": mode, "mean": mean, "std": std}
    raise ValueError(f"Unknown EEG normalization: {mode}")


def apply_eeg_normalization(eeg: np.ndarray, stats: dict[str, np.ndarray | float | str]) -> np.ndarray:
    mode = str(stats["mode"])
    if mode == "none":
        return eeg.astype(np.float32)
    if mode == "max_abs":
        return (eeg / float(stats["scale"])).astype(np.float32)
    if mode == "zscore":
        return ((eeg - np.asarray(stats["mean"], dtype=np.float32)) / np.asarray(stats["std"], dtype=np.float32)).astype(np.float32)
    raise ValueError(f"Unknown EEG normalization: {mode}")


def fit_face_frames(face_sequence: np.ndarray, n_frames: int) -> np.ndarray:
    face_sequence = normalize_face(face_sequence)
    if face_sequence.ndim == 4 and face_sequence.shape[-2] == face_sequence.shape[-1]:
        face_sequence = face_sequence.reshape(-1, face_sequence.shape[-2], face_sequence.shape[-1])
    if face_sequence.ndim == 4 and face_sequence.shape[1] == 1:
        face_sequence = face_sequence[:, 0]
    if face_sequence.ndim == 4 and face_sequence.shape[-1] == 1:
        face_sequence = face_sequence[..., 0]
    if face_sequence.ndim == 2:
        face_sequence = face_sequence[None, :, :]
    if face_sequence.ndim != 3:
        raise ValueError(f"Expected face sequence with shape (F,H,W), got {face_sequence.shape}.")
    if face_sequence.shape[0] == n_frames:
        return face_sequence
    if face_sequence.shape[0] > n_frames:
        indices = np.linspace(0, face_sequence.shape[0] - 1, n_frames, dtype=np.int64)
        return face_sequence[indices]
    pad = np.repeat(face_sequence[-1:], n_frames - face_sequence.shape[0], axis=0)
    return np.concatenate([face_sequence, pad], axis=0)


def valid_frame_ids(total_samples: int, n_frames: int, sampling_rate: float, trial_seconds: float, window_seconds: float) -> list[int]:
    window_samples = int(round(window_seconds * sampling_rate))
    ids: list[int] = []
    for frame_id in range(n_frames):
        frame_time = (frame_id + 0.5) * (trial_seconds / n_frames)
        center = int(round(frame_time * sampling_rate))
        start = center - window_samples // 2
        end = start + window_samples
        if start >= 0 and end <= total_samples:
            ids.append(frame_id)
    return ids


def center_window(eeg: np.ndarray, frame_id: int, n_frames: int, sampling_rate: float, trial_seconds: float, window_seconds: float) -> np.ndarray:
    eeg = as_channel_first(eeg)
    window_samples = int(round(window_seconds * sampling_rate))
    frame_time = (frame_id + 0.5) * (trial_seconds / n_frames)
    center = int(round(frame_time * sampling_rate))
    start = center - window_samples // 2
    end = start + window_samples
    if start < 0 or end > eeg.shape[1]:
        raise ValueError(f"Frame {frame_id} requires EEG window [{start}, {end}), but trial length is {eeg.shape[1]}.")
    return eeg[:, start:end]


class ReferenceEEGFacePool:
    def __init__(self, config: Config) -> None:
        self.records: list[tuple[int, int, int, int]] = []
        self.identity_to_id: dict[tuple[int, int, int], int] = {}
        self.subject_trials: dict[int, dict[str, object]] = {}
        self.skipped_subjects: dict[int, str] = {}
        subject_ids = parse_id_list(config.data.subject_ids) or get_valid_subject_ids(config.data.eeg_dir)
        explicit_train = parse_id_list(config.data.train_subjects)
        explicit_test = parse_id_list(config.data.test_subjects)
        if explicit_train or explicit_test:
            subject_ids = sorted(set((explicit_train or []) + (explicit_test or [])))

        if config.data.repeat < 1:
            raise ValueError("--repeat must be at least 1.")
        if config.data.split_mode == "paired_reference" and config.data.repeat < 2:
            raise ValueError("--split-mode paired_reference requires --repeat >= 2.")

        for subject_id in subject_ids:
            self.load_subject(subject_id, config)
        if not self.records:
            raise RuntimeError("No EEG-to-face records were loaded.")

    def load_subject(self, subject_id: int, config: Config) -> None:
        eeg_path = find_subject_file(config.data.eeg_dir, subject_id, "eeg")
        face_path = find_subject_file(config.data.face_root, subject_id, "face")
        if eeg_path is None or face_path is None:
            self.skipped_subjects[subject_id] = f"missing_eeg_or_face_file: eeg={eeg_path}, face={face_path}"
            LOGGER.warning("Skipping subject %02d because EEG or face file is missing. eeg=%s face=%s", subject_id, eeg_path, face_path)
            return

        try:
            with eeg_path.open("rb") as handle:
                eeg_payload = pickle.load(handle)
            with face_path.open("rb") as handle:
                face_payload = pickle.load(handle)
        except Exception as error:
            self.skipped_subjects[subject_id] = f"load_failed: {error}"
            LOGGER.warning("Skipping subject %02d because loading failed: %s", subject_id, error)
            return

        eeg_trials = trial_list_from_payload(eeg_payload, "eeg")
        face_trials = trial_list_from_payload(face_payload, "face")
        face_trials = align_face_trials_to_eeg(eeg_trials, face_trials)
        n_trials = min(len(eeg_trials), len(face_trials))
        if n_trials == 0:
            self.skipped_subjects[subject_id] = f"empty_trials: eeg={len(eeg_trials)}, face={len(face_trials)}"
            LOGGER.warning("Skipping subject %02d because no paired EEG/face trials were found. eeg=%d face=%d", subject_id, len(eeg_trials), len(face_trials))
            return
        eeg_trials = eeg_trials[:n_trials]
        face_trials = face_trials[:n_trials]
        stats = normalize_eeg_trials(eeg_trials, config.data.eeg_normalization)
        self.subject_trials[subject_id] = {"eeg": eeg_trials, "face": face_trials, "stats": stats, "n_trials": n_trials}

        valid_count = 0
        for trial_id in range(n_trials):
            eeg = as_channel_first(eeg_trials[trial_id])
            face_sequence = fit_face_frames(face_trials[trial_id], config.data.n_frames)
            frame_ids = valid_frame_ids(
                total_samples=eeg.shape[1],
                n_frames=face_sequence.shape[0],
                sampling_rate=config.data.sampling_rate,
                trial_seconds=config.data.trial_seconds,
                window_seconds=config.data.eeg_window_seconds,
            )
            for frame_id in frame_ids:
                identity = (subject_id, trial_id, frame_id)
                if identity not in self.identity_to_id:
                    self.identity_to_id[identity] = len(self.identity_to_id)
                for copy_id in range(config.data.repeat):
                    self.records.append((subject_id, trial_id, frame_id, copy_id))
                valid_count += 1
        LOGGER.info("Loaded subject %02d from %s and %s: %d trials, %d aligned samples, %d repeated records.", subject_id, eeg_path.name, face_path.name, n_trials, valid_count, valid_count * config.data.repeat)


class ReferenceEEGFaceDataset(Dataset):
    def __init__(self, pool: ReferenceEEGFacePool, indices: np.ndarray, config: Config) -> None:
        self.pool = pool
        self.indices = np.asarray(indices, dtype=np.int64)
        self.config = config

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        subject_id, trial_id, frame_id, copy_id = self.pool.records[int(self.indices[index])]
        item = self.pool.subject_trials[subject_id]
        eeg_trials = item["eeg"]
        face_trials = item["face"]
        stats = item["stats"]
        face_sequence = fit_face_frames(face_trials[trial_id], self.config.data.n_frames)
        eeg = center_window(
            eeg_trials[trial_id],
            frame_id,
            face_sequence.shape[0],
            self.config.data.sampling_rate,
            self.config.data.trial_seconds,
            self.config.data.eeg_window_seconds,
        )
        eeg = apply_eeg_normalization(eeg, stats)
        face = face_sequence[frame_id]
        identity_id = self.pool.identity_to_id[(subject_id, trial_id, frame_id)]
        return {
            "eeg": torch.from_numpy(eeg),
            "face": torch.from_numpy(face[None, :, :].astype(np.float32)),
            "subject_id": torch.tensor(subject_id, dtype=torch.long),
            "trial_id": torch.tensor(trial_id, dtype=torch.long),
            "trial_key": torch.tensor(subject_id * self.config.data.max_trials_per_subject + trial_id, dtype=torch.long),
            "frame_id": torch.tensor(frame_id, dtype=torch.long),
            "identity_id": torch.tensor(identity_id, dtype=torch.long),
            "copy_id": torch.tensor(copy_id, dtype=torch.long),
        }


def record_identity(record: tuple[int, int, int, int]) -> tuple[int, int, int]:
    subject_id, trial_id, frame_id, _ = record
    return subject_id, trial_id, frame_id


def split_indices(pool: ReferenceEEGFacePool, config: Config, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if config.data.split_mode == "random":
        all_indices = np.arange(len(pool.records), dtype=np.int64)
        rng.shuffle(all_indices)
        n_test = int(round(len(all_indices) * config.data.test_ratio))
        return all_indices[n_test:], all_indices[:n_test]

    grouped: dict[tuple[int, int, int], list[int]] = {}
    for index, record in enumerate(pool.records):
        grouped.setdefault(record_identity(record), []).append(index)
    test_copies = max(1, min(config.data.repeat - 1, int(round(config.data.repeat * config.data.test_ratio))))
    train_indices: list[int] = []
    test_indices: list[int] = []
    for indices in grouped.values():
        shuffled = np.asarray(indices, dtype=np.int64)
        rng.shuffle(shuffled)
        test_indices.extend(shuffled[:test_copies].tolist())
        train_indices.extend(shuffled[test_copies:].tolist())
    train = np.asarray(train_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def leakage_report(pool: ReferenceEEGFacePool, train_indices: np.ndarray, test_indices: np.ndarray) -> dict[str, float | int]:
    train_keys = {record_identity(pool.records[int(index)]) for index in train_indices}
    test_keys = {record_identity(pool.records[int(index)]) for index in test_indices}
    leaked_test_records = sum(1 for index in test_indices if record_identity(pool.records[int(index)]) in train_keys)
    leaked_identities = len(test_keys & train_keys)
    return {
        "unique_aligned_samples": len({record_identity(record) for record in pool.records}),
        "unique_train_identities": len(train_keys),
        "unique_test_identities": len(test_keys),
        "shared_test_records": leaked_test_records,
        "shared_test_record_fraction": leaked_test_records / max(1, len(test_indices)),
        "shared_test_identity_fraction": leaked_identities / max(1, len(test_keys)),
    }


def create_dataloaders(config: Config) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    pool = ReferenceEEGFacePool(config)
    rng = np.random.default_rng(config.train.seed)
    train_indices, test_indices = split_indices(pool, config, rng)
    train_set = ReferenceEEGFaceDataset(pool, train_indices, config)
    test_set = ReferenceEEGFaceDataset(pool, test_indices, config)
    loader_args = dict(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    split_info = {
        "dataset": config.data.dataset,
        "protocol": config.data.split_mode,
        "subjects": sorted(pool.subject_trials),
        "skipped_subjects": pool.skipped_subjects,
        "eeg_channels": config.data.eeg_channels,
        "sampling_rate": config.data.sampling_rate,
        "trial_seconds": config.data.trial_seconds,
        "n_frames": config.data.n_frames,
        "window_seconds": config.data.eeg_window_seconds,
        "repeat": config.data.repeat,
        "total_records": len(pool.records),
        "train_samples": len(train_set),
        "test_samples": len(test_set),
        **leakage_report(pool, train_indices, test_indices),
    }
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_args)
    return (
        DataLoader(train_set, shuffle=True, drop_last=False, **loader_args),
        test_loader,
        test_loader,
        split_info,
    )
