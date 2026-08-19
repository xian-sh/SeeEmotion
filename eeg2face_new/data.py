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
class AlignedRecord:
    subject_id: int
    trial_id: int
    frame_id: int


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
        ])

    seen: set[Path] = set()
    existing: list[Path] = []
    for item in roots:
        if item not in seen and item.exists():
            existing.append(item)
            seen.add(item)
    return existing


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
        looks_like_eav_tuple = False
        if len(payload) >= 4:
            first = np.asarray(payload[0])
            second = np.asarray(payload[1])
            third = np.asarray(payload[2])
            fourth = np.asarray(payload[3])
            looks_like_eav_tuple = first.ndim >= 3 and third.ndim >= 3 and second.ndim <= 2 and fourth.ndim <= 2
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
    return face_sequence


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


def normalize_eeg_trials(eeg_trials: list[np.ndarray], mode: str) -> dict[str, np.ndarray | float | str]:
    if mode == "none":
        return {"mode": mode, "scale": 1.0}
    if mode == "max_abs":
        max_abs = 0.0
        for trial in eeg_trials:
            max_abs = max(max_abs, float(np.max(np.abs(as_channel_first(trial)), initial=0.0)))
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


class EEGFacePool:
    def __init__(self, config: Config) -> None:
        self.records: list[AlignedRecord] = []
        self.subject_trials: dict[int, dict[str, object]] = {}
        self.skipped_subjects: dict[int, str] = {}
        self.eeg_stats: dict[str, np.ndarray | float | str] | None = None

        subject_ids = parse_id_list(config.data.subject_ids) or get_valid_subject_ids(config.data.eeg_dir)
        explicit_train = parse_id_list(config.data.train_subjects)
        explicit_val = parse_id_list(config.data.val_subjects)
        explicit_test = parse_id_list(config.data.test_subjects)
        if explicit_train or explicit_val or explicit_test:
            subject_ids = sorted(set((explicit_train or []) + (explicit_val or []) + (explicit_test or [])))

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
            LOGGER.warning("Skipping subject %02d because no paired EEG/face trials were found.", subject_id)
            return

        eeg_trials = eeg_trials[:n_trials]
        face_trials = face_trials[:n_trials]
        self.subject_trials[subject_id] = {"eeg": eeg_trials, "face": face_trials, "n_trials": n_trials}

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
                self.records.append(AlignedRecord(subject_id, trial_id, frame_id))
                valid_count += 1
        LOGGER.info("Loaded subject %02d from %s and %s: %d trials, %d aligned samples.", subject_id, eeg_path.name, face_path.name, n_trials, valid_count)

    def fit_normalization(self, train_indices: np.ndarray, mode: str) -> None:
        train_trial_keys = {(self.records[int(index)].subject_id, self.records[int(index)].trial_id) for index in train_indices}
        eeg_trials = [self.subject_trials[s]["eeg"][t] for s, t in sorted(train_trial_keys)]
        self.eeg_stats = normalize_eeg_trials(eeg_trials, mode)


class EEGFaceDataset(Dataset):
    def __init__(self, pool: EEGFacePool, indices: np.ndarray, config: Config) -> None:
        self.pool = pool
        self.indices = np.asarray(indices, dtype=np.int64)
        self.config = config

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.pool.records[int(self.indices[index])]
        item = self.pool.subject_trials[record.subject_id]
        eeg_trials = item["eeg"]
        face_trials = item["face"]
        face_sequence = fit_face_frames(face_trials[record.trial_id], self.config.data.n_frames)
        eeg = center_window(
            eeg_trials[record.trial_id],
            record.frame_id,
            face_sequence.shape[0],
            self.config.data.sampling_rate,
            self.config.data.trial_seconds,
            self.config.data.eeg_window_seconds,
        )
        if self.pool.eeg_stats is None:
            raise RuntimeError("EEG normalization statistics have not been fitted.")
        eeg = apply_eeg_normalization(eeg, self.pool.eeg_stats)
        face = face_sequence[record.frame_id]
        return {
            "eeg": torch.from_numpy(eeg),
            "face": torch.from_numpy(face[None, :, :].astype(np.float32)),
            "subject_id": torch.tensor(record.subject_id, dtype=torch.long),
            "trial_id": torch.tensor(record.trial_id, dtype=torch.long),
            "trial_key": torch.tensor(record.subject_id * self.config.data.max_trials_per_subject + record.trial_id, dtype=torch.long),
            "frame_id": torch.tensor(record.frame_id, dtype=torch.long),
            "identity_id": torch.tensor(0, dtype=torch.long),
        }


def split_list(items: list[Any], train_ratio: float, val_ratio: float, test_ratio: float, rng: np.random.Generator) -> tuple[list[Any], list[Any], list[Any]]:
    values = list(items)
    rng.shuffle(values)
    n_items = len(values)
    if n_items < 3:
        raise ValueError("At least 3 groups are required for a train/val/test split.")
    ratio_sum = train_ratio + val_ratio + test_ratio
    train_ratio = train_ratio / ratio_sum
    val_ratio = val_ratio / ratio_sum
    n_test = max(1, int(round(n_items * (test_ratio / ratio_sum))))
    n_val = max(1, int(round(n_items * val_ratio)))
    if n_val + n_test >= n_items:
        n_val = max(1, min(n_val, n_items - 2))
        n_test = max(1, min(n_test, n_items - n_val - 1))
    n_train = n_items - n_val - n_test
    return values[:n_train], values[n_train:n_train + n_val], values[n_train + n_val:]


def split_indices(pool: EEGFacePool, config: Config, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    loaded_subjects = sorted(pool.subject_trials)
    explicit_train = parse_id_list(config.data.train_subjects)
    explicit_val = parse_id_list(config.data.val_subjects)
    explicit_test = parse_id_list(config.data.test_subjects)

    if config.data.split_mode == "subject":
        if explicit_train or explicit_val or explicit_test:
            train_subjects = explicit_train or []
            val_subjects = explicit_val or []
            test_subjects = explicit_test or []
            if not train_subjects or not val_subjects or not test_subjects:
                raise ValueError("--split-mode subject requires all of --train-subjects, --val-subjects, and --test-subjects when explicit subjects are used.")
        else:
            if len(loaded_subjects) < 3:
                LOGGER.warning("Subject-level split needs at least 3 subjects; falling back to trial-level split.")
                return split_indices_by_trial(pool, config, rng, protocol="trial")
            train_subjects, val_subjects, test_subjects = split_list(
                loaded_subjects,
                config.data.train_ratio,
                config.data.val_ratio,
                config.data.test_ratio,
                rng,
            )
        train_set, val_set, test_set = set(train_subjects), set(val_subjects), set(test_subjects)
        train = [idx for idx, record in enumerate(pool.records) if record.subject_id in train_set]
        val = [idx for idx, record in enumerate(pool.records) if record.subject_id in val_set]
        test = [idx for idx, record in enumerate(pool.records) if record.subject_id in test_set]
        return np.asarray(train), np.asarray(val), np.asarray(test), {
            "protocol": "subject",
            "train_subjects": sorted(train_set),
            "val_subjects": sorted(val_set),
            "test_subjects": sorted(test_set),
        }

    return split_indices_by_trial(pool, config, rng, protocol="trial")


def split_indices_by_trial(pool: EEGFacePool, config: Config, rng: np.random.Generator, protocol: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    grouped: dict[tuple[int, int], list[int]] = {}
    for index, record in enumerate(pool.records):
        grouped.setdefault((record.subject_id, record.trial_id), []).append(index)

    train_trials, val_trials, test_trials = split_list(
        list(grouped),
        config.data.train_ratio,
        config.data.val_ratio,
        config.data.test_ratio,
        rng,
    )
    train = [index for key in train_trials for index in grouped[key]]
    val = [index for key in val_trials for index in grouped[key]]
    test = [index for key in test_trials for index in grouped[key]]
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return np.asarray(train), np.asarray(val), np.asarray(test), {
        "protocol": protocol,
        "train_trials": len(train_trials),
        "val_trials": len(val_trials),
        "test_trials": len(test_trials),
    }


def split_audit(pool: EEGFacePool, train_indices: np.ndarray, val_indices: np.ndarray, test_indices: np.ndarray) -> dict[str, float | int]:
    def trial_keys(indices: np.ndarray) -> set[tuple[int, int]]:
        return {(pool.records[int(index)].subject_id, pool.records[int(index)].trial_id) for index in indices}

    train_trials = trial_keys(train_indices)
    val_trials = trial_keys(val_indices)
    test_trials = trial_keys(test_indices)
    train_subjects = {subject_id for subject_id, _ in train_trials}
    val_subjects = {subject_id for subject_id, _ in val_trials}
    test_subjects = {subject_id for subject_id, _ in test_trials}
    return {
        "unique_aligned_samples": len(pool.records),
        "unique_train_trials": len(train_trials),
        "unique_val_trials": len(val_trials),
        "unique_test_trials": len(test_trials),
        "shared_train_val_trials": len(train_trials & val_trials),
        "shared_train_test_trials": len(train_trials & test_trials),
        "shared_val_test_trials": len(val_trials & test_trials),
        "shared_train_val_subjects": len(train_subjects & val_subjects),
        "shared_train_test_subjects": len(train_subjects & test_subjects),
        "shared_val_test_subjects": len(val_subjects & test_subjects),
    }


def create_dataloaders(config: Config) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    pool = EEGFacePool(config)
    rng = np.random.default_rng(config.train.seed)
    train_indices, val_indices, test_indices, split_detail = split_indices(pool, config, rng)
    pool.fit_normalization(train_indices, config.data.eeg_normalization)
    train_set = EEGFaceDataset(pool, train_indices, config)
    val_set = EEGFaceDataset(pool, val_indices, config)
    test_set = EEGFaceDataset(pool, test_indices, config)
    loader_args = dict(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    split_info = {
        "dataset": config.data.dataset,
        "split_mode": config.data.split_mode,
        "subjects": sorted(pool.subject_trials),
        "skipped_subjects": pool.skipped_subjects,
        "eeg_channels": config.data.eeg_channels,
        "sampling_rate": config.data.sampling_rate,
        "trial_seconds": config.data.trial_seconds,
        "n_frames": config.data.n_frames,
        "window_seconds": config.data.eeg_window_seconds,
        "train_ratio": config.data.train_ratio,
        "val_ratio": config.data.val_ratio,
        "test_ratio": config.data.test_ratio,
        "total_records": len(pool.records),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "test_samples": len(test_set),
        **split_detail,
        **split_audit(pool, train_indices, val_indices, test_indices),
    }
    return (
        DataLoader(train_set, shuffle=True, drop_last=False, **loader_args),
        DataLoader(val_set, shuffle=False, drop_last=False, **loader_args),
        DataLoader(test_set, shuffle=False, drop_last=False, **loader_args),
        split_info,
    )
