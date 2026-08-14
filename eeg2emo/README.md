# Clean EEG Emotion Recognition

This is a compact EEG-only emotion recognition codebase. The pipeline now focuses on EEG input, emotion labels, classification training, and evaluation.

## Files

- `dataset.py`: EEG-only dataset loaders for EAV, MMER, and SEED.
- `loader.py`: single-subject and cross-subject dataloader construction.
- `model_eeg.py`: custom EEG classifier that returns classification logits only.
- `engine.py`: classification training, evaluation, and feature extraction.
- `run_single.py`: single-subject training entry point.
- `run_cross.py`: cross-subject training entry point.
- `config.py`: dataset, model, and EEG-only ablation configs.
- `compute.py`, `util.py`: experiment utilities.

## Examples

```bash
python run_single.py --dataset MMER --run 1 --epochs 50
python run_cross.py --dataset MMER --run 1 --epochs 30
python run_single.py --dataset EAV --run 1 --ablation_study --ablation_type depth
```

If you want to use baseline models (`--run 2` and above), place your original `backbones.py` next to these files. The custom model (`--run 1`) does not require `backbones.py`.
