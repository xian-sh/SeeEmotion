# EEG2Face Reference

This folder contains a reference EEG-to-facial-emoji reconstruction pipeline.
It keeps only the EEG-to-image branch:

```text
EEG window -> EEG backbone -> AutoencoderKL latent -> landmark emoji image
```

The Face2Face AutoencoderKL is frozen. The trainable part maps EEG windows to
the pretrained KL latent space.

This version is kept for reference and compatibility with earlier reported
settings. For new experiments and public benchmarking, use the latest
`eeg2face` implementation as the recommended version.

## Files

```text
eeg2face_ref/
+-- config.py
+-- data.py
+-- autoencoder.py
+-- model.py
+-- losses.py
+-- metrics.py
+-- train_utils.py
+-- train_single.py
+-- train_cross.py
+-- evaluate.py
+-- requirements.txt
```

## Supported Datasets

### EAV

Expected files:

```text
EEG/
+-- subject_01_eeg.pkl
+-- subject_02_eeg.pkl

Vision_Landmarks_25x56x56/
+-- emoji_vis_subject_01.pkl
+-- emoji_vis_subject_02.pkl
```

Default EAV settings:

```text
EEG channels: 30
Sampling rate: 100 Hz
Trial length: 5 s
Face frames: 25
Image size: 56 x 56
```

### MMER

Expected EEG files follow the MMER style used by `eeg2emo`:

```text
MMER/
+-- EEG/
|   +-- 1_eeg_20s.pkl
|   +-- 5_eeg_20s.pkl
+-- Aligned_data/
    +-- Landmarks/
        +-- 1_landmarks.pkl
        +-- 5_landmarks.pkl
```

The loader also accepts common EAV-style names such as
`subject_01_eeg.pkl`, `emoji_vis_subject_01.pkl`, and
`subject_01_landmarks.pkl`. If `--eeg-dir` is set to
`MMER/Aligned_data/EEG`, the loader will also check the sibling
`MMER/EEG` folder automatically.

Default MMER settings:

```text
EEG channels: 18
Sampling rate: 300 Hz
Trial length: 2 s
Face frames: 4
Image size: 64 x 64
```

If your MMER landmark files contain a different number of frames, pass
`--n-frames` explicitly.

## Data Protocol

The default protocol follows the reference setting used by this folder:

```text
original train/test trials are merged
frame-level EEG-image samples are created
each aligned sample is repeated
copies are split into train and test
best checkpoint is selected by test SSIM
```

This is intended to reproduce the reference behavior and inspect the upper
range of the pipeline. Use the latest strict `eeg2face` version for a cleaner
generalization protocol.

## EAV Training

```bash
python train_cross.py \
  --dataset EAV \
  --eeg-dir /path/to/EAV/EEG \
  --face-root /path/to/EAV/Vision_Landmarks_25x56x56 \
  --subject-ids 1-42 \
  --autoencoder /path/to/pretrained_face_autoencoder_AutoencoderKL_light.pth \
  --autoencoder-type light \
  --epochs 300 \
  --batch-size 128 \
  --best-metric ssim \
  --results-dir ./eeg2face_reference_results
```

Stronger reference run:

```bash
python train_cross.py \
  --dataset EAV \
  --subject-ids 1-42 \
  --autoencoder /path/to/pretrained_face_autoencoder_AutoencoderKL_light.pth \
  --sample-embedding-dim 512 \
  --epochs 300
```

## MMER Training

```bash
python train_cross.py \
  --dataset MMER \
  --eeg-dir /path/to/MMER/EEG \
  --face-root /path/to/MMER/Aligned_data/Landmarks \
  --subject-ids 1,5,11,12,19,20,22,23,24,25,29,32,33,38 \
  --autoencoder /path/to/pretrained_face_autoencoder_AutoencoderKL_light.pth \
  --autoencoder-type light \
  --n-frames 2 \
  --epochs 300 \
  --batch-size 128 \
  --best-metric ssim
```

## Evaluation

```bash
python evaluate.py \
  --dataset EAV \
  --checkpoint ./eeg2face_reference_results/eav/reference/best_eeg2face.pt \
  --autoencoder /path/to/pretrained_face_autoencoder_AutoencoderKL_light.pth \
  --eeg-dir /path/to/EAV/EEG \
  --face-root /path/to/EAV/Vision_Landmarks_25x56x56 \
  --subject-ids 1-42
```

## Outputs

```text
best_eeg2face.pt
history.csv
results.csv
results.json
train_reconstructions.png
test_reconstructions.png
```

Logged metrics include:

```text
MSE
MAE
PSNR
SSIM
foreground MSE
crop SSIM
Dice
IoU
```
