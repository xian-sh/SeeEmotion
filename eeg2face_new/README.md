# EEG2Face Clean

This is the recommended EEG-to-facial-emoji reconstruction pipeline for new experiments.
It uses a clean train/validation/test protocol while keeping the implementation compact.

This folder contains a clean EEG-to-facial-emoji reconstruction pipeline.
It keeps only the EEG-to-image branch and uses a frozen Face2Face AutoencoderKL
as the image decoder:

```text
EEG window -> EEG backbone -> KL latent prediction -> frozen AutoencoderKL -> emoji image
```

Compared with `eeg2face_ref`, this version is intended for cleaner experiments:

- train/validation/test are separated;
- best checkpoint is selected by validation metrics;
- test metrics are computed only after loading the best checkpoint;
- trial/video-level splitting keeps all frames from one trial in the same split;
- EEG normalization statistics are fitted from the training split only;
- sample identity embedding is disabled by default.

## Files

```text
eeg2face_new/
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

## Data

### EAV

Default paths:

```text
/home/devuser/hjj/seeemotion/data/EAV/EEG
/home/devuser/hjj/seeemotion/data/EAV/Vision_Landmarks_25x56x56
```

Expected files:

```text
EEG/
+-- subject_01_eeg.pkl
+-- subject_02_eeg.pkl

Vision_Landmarks_25x56x56/
+-- emoji_vis_subject_01.pkl
+-- emoji_vis_subject_02.pkl
```

Default setting: 30 EEG channels, 100 Hz, 5 s trials, 25 frames, 56 x 56 images.

### MMER

Default paths:

```text
/home/devuser/hjj/seeemotion/data/MMER/EEG
/home/devuser/hjj/seeemotion/data/MMER/Landmarks_64x64
```

Expected files:

```text
MMER/
+-- EEG/
|   +-- 1_eeg_20s.pkl
|   +-- 5_eeg_20s.pkl
+-- Landmarks_64x64/
    +-- 1_landmarks.pkl
    +-- 5_landmarks.pkl
```

The loader also accepts common landmark folder names such as
`Aligned_data/Landmarks` and common EAV-style filenames such as
`emoji_vis_subject_01.pkl`.

Default setting: 18 EEG channels, 300 Hz, 20 s trials, 40 frames, 64 x 64 images.
Each landmark image is aligned with a 0.5-second EEG window.

The common MMER aligned landmark file is stored as:

```text
EEG:  (32, 18, 6000)
Face: (32 * 20 * 2, 64, 64), namely (1280, 64, 64)
```

The loader groups it back to `(32, 40, 64, 64)` before creating frame-level EEG-image samples.

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Single-subject training uses trial-level splitting by default. All frames from the same trial stay in the same split:

```bash
python train_single.py \
  --dataset EAV \
  --subject-ids 1 \
  --autoencoder /home/devuser/hjj/seeemotion/models/AutoencoderKL.pth \
  --autoencoder-type light \
  --epochs 300 \
  --batch-size 128
```

Multi-subject training can also use trial-level splitting:

```bash
python train_single.py \
  --dataset EAV \
  --subject-ids 1-42 \
  --autoencoder /home/devuser/hjj/seeemotion/models/AutoencoderKL.pth \
  --autoencoder-type light \
  --epochs 300 \
  --batch-size 128
```

Cross-subject training uses subject-level splitting by default and is the stricter setting for subject generalization:

```bash
python train_cross.py \
  --dataset MMER \
  --subject-ids 1,5,11,12,19,20,22,23,24,25,29,32,33,38 \
  --autoencoder /home/devuser/hjj/seeemotion/models/AutoencoderKL.pth \
  --autoencoder-type light \
  --epochs 300 \
  --batch-size 128
```

You can set explicit subject splits:

```bash
python train_cross.py \
  --dataset MMER \
  --train-subjects 1,5,11,12,19,20,22,23,24,25 \
  --val-subjects 29,32 \
  --test-subjects 33,38 \
  --autoencoder /home/devuser/hjj/seeemotion/models/AutoencoderKL.pth
```

## Evaluation

```bash
python evaluate.py \
  --dataset EAV \
  --checkpoint ./eeg2face_clean_results/eav/single/best_eeg2face_clean.pt \
  --autoencoder /home/devuser/hjj/seeemotion/models/AutoencoderKL.pth \
  --subject-ids 1
```

## Outputs

Results are saved under:

```text
./eeg2face_clean_results/{dataset}/{single|cross}/
```

Main outputs:

```text
best_eeg2face_clean.pt
history.csv
results.csv
results.json
train_reconstructions.png
val_reconstructions.png
test_reconstructions.png
```

Logged metrics include MSE, MAE, PSNR, SSIM, foreground MSE, crop SSIM, Dice,
and IoU.
