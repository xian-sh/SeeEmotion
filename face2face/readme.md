
<img width="624" height="165" alt="image" src="https://github.com/user-attachments/assets/6bddee71-3122-449a-b1e4-baa3af4ffc48" />

# Face2Face Autoencoder

The code trains a face-to-face autoencoder on generated facial landmark emoji images. It collects all valid subjects, loads their landmark face images, and pretrains a reconstruction model.

## 1. Trained Weights

The trained Face2Face AutoencoderKL weights are available here:

https://www.kaggle.com/models/jingjinghuhu/face2face

You can directly use the trained weights without rerunning the training script.

## 2. File Structure

```text
face2face/
+-- config.py
+-- data_preparation.py
+-- autoencoders.py
+-- train_face_autoencoder.py
```

## 3. Train by Yourself

You can also run the code to train your own Face2Face AutoencoderKL.

### Set Paths

Before running, update the paths in `config.py` or pass them from the command line.

```python
eeg_dir = "/kaggle/input/datasets/jingjinghuhu/eva-feat/Input_images/eeg"

emoji_root_template = (
    "/kaggle/input/notebooks/jingjinghuhu/eav-binary-{emoji_size}/"
    "Vision_Landmarks_sampled_25x{emoji_size}x{emoji_size}"
)

face_emotion_labels = "/kaggle/input/datasets/jingjinghuhu/eav-image-labels/face_emotions.json"
```

Required files:

* EEG files: `subject_xx_eeg.pkl`
* Landmark face files: `emoji_vis_subject_xx.pkl`
* Face emotion labels: `face_emotions.json`

### Run Training

Default training:

```bash
python train_face_autoencoder.py
```

Custom training:

```bash
python train_face_autoencoder.py \
  --emoji-size 56 \
  --model-type light \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-4 \
  --results-dir ./eeg2face_multitask_results_multi
```

Available model sizes:

```text
light
medium
heavy
```

## 4. Processing Logic

The training script:

1. Finds valid subject IDs from the EEG folder.
2. Loads each subject's landmark face images.
3. Flattens `(videos, frames, H, W)` into frame-level face samples.
4. Normalizes face images to `[0, 1]`.
5. Builds a Diffusers autoencoder.
6. Trains the model with MSE reconstruction loss.
7. Saves the best checkpoint and reconstruction examples.

## 5. Outputs

Generated files are saved under:

```text
./eeg2face_multitask_results_multi/face_autoencoder/
```

Main outputs:

```text
best_face_autoencoder.pth
pretrained_face_autoencoder_AutoencoderKL_{model_type}.pth
loss_curve.png
reconstruction_epoch*.png
```

## 6. Notes

If `diffusers` is unavailable, install it before running the training script.

## 7. Evaluate a Trained Model

Evaluate an AutoencoderKL checkpoint on the test landmark-face images:

```bash
python evaluate.py \
  --model AutoencoderKL.pth \
  --model-type light \
  --emoji-size 56 \
  --latent-dim 512 \
  --batch-size 32 \
  --eeg-dir /path/to/eeg \
  --emoji-root-template "/path/to/Vision_Landmarks_sampled_25x{emoji_size}x{emoji_size}" \
  --results-dir ./face2face_results
```

The evaluator uses `subject_xx_eeg.pkl` filenames to discover subject IDs. If the EEG directory is unavailable, provide IDs directly:

```bash
python evaluate_face_kl.py \
  --model /path/to/checkpoint.pth \
  --subject-ids 1,2,5-8 \
  --emoji-root-template "/path/to/Vision_Landmarks_sampled_25x{emoji_size}x{emoji_size}"
```

By default, reconstruction uses the KL posterior mean/mode for deterministic metrics. Add `--sample-latent` to evaluate stochastic posterior samples. Use `--max-samples 1000` for a quick subset check.

The evaluation reports image-level mean MSE, PSNR, and SSIM over the complete test set. It also saves reconstruction examples, a latent-value histogram, a normal Q-Q diagnostic, and a JSON summary.

### Evaluation Results

The complete dataset contains 294,000 training images and 126,000 test images. Evaluation on the full test set (`126000 x 1 x 56 x 56`) with a batch size of 32 produced the following reconstruction results:

| Metric | Result |
| --- | ---: |
| MSE | 0.000001 |
| PSNR | 64.404443 dB |
| SSIM | 0.999942 |

The evaluation processed 3,938 batches and required approximately 2 h 24 min.

Evaluation outputs are saved under:

```text
./face2face_results/face_autoencoder/evaluation_kl/
```

Main evaluation outputs:

```text
kl_evaluation_results.json
kl_reconstruction_samples.png
kl_latent_distribution.png
kl_normality_check.png
```

Install `torch`, `diffusers`, `numpy`, `matplotlib`, `scikit-image`, and `tqdm` before evaluation. The model architecture settings (`model-type`, `emoji-size`, and `latent-dim`) must match the checkpoint.

