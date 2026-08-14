# See the Emotion

Official PyTorch implementation for:

**See the Emotion: A Facial Emoji Proxy Modeling for EEG Emotion Recognition**  
Accepted at ICML 2026

Jingjing Hu, Dan Guo, Haofan Cheng, Ying Zeng, Zhan Si, Jinxing Zhou, Meng Wang

> Links: [Paper (arXiv)]() | [Project Page]() | [EAV dataset](https://www.kaggle.com/datasets/jingjinghuhu/eva-feat) | [Face2Face weights](https://www.kaggle.com/models/jingjinghuhu/face2face)

<img width="1866" height="1234" alt="image" src="https://github.com/user-attachments/assets/3ef48dcf-568f-424c-8a3d-0afb669410f8" />



## Framework

Existing EEG-based emotion recognition models remain opaque "black boxes," lacking semantic grounding between abstract neural features and human-interpretable states. This paper reframes EEG explainability as a **cross-modal generation task**, shifting the paradigm from feature attribution to behavioral visualization.

We introduce **Facial Emoji Proxy Modeling (FELB)** , a novel framework that translates high-dimensional EEG signals into identity-anonymized facial emojis. Our approach:

- **FMENet**: A specialized backbone capturing expression-relevant spatial synergies and multi-scale temporal dynamics
- **FELB**: A facial emoji learning branch that treats emoji reconstruction as a structured semantic regularizer
- **Privacy-preserving**: Generates identity-anonymized emoji visualizations while achieving state-of-the-art accuracy

<p align="center">
  <img src="figs/fig_main_icml_260129_01.png" width="80%">
  <br>
  <em>Figure: EEG-to-Emoji translation framework overview.</em>
</p>

## Overview

This repository studies EEG emotion recognition with a facial emoji proxy. The code is organized into small modules: EEG emotion classification, facial landmark emoji preprocessing, MMER preprocessing, and Face2Face AutoencoderKL training/evaluation.

The homepage is intentionally kept short. For implementation details, path settings, and full command examples, open the README inside each subfolder.

## Repository Guide

| Folder | Purpose | Details |
| --- | --- | --- |
| [`eeg2emo/`](eeg2emo/) | EEG-only emotion recognition on EAV, MMER, and SEED | [`eeg2emo/README.md`](eeg2emo/README.md) |
| [`face2face/`](face2face/) | Face2Face AutoencoderKL training and reconstruction evaluation | [`face2face/readme.md`](face2face/readme.md) |
| [`preprocess_eav/`](preprocess_eav/) | EAV facial landmark emoji generation and image-label preparation | [`preprocess_eav/readme.md`](preprocess_eav/readme.md) |
| [`preprocess_mmer/`](preprocess_mmer/) | MMER EEG/video/face preprocessing pipeline | [`preprocess_mmer/readme.md`](preprocess_mmer/readme.md) |

## Datasets

This work uses three datasets:

| Dataset | EEG Channels | Emotions | Subjects | Face Data |
|---------|-------------|----------|----------|-----------|
| **[EAV](https://www.nature.com/articles/s41597-024-03838-4)** | 30 | 5 (Neutral, Anger, Happiness, Sadness, Calmness) | 42 | ✅ |
| **[MMER](https://www.nature.com/articles/s41597-024-03676-4)** | 18 | 3 (Positive, Negative, Mixed) | 38 | ✅ |
| **[SEED](https://bcmi.sjtu.edu.cn/~seed/)** | 62 | 3 (Positive, Neutral, Negative) | 15 | ❌ (zero-shot) |

## Quick Start

Install the EEG training dependencies:

```bash
cd eeg2emo
pip install -r requirements.txt
```

Run EEG emotion recognition:

```bash
python run_single.py --dataset MMER --run 1 --epochs 50
python run_cross.py --dataset MMER --run 1 --epochs 30
```

For Face2Face AutoencoderKL:

```bash
cd face2face
python train_face_autoencoder.py
python evaluation.py --model /path/to/checkpoint.pth
```

See [`face2face/readme.md`](face2face/readme.md) for required paths, pretrained weights, and evaluation options.

## Data And Weights

| Resource | Link |
| --- | --- |
| EAV EEG/vision features | [Kaggle: eva-feat](https://www.kaggle.com/datasets/jingjinghuhu/eva-feat) |
| EAV landmark emojis | [Kaggle output: eav-binary-56](https://www.kaggle.com/code/jingjinghuhu/eav-binary-56/output) |
| EAV image labels | [Kaggle: eav-image-labels](https://www.kaggle.com/datasets/jingjinghuhu/eav-image-labels) |
| MMER processed features | [Kaggle: mmer-feat](https://www.kaggle.com/datasets/jingjinghuhu/mmer-feat) |
| MMER landmarks | [Kaggle: mmer-landmark](https://www.kaggle.com/datasets/jingjinghuhu/mmer-landmark) |
| Face2Face AutoencoderKL | [Kaggle model: face2face](https://www.kaggle.com/models/jingjinghuhu/face2face) |

## Face2Face Evaluation Snapshot

AutoencoderKL reconstruction quality on the 56x56 test landmark-face set:

```text
Train samples: 294,000
Test samples: 126,000
Test shape: (126000, 1, 56, 56)
MSE: 0.000001
PSNR: 64.404443
SSIM: 0.999942
```

Full evaluation usage and outputs are documented in [`face2face/readme.md`](face2face/readme.md).

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{hu2026see,
  title={See the Emotion: A Facial Emoji Proxy Modeling for EEG Emotion Recognition},
  author={Hu, Jingjing and Guo, Dan and Cheng, Haofan and Zeng, Ying and Si, Zhan and Zhou, Jinxing and Wang, Meng},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

## Contact

For questions or issues, please open a GitHub issue or contact Jingjing Hu at [xianhjj623@gmail.com](mailto:xianhjj623@gmail.com).

## Ethical Considerations

This work is intended for research in affective computing and brain-computer interfaces. The facial emoji proxy is designed to be identity-anonymized and should not be used for non-consensual surveillance, emotional profiling, or high-stakes decision-making without proper oversight.
