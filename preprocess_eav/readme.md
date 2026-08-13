# Vision Landmarks Sampling using `face2emoji_eav.py`

This script converts EVA-Feat face videos into 25-frame facial landmark images.

## 1. Processed Data

The processed data has been uploaded here:

https://www.kaggle.com/code/jingjinghuhu/eav-binary-56/output

Please use the folder:

```text
Vision_Landmarks_sampled_25x56x56
````

Each subject file contains:

```python
[
    emojis_train,
    labels_train,
    emojis_test,
    labels_test
]
```

Data shape:

```text
emojis_train: (280, 25, 56, 56)
emojis_test:  (120, 25, 56, 56)
```

## 2. Generate by Yourself

You can also run `face2emoji_eav.py` to generate the data.

Before running the script, modify these paths:

```python
vis = pickle.load(open(fr'/eva-feat/Input_images/Vision/subject_{sub:02d}_vis.pkl', 'rb'))
predictor_path = 'shape_predictor_68_face_landmarks.dat'
```

Required files:

* EVA-Feat vision data: `/eva-feat/Input_images/Vision/subject_xx_vis.pkl`
* Dlib landmark model: `shape_predictor_68_face_landmarks.dat`

Download links:

* EVA-Feat: [https://www.kaggle.com/datasets/jingjinghuhu/eva-feat](https://www.kaggle.com/datasets/jingjinghuhu/eva-feat)
* Landmark model: [https://www.kaggle.com/models/tranthaitoanb2103447/shape_predictor_68_face_landmarks.dat/TensorFlow2/default/1](https://www.kaggle.com/models/tranthaitoanb2103447/shape_predictor_68_face_landmarks.dat/TensorFlow2/default/1)

## 3. Processing Logic

For each subject, the script:

1. Loads train/test video data.
2. Uniformly samples 25 frames from each video.
3. Extracts 68 facial landmarks using Dlib.
4. Draws landmark-based face images.
5. If a sampled frame fails, it is replaced by the nearest successfully extracted frame.
6. Saves the processed landmark data as `.pkl`.

## 4. Output

If you run the script yourself, the generated data is saved to:

```text
./Vision_Landmarks_sampled_25x56x56/
```

---

# Getting Image Labels using `generate_labels_to_json.py`

The generated image labels are available here:

https://www.kaggle.com/datasets/jingjinghuhu/eav-image-labels


### Optional: Generate Labels by Yourself

You can also run the provided label generation code. Set the paths first:

```python
VISION_BASE = Path("./EAV/data/Inputs/Vision")   #download from https://www.kaggle.com/datasets/jingjinghuhu/eva-feat

LABELS_DIR = Path(
    "./labels_face"   # output
)

OUTPUT_JSON = Path(
    "./face_emotions.json"  # output
)

MODEL_NAME = Path(
    "./cache_dir/face-model"   # download from https://huggingface.co/dima806/facial_emotions_image_detection
)
````

The provided code is only a reference. You can either directly use the generated Kaggle labels or rerun the code to generate your own labels.

---

# Training Emoji Emotion Classifier using `emoji_resnet18_classifier.py`

This script trains a ResNet18 classifier on the generated facial landmark emoji images.

The trained ResNet18 weights are available here:

https://www.kaggle.com/models/jingjinghuhu/resnet18-binary

### Optional: Train by Yourself

You can also run `emoji_resnet18_classifier.py` to train the model yourself. Set the data paths first:

```python
DATA_DIRS = {
    56: "/kaggle/input/notebooks/jingjinghuhu/eav-binary-56/Vision_Landmarks_sampled_25x56x56"
}

EMOTION_LABELS_PATH = "/kaggle/input/datasets/jingjinghuhu/eav-image-labels/face_emotions.json"

RESULTS_DIR = "./emotion_classifier_results"
````

Required files:

* Processed landmark data: [`/kaggle/input/notebooks/jingjinghuhu/eav-binary-56/Vision_Landmarks_sampled_25x56x56`](https://www.kaggle.com/code/jingjinghuhu/eav-binary-56/output)
* Image labels: [`/kaggle/input/datasets/jingjinghuhu/eav-image-labels/face_emotions.json`](https://www.kaggle.com/datasets/jingjinghuhu/eav-image-labels)

Generated outputs:

* Best model weights: `./emotion_classifier_results/emoji_size_56_ResNet18_*/best_model.pth`
* Training curve: `training_history.png`
* Confusion matrix: `test_confusion_matrix.png`
* Classification report: `test_classification_report.txt`

The provided code only keeps ResNet18. You can either directly use the trained Kaggle weights or rerun the code to train your own model.

