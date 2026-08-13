# Vision Landmarks Sampling using `face2emoji_eav.py`

This script converts EVA-Feat face videos into 25-frame facial landmark images.

## 1. Set Paths

Before running the script, modify these paths:

```python
vis = pickle.load(open(fr'/eva-feat/Input_images/Vision/subject_{sub:02d}_vis.pkl', 'rb'))
predictor_path = 'shape_predictor_68_face_landmarks.dat'
````

Required files:

* EVA-Feat vision data: `/eva-feat/Input_images/Vision/subject_xx_vis.pkl`
* Dlib landmark model: `shape_predictor_68_face_landmarks.dat`

Download links:

* EVA-Feat: [https://www.kaggle.com/datasets/jingjinghuhu/eva-feat](https://www.kaggle.com/datasets/jingjinghuhu/eva-feat)
* Landmark model: [https://www.kaggle.com/models/tranthaitoanb2103447/shape_predictor_68_face_landmarks.dat/TensorFlow2/default/1](https://www.kaggle.com/models/tranthaitoanb2103447/shape_predictor_68_face_landmarks.dat/TensorFlow2/default/1)

## 2. Processing Logic

For each subject, the script:

1. Loads train/test video data.
2. Uniformly samples 25 frames from each video.
3. Extracts 68 facial landmarks using Dlib.
4. Draws landmark-based face images.
5. If a sampled frame fails, it is replaced by the nearest successfully extracted frame.
6. Saves the processed landmark data as `.pkl`.

## 3. Output

The generated data is saved to:

```text
./Vision_Landmarks_sampled_25x56x56/
```

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

## 4. Processed Data

The processed data has been uploaded here:

[https://www.kaggle.com/code/jingjinghuhu/eav-binary-56/output](https://www.kaggle.com/code/jingjinghuhu/eav-binary-56/output)

Please use the folder:

```text
Vision_Landmarks_sampled_25x56x56
```
