
<img width="552" height="650" alt="image" src="https://github.com/user-attachments/assets/2a1707af-d78f-4da0-826c-aa958194cae7" />


# MMER Preprocessing Pipeline

This folder contains the preprocessing scripts for MMER EEG-video-face data.

## 1. Processed Data

The processed MMER-Feat data has been uploaded here:

https://www.kaggle.com/datasets/jingjinghuhu/mmer-feat

https://www.kaggle.com/datasets/jingjinghuhu/mmer-landmark

The dataset contains:

```text
MMER-Feat/
+-- EEG/
+-- Images_0.5/
+-- Labels_AV/
+-- Labels_Emotion/
+-- face_emotion_labels_0.5.json
+-- MMER-Landmark/
```

You can directly use these files without rerunning the preprocessing scripts.

---

## 2. Generate by Yourself

You can also rerun the full preprocessing pipeline with the scripts below.

Recommended running order:

```text
0_extract.py
1_video2img.py
2_copy.py
3_face2small.py
4_visface.py
generate_labels_to_json.py
```

---

# 2.1 Extract EEG and 20s Videos using `0_extract.py`

This script extracts the first 20 seconds of EEG data and cuts each video to 20 seconds.

Before running the script, set the subject list and data path:

```python
mat_list = [1, 5, 11, 12, 19, 20, 22, 23, 24, 25, 29, 32, 33, 38]

mat_path = f"Aligned_data/{m}/datas.mat"
video_dir = f"Aligned_data/{subject_id}"
output_dir = f"Aligned_data/{subject_id}/videos_20s"
```

Required files from original [MMER dataset](https://www.nature.com/articles/s41597-024-03676-4):

* EEG data: `Aligned_data/{subject_id}/datas.mat`
* Raw videos: `Aligned_data/{subject_id}/{video_id}.mp4`

Generated outputs:

* EEG data: `Aligned_data/{subject_id}/eeg_20s.pkl`
* 20s videos: `Aligned_data/{subject_id}/videos_20s/{video_id}_20s.mp4`
* Processing info: `Aligned_data/{subject_id}/processing_info.pkl`

---

# 2.2 Extract Face Images using `1_video2img.py`

This script extracts face images from the 20s videos using Dlib.

Before running the script, set:

```python
predictor_path = "shape_predictor_68_face_landmarks.dat"
video_dir = f"Aligned_data/{subject_id}/videos_20s"
face_output_dir = f"Aligned_data/{subject_id}/face_images"
```

Required files:

* 20s videos: [`Aligned_data/{subject_id}/videos_20s/{video_id}_20s.mp4`](https://www.kaggle.com/datasets/jingjinghuhu/mmer-feat)
* Dlib landmark model: [`shape_predictor_68_face_landmarks.dat`](https://www.kaggle.com/models/tranthaitoanb2103447/shape_predictor_68_face_landmarks.dat/TensorFlow2/default/1)

Generated outputs:

* Face images: `Aligned_data/{subject_id}/face_images/`
* Extraction info: `Aligned_data/{subject_id}/face_extraction_info.pkl`
* Final summary: `face_extraction_final_summary.pkl`

---

# 2.3 Reorganize Face Images using `2_copy.py`

This script moves each subject's face images into a unified folder.

Set the paths:

```python
base_dir = "Aligned_data"
target_dir = "./Aligned_data/Images_0.5"
```

Input:

```text
Aligned_data/{subject_id}/face_images/
```

Output:

```text
Aligned_data/Images_0.5/{subject_id}_face_imgs/
```

---

# 2.4 Extract Small Landmark Images using `3_face2small.py`

This script converts face images into fixed-length landmark arrays.

Set the paths:

```python
images_dir = "./Aligned_data/Images_0.5"
output_dir = "./Aligned_data/Landmarks_64x64"
predictor_path = "shape_predictor_68_face_landmarks.dat"
```

Main settings:

```python
face_size = 64
output_size = 32
expected_length = 640
detect_size = 224
```

Generated outputs:

* Landmark files: `Aligned_data/Landmarks_64x64/{subject_id}_landmarks.pkl`
* Summary file: `Aligned_data/Landmarks_64x64/landmarks_extraction_summary.pkl`
* Sample images: `Aligned_data/Landmarks_64x64/samples/`

---

# 2.5 Visualize and Save Face Images using `4_visface.py`

This script reads landmark `.pkl` files and saves visual face images with MMER emotion labels.

Set the paths:

```python
landmarks_directory = "./Aligned_data/Landmarks_64x64"
output_directory = "./Face_Images_All_64x64"
labels_file = "./Aligned_data/Labels_Emotion/1_Emotions.csv"
```

Required files:

* Landmark files: [`Aligned_data/Landmarks_64x64/{subject_id}_landmarks.pkl`](https://www.kaggle.com/code/jingjinghuhu/mmer-landmark-64)
* MMER labels: `Aligned_data/Labels_Emotion/1_Emotions.csv`

Generated outputs:

* Face images: `Face_Images_All_64x64/{subject_id}_face_images/`
* Statistics: `{subject_id}_face_images_stats.pkl`

Label rule:

```text
amusement > disgust  -> positive
amusement < disgust  -> negative
amusement = disgust  -> mixed
```

---

# 2.6 Generate Image Labels using `generate_labels_to_json.py`

This script uses a facial emotion recognition model to generate image-level labels.

Set the paths:

```python
face_img_base_dir = "./Aligned_data/Images_0.5"
out_file = "face_emotion_labels_0.5.json"

model_name = "./cache_dir/face-model"
```

Required files:

* Face image folders: [`Aligned_data/Images_0.5/{subject_id}_face_imgs/`](https://www.kaggle.com/datasets/jingjinghuhu/mmer-feat)
* Facial emotion model: `./cache_dir/face-model`

Model source:

```text
https://huggingface.co/dima806/facial_emotions_image_detection
```

Generated output:

```text
face_emotion_labels_0.5.json
```

The output JSON contains:

```python
{
    "emotion_id": int,
    "emotion": str,
    "confidence": float
}
