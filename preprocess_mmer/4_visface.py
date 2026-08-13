import cv2
import dlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path
import glob
import logging
from datetime import datetime
import pandas as pd

def load_mmer_labels(label_path):
    """Load the label file of the MMER dataset."""
    try:
        label_df = pd.read_csv(label_path)
        emotion_labels = {}
        
        for idx, row in label_df.iterrows():
            video_id = int(row.iloc[0])  # The first column is video ID
            amusement = float(row.iloc[1])  # The second column is amusement
            disgust = float(row.iloc[2])  # The third column is disgust
            
            # Generate emotion labels according to the rule
            if amusement > disgust:
                emotion_label = 0  # Positive
            elif amusement < disgust:
                emotion_label = 2  # Negative
            else:
                emotion_label = 1  # Mixed
            
            emotion_labels[video_id] = {
                'emotion_code': emotion_label,
                'emotion_name': ['positive', 'mixed', 'negative'][emotion_label],
                'amusement': amusement,
                'disgust': disgust
            }
        
        return emotion_labels
    except Exception as e:
        print(f"Failed to load MMER label file: {e}")
        return {}

def extract_video_id_from_filename(filename):
    """Extract video ID from filename with support for multiple formats."""
    import re
    
    # Try extracting video_number format
    match = re.search(r'video[_\s]*(\d+)', filename.lower())
    if match:
        return int(match.group(1))
    
    # Try extracting leading numbers
    match = re.search(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    
    # Try extracting any number and use the first one
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[0])
    
    return None

def get_emotion_for_filename(filename, emotion_labels):
    """Get the emotion label by filename."""
    video_id = extract_video_id_from_filename(filename)
    if video_id is not None and video_id in emotion_labels:
        return emotion_labels[video_id]
    return {'emotion_code': -1, 'emotion_name': 'unknown', 'amusement': 0, 'disgust': 0}

def save_face_images_from_pkl(pkl_file_path, output_face_dir, label_path=None, 
                             save_white_background=True, save_black_background=False):
    """
    Read landmark data from a PKL file and save face images with emotion labels.
    
    Args:
    - pkl_file_path: PKL file path.
    - output_face_dir: Output face image directory.
    - label_path: MMER label file path.
    - save_white_background: Whether to save images with white background and black lines.
    - save_black_background: Whether to save images with black background and white lines.
    """
    
    # Create output directory
    os.makedirs(output_face_dir, exist_ok=True)
    
    # Load PKL data
    print(f"Loading PKL file: {pkl_file_path}")
    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Failed to load PKL file: {e}")
        return
    
    # Support both new and old formats
    if 'landmarks_main_array' in data:
        landmarks_main_array = data['landmarks_main_array']
        landmarks_small_array = data['landmarks_small_array']
        filenames_list = data['filenames_list']
        subject_id = data.get('subject_id', 'unknown')
        face_size = data.get('face_size', landmarks_main_array.shape[1])
    elif 'landmarks_stick_array' in data:  # Support old format
        landmarks_main_array = data['landmarks_stick_array']
        landmarks_small_array = data['landmarks_emoji_array']
        filenames_list = data['filenames_list']
        subject_id = data.get('subject_id', 'unknown')
        face_size = landmarks_main_array.shape[1]
    else:
        print("Unsupported PKL file format")
        return
    
    # Load emotion labels
    emotion_labels = {}
    if label_path and os.path.exists(label_path):
        print(f"Loading emotion labels: {label_path}")
        emotion_labels = load_mmer_labels(label_path)
        print(f"Loaded {len(emotion_labels)} video emotion labels")
    else:
        print("No valid label file provided. Default emotion labels will be used.")
    
    # Statistics
    emotion_counts = {'positive': 0, 'mixed': 0, 'negative': 0, 'unknown': 0}
    saved_count = 0
    
    print(f"Start saving face images, subject: {subject_id}")
    print(f"   Array shape: {landmarks_main_array.shape}")
    print(f"   Face size: {face_size}x{face_size}")
    
    # Process and save images one by one
    for i in range(landmarks_main_array.shape[0]):
        main_img = landmarks_main_array[i]
        filename = filenames_list[i] if i < len(filenames_list) else f"frame_{i:04d}"
        
        # Skip blank images
        if not np.any(main_img):
            continue
        
        # Get emotion label
        emotion_info = get_emotion_for_filename(filename, emotion_labels)
        emotion_name = emotion_info['emotion_name']
        emotion_code = emotion_info['emotion_code']
        
        # Update statistics
        emotion_counts[emotion_name] += 1
        
        # Generate new filename
        original_name = os.path.splitext(filename)[0]
        
        if save_white_background:
            # White background with black lines
            white_bg_img = 255 - main_img
            white_filename = f"{subject_id}_{i:04d}_{original_name}_emotion{emotion_code}_{emotion_name}_white.png"
            white_path = os.path.join(output_face_dir, white_filename)
            
            white_img_pil = Image.fromarray(white_bg_img.astype(np.uint8))
            white_img_pil.save(white_path)
            
        if save_black_background:
            # Black background with white lines
            black_bg_img = main_img
            black_filename = f"{subject_id}_{i:04d}_{original_name}_emotion{emotion_code}_{emotion_name}_black.png"
            black_path = os.path.join(output_face_dir, black_filename)
            
            black_img_pil = Image.fromarray(black_bg_img.astype(np.uint8))
            black_img_pil.save(black_path)
        
        saved_count += 1
        
        # Print progress every 100 images
        if saved_count % 100 == 0:
            print(f"   Saved {saved_count} images...")
    
    # Print statistics
    print(f"\nSaving completed.")
    print(f"   Subject: {subject_id}")
    print(f"   Total saved: {saved_count} images")
    print(f"   Output directory: {output_face_dir}")
    print(f"\n Emotion distribution:")
    print(f"   Positive (0): {emotion_counts['positive']}")
    print(f"   Mixed (1):    {emotion_counts['mixed']}") 
    print(f"   Negative (2): {emotion_counts['negative']}")
    print(f"   Unknown:      {emotion_counts['unknown']}")
    
    # Save statistics
    stats = {
        'subject_id': subject_id,
        'total_saved': saved_count,
        'emotion_counts': emotion_counts,
        'face_size': face_size,
        'save_white_background': save_white_background,
        'save_black_background': save_black_background,
        'label_path': label_path,
        'processing_time': datetime.now()
    }
    
    stats_file = os.path.join(output_face_dir, f"{subject_id}_face_images_stats.pkl")
    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f)
    print(f"Statistics saved to: {stats_file}")

def batch_process_all_subjects(landmarks_dir, output_base_dir, label_path=None,
                              save_white_background=True, save_black_background=False):
    """
    Batch process PKL files of all subjects and save face images.
    
    Args:
    - landmarks_dir: Directory containing PKL files.
    - output_base_dir: Base output directory.
    - label_path: MMER label file path.
    - save_white_background: Whether to save white-background images with black lines.
    - save_black_background: Whether to save black-background images with white lines.
    """
    
    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Find all PKL files
    pkl_files = glob.glob(os.path.join(landmarks_dir, "*_landmarks.pkl"))
    pkl_files.sort()
    
    if not pkl_files:
        print(f"No PKL files found in {landmarks_dir} ")
        return
    
    print(f"Found {len(pkl_files)} PKL files")
    print(f"Base output directory: {output_base_dir}")
    
    # Process each PKL file
    for pkl_file in pkl_files:
        print(f"\n{'='*60}")
        subject_id = os.path.basename(pkl_file).replace('_landmarks.pkl', '')
        output_face_dir = os.path.join(output_base_dir, f"{subject_id}_face_images")
        
        save_face_images_from_pkl(
            pkl_file_path=pkl_file,
            output_face_dir=output_face_dir,
            label_path=label_path,
            save_white_background=save_white_background,
            save_black_background=save_black_background
        )
    
    print(f"\nAll subjects processed.")
    print(f"Output directory: {output_base_dir}")

def visualize_face_samples(face_images_dir, num_samples=12, emotion_filter=None):
    """
    Visualize saved face image samples.
    
    Args:
    - face_images_dir: Face image directory.
    - num_samples: Number of samples to display.
    - emotion_filter: Filter a specific emotion ('positive', 'mixed', 'negative'); None means all.
    """
    
    # Get all image files
    image_files = glob.glob(os.path.join(face_images_dir, "*.png"))
    
    if emotion_filter:
        # Filter a specific emotion
        filtered_files = [f for f in image_files if f"_{emotion_filter}_" in f]
        image_files = filtered_files
    
    if not image_files:
        print(f"No image files found")
        return
    
    # Randomly select samples
    import random
    random.shuffle(image_files)
    sample_files = image_files[:num_samples]
    
    # Compute grid size
    cols = 4
    rows = (len(sample_files) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, img_path in enumerate(sample_files):
        row, col = i // cols, i % cols
        
        # Load image
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Display image
        axes[row, col].imshow(img_array, cmap='gray')
        
        # Extract information from filename
        filename = os.path.basename(img_path)
        parts = filename.split('_')
        
        # Try extracting emotion information
        emotion_info = "unknown"
        for part in parts:
            if part.startswith('emotion') and len(part) > 7:
                emotion_code = part[7]
                emotion_name = part[8:].split('_')[0] if '_' in part[8:] else part[8:]
                emotion_info = f"{emotion_name}({emotion_code})"
                break
        
        axes[row, col].set_title(f"{emotion_info}", fontsize=8)
        axes[row, col].axis('off')
    
    # Hide extra subplots
    for i in range(len(sample_files), rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    title = f"Face Image Samples"
    if emotion_filter:
        title += f" - {emotion_filter.capitalize()}"
    title += f" (total {len(image_files)} images, showing {len(sample_files)} images)"
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

# Usage example
if __name__ == "__main__":
    
    # Example 1: process a single subject
    # single_pkl_path = "./Aligned_data/Landmarks_128x128/1_landmarks.pkl"
    # single_output_dir = "./Face_Images/subject_1"
    # label_file_path = "./Aligned_data/Labels_Emotion/1_Emotions.csv"  # Replace with your label file path
    
    # if os.path.exists(single_pkl_path):
    #     print("Single-subject processing example:")
    #     save_face_images_from_pkl(
    #         pkl_file_path=single_pkl_path,
    #         output_face_dir=single_output_dir,
    #         label_path=label_file_path if os.path.exists(label_file_path) else None,
    #         save_white_background=True,   # Save white-background images with black lines
    #         save_black_background=False   # Do not save black-background images with white lines
    #     )
        
    #     # Visualize results
    #     if os.path.exists(single_output_dir):
    #         print("\nVisualize saved face images:")
    #         visualize_face_samples(single_output_dir, num_samples=12)
            
    #         # Visualize different emotions
    #         for emotion in ['positive', 'mixed', 'negative']:
    #             visualize_face_samples(single_output_dir, num_samples=8, emotion_filter=emotion)
    
    # Example 2: batch process all subjects
    print("\n" + "="*60)
    print("Batch process all subjects:")
    
    landmarks_directory = "./Aligned_data/Landmarks_64x64"  # PKL file directory
    output_directory = "./Face_Images_All_64x64"                  # Output directory
    labels_file = "./Aligned_data/Labels_Emotion/1_Emotions.csv"       # Label file path
    
    if os.path.exists(landmarks_directory):
        batch_process_all_subjects(
            landmarks_dir=landmarks_directory,
            output_base_dir=output_directory,
            label_path=labels_file if os.path.exists(labels_file) else None,
            save_white_background=True,   # Save white-background images with black lines
            save_black_background=False   # Do not save black-background images with white lines
        )
    else:
        print(f"PKL directory does not exist: {landmarks_directory}")
    
    print("\nProcessing completed.")
