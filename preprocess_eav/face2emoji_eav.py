import cv2
import dlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle

import logging
import os

def setup_logger(logfile="face2emoji.log"):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        handlers=[
            logging.FileHandler(logfile, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    return logger

logger = setup_logger()

# Modified function to support custom output resolution
def draw_stick_emoji(landmarks, face_res=64, out_res=8, line_width=2):
    """
    Draw abstract emoji from facial landmarks
    Args:
        landmarks: Facial landmarks (68, 2)
        face_res: Face image resolution (64, 128, etc.)
        out_res: Final output small image resolution
        line_width: Line width
    """
    image = np.zeros((face_res, face_res), dtype=np.uint8)
    
    # Adjust line width based on resolution
    if face_res >= 128:
        line_width = max(3, line_width)
    elif face_res >= 64:
        line_width = max(2, line_width)
    
    # Face contour
    contour = landmarks[0:17]
    cv2.polylines(image, [contour], False, 255, line_width)
    
    # Eyebrows
    left_brow = landmarks[17:22]
    right_brow = landmarks[22:27]
    cv2.polylines(image, [left_brow], False, 255, line_width)
    cv2.polylines(image, [right_brow], False, 255, line_width)
    
    # Nose (both upper and lower parts)
    nose_bridge = landmarks[27:31]    # Bridge from between eyes to nose tip
    nose_bottom = landmarks[31:36]    # Nostrils
    cv2.polylines(image, [nose_bridge], False, 255, line_width)
    cv2.polylines(image, [nose_bottom], True, 255, line_width)
    
    # Eyes
    for eye_indices in [(36, 42), (42, 48)]:
        eye_pts = landmarks[eye_indices[0]:eye_indices[1]]
        center = np.mean(eye_pts, axis=0).astype(int)
        axes = (
            max(int(np.ptp(eye_pts[:,0])//2), 1), 
            max(int(np.ptp(eye_pts[:,1])//2), 1)
        )
        cv2.ellipse(image, tuple(center), axes, 0, 0, 360, 255, line_width)
    
    # Mouth
    mouth_outer = landmarks[48:60]
    cv2.polylines(image, [mouth_outer], True, 255, line_width)
    
    # Resize to target resolution
    small_img = cv2.resize(image, (out_res, out_res), interpolation=cv2.INTER_NEAREST)
    return image, small_img

# Modified function to support custom face resolution
def extract_landmarks_from_np(img_np, detector, predictor, face_res=64):
    """
    Extract facial landmarks from an image
    Args:
        img_np: Input image
        detector: dlib face detector
        predictor: dlib landmark predictor
        face_res: Target face resolution
    """
    # Convert to grayscale
    if img_np.ndim == 3:
        if img_np.shape[2] == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np[:,:,0]
    else:
        gray = img_np
    
    # Upscale to 224x224 for face detection
    big_gray = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_CUBIC)
    faces = detector(big_gray)
    
    if len(faces)==0:
        return None, big_gray, None
    
    shape = predictor(big_gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    
    # Scale landmarks to target resolution
    landmarks_target = (landmarks * (face_res / 224)).astype(np.int32)
    
    return landmarks_target, big_gray, landmarks

def extract_landmarks_with_sampling(frames, detector, predictor, face_res=64, 
                                   num_samples=5, output_size=8, line_width=2):
    """
    Extract landmarks from uniformly sampled frames, replace failures with nearest neighbor
    
    Args:
        frames: Input frame array (T, H, W, C)
        detector: dlib face detector
        predictor: dlib landmark predictor  
        face_res: Face processing resolution
        num_samples: Number of frames to sample
        output_size: Final output image size
        line_width: Line width
    
    Returns:
        sampled_emojis: (num_samples, face_res, face_res) numpy array
        sampled_indices: Actual sampled frame indices
        extraction_info: Extraction process information
    """
    total_frames = len(frames)
    
    # Uniform sampling indices
    if total_frames <= num_samples:
        # If total frames are insufficient, take all frames
        sampled_indices = list(range(total_frames))
        # Pad to num_samples
        while len(sampled_indices) < num_samples:
            sampled_indices.append(sampled_indices[-1])
    else:
        # Uniform sampling
        step = total_frames / num_samples
        sampled_indices = [int(i * step) for i in range(num_samples)]
    
    logger.info(f"    Sampling strategy: total_frames={total_frames}, num_samples={num_samples}, indices={sampled_indices}")
    
    # Store results
    sampled_emojis = np.zeros((num_samples, face_res, face_res), dtype=np.uint8)
    successful_landmarks = {}  # Store successfully extracted landmarks
    failed_indices = []
    
    extraction_info = {
        'total_frames': total_frames,
        'sampled_indices': sampled_indices,
        'successful_count': 0,
        'failed_count': 0,
        'failed_indices': [],
        'replacement_map': {}  # Record which indices were replaced
    }
    
    # Phase 1: Try to extract landmarks from all sampled frames
    for i, frame_idx in enumerate(sampled_indices):
        try:
            frame = frames[frame_idx]
            landmarks, _, _ = extract_landmarks_from_np(frame, detector, predictor, face_res=face_res)
            
            if landmarks is not None:
                # Generate landmark image
                main_img, small_img = draw_stick_emoji(landmarks, face_res=face_res, 
                                                     out_res=output_size, line_width=line_width)
                sampled_emojis[i] = main_img
                successful_landmarks[i] = landmarks
                extraction_info['successful_count'] += 1
                logger.debug(f"      Index {i} (frame {frame_idx}): Landmark extraction successful")
            else:
                failed_indices.append(i)
                extraction_info['failed_count'] += 1
                extraction_info['failed_indices'].append(i)
                logger.warning(f"      Index {i} (frame {frame_idx}): No face detected")
                
        except Exception as e:
            failed_indices.append(i)
            extraction_info['failed_count'] += 1
            extraction_info['failed_indices'].append(i)
            logger.error(f"      Index {i} (frame {frame_idx}): Extraction failed - {e}")
    
    # Phase 2: Find nearest neighbor replacement for failed indices
    if failed_indices and successful_landmarks:
        logger.info(f"    Starting nearest neighbor replacement: {len(failed_indices)} failed indices")
        
        for failed_idx in failed_indices:
            # Find the nearest successful index
            min_distance = float('inf')
            nearest_idx = None
            
            for success_idx in successful_landmarks.keys():
                distance = abs(success_idx - failed_idx)
                if distance < min_distance:
                    min_distance = distance
                    nearest_idx = success_idx
            
            if nearest_idx is not None:
                # Copy landmarks
                source_landmarks = successful_landmarks[nearest_idx]
                main_img, small_img = draw_stick_emoji(source_landmarks, face_res=face_res,
                                                     out_res=output_size, line_width=line_width)
                sampled_emojis[failed_idx] = main_img
                extraction_info['replacement_map'][failed_idx] = nearest_idx
                logger.info(f"      Index {failed_idx} <- Copied from index {nearest_idx} (distance: {min_distance})")
            else:
                logger.warning(f"      Index {failed_idx}: No available replacement source")
    
    elif failed_indices and not successful_landmarks:
        logger.error(f"    No successful landmarks available for replacement, indices {failed_indices} will remain blank")
    
    logger.info(f"    Result: {extraction_info['successful_count']} successful, "
               f"{len(extraction_info['replacement_map'])} replaced, "
               f"{extraction_info['failed_count'] - len(extraction_info['replacement_map'])} failed")
    
    return sampled_emojis, sampled_indices, extraction_info

# Modified batch processing function
def batch_faces2emojis_sampled(video_data, face_res=64, output_size=8, 
                              predictor_path='shape_predictor_68_face_landmarks.dat',
                              num_samples=5, line_width=2):
    """
    Batch process video data, uniformly sample frames from each example
    
    Args:
        video_data: (N, T, H, W, C) video data
        face_res: Face processing resolution
        output_size: Final output image size  
        predictor_path: dlib predictor path
        num_samples: Number of frames to sample per example
        line_width: Line width
    
    Returns:
        all_emojis: (N, num_samples, face_res, face_res) numpy array
        all_extraction_info: Extraction info for each example
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    N = len(video_data)
    all_emojis = np.zeros((N, num_samples, face_res, face_res), dtype=np.uint8)
    all_extraction_info = []
    
    logger.info(f"Starting batch processing: {N} examples, {num_samples} frames each, resolution {face_res}x{face_res}")
    
    for i in range(N):
        logger.info(f"  Processing example {i+1}/{N}")
        frames = video_data[i]  # (T, H, W, C)
        
        try:
            sampled_emojis, sampled_indices, extraction_info = extract_landmarks_with_sampling(
                frames, detector, predictor, face_res=face_res, 
                num_samples=num_samples, output_size=output_size, line_width=line_width
            )
            
            all_emojis[i] = sampled_emojis
            all_extraction_info.append(extraction_info)
            
        except Exception as e:
            logger.error(f"  Example {i} processing failed: {e}")
            # Create empty extraction_info
            extraction_info = {
                'total_frames': len(frames) if frames is not None else 0,
                'sampled_indices': list(range(num_samples)),
                'successful_count': 0,
                'failed_count': num_samples,
                'failed_indices': list(range(num_samples)),
                'replacement_map': {},
                'error': str(e)
            }
            all_extraction_info.append(extraction_info)
    
    return all_emojis, all_extraction_info

def visualize_sampling_results(video_data, emojis_data, extraction_info, 
                              sample_idx=0, face_res=64):
    """
    Visualize sampling and landmark extraction results
    
    Args:
        video_data: Original video data (N, T, H, W, C)
        emojis_data: Landmark image data (N, num_samples, face_res, face_res)  
        extraction_info: List of extraction info
        sample_idx: Index of example to visualize
        face_res: Face resolution
    """
    if sample_idx >= len(video_data) or sample_idx >= len(emojis_data):
        print(f"Example index {sample_idx} out of range")
        return
    
    frames = video_data[sample_idx]  # (T, H, W, C)
    emojis = emojis_data[sample_idx]  # (num_samples, face_res, face_res)
    info = extraction_info[sample_idx]
    
    sampled_indices = info['sampled_indices']
    replacement_map = info.get('replacement_map', {})
    
    num_samples = len(sampled_indices)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle(f'Example {sample_idx} - Sampling Results (Total Frames: {info["total_frames"]})', fontsize=14)
    
    for i in range(num_samples):
        frame_idx = sampled_indices[i]
        emoji = emojis[i]
        
        # Show original frame
        if frame_idx < len(frames):
            frame = frames[frame_idx]
            if frame.ndim == 3 and frame.shape[2] == 3:
                axes[0, i].imshow(frame.astype(np.uint8))
            else:
                axes[0, i].imshow(frame, cmap='gray')
        else:
            axes[0, i].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[0, i].transAxes)
        
        # Add title information
        title = f'Frame {frame_idx}'
        if i in replacement_map:
            title += f'\n(Copied from {replacement_map[i]})'
            axes[0, i].set_title(title, color='red', fontsize=10)
        elif i in info.get('failed_indices', []):
            title += f'\n(Failed)'
            axes[0, i].set_title(title, color='orange', fontsize=10)
        else:
            title += f'\n(Success)'
            axes[0, i].set_title(title, color='green', fontsize=10)
        
        axes[0, i].axis('off')
        
        # Show landmark image
        axes[1, i].imshow(255 - emoji, cmap='gray')  # Invert colors
        axes[1, i].set_title(f'Landmarks\n{face_res}x{face_res}', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nExample {sample_idx} Statistics:")
    print(f"  Total Frames: {info['total_frames']}")
    print(f"  Sampled Frames: {num_samples}")
    print(f"  Sampled Indices: {sampled_indices}")
    print(f"  Successful Extractions: {info['successful_count']}")
    print(f"  Failed Count: {info['failed_count']}")
    if replacement_map:
        print(f"  Replacement Map: {replacement_map}")

def save_emoji_samples_new(emojis, subject_id, save_dir="./samples", max_samples=6, face_res=64):
    """Save emoji sample images"""
    out_dir = os.path.join(save_dir, f"subject_{subject_id:02d}_{face_res}x{face_res}")
    os.makedirs(out_dir, exist_ok=True)
    
    # emojis shape: (num_samples, face_res, face_res)
    n = min(max_samples, emojis.shape[0])
    for i in range(n):
        img = 255 - emojis[i]  # Invert colors (black lines on white)
        Image.fromarray(img).save(os.path.join(out_dir, f"emoji_{i}.png"))

# Main processing loop - modified for sampling mode
frame_indices = None  # No longer using fixed frame_indices, using dynamic sampling instead
num_samples = 25  # Sample 25 frames per example

# Can choose different resolutions
face_resolutions = [56]

for face_res in face_resolutions:
    logger.info(f"Processing with face resolution: {face_res}x{face_res}, sampling {num_samples} frames")
    
    for sub in range(1, 43):
        logger.info(f"Processing subject {sub:02d} with {face_res}x{face_res} ...")
        
        try:
            vis = pickle.load(open(fr'/eva-feat/Input_images/Vision/subject_{sub:02d}_vis.pkl', 'rb'))   # download from https://www.kaggle.com/datasets/jingjinghuhu/eva-feat
            video_train = vis[0]  # (280, 25, 56, 56, 3)
            video_test = vis[2]   # (120, 25, 56, 56, 3)
            predictor_path = 'shape_predictor_68_face_landmarks.dat'  # download from https://www.kaggle.com/models/tranthaitoanb2103447/shape_predictor_68_face_landmarks.dat/TensorFlow2/default/1

            # Use new sampling method
            emojis_train, train_info = batch_faces2emojis_sampled(
                video_train, face_res=face_res, output_size=face_res,
                predictor_path=predictor_path, num_samples=num_samples
            )
            logger.info(f"Subject {sub:02d} train: shape {emojis_train.shape}")

            emojis_test, test_info = batch_faces2emojis_sampled(
                video_test, face_res=face_res, output_size=face_res,
                predictor_path=predictor_path, num_samples=num_samples
            )
            logger.info(f"Subject {sub:02d} test: shape {emojis_test.shape}")

            # Modified save format using the same list format as original pkl
            emoji_vis = [
                emojis_train,
                vis[1],  # labels_train
                emojis_test,
                vis[3]   # labels_test
            ]
            
            # Save path based on resolution
            out_path = f'./Vision_Landmarks_sampled_25x{face_res}x{face_res}/emoji_vis_subject_{sub:02d}.pkl'
            
            # Create directory
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            with open(out_path, 'wb') as f:
                pickle.dump(emoji_vis, f)
                logger.info(f"Saved to {out_path}")

            # Save sample images
            save_emoji_samples_new(emojis_train[0], sub, save_dir='./samples', 
                                  max_samples=num_samples, face_res=face_res)
            
            # Additionally save extraction info (optional, doesn't affect main data format)
            info_path = f'./Vision_Landmarks_sampled_25x{face_res}x{face_res}/emoji_extraction_info_subject_{sub:02d}.pkl'
            extraction_info = {
                'train_info': train_info,
                'test_info': test_info,
                'face_res': face_res,
                'num_samples': num_samples,
                'subject_id': sub
            }
            with open(info_path, 'wb') as f:
                pickle.dump(extraction_info, f)
            
            # Visualize first training example (optional)
            if sub == 1:  # Only visualize for the first subject
                logger.info(f"Visualizing subject {sub:02d} sample...")
                visualize_sampling_results(video_train, emojis_train, train_info, 
                                          sample_idx=0, face_res=face_res)
                
        except Exception as e:
            logger.error(f"Subject {sub:02d} processing failed: {e}")
            continue

logger.info("All subjects processed successfully!")