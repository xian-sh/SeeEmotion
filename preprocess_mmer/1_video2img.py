import dlib
from PIL import Image
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import cv2
import logging
from datetime import datetime

def setup_logging(log_file="face_extraction.log"):
    """Set up logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def extract_face_images_from_video(video_path, output_dir, video_id, predictor_path, interval=0.1, target_duration=20):
    """
    Extract face images from a video at fixed time intervals and keep the target count.
    
    Args:
    - video_path: Video file path.
    - output_dir: Output directory.
    - video_id: Video ID.
    - predictor_path: Path to the dlib facial landmark predictor.
    - interval: Extraction interval in seconds.
    - target_duration: Target duration in seconds.
    
    Returns:
    - extracted_images: List of extracted image paths.
    - extraction_log: Detailed extraction log.
    """
    logger = logging.getLogger(__name__)
    
    # Compute the target number of extracted images
    target_count = int(target_duration / interval)  # 20 seconds / 0.1 seconds = 200 images
    
    extraction_log = {
        'video_id': video_id,
        'interval': interval,
        'target_duration': target_duration,
        'target_count': target_count,  # 200 images
        'extracted_count': 0,
        'face_detected_count': 0,
        'duplicated_count': 0,
        'failed_frames': [],
        'successful_frames': [],
        'duplication_info': []
    }
    
    try:
        # Initialize dlib face detector and landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return [], extraction_log
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Compute the frame interval for extraction
        frame_interval = fps * interval  # example: 30 fps * 0.1 s = 3 frames
        
        extracted_images = []
        valid_face_images = []  # Store images with valid face detections
        
        logger.info(f"Video {video_id}: start face extraction, target {target_count} images (one image every {interval} seconds)")
        
        # Phase 1: extract at fixed intervals and record successful/failed frames
        for i in range(target_count):
            frame_number = int(i * frame_interval)  # Ensure integer frame index
            time_point = i * interval  # Current time point
            
            if frame_number >= total_frames:
                logger.warning(f"Video {video_id}: time point {time_point:.1f}s exceeds video length")
                extraction_log['failed_frames'].append(time_point)
                continue
            
            # Jump to the target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Video {video_id}: failed to read frame {frame_number} at {time_point:.1f}s")
                extraction_log['failed_frames'].append(time_point)
                continue
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = detector(rgb_frame)
            
            if len(faces) == 0:
                logger.debug(f"Video {video_id} at {time_point:.1f}s: no face detected")
                extraction_log['failed_frames'].append(time_point)
                continue
            
            # Select the largest face
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Get facial landmarks
            landmarks = predictor(rgb_frame, face)
            
            # Compute face bounding box
            face_points = []
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                face_points.append((x, y))
            
            face_points = np.array(face_points)
            
            # Compute a tighter bounding box
            x_min, y_min = np.min(face_points, axis=0)
            x_max, y_max = np.max(face_points, axis=0)
            
            # Add margin
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(rgb_frame.shape[1], x_max + margin)
            y_max = min(rgb_frame.shape[0], y_max + margin)
            
            # Crop face region
            face_image = rgb_frame[y_min:y_max, x_min:x_max]
            
            if face_image.size == 0:
                logger.warning(f"Video {video_id} at {time_point:.1f}s: empty face region")
                extraction_log['failed_frames'].append(time_point)
                continue
            
            # Save image with a filename that supports decimal-second timestamps
            output_filename = f"video_{video_id:02d}_{time_point:.1f}s.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            pil_image = Image.fromarray(face_image)
            pil_image.save(output_path, 'JPEG', quality=95)
            
            extracted_images.append(output_path)
            valid_face_images.append((output_path, time_point))  # Record path and time point
            extraction_log['successful_frames'].append(time_point)
            extraction_log['face_detected_count'] += 1
            
            # Log progress occasionally to avoid excessive output
            if (i + 1) % 50 == 0 or i == 0:
                logger.info(f"Video {video_id}: processed {i+1}/{target_count} images, successful detections: {extraction_log['face_detected_count']}")
        
        cap.release()
        
        # Phase 2: if the count is insufficient, fill with nearest valid images
        current_count = len(extracted_images)
        
        if current_count < target_count:
            need_duplicate = target_count - current_count
            logger.warning(f"Video {video_id}: only extracted {current_count} images, need to duplicate {need_duplicate}")
            
            if len(valid_face_images) == 0:
                logger.error(f"Video {video_id}: no valid face image is available for duplication")
                extraction_log['extracted_count'] = current_count
                return extracted_images, extraction_log
            
            # Duplication strategy: prefer images from nearby time points
            failed_time_points = extraction_log['failed_frames']
            
            for failed_time in failed_time_points:
                if len(extracted_images) >= target_count:
                    break
                
                # Find the nearest valid image
                best_source = None
                min_distance = float('inf')
                
                for img_path, success_time in valid_face_images:
                    distance = abs(success_time - failed_time)
                    if distance < min_distance:
                        min_distance = distance
                        best_source = (img_path, success_time)
                
                if best_source:
                    source_path, source_time = best_source
                    
                    # Create duplicate filename
                    duplicate_filename = f"video_{video_id:02d}_{failed_time:.1f}s.jpg"
                    duplicate_path = os.path.join(output_dir, duplicate_filename)
                    
                    # Duplicate image
                    source_img = Image.open(source_path)
                    source_img.save(duplicate_path, 'JPEG', quality=95)
                    
                    extracted_images.append(duplicate_path)
                    extraction_log['duplicated_count'] += 1
                    extraction_log['duplication_info'].append({
                        'target_time': failed_time,
                        'source_time': source_time,
                        'source_file': os.path.basename(source_path),
                        'duplicate_file': duplicate_filename
                    })
                    
                    if len(extraction_log['duplication_info']) % 20 == 1:  # Reduce logging frequency
                        logger.info(f"Video {video_id}: duplication progress {extraction_log['duplicated_count']}/{need_duplicate}")
        
        extraction_log['extracted_count'] = len(extracted_images)
        logger.info(f"Video {video_id}: final extracted count: {len(extracted_images)} face images "
                   f"(original {extraction_log['face_detected_count']}, duplicated {extraction_log['duplicated_count']})")
        
        return extracted_images, extraction_log
        
    except Exception as e:
        logger.error(f"Video {video_id} face image extraction failed: {e}")
        extraction_log['error'] = str(e)
        return [], extraction_log

def process_all_face_extraction(subject_id, predictor_path, interval=0.5, target_duration=20):
    """
    Extract face images from all videos of a given subject.
    
    Parameter notes:
    - interval: extraction interval in seconds.
    - target_duration: target duration is 20 seconds.
    
    Result: target count is determined by target_duration / interval for each video.
    """
    logger = logging.getLogger(__name__)
    video_dir = f'Aligned_data/{subject_id}/videos_20s'
    
    if not os.path.exists(video_dir):
        logger.error(f"Video directory does not exist: {video_dir}")
        return {}
    
    # Create a unified output directory for face images
    face_output_dir = f'Aligned_data/{subject_id}/face_images'
    os.makedirs(face_output_dir, exist_ok=True)
    
    processed_videos = {}
    all_face_images = []
    target_per_video = int(target_duration / interval)  # Target images per video
    target_total = 32 * target_per_video  # 6400 images in total
    
    total_extraction_log = {
        'subject_id': subject_id,
        'interval': interval,
        'target_duration': target_duration,
        'target_per_video': target_per_video,
        'target_total_images': target_total,
        'actual_total_images': 0,
        'total_original_faces': 0,
        'total_duplicated_faces': 0,
        'successful_videos': 0,
        'failed_videos': 0,
        'video_logs': {}
    }
    
    logger.info(f"Start processing subject {subject_id}, target extraction count: {target_total} face images (per video: {target_per_video})")
    
    for video_id in range(32):
        video_path = os.path.join(video_dir, f'{video_id}_20s.mp4')
        
        if not os.path.exists(video_path):
            logger.warning(f"Video does not exist: {video_path}")
            total_extraction_log['failed_videos'] += 1
            continue
        
        logger.info(f"Processing video {video_id}/{31} (target: {target_per_video})")
        extracted_images, extraction_log = extract_face_images_from_video(
            video_path, face_output_dir, video_id, predictor_path, interval, target_duration
        )
        
        processed_videos[video_id] = {
            'video_path': video_path,
            'face_images': extracted_images,
            'image_count': len(extracted_images),
            'extraction_log': extraction_log
        }
        
        # Update global log
        total_extraction_log['video_logs'][video_id] = extraction_log
        total_extraction_log['actual_total_images'] += len(extracted_images)
        total_extraction_log['total_original_faces'] += extraction_log['face_detected_count']
        total_extraction_log['total_duplicated_faces'] += extraction_log['duplicated_count']
        
        if len(extracted_images) > 0:
            total_extraction_log['successful_videos'] += 1
        else:
            total_extraction_log['failed_videos'] += 1
        
        all_face_images.extend(extracted_images)
        
        logger.info(f"Video {video_id} finished: {len(extracted_images)}/{target_per_video} images "
                   f"(original {extraction_log['face_detected_count']}, duplicated {extraction_log['duplicated_count']})")
    
    # Save processing information
    face_info = {
        'subject_id': subject_id,
        'processed_videos': processed_videos,
        'total_videos': len(processed_videos),
        'total_face_images': len(all_face_images),
        'all_face_images': all_face_images,
        'interval_seconds': interval,
        'target_duration': target_duration,
        'target_per_video': target_per_video,
        'face_images_dir': face_output_dir,
        'extraction_log': total_extraction_log
    }
    
    with open(f'Aligned_data/{subject_id}/face_extraction_info.pkl', 'wb') as f:
        pickle.dump(face_info, f)
    
    # Detailed statistics report
    logger.info(f"\n{'='*60}")
    logger.info(f"Subject {subject_id} face extraction summary:")
    logger.info(f"Target images: {total_extraction_log['target_total_images']}")
    logger.info(f"Actual images: {total_extraction_log['actual_total_images']}")
    logger.info(f"Original detections: {total_extraction_log['total_original_faces']}")
    logger.info(f"Duplicated fill: {total_extraction_log['total_duplicated_faces']}")
    logger.info(f"Successful videos: {total_extraction_log['successful_videos']}/32")
    logger.info(f"Failed videos: {total_extraction_log['failed_videos']}/32")
    logger.info(f"Image output directory: {face_output_dir}")
    logger.info(f"{'='*60}")
    
    return processed_videos

def print_detailed_extraction_summary(subject_id):
    """
    Print a detailed extraction summary.
    """
    logger = logging.getLogger(__name__)
    info_path = f'Aligned_data/{subject_id}/face_extraction_info.pkl'
    
    if not os.path.exists(info_path):
        logger.error(f"No extraction information found for subject {subject_id}")
        return None
    
    with open(info_path, 'rb') as f:
        info = pickle.load(f)
    
    extraction_log = info['extraction_log']
    target_per_video = extraction_log['target_per_video']
    interval = extraction_log['interval']
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Subject {subject_id} detailed extraction report (one image every {interval} seconds)")
    logger.info(f"{'='*80}")
    
    # Per-video statistics
    for video_id in range(32):
        if video_id in extraction_log['video_logs']:
            vlog = extraction_log['video_logs'][video_id]
            logger.info(f"Video {video_id:2d}: {vlog['extracted_count']:3d}/{target_per_video:3d} images "
                       f"(original {vlog['face_detected_count']:3d}, duplicated {vlog['duplicated_count']:3d}) "
                       f"failed frames: {len(vlog['failed_frames'])}")
            
            # Show only the first few duplication operations to keep logs concise
            if vlog['duplicated_count'] > 0:
                shown_count = min(3, len(vlog['duplication_info']))
                for i in range(shown_count):
                    dup_info = vlog['duplication_info'][i]
                    logger.info(f"    {dup_info['target_time']:.1f}s <- copied from {dup_info['source_time']:.1f}s")
                if vlog['duplicated_count'] > 3:
                    logger.info(f"    ... plus {vlog['duplicated_count'] - 3} more duplication operations")
        else:
            logger.info(f"Video {video_id:2d}: not processed")
    
    target_total = extraction_log['target_total_images']
    actual_total = extraction_log['actual_total_images']
    logger.info(f"\nTotal: {actual_total}/{target_total} face images")
    logger.info(f"Success rate: {actual_total/target_total*100:.1f}%")
    
    return info

def visualize_face_extraction_results(subject_id, images_per_video=4, max_videos_to_show=3):
    """
    Visualize face extraction results.
    """
    info_path = f'Aligned_data/{subject_id}/face_extraction_info.pkl'
    
    if not os.path.exists(info_path):
        print(f"Face extraction information not found: {info_path}")
        return
    
    with open(info_path, 'rb') as f:
        info = pickle.load(f)
    
    processed_videos = info['processed_videos']
    video_ids = list(processed_videos.keys())[:max_videos_to_show]
    interval = info.get('interval_seconds', 1.0)
    
    if not video_ids:
        print("No valid face image data")
        return
    
    fig, axes = plt.subplots(len(video_ids), images_per_video, figsize=(5*images_per_video, 4*len(video_ids)))
    if len(video_ids) == 1:
        axes = axes.reshape(1, -1)
    
    for row, video_id in enumerate(video_ids):
        face_images = processed_videos[video_id]['face_images']
        extraction_log = processed_videos[video_id]['extraction_log']
        
        # Display a fixed number of face images with roughly even spacing
        total_images = len(face_images)
        indices = np.linspace(0, total_images-1, min(images_per_video, total_images), dtype=int) if total_images > 0 else []
        
        for col in range(images_per_video):
            if col < len(indices):
                img_idx = indices[col]
                img_path = face_images[img_idx]
                if os.path.exists(img_path):
                    img = plt.imread(img_path)
                    axes[row, col].imshow(img)
                    # Extract time information from filename
                    filename = os.path.basename(img_path)
                    time_info = filename.split('_')[-1].replace('.jpg', '')
                    
                    # Check whether the image is duplicated
                    time_point = float(time_info.replace('s', ''))
                    is_duplicate = any(abs(d['target_time'] - time_point) < 0.01 for d in extraction_log['duplication_info'])
                    title_suffix = " (Duplicated)" if is_duplicate else ""
                    
                    axes[row, col].set_title(f'Video {video_id}\n{time_info}{title_suffix}')
                else:
                    axes[row, col].text(0.5, 0.5, 'Image\nNot Found', 
                                      ha='center', va='center', transform=axes[row, col].transAxes)
            else:
                axes[row, col].text(0.5, 0.5, 'No Image', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
            
            axes[row, col].axis('off')
    
    target_total = info["extraction_log"]["target_total_images"]
    actual_total = info["total_face_images"]
    original_total = info["extraction_log"]["total_original_faces"]
    duplicated_total = info["extraction_log"]["total_duplicated_faces"]
    
    plt.suptitle(f'Subject {subject_id} - Face Extraction Results (one image every {interval} seconds)\n'
                f'Total: {actual_total}/{target_total} Images '
                f'(Original: {original_total}, Duplicated: {duplicated_total})', fontsize=16)
    plt.tight_layout()
    plt.show()

# Main processing workflow
mat_list = [1,5,11,12,19,20,22,23,24,25,29,32,33,38]
fs = 300  # Hz
time_window = 20 # s

# Set up logging
logger = setup_logging("face_extraction.log")
logger.info(f"Start face extraction, time: {datetime.now()}")

# Predictor path
predictor_path = 'shape_predictor_68_face_landmarks.dat'

# Check whether the predictor file exists
if not os.path.exists(predictor_path):
    logger.error(f"Facial landmark predictor file does not exist: {predictor_path}")
    logger.error("Please make sure shape_predictor_68_face_landmarks.dat has been downloaded")
else:
    logger.info("Start face extraction...")
    
    all_subjects_summary = {}
    
    for i, m in enumerate(mat_list):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing subject {m} ({i+1}/{len(mat_list)})")
        logger.info(f"{'='*80}")
        
        try:
            # Run face extraction
            processed_videos = process_all_face_extraction(
                m, 
                predictor_path, 
                interval=0.5,      #  extract one image every 0.1 seconds
                target_duration=20  # 20-second video
            )
            
            if processed_videos:
                # Print detailed statistics
                info = print_detailed_extraction_summary(m)
                all_subjects_summary[m] = info['extraction_log'] if info else None
                
                # Visualize partial results
                logger.info(f"Generate visualization for subject {m}...")
                visualize_face_extraction_results(m, images_per_video=4, max_videos_to_show=2)
            else:
                logger.error(f"Subject {m}: face extraction failed")
                all_subjects_summary[m] = None
                
        except Exception as e:
            logger.error(f"Subject {m} face extraction error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            all_subjects_summary[m] = None
            continue

logger.info("All face extraction processing completed.")

# Final statistics for all subjects
def print_final_summary(mat_list, all_subjects_summary):
    """
    Print the final summary for all subjects.
    """
    logger.info(f"\n{'='*80}")
    logger.info("Final summary")
    logger.info(f"{'='*80}")
    
    total_images = 0
    total_original = 0
    total_duplicated = 0
    successful_subjects = 0
    
    # Target: 200 images per video, 32 videos, 6400 images per subject
    target_per_subject = 32 * 200  # 6400 images
    target_total = len(mat_list) * target_per_subject
    
    logger.info("Subject ID | Actual Images | Original | Duplicated | Success Rate")
    logger.info("-" * 65)
    
    for m in mat_list:
        if m in all_subjects_summary and all_subjects_summary[m]:
            summary = all_subjects_summary[m]
            actual = summary['actual_total_images']
            original = summary['total_original_faces']
            duplicated = summary['total_duplicated_faces']
            success_rate = actual / target_per_subject * 100
            
            successful_subjects += 1
            total_images += actual
            total_original += original
            total_duplicated += duplicated
            
            logger.info(f"Subject {m:2d} | {actual:5d}/{target_per_subject:4d} | {original:5d}    | {duplicated:5d}    | {success_rate:5.1f}%")
        else:
            logger.info(f"Subject {m:2d} | processing failed")
    
    logger.info("-" * 65)
    logger.info(f"Total: {total_images}/{target_total}  images")
    logger.info(f"Successfully processed subjects: {successful_subjects}/{len(mat_list)}")
    logger.info(f"Average per subject: {total_images/successful_subjects:.1f}  images" if successful_subjects > 0 else "")
    logger.info(f"Total original detections: {total_original}")
    logger.info(f"Total duplicated fills: {total_duplicated}")
    logger.info(f"Overall success rate: {total_images/target_total*100:.1f}%")
    
    # Save final statistics
    final_summary = {
        'processing_time': datetime.now(),
        'interval': 0.1,
        'target_per_video': 200,
        'target_per_subject': target_per_subject,
        'target_subjects': len(mat_list),
        'successful_subjects': successful_subjects,
        'target_total_images': target_total,
        'actual_total_images': total_images,
        'total_original_faces': total_original,
        'total_duplicated_faces': total_duplicated,
        'subjects_summary': all_subjects_summary
    }
    
    with open('face_extraction_final_summary.pkl', 'wb') as f:
        pickle.dump(final_summary, f)
    
    logger.info(f"Final summary saved to: face_extraction_final_summary.pkl")

print_final_summary(mat_list, all_subjects_summary)
logger.info(f"Processing finished at: {datetime.now()}")