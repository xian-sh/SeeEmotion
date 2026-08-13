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

def setup_landmarks_logging(log_file="landmarks_extraction.log"):
    """Set up logging for landmark extraction"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Add base_size to make the image size configurable
def draw_stick_emoji(landmarks, base_size=56, out_res=8, line_width=2):
    """
    Draw abstract emoji landmarks.
    
    Args:
    - landmarks: Landmark coordinates.
    - base_size: Base image size, default is 56.
    - out_res: Output small image size.
    - line_width: Line width.
    
    Returns:
    - image: Image with shape (base_size, base_size).
    - small_img: Small image with shape (out_res, out_res).
    """
    image = np.zeros((base_size, base_size), dtype=np.uint8)
    
    # Face contour
    contour = landmarks[0:17]
    cv2.polylines(image, [contour], False, 255, line_width)
    
    # Eyebrows
    left_brow = landmarks[17:22]
    right_brow = landmarks[22:27]
    cv2.polylines(image, [left_brow], False, 255, line_width)
    cv2.polylines(image, [right_brow], False, 255, line_width)
    
    # Nose, including bridge and bottom
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
    
    # Resize
    small_img = cv2.resize(image, (out_res, out_res), interpolation=cv2.INTER_NEAREST)
    return image, small_img

# Add base_size parameter
def extract_landmarks_from_np(img_np, detector, predictor, base_size=56, detect_size=224):
    """
    Extract landmarks from a numpy array.
    
    Args:
    - img_np: Input image.
    - detector: dlib detector.
    - predictor: dlib predictor.
    - base_size: Final landmark image size.
    - detect_size: Image size used for face detection.
    """
    # Convert to grayscale
    if img_np.ndim == 3:
        if img_np.shape[2] == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        elif img_np.shape[2] == 4:  # RGBA
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
        else:
            gray = img_np[:,:,0]
    else:
        gray = img_np
    
    # Resize to detect_size for face detection
    big_gray = cv2.resize(gray, (detect_size, detect_size), interpolation=cv2.INTER_CUBIC)
    faces = detector(big_gray)
    if len(faces)==0:
        return None, big_gray, None
    
    shape = predictor(big_gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    
    # Scale landmarks to base_size
    landmarks_scaled = (landmarks * (base_size / detect_size)).astype(np.int32)
    return landmarks_scaled, big_gray, landmarks

def load_image_safely(image_path):
    """Load an image file safely."""
    try:
        # Try loading with PIL
        img_pil = Image.open(image_path)
        # Convert to RGB
        if img_pil.mode == 'RGBA':
            img_pil = img_pil.convert('RGB')
        elif img_pil.mode == 'L':  # grayscale image
            img_pil = img_pil.convert('RGB')
        
        # Convert to numpy array
        img_np = np.array(img_pil)
        return img_np
    except Exception as e:
        print(f"   Failed to load image: {e}")
        return None

def parse_filename_timestamp(filename):
    """Parse timestamp from filename with support for multiple formats."""
    import re
    
    # Try extracting numbers
    numbers = re.findall(r'\d+', filename)
    
    # Handle video_XX_YYs.jpg format
    if 'video_' in filename and '_' in filename and 's.' in filename:
        parts = filename.split('_')
        if len(parts) >= 3:
            try:
                return int(parts[2].replace('s.jpg', '').replace('s.png', ''))
            except:
                pass
    
    # If multiple numbers exist, use the last one
    if numbers:
        return int(numbers[-1])
    
    return 0  # Default value

def find_nearest_valid_landmarks(failed_indices, valid_landmarks_indices, all_files):
    """Find nearest valid landmarks for failed files."""
    logger = logging.getLogger(__name__)
    nearest_map = {}
    
    if not valid_landmarks_indices:
        logger.warning("No valid landmarks available for duplication")
        return nearest_map
    
    for failed_idx in failed_indices:
        failed_file = all_files[failed_idx]
        failed_timestamp = parse_filename_timestamp(failed_file)
        
        # Find the nearest valid file
        min_distance = float('inf')
        nearest_idx = None
        
        for valid_idx in valid_landmarks_indices:
            valid_file = all_files[valid_idx]
            valid_timestamp = parse_filename_timestamp(valid_file)
            distance = abs(valid_timestamp - failed_timestamp)
            
            if distance < min_distance:
                min_distance = distance
                nearest_idx = valid_idx
        
        if nearest_idx is not None:
            nearest_map[failed_idx] = nearest_idx
            logger.info(f"    Index {failed_idx}({failed_file}) <- copied from index Index {nearest_idx}({all_files[nearest_idx]}) (distance:{min_distance})")
        else:
            logger.warning(f"    Index {failed_idx}({failed_file}): no valid duplication source found")
    
    return nearest_map

# Add face_size parameter
def extract_landmarks_from_image_folder(face_dir, detector, predictor, face_size=56, output_size=32, 
                                       line_width=2, expected_length=640, detect_size=224):
    """
    Extract binary landmark images from a face image folder and return fixed-length arrays.
    
    Args:
    - face_dir: Face image folder path.
    - detector: dlib face detector.
    - predictor: dlib landmark predictor.
    - face_size: Main landmark image size.
    - output_size: Output small binary image size.
    - line_width: Line width.
    - expected_length: Expected sequence length.
    - detect_size: Image size used for face detection.
    
    Returns:
    - landmarks_main_array: numpy array with shape (expected_length, face_size, face_size).
    - landmarks_small_array: numpy array with shape (expected_length, output_size, output_size).
    - filenames_list: Corresponding filename list.
    - extraction_log: Detailed extraction log.
    """
    logger = logging.getLogger(__name__)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(face_dir, ext)))
        image_files.extend(glob.glob(os.path.join(face_dir, ext.upper())))
    
    # Sort by filename timestamp
    image_files.sort(key=lambda x: parse_filename_timestamp(os.path.basename(x)))
    
    if not image_files:
        logger.error(f"   No image files found")
        return None, None, [], {'error': 'No image files found'}
    
    # If there are more files than expected_length, keep the first expected_length files
    if len(image_files) > expected_length:
        logger.warning(f"  File count ({len(image_files)}) exceeds expected length ({expected_length}), keeping the first {expected_length}")
        image_files = image_files[:expected_length]
    
    # Warn if the file count is smaller than expected_length
    actual_length = len(image_files)
    if actual_length < expected_length:
        logger.warning(f"  File count ({actual_length}) is less than expected length ({expected_length}), will be filled using nearest neighbors")
    
    extraction_log = {
        'total_files': actual_length,
        'expected_length': expected_length,
        'successful_extractions': 0,
        'failed_extractions': 0,
        'duplicated_landmarks': 0,
        'failed_indices': [],
        'successful_indices': [],
        'duplication_map': {}
    }
    
    # Initialize output arrays using face_size
    landmarks_main_array = np.zeros((expected_length, face_size, face_size), dtype=np.uint8)
    landmarks_small_array = np.zeros((expected_length, output_size, output_size), dtype=np.uint8)
    filenames_list = [''] * expected_length
    
    valid_landmarks = {}  # Store successfully extracted landmarks {index: landmarks}
    valid_landmarks_indices = set()
    failed_indices = []
    
    logger.info(f"  Found {actual_length} images, expected length {expected_length}, face size {face_size}x{face_size}, start landmark extraction...")
    
    # Phase 1: try extracting landmarks from all images
    for i, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        filenames_list[i] = filename
        
        # Load image
        img_np = load_image_safely(img_path)
        if img_np is None:
            failed_indices.append(i)
            extraction_log['failed_indices'].append(i)
            extraction_log['failed_extractions'] += 1
            continue
        
        try:
            # Extract landmarks with new parameters
            landmarks, big_gray, landmarks_big = extract_landmarks_from_np(
                img_np, detector, predictor, base_size=face_size, detect_size=detect_size
            )
            
            if landmarks is not None:
                # Generate binary images with new parameters
                main_img, small_img = draw_stick_emoji(
                    landmarks, base_size=face_size, out_res=output_size, line_width=line_width
                )
                landmarks_main_array[i] = main_img
                landmarks_small_array[i] = small_img
                valid_landmarks[i] = landmarks  # Save original landmarks for duplication
                valid_landmarks_indices.add(i)
                extraction_log['successful_indices'].append(i)
                extraction_log['successful_extractions'] += 1
                
                logger.debug(f"     Index {i}({filename}): landmark extraction succeeded")
            else:
                failed_indices.append(i)
                extraction_log['failed_indices'].append(i)
                extraction_log['failed_extractions'] += 1
                logger.warning(f"     Index {i}({filename}): no face detected")
                
        except Exception as e:
            failed_indices.append(i)
            extraction_log['failed_indices'].append(i)
            extraction_log['failed_extractions'] += 1
            logger.error(f"     Index {i}({filename}): processing failed - {e}")
    
    logger.info(f"  Phase 1 completed: success {extraction_log['successful_extractions']}, failed {extraction_log['failed_extractions']}")
    
    # Phase 2: find nearest-neighbor duplications for failed files
    if failed_indices and valid_landmarks_indices:
        logger.info(f"  Start nearest-neighbor landmark duplication...")
        
        all_filenames = [os.path.basename(f) for f in image_files]
        nearest_map = find_nearest_valid_landmarks(failed_indices, valid_landmarks_indices, all_filenames)
        
        for failed_idx, source_idx in nearest_map.items():
            try:
                # Duplicate landmarks
                source_landmarks = valid_landmarks[source_idx]
                main_img, small_img = draw_stick_emoji(
                    source_landmarks, base_size=face_size, out_res=output_size, line_width=line_width
                )
                landmarks_main_array[failed_idx] = main_img
                landmarks_small_array[failed_idx] = small_img
                
                extraction_log['duplicated_landmarks'] += 1
                extraction_log['duplication_map'][failed_idx] = source_idx
                
                logger.info(f"     Index {failed_idx}: copied from index Index {source_idx}")
                
            except Exception as e:
                logger.error(f"     Index {failed_idx}: duplication failed - {e}")
                # Keep zero arrays, already initialized
    
    elif failed_indices and not valid_landmarks_indices:
        logger.error(f"  No valid landmarks available for duplication. Indices {failed_indices} will remain blank")
    
    # Phase 3: if file count is smaller than expected_length, fill with the last valid landmarks
    if actual_length < expected_length and valid_landmarks_indices:
        # Find the last valid landmarks
        last_valid_idx = max(valid_landmarks_indices)
        last_valid_landmarks = valid_landmarks[last_valid_idx]
        
        logger.info(f"  Use index {last_valid_idx} landmarks to fill the remaining {expected_length - actual_length} positions")
        
        for i in range(actual_length, expected_length):
            main_img, small_img = draw_stick_emoji(
                last_valid_landmarks, base_size=face_size, out_res=output_size, line_width=line_width
            )
            landmarks_main_array[i] = main_img
            landmarks_small_array[i] = small_img
            filenames_list[i] = f"padded_from_{last_valid_idx}"
            extraction_log['duplicated_landmarks'] += 1
            extraction_log['duplication_map'][i] = last_valid_idx
    
    # Update final statistics
    final_landmarks_count = np.sum(np.any(landmarks_main_array.reshape(expected_length, -1), axis=1))
    extraction_log['final_landmarks_count'] = int(final_landmarks_count)
    extraction_log['completion_rate'] = final_landmarks_count / expected_length * 100
    
    logger.info(f"  Final result: {final_landmarks_count}/{expected_length} positions have landmarks "
               f"(original {extraction_log['successful_extractions']}, "
               f"duplicated {extraction_log['duplicated_landmarks']})")
    
    return landmarks_main_array, landmarks_small_array, filenames_list, extraction_log

# Add face_size parameter
def process_all_subjects(images_dir='Images', output_dir='Landmarks', 
                        predictor_path='shape_predictor_68_face_landmarks.dat',
                        face_size=128, output_size=32, line_width=2, 
                        expected_length=640, detect_size=224):
    """
    Process face images of all subjects, extract binary landmark images, and save fixed-length arrays.
    
    Args:
    - face_size: Main landmark image size, configurable.
    - output_size: Small landmark image size.
    - detect_size: Image size used for face detection.
    """
    logger = setup_landmarks_logging("landmarks_extraction.log")
    logger.info(f"Start landmark extraction, time: {datetime.now()}")
    logger.info(f"Face size: {face_size}x{face_size}, small size: {output_size}x{output_size}")
    
    # Initialize dlib
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        logger.info(f" Successfully loaded dlib model: {predictor_path}")
    except Exception as e:
        logger.error(f" Failed to load dlib model: {e}")
        logger.error("Please make sure shape_predictor_68_face_landmarks.dat has been downloaded")
        return
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Get all subject folders
    if not os.path.exists(images_dir):
        logger.error(f" Images directory does not exist: {images_dir}")
        return
    
    # Support two directory structures
    face_dirs = []
    
    # Option 1: find *_face_imgs folders directly under images_dir
    for item in os.listdir(images_dir):
        item_path = os.path.join(images_dir, item)
        if os.path.isdir(item_path) and item.endswith('_face_imgs'):
            face_dirs.append((item, item_path))
    
    # Option 2: find face_images folders under subdirectories of images_dir
    if not face_dirs:
        for item in os.listdir(images_dir):
            item_path = os.path.join(images_dir, item)
            if os.path.isdir(item_path):
                face_img_path = os.path.join(item_path, 'face_images')
                if os.path.exists(face_img_path):
                    face_dirs.append((f"{item}_face_images", face_img_path))
    
    face_dirs.sort()
    
    if not face_dirs:
        logger.error(f" No face image folders found")
        logger.info("Please make sure the directory structure is one of the following:")
        logger.info("1. Images/subject_id_face_imgs/")
        logger.info("2. Images/subject_id/face_images/")
        return
    
    logger.info(f"Found {len(face_dirs)} subject folders")
    
    processed_subjects = []
    failed_subjects = []
    all_subjects_log = {}
    
    for face_dir_name, face_dir_path in face_dirs:
        # Extract subject ID
        if face_dir_name.endswith('_face_imgs'):
            subject_id = face_dir_name.replace('_face_imgs', '')
        elif face_dir_name.endswith('_face_images'):
            subject_id = face_dir_name.replace('_face_images', '')
        else:
            subject_id = face_dir_name
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing subject {subject_id}")
        logger.info(f"Processing directory: {face_dir_path}")
        logger.info(f"{'='*60}")
        
        try:
            # Extract landmarks with new parameters
            landmarks_main_array, landmarks_small_array, filenames_list, extraction_log = extract_landmarks_from_image_folder(
                face_dir_path, detector, predictor, 
                face_size=face_size, output_size=output_size, 
                line_width=line_width, expected_length=expected_length, 
                detect_size=detect_size
            )
            
            if landmarks_main_array is not None:
                # Save results
                output_file = os.path.join(output_dir, f'{subject_id}_landmarks.pkl')
                
                # Save landmark data and extraction log
                save_data = {
                    'landmarks_main_array': landmarks_main_array,      # (expected_length, face_size, face_size)
                    'landmarks_small_array': landmarks_small_array,    # (expected_length, output_size, output_size)
                    'filenames_list': filenames_list,                 # expected_length filenames
                    'extraction_log': extraction_log,
                    'subject_id': subject_id,
                    'face_size': face_size,                           # Save face_size config
                    'output_size': output_size,
                    'line_width': line_width,
                    'expected_length': expected_length,
                    'detect_size': detect_size,
                    'processing_time': datetime.now(),
                    'array_shapes': {
                        'main_shape': landmarks_main_array.shape,
                        'small_shape': landmarks_small_array.shape
                    }
                }
                
                with open(output_file, 'wb') as f:
                    pickle.dump(save_data, f)
                
                logger.info(f"   Saved landmark arrays to: {output_file}")
                logger.info(f"     Main array shape: {landmarks_main_array.shape}")
                logger.info(f"     Small array shape: {landmarks_small_array.shape}")
                
                processed_subjects.append(subject_id)
                all_subjects_log[subject_id] = extraction_log
                
                # Save sample images
                save_sample_images_from_arrays(landmarks_main_array, landmarks_small_array, 
                                              filenames_list, subject_id, output_dir, num_samples=6)
                
                # Print detailed statistics
                logger.info(f"  Statistics: expected length {expected_length}, actual files {extraction_log['total_files']}, "
                           f"successful extractions {extraction_log['successful_extractions']}, "
                           f"duplicated {extraction_log['duplicated_landmarks']}, "
                           f"completion rate{extraction_log['completion_rate']:.1f}%")
                
            else:
                logger.error(f"   No landmarks extracted")
                failed_subjects.append(subject_id)
                all_subjects_log[subject_id] = {'error': 'No landmarks extracted'}
                
        except Exception as e:
            logger.error(f"   processing failed: {e}")
            failed_subjects.append(subject_id)
            all_subjects_log[subject_id] = {'error': str(e)}
    
    # Save global log
    final_log = {
        'processing_time': datetime.now(),
        'total_subjects': len(face_dirs),
        'successful_subjects': len(processed_subjects),
        'failed_subjects': len(failed_subjects),
        'processed_subjects': processed_subjects,
        'failed_subjects': failed_subjects,
        'subjects_log': all_subjects_log,
        'face_size': face_size,                  # Save config information
        'output_size': output_size,
        'line_width': line_width,
        'expected_length': expected_length,
        'detect_size': detect_size
    }
    
    final_log_file = os.path.join(output_dir, 'landmarks_extraction_summary.pkl')
    with open(final_log_file, 'wb') as f:
        pickle.dump(final_log, f)
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("Landmark Extraction Summary")
    logger.info(f"{'='*80}")
    logger.info(f"Total subjects: {len(face_dirs)}")
    logger.info(f"Successfully processed: {len(processed_subjects)}")
    logger.info(f"processing failed: {len(failed_subjects)}")
    logger.info(f"Success rate: {len(processed_subjects)/len(face_dirs)*100:.1f}%")
    logger.info(f"Array shape: Main array({expected_length}, {face_size}, {face_size}), Small array({expected_length}, {output_size}, {output_size})")
    
    if processed_subjects:
        logger.info(f"\nProcessed subjects: {processed_subjects}")
        
        # Global landmark statistics
        total_positions = 0
        total_original = 0
        total_duplicated = 0
        
        for subject_id in processed_subjects:
            if subject_id in all_subjects_log:
                log = all_subjects_log[subject_id]
                total_positions += log.get('final_landmarks_count', 0)
                total_original += log.get('successful_extractions', 0)
                total_duplicated += log.get('duplicated_landmarks', 0)
        
        logger.info(f"\nGlobal statistics:")
        logger.info(f"Total landmark positions: {total_positions}")
        logger.info(f"Original extractions: {total_original}")
        logger.info(f"duplicated fill: {total_duplicated}")
        
        if total_positions > 0:
            logger.info(f"Original extraction rate: {total_original/total_positions*100:.1f}%")
            logger.info(f"Duplication fill rate: {total_duplicated/total_positions*100:.1f}%")
    
    if failed_subjects:
        logger.info(f"\nFailed subjects: {failed_subjects}")
    
    logger.info(f"\nProcessing finished at: {datetime.now()}")
    logger.info(f"Global log saved to: {final_log_file}")

def save_sample_images_from_arrays(landmarks_main_array, landmarks_small_array, filenames_list, 
                                  subject_id, output_dir, num_samples=6):
    """Save sample images from arrays for visualization."""
    
    sample_dir = os.path.join(output_dir, 'samples', subject_id)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    
    # Use the first non-zero samples
    non_zero_indices = []
    for i in range(min(len(filenames_list), landmarks_main_array.shape[0])):
        if np.any(landmarks_main_array[i]) and filenames_list[i]:
            non_zero_indices.append(i)
        if len(non_zero_indices) >= num_samples:
            break
    
    for i in non_zero_indices:
        filename = filenames_list[i] if i < len(filenames_list) else f"index_{i}"
        main_img = landmarks_main_array[i]
        small_img = landmarks_small_array[i]
        
        # Save main-size version
        main_img_pil = Image.fromarray(255 - main_img)  # Invert colors to make lines black
        main_path = os.path.join(sample_dir, f'{i:03d}_{filename}_main_{main_img.shape[0]}x{main_img.shape[0]}.png')
        main_img_pil.save(main_path)
        
        # Save small-size version with enlargement
        small_img_pil = Image.fromarray(255 - small_img)  # Invert colors to make lines black
        small_enlarged = small_img_pil.resize((128, 128), Image.NEAREST)  # enlarged display
        small_path = os.path.join(sample_dir, f'{i:03d}_{filename}_small_{small_img.shape[0]}x{small_img.shape[0]}.png')
        small_enlarged.save(small_path)

def visualize_landmarks_results(landmarks_file, num_display=6, start_index=0):
    """Visualize landmark extraction results and show duplication information."""
    
    with open(landmarks_file, 'rb') as f:
        data = pickle.load(f)
    
    # Support both new and old formats
    if 'landmarks_main_array' in data:
        landmarks_main_array = data['landmarks_main_array']
        landmarks_small_array = data['landmarks_small_array']
        filenames_list = data['filenames_list']
        extraction_log = data.get('extraction_log', {})
        face_size = data.get('face_size', landmarks_main_array.shape[1])  # Infer from array shape
    elif 'landmarks_stick_array' in data:  # Support old version
        landmarks_main_array = data['landmarks_stick_array']
        landmarks_small_array = data['landmarks_emoji_array']
        filenames_list = data['filenames_list']
        extraction_log = data.get('extraction_log', {})
        face_size = landmarks_main_array.shape[1]
    else:
        print("Unsupported file format")
        return
    
    duplication_map = extraction_log.get('duplication_map', {})
    
    # Select display indices
    end_index = min(start_index + num_display, landmarks_main_array.shape[0])
    display_indices = list(range(start_index, end_index))
    
    fig, axes = plt.subplots(2, len(display_indices), figsize=(3*len(display_indices), 6))
    if len(display_indices) == 1:
        axes = axes.reshape(2, 1)
    
    for j, i in enumerate(display_indices):
        main_img = landmarks_main_array[i]
        small_img = landmarks_small_array[i]
        filename = filenames_list[i] if i < len(filenames_list) else f"index_{i}"
        
        # Check whether landmarks are duplicated
        is_duplicate = i in duplication_map
        source_info = f"\n(copied from: {duplication_map[i]})" if is_duplicate else ""
        
        # Show main-size version
        axes[0, j].imshow(255 - main_img, cmap='gray')
        axes[0, j].set_title(f'[{i}] {filename}\n{face_size}x{face_size}{source_info}', fontsize=8)
        axes[0, j].axis('off')
        
        # Show small-size version
        axes[1, j].imshow(255 - small_img, cmap='gray')
        axes[1, j].set_title(f'{small_img.shape[0]}x{small_img.shape[0]}')
        axes[1, j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print array information
    print(f"Landmark array shapes:")
    print(f"  Main array: {landmarks_main_array.shape}")
    print(f"  Small array: {landmarks_small_array.shape}")
    print(f"  Display indices: {start_index}-{end_index-1}")

def print_landmarks_summary(output_dir='Landmarks'):
    """Print landmark extraction summary for all subjects."""
    
    summary_file = os.path.join(output_dir, 'landmarks_extraction_summary.pkl')
    
    if not os.path.exists(summary_file):
        print("Landmark extraction summary file not found")
        return
    
    with open(summary_file, 'rb') as f:
        summary = pickle.load(f)
    
    print(f"\n{'='*80}")
    print("Detailed Landmark Extraction Summary")
    print(f"{'='*80}")
    
    subjects_log = summary.get('subjects_log', {})
    expected_length = summary.get('expected_length', 640)
    face_size = summary.get('face_size', 'unknown')
    output_size = summary.get('output_size', 'unknown')
    
    print(f"Configuration:")
    print(f"  Face size: {face_size}x{face_size}")
    print(f"  small size: {output_size}x{output_size}")
    print(f"  expected length: {expected_length}")
    print(f"\nSubject ID | Files | Success | Duplicated | Completion")
    print("-" * 65)
    
    for subject_id in summary.get('processed_subjects', []):
        if subject_id in subjects_log:
            log = subjects_log[subject_id]
            total = log.get('total_files', 0)
            original = log.get('successful_extractions', 0)
            duplicated = log.get('duplicated_landmarks', 0)
            completion = log.get('completion_rate', 0)
            
            print(f"Subject{subject_id:2s} | {total:4d}     | {original:4d}     | {duplicated:4d}     | {completion:5.1f}%")
    
    print("-" * 65)
    print(f"Successfully processed: {summary.get('successful_subjects', 0)}/{summary.get('total_subjects', 0)}")

def load_landmarks_arrays(landmarks_file):
    """Convenience function for loading landmark arrays."""
    with open(landmarks_file, 'rb') as f:
        data = pickle.load(f)
    
    # Support both new and old formats
    if 'landmarks_main_array' in data:
        return (data['landmarks_main_array'], 
                data['landmarks_small_array'], 
                data['filenames_list'], 
                data['extraction_log'],
                data.get('face_size', data['landmarks_main_array'].shape[1]))
    elif 'landmarks_stick_array' in data:
        return (data['landmarks_stick_array'], 
                data['landmarks_emoji_array'], 
                data['filenames_list'], 
                data['extraction_log'],
                data['landmarks_stick_array'].shape[1])
    else:
        raise ValueError("Unsupported file format")

# Usage example
if __name__ == "__main__":
    # Process all subjects. face_size is now configurable.
    process_all_subjects(
        images_dir='./Aligned_data/Images1',
        output_dir='./Aligned_data/Landmarks_64x64',
        predictor_path='shape_predictor_68_face_landmarks.dat',
        face_size=64,        #  Main size is now configurable.
        output_size=32,       # Small size
        line_width=2,
        expected_length=640,
        detect_size=224       # Size used for detection
    )
    
    # Print summary
    print_landmarks_summary('./Aligned_data/Landmarks_128x128')
    
    # Visualize one subject if the file exists
    sample_landmarks_file = './Aligned_data/Landmarks_128x128/1_landmarks.pkl'
    if os.path.exists(sample_landmarks_file):
        print(f"\nVisualize landmark results for subject 1:")
        visualize_landmarks_results(sample_landmarks_file, num_display=6, start_index=0)
        
        # Example of loading arrays
        main_array, small_array, filenames, log, face_size = load_landmarks_arrays(sample_landmarks_file)
        print(f"\nArray loading example:")
        print(f"Main array: {main_array.shape}, dtype: {main_array.dtype}")
        print(f"Small array: {small_array.shape}, dtype: {small_array.dtype}")
        print(f"Face size: {face_size}x{face_size}")
        print(f"Number of filenames: {len(filenames)}")