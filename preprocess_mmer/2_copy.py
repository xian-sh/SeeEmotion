import os
import shutil
from pathlib import Path

def reorganize_face_images(mat_list, base_dir='Aligned_data', target_dir='Images'):
    """
    Reorganize the face image file structure.
    Rename each subject's face_images folder to {subject_id}_face_imgs and move it to the Images folder.
    
    Args:
    - mat_list: Subject ID list.
    - base_dir: Original data directory.
    - target_dir: Target Images directory.
    """
    
    # Create target Images directory
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")
    
    moved_subjects = []
    failed_subjects = []
    
    for subject_id in mat_list:
        source_face_dir = os.path.join(base_dir, str(subject_id), 'face_images')
        target_face_dir = os.path.join(target_dir, f'{subject_id}_face_imgs')
        
        print(f"\nProcessing subject {subject_id}...")
        print(f"Source directory: {source_face_dir}")
        print(f"Target directory: {target_face_dir}")
        
        # Check whether the source directory exists
        if not os.path.exists(source_face_dir):
            print(f"  Source directory does not exist. Skip.")
            failed_subjects.append(subject_id)
            continue
        
        # Check whether the source directory contains image files
        image_files = [f for f in os.listdir(source_face_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"  No image files found in source directory. Skip.")
            failed_subjects.append(subject_id)
            continue
        
        try:
            # If the target directory already exists, remove it first
            if os.path.exists(target_face_dir):
                print(f"  Target directory already exists and will be overwritten.")
                shutil.rmtree(target_face_dir)
            
            # Move and rename the directory
            shutil.move(source_face_dir, target_face_dir)
            
            # Verify the move result
            moved_files = [f for f in os.listdir(target_face_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"  Successfully moved {len(moved_files)}  images")
            moved_subjects.append(subject_id)
            
        except Exception as e:
            print(f"  Move failed: {e}")
            failed_subjects.append(subject_id)
    
    # Print summary
    print(f"\n{'='*60}")
    print("File Reorganization Summary")
    print(f"{'='*60}")
    print(f"Successfully processed subjects: {len(moved_subjects)}")
    print(f"Failed subjects: {len(failed_subjects)}")
    
    if moved_subjects:
        print(f"\nMoved subjects: {moved_subjects}")
    
    if failed_subjects:
        print(f"\nFailed subjects: {failed_subjects}")
    
    return moved_subjects, failed_subjects

def verify_reorganized_structure(mat_list, target_dir='Images'):
    """
    Verify the reorganized file structure.
    """
    print(f"\n{'='*60}")
    print("Verify the reorganized file structure.")
    print(f"{'='*60}")
    
    if not os.path.exists(target_dir):
        print(f"Target directory does not exist: {target_dir}")
        return
    
    total_images = 0
    valid_subjects = 0
    
    for subject_id in mat_list:
        face_dir = os.path.join(target_dir, f'{subject_id}_face_imgs')
        
        if os.path.exists(face_dir):
            image_files = [f for f in os.listdir(face_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_count = len(image_files)
            total_images += image_count
            valid_subjects += 1
            
            print(f"Subject {subject_id:2d}: {image_count:3d}  images - {face_dir}")
            
            # Show the first few filenames as examples
            if image_files:
                sample_files = image_files[:3]
                print(f"         Examples: {', '.join(sample_files)}")
        else:
            print(f"Subject {subject_id:2d}: Directory does not exist")
    
    print(f"\nSummary:")
    print(f"Valid subjects: {valid_subjects}/{len(mat_list)}")
    print(f"Total images: {total_images}")
    print(f"Average per subject: {total_images/valid_subjects:.1f} images" if valid_subjects > 0 else "")

def list_images_directory_structure(target_dir='./Aligned_data/Images'):
    """
    List the complete Images directory structure.
    """
    print(f"\n{'='*60}")
    print(f"Images Directory Structure")
    print(f"{'='*60}")
    
    if not os.path.exists(target_dir):
        print(f"Directory does not exist: {target_dir}")
        return
    
    subdirs = [d for d in os.listdir(target_dir) 
               if os.path.isdir(os.path.join(target_dir, d)) and d.endswith('_face_imgs')]
    
    subdirs.sort()
    
    print(f"Images/")
    for subdir in subdirs:
        subdir_path = os.path.join(target_dir, subdir)
        image_files = [f for f in os.listdir(subdir_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"- {subdir}/ ({len(image_files)} images)")

# Usage example
if __name__ == "__main__":
    # Assume mat_list is already available
    mat_list = [1,5,11,12,19,20,22,23,24,25,29,32,33,38]
    fs = 300  # Hz
    time_window = 20 # s
    
    print("Start reorganizing the face image file structure...")
    
    # Reorganize file structure
    moved_subjects, failed_subjects = reorganize_face_images(mat_list,target_dir='./Aligned_data/Images_0.1')
    
    # Verify results
    verify_reorganized_structure(mat_list)
    
    # Show directory structure
    list_images_directory_structure()
    
    print(f"\nFile reorganization completed.")
    print(f"All face images are now under the Images/ directory")