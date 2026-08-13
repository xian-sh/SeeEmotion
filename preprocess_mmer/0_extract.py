import os
import pickle
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io


def load_and_process_mat_data(mat_path):
    """
    Load and process EEG data from a .mat file.
    """
    mat_data = scipy.io.loadmat(mat_path)

    eeg_datas = mat_data["eeg_datas"]  # 20 x N
    video_ids = eeg_datas[18]  # Row 19: video IDs
    time_info = eeg_datas[19]  # Row 20: time information
    eeg_signals = eeg_datas[:18]  # First 18 rows: EEG signals

    fs_eeg = mat_data["fs_eeg"][0, 0]  # Sampling rate

    print(
        f"Original data shape: EEG signals {eeg_signals.shape}, "
        f"video IDs {video_ids.shape}, time information {time_info.shape}"
    )
    print(f"Sampling rate: {fs_eeg} Hz")

    return eeg_signals, video_ids, time_info, fs_eeg


def segment_eeg_by_video_trial(eeg_signals, video_ids, time_info, fs, valid_video_ids=None):
    """
    Segment EEG data by video ID and time information.
    """
    if valid_video_ids is None:
        valid_video_ids = list(range(32))

    valid_mask = np.isin(video_ids, valid_video_ids)

    valid_eeg = eeg_signals[:, valid_mask]
    valid_video_ids_filtered = video_ids[valid_mask]
    valid_time_info = time_info[valid_mask]

    segmented_data = {}

    for video_id in valid_video_ids:
        video_mask = valid_video_ids_filtered == video_id
        video_indices = np.where(video_mask)[0]

        if len(video_indices) == 0:
            print(f"Video ID {video_id}: no data")
            segmented_data[video_id] = None
            continue

        video_time = valid_time_info[video_indices]
        time_change_points = np.where(video_time >= 0)[0]

        if len(time_change_points) == 0:
            print(f"Video ID {video_id}: no valid time information")
            segmented_data[video_id] = None
            continue

        segments = []
        current_segment_start = 0

        for i in range(1, len(time_change_points)):
            current_point = time_change_points[i]
            prev_point = time_change_points[i - 1]

            if video_time[current_point] - video_time[prev_point] > 1.5:
                segment_indices = video_indices[current_segment_start:prev_point + 1]
                if len(segment_indices) > 0:
                    segment_data = valid_eeg[:, segment_indices]
                    segments.append(segment_data)
                current_segment_start = current_point

        if current_segment_start < len(video_indices):
            segment_indices = video_indices[current_segment_start:]
            if len(segment_indices) > 0:
                segment_data = valid_eeg[:, segment_indices]
                segments.append(segment_data)

        if segments:
            longest_segment = max(segments, key=lambda x: x.shape[1])
            segmented_data[video_id] = longest_segment
        else:
            segmented_data[video_id] = None
            print(f"Video ID {video_id}: failed to extract a valid segment")

    return segmented_data


def extract_first_20s_eeg_data(segmented_data, fs, target_duration=20):
    """
    Extract the first 20 seconds of EEG data for each video.
    """
    target_points = int(target_duration * fs)
    eeg_20s = np.zeros((32, 18, target_points))

    missing_videos = []

    for video_id in range(32):
        if video_id in segmented_data and segmented_data[video_id] is not None:
            data = segmented_data[video_id]

            if data.shape[1] >= target_points:
                eeg_20s[video_id] = data[:, :target_points]
            else:
                print(
                    f"Warning: EEG data for video ID {video_id} is shorter than "
                    f"{target_duration} seconds. Actual duration: {data.shape[1] / fs:.2f} seconds"
                )
                eeg_20s[video_id, :, :data.shape[1]] = data
        else:
            missing_videos.append(video_id)
            print(f"Warning: video ID {video_id} has no valid EEG data")

    if missing_videos:
        print(f"Video IDs with missing EEG data: {missing_videos}")

    return eeg_20s


def cut_video_first_20s_opencv(input_video_path, output_video_path, target_duration=20):
    """
    Cut the first 20 seconds of a video using OpenCV.

    Args:
        input_video_path: Input video path.
        output_video_path: Output video path.
        target_duration: Target duration in seconds.
    """
    try:
        cap = cv2.VideoCapture(input_video_path)

        if not cap.isOpened():
            print(f"Failed to open video file: {input_video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_duration = total_frames / fps

        target_frames = int(target_duration * fps)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frames_written = 0

        for _ in range(min(target_frames, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            out.write(frame)
            frames_written += 1

        cap.release()
        out.release()

        actual_duration = frames_written / fps

        if actual_duration >= target_duration:
            print(
                f"Video cut successfully: {input_video_path} -> "
                f"{output_video_path} ({actual_duration:.2f} seconds)"
            )
        else:
            print(
                f"Warning: video {input_video_path} is shorter than target duration. "
                f"Only {actual_duration:.2f} seconds were cut"
            )

        return True

    except Exception as e:
        print(f"Failed to cut video {input_video_path}: {e}")
        return False


def check_video_duration(video_path):
    """
    Check video duration.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        return duration
    except Exception:
        return 0


def process_all_videos_for_subject(subject_id, target_duration=20):
    """
    Process all 32 videos for one subject.
    """
    video_dir = f"Aligned_data/{subject_id}"
    output_dir = f"Aligned_data/{subject_id}/videos_20s"
    os.makedirs(output_dir, exist_ok=True)

    processed_videos = []
    missing_videos = []
    short_videos = []

    for video_id in range(32):
        input_video_path = os.path.join(video_dir, f"{video_id}.mp4")
        output_video_path = os.path.join(output_dir, f"{video_id}_20s.mp4")

        if os.path.exists(input_video_path):
            duration = check_video_duration(input_video_path)
            if duration < target_duration:
                short_videos.append((video_id, duration))
                print(
                    f"Warning: video {video_id} is shorter than {target_duration} seconds. "
                    f"Actual duration: {duration:.2f} seconds"
                )

            if cut_video_first_20s_opencv(input_video_path, output_video_path, target_duration):
                processed_videos.append(video_id)
            else:
                missing_videos.append(video_id)
        else:
            print(f"Video file does not exist: {input_video_path}")
            missing_videos.append(video_id)

    print(f"Subject {subject_id}: successfully processed {len(processed_videos)} videos")
    if missing_videos:
        print(f"Missing video IDs: {missing_videos}")
    if short_videos:
        print(f"Short video IDs: {[vid for vid, _ in short_videos]}")

    return processed_videos, missing_videos, short_videos


def save_eeg_data_as_pkl(eeg_data, output_path):
    """
    Save EEG data as a pkl file.
    """
    with open(output_path, "wb") as f:
        pickle.dump(eeg_data, f)
    print(f"EEG data saved to: {output_path}")


def analyze_eeg_data(eeg_data, fs):
    """
    Analyze EEG data quality.
    """
    print(f"\nEEG data shape: {eeg_data.shape}")
    print(f"Number of videos: {eeg_data.shape[0]}")
    print(f"Number of EEG channels: {eeg_data.shape[1]}")
    print(f"Data points per video: {eeg_data.shape[2]}")
    print(f"Duration per video: {eeg_data.shape[2] / fs:.2f} seconds")

    zero_videos = []
    for vid in range(eeg_data.shape[0]):
        if np.all(eeg_data[vid] == 0):
            zero_videos.append(vid)

    if zero_videos:
        print(f"All-zero video IDs (missing): {zero_videos}")
    else:
        print("All 32 videos have valid data")


mat_list = [1, 5, 11, 12, 19, 20, 22, 23, 24, 25, 29, 32, 33, 38]
fs = 300
time_window = 20
target_points = int(time_window * fs)


try:
    import cv2

    print("OpenCV is available. Start processing...")
except ImportError:
    print("Error: OpenCV is not installed. Please run: pip install opencv-python")
    exit(1)


for m in mat_list:
    print(f"\n{'=' * 60}")
    print(f"Processing subject {m}")
    print(f"{'=' * 60}")

    mat_path = f"Aligned_data/{m}/datas.mat"

    if not os.path.exists(mat_path):
        print(f"File does not exist: {mat_path}")
        continue

    try:
        print("Step 1: Processing EEG data...")
        eeg_signals, video_ids, time_info, fs_actual = load_and_process_mat_data(mat_path)

        if fs_actual != fs:
            print(f"Warning: actual sampling rate {fs_actual} Hz differs from expected {fs} Hz")

        segmented_data = segment_eeg_by_video_trial(eeg_signals, video_ids, time_info, fs_actual)
        eeg_20s = extract_first_20s_eeg_data(segmented_data, fs_actual, time_window)
        analyze_eeg_data(eeg_20s, fs_actual)

        eeg_output_path = f"Aligned_data/{m}/eeg_20s.pkl"
        save_eeg_data_as_pkl(eeg_20s, eeg_output_path)

        print("\nStep 2: Processing video data...")
        processed_videos, missing_videos, short_videos = process_all_videos_for_subject(
            m, time_window
        )

        info = {
            "subject_id": m,
            "eeg_shape": eeg_20s.shape,
            "fs": fs_actual,
            "duration_seconds": time_window,
            "processed_videos": processed_videos,
            "missing_videos": missing_videos,
            "short_videos": short_videos,
            "total_videos": len(processed_videos) + len(missing_videos),
        }

        with open(f"Aligned_data/{m}/processing_info.pkl", "wb") as f:
            pickle.dump(info, f)

        print(f"\nSubject {m} processing completed")
        print(f"EEG data: {eeg_20s.shape}")
        print(f"Processed videos: {len(processed_videos)}")
        print(f"Missing videos: {len(missing_videos)}")
        print(f"Short videos: {len(short_videos)}")

        if len(processed_videos) > 0:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            valid_count = 0
            for vid in range(min(3, eeg_20s.shape[0])):
                if not np.all(eeg_20s[vid] == 0):
                    display_points = min(1500, eeg_20s.shape[2])
                    plt.plot(eeg_20s[vid, 0, :display_points], label=f"Video {vid}")
                    valid_count += 1
            if valid_count > 0:
                plt.title(f"Subject {m} - EEG Signal Preview")
                plt.xlabel("Data Point")
                plt.ylabel("Amplitude")
                plt.legend()

            plt.subplot(1, 2, 2)
            labels = ["Processed", "Missing", "Short"]
            sizes = [len(processed_videos), len(missing_videos), len(short_videos)]
            colors = ["lightgreen", "lightcoral", "orange"]
            plt.pie(
                [s for s in sizes if s > 0],
                labels=[label for i, label in enumerate(labels) if sizes[i] > 0],
                colors=[color for i, color in enumerate(colors) if sizes[i] > 0],
                autopct="%1.1f%%",
                startangle=90,
            )
            plt.title("Video Processing Statistics")

            plt.tight_layout()
            plt.show()
        else:
            print("No valid video data. Skip visualization.")

    except Exception as e:
        print(f"Error while processing subject {m}: {e}")
        traceback.print_exc()
        continue


print("\nAll processing completed")


def print_final_statistics(mat_list):
    """
    Print final processing statistics.
    """
    print(f"\n{'=' * 60}")
    print("Final Processing Statistics")
    print(f"{'=' * 60}")

    total_subjects = len(mat_list)
    successful_subjects = 0

    for m in mat_list:
        info_path = f"Aligned_data/{m}/processing_info.pkl"
        if os.path.exists(info_path):
            with open(info_path, "rb") as f:
                info = pickle.load(f)
            successful_subjects += 1
            print(f"Subject {m}: {info['processed_videos']} videos processed successfully")
        else:
            print(f"Subject {m}: failed or not processed")

    print(f"\nSummary: {successful_subjects}/{total_subjects} subjects processed successfully")


print_final_statistics(mat_list)