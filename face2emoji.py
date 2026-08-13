import cv2
import dlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle


# Draw an abstract facial expression using landmarks only (including the nose and contour)
def draw_stick_emoji(landmarks, out_res=8, line_width=2):
    image = np.zeros((56, 56), dtype=np.uint8)
    # Face contour
    contour = landmarks[0:17]
    cv2.polylines(image, [contour], False, 255, line_width)
    # Eyebrows
    left_brow = landmarks[17:22]
    right_brow = landmarks[22:27]
    cv2.polylines(image, [left_brow], False, 255, line_width)
    cv2.polylines(image, [right_brow], False, 255, line_width)
    # Nose (draw both the bridge and lower part)
    nose_bridge = landmarks[27:31]    # From between the eyebrows to the nose tip
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
    # Downsample
    small_img = cv2.resize(image, (out_res, out_res), interpolation=cv2.INTER_NEAREST)
    return image, small_img

# Detect facial landmarks directly from a NumPy array
def extract_landmarks_from_np(img_np, detector, predictor):
    # Convert to grayscale
    if img_np.ndim == 3:
        if img_np.shape[2] == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np[:,:,0]
    else:
        gray = img_np
    # Resize to 224x224 for face detection
    big_gray = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_CUBIC)
    faces = detector(big_gray)
    if len(faces)==0:
        return None, big_gray, None
    shape = predictor(big_gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    # Scale the landmarks back to 56x56
    landmarks_small = (landmarks * (56 / 224)).astype(np.int32)
    return landmarks_small, big_gray, landmarks

# Visualize the original image, landmarks, and abstract emoji
def visualize_all_steps(frames, emojis, big_grays, landmarkss, idxs=None):
    if idxs is None:
        idxs = range(min(6, len(frames)))
    for i in idxs:
        plt.figure(figsize=(15,3))
        # Step 0: Original RGB image
        plt.subplot(1,5,1)
        img = frames[i]
        # Check whether the image is grayscale or RGB
        if img.ndim == 3 and img.shape[2] == 3:
            plt.imshow(img.astype(np.uint8))
        else:
            plt.imshow(img, cmap='gray')
        plt.title('Original RGB')
        plt.axis('off')
        # Step 1: Enlarged grayscale face
        plt.subplot(1,5,2)
        plt.imshow(big_grays[i], cmap='gray')
        plt.title('224x224 Gray')
        plt.axis('off')
        # Step 2: Plot the 68 facial landmarks
        plt.subplot(1,5,3)
        plt.imshow(big_grays[i], cmap='gray')
        if landmarkss[i] is not None:
            pts = landmarkss[i]
            plt.scatter(pts[:,0], pts[:,1], s=10, c='r')
        plt.title('Landmarks')
        plt.axis('off')
        # Step 3: 56x56 abstract facial line drawing
        plt.subplot(1,5,4)
        plt.imshow(255-emojis[i][0], cmap='gray')
        plt.title('Stick Emoji 56x56')
        plt.axis('off')
        # Step 4: Enlarged view of the final 8x8 image
        plt.subplot(1,5,5)
        plt.imshow(255-emojis[i][1], cmap='gray', vmin=0, vmax=255)
        plt.title('Emoji 8x8 enlarged')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Fill failed detections from the temporally nearest valid frame in the same clip
def fill_missing_emojis_with_nearest(emojis, output_size):
    valid_indices = [idx for idx, item in enumerate(emojis) if item is not None]

    # Keep zero-valued features when no frame in the clip has valid landmarks
    if not valid_indices:
        return [
            (
                np.zeros((56, 56), dtype=np.uint8),
                np.zeros((output_size, output_size), dtype=np.uint8),
            )
            for _ in emojis
        ]

    filled_emojis = list(emojis)
    for missing_idx, item in enumerate(filled_emojis):
        if item is not None:
            continue

        # Use frame-index distance as temporal distance; prefer the earlier
        # valid frame when the previous and next candidates are equally close.
        source_idx = min(
            valid_indices,
            key=lambda valid_idx: (abs(valid_idx - missing_idx), valid_idx),
        )
        source_stick, source_emoji = emojis[source_idx]
        filled_emojis[missing_idx] = (
            source_stick.copy(),
            source_emoji.copy(),
        )

    return filled_emojis


# Batch processing
def batch_faces2emojis(frames, output_size=8, predictor_path='shape_predictor_68_face_landmarks.dat'):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    emojis = [None] * len(frames)
    big_grays = [None] * len(frames)
    landmarkss = [None] * len(frames)

    # First pass: extract landmarks independently for every frame
    for i, img in enumerate(frames):
        try:
            landmarks, big_gray, landmarks_big = extract_landmarks_from_np(img, detector, predictor)
            big_grays[i] = big_gray
            landmarkss[i] = landmarks_big
            if landmarks is not None:
                stick, emoji = draw_stick_emoji(landmarks, out_res=output_size, line_width=2)
                emojis[i] = (stick, emoji)
        except Exception as e:
            print(f"Frame {i} error:", e)
            big_grays[i] = np.zeros((224, 224), dtype=np.uint8)

    # Second pass: compensate failed detections using the nearest valid frame
    # within this clip only, preventing information leakage across samples.
    emojis = fill_missing_emojis_with_nearest(emojis, output_size)

    return emojis, big_grays, landmarkss


def get_emojis(video, output_size=8, predictor_path='shape_predictor_68_face_landmarks.dat'):
    # video: (N, 25, 56, 56, 3)
    all_emojis = []
    for i in range(len(video)):
        emojis, _, _ = batch_faces2emojis(video[i], output_size=output_size, predictor_path=predictor_path)
        # Keep only the downsampled images
        # emojis: list of (56x56, output_size x output_size)
        emoji_arr = np.stack([item[1] for item in emojis], axis=0)  # (25, out, out)
        all_emojis.append(emoji_arr)
    return np.stack(all_emojis, axis=0)  # (N, 25, out, out)


# Extract small-face features for 42 subjects
for sub in range (1,43):
    print(f"Start process subject {sub:02d}...")
    vis = pickle.load(open(fr'/EAV/data/Inputs/Vision/subject_{sub:02d}_vis.pkl', 'rb'))
    video_train = vis[0]  # (280, 25, 56, 56, 3)
    video_test = vis[2]   # (120, 25, 56, 56, 3)

    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    emojis_train = get_emojis(video_train, output_size=56, predictor_path=predictor_path)  # (280, 25, 56, 56)
    emojis_test = get_emojis(video_test, output_size=56, predictor_path=predictor_path)    # (120, 25, 56, 56)

    # Preserve the same structure as vis
    emoji_vis = [emojis_train, vis[1], emojis_test, vis[3]]

    with open('output/face_emoji/emoji_vis_subject_%02d.pkl' % sub, 'wb') as f:
        pickle.dump(emoji_vis, f)

# Main program
# if __name__ == "__main__":
#     import pickle

#     subject = 1
#     vis = pickle.load(open(f'Input_images/Vision/subject_{subject:02d}_vis.pkl', 'rb'))
#     video_frames = vis[0][200][::2]  # For example, 25 frames, each sized 56x56 or 56x56x3

#     # Generate emojis in batches
#     emojis, big_grays, landmarkss = batch_faces2emojis(
#         video_frames, output_size=32, 
#         predictor_path='shape_predictor_68_face_landmarks.dat'
#     )

#     # Visualize the first six frames
#     visualize_all_steps(video_frames, emojis, big_grays, landmarkss, idxs=range(6))

#     # Display the first 12 final emojis (enlarged to 32x32)
#     fig, axes = plt.subplots(1, 12, figsize=(18, 2))
#     for i, ax in enumerate(axes):
#         ax.imshow(255-emojis[i][1], cmap='gray', vmin=0, vmax=255)
#         ax.set_title(f"Frame {i}")
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()

#     # Save frame 0
#     Image.fromarray(emojis[0][1]).save("emoji_0.png")
#     print("Saved emoji_0.png")
