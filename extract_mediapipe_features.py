import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from constants import (
    FACE_2D_IDS,
    FACE_3D_MODEL_PTS,
    GEOMETRY_CSV_PATH,
    LEFT_EYE,
    RIGHT_EYE,
)


def extract_features(pose_lm, face_lm, img_w, img_h):
    """
    pose_lm : list of NormalizedLandmark (from PoseLandmarker result)
    face_lm : list of NormalizedLandmark (from FaceLandmarker result)
    Returns a 1D float32 numpy array of scale-invariant features.
    """
    features = []

    # --- Scale reference ---
    l_sh = np.array([pose_lm[11].x, pose_lm[11].y], dtype=np.float32)
    r_sh = np.array([pose_lm[12].x, pose_lm[12].y], dtype=np.float32)
    shoulder_dist = float(np.linalg.norm(l_sh - r_sh) + 1e-6)

    # --- Inter-landmark distances normalised by shoulder width ---
    def dist(a, b):
        pa = np.array([pose_lm[a].x, pose_lm[a].y], dtype=np.float32)
        pb = np.array([pose_lm[b].x, pose_lm[b].y], dtype=np.float32)
        return float(np.linalg.norm(pa - pb) / shoulder_dist)

    nose_to_l_ear = dist(0, 7)
    nose_to_r_ear = dist(0, 8)
    l_ear_to_l_sh = dist(7, 11)
    r_ear_to_r_sh = dist(8, 12)
    l_sh_to_r_sh = dist(11, 12)
    l_elbow_to_l_sh = dist(13, 11)
    r_elbow_to_r_sh = dist(14, 12)
    l_wrist_to_l_sh = dist(15, 11)
    r_wrist_to_r_sh = dist(16, 12)

    features.extend(
        [
            nose_to_l_ear,
            nose_to_r_ear,
            l_ear_to_l_sh,
            r_ear_to_r_sh,
            l_sh_to_r_sh,
            l_elbow_to_l_sh,
            r_elbow_to_r_sh,
            l_wrist_to_l_sh,
            r_wrist_to_r_sh,
        ]
    )

    # --- Joint angles (cosine, scale-free) ---
    def angle(a_idx, b_idx, c_idx):
        a = np.array([pose_lm[a_idx].x, pose_lm[a_idx].y], dtype=np.float32)
        b = np.array([pose_lm[b_idx].x, pose_lm[b_idx].y], dtype=np.float32)
        c = np.array([pose_lm[c_idx].x, pose_lm[c_idx].y], dtype=np.float32)
        ba, bc = a - b, c - b
        cos_a = float(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6))
        return float(np.clip(cos_a, -1.0, 1.0))

    l_elbow_angle = angle(15, 13, 11)  # wrist–elbow–shoulder
    r_elbow_angle = angle(16, 14, 12)
    l_sh_tilt = angle(7, 11, 12)  # ear–shoulder–shoulder (lean)
    r_sh_tilt = angle(8, 12, 11)

    features.extend([l_elbow_angle, r_elbow_angle, l_sh_tilt, r_sh_tilt])

    # --- Eye Aspect Ratio (EAR) ---
    def ear(eye_ids):
        p = [np.array([face_lm[i].x, face_lm[i].y], dtype=np.float32) for i in eye_ids]
        a = float(np.linalg.norm(p[1] - p[5]))
        b = float(np.linalg.norm(p[2] - p[4]))
        c = float(np.linalg.norm(p[0] - p[3]))
        return (a + b) / (2.0 * c + 1e-6)

    mean_ear = (ear(LEFT_EYE) + ear(RIGHT_EYE)) / 2.0
    features.append(float(mean_ear))

    # --- Head pose via solvePnP (pitch, yaw, roll in degrees) ---
    pts_2d = np.array(
        [[face_lm[i].x * img_w, face_lm[i].y * img_h] for i in FACE_2D_IDS],
        dtype=np.float64,
    )
    focal = float(img_w)
    cam_matrix = np.array(
        [[focal, 0, img_w / 2], [0, focal, img_h / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        FACE_3D_MODEL_PTS,
        pts_2d,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        pitch = yaw = roll = 0.0
    else:
        rmat, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])

    features.extend([pitch, yaw, roll])

    return np.array(features, dtype=np.float32)


def main():
    logging.basicConfig(filename="failed_detections.log", level=logging.WARNING)

    # --- MediaPipe Tasks API initialisation (NO legacy mp.solutions) ---
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path="pose_landmarker_lite.task"),
            output_segmentation_masks=False,
            num_poses=1,
        )
    )
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path="face_landmarker.task"),
            num_faces=1,
        )
    )

    class_dirs = {
        "Engaged": 0,
        "Bored": 1,
        "Drowsy": 2,
        "LookingAway": 3,
    }

    rows = []
    failed_frames = []
    total = 0

    for class_name, label in class_dirs.items():
        image_dir = Path(f"data/images/{class_name}")
        if not image_dir.exists():
            continue

        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            img_paths.extend(sorted(image_dir.glob(ext)))

        for img_path in img_paths:
            total += 1
            frame = cv2.imread(str(img_path))
            if frame is None:
                logging.warning(f"Detection failed: {img_path} — unreadable image — skipped")
                failed_frames.append(str(img_path))
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result_pose = pose_landmarker.detect(mp_image)
            result_face = face_landmarker.detect(mp_image)

            if not result_pose.pose_landmarks or not result_face.face_landmarks:
                logging.warning(f"Detection failed: {img_path} — skipped")
                failed_frames.append(str(img_path))
                continue

            feats = extract_features(
                result_pose.pose_landmarks[0],
                result_face.face_landmarks[0],
                frame.shape[1],
                frame.shape[0],
            )
            rows.append(list(feats) + [label])

    if not rows:
        print("No rows extracted. Check that images exist under data/images/{Engaged,Bored,Drowsy,LookingAway}.")
        print(f"Skipped {len(failed_frames)} / {total} frames")
        return

    n_features = len(rows[0]) - 1
    col_names = [f"f{i}" for i in range(n_features)] + ["label"]
    df = pd.DataFrame(rows, columns=col_names)
    out_path = Path(GEOMETRY_CSV_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows, {n_features} features to {GEOMETRY_CSV_PATH}")
    print(f"Class distribution:\n{df['label'].value_counts().sort_index()}")
    print(f"Skipped {len(failed_frames)} / {total} frames")
    print(f"Set EXPECTED_FEATURE_DIM = {n_features} in constants.py after verifying this count.")


if __name__ == "__main__":
    main()

