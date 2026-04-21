import csv
import logging
from collections import deque
from datetime import datetime

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO

from constants import *
from extract_mediapipe_features import extract_features

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

from mediapipe.framework.formats import landmark_pb2

def convert_to_landmark_list(landmarks):
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    for lm in landmarks:
        landmark = landmark_list.landmark.add()
        landmark.x = lm.x
        landmark.y = lm.y
        landmark.z = lm.z
    return landmark_list

def resolve_state(phone_flag, geometry_pred, mean_ear, yaw_deg):
    """
    Returns an integer label (0–4) or None (do not update buffer).
    phone_flag     : bool — cached YOLO result
    geometry_pred  : int 0–3 from geometry classifier, or None if detection failed
    mean_ear       : float — current EAR value
    yaw_deg        : float — head yaw in degrees from solvePnP
    """
    if phone_flag and geometry_pred != 0:
        return 4

    if geometry_pred is None:
        return None

    if geometry_pred == 2:
        return 2 if (mean_ear < EAR_DROWSY_THRESHOLD) else 0

    if geometry_pred == 3:
        return 3 if (abs(yaw_deg) > HEAD_YAW_AWAY_DEG) else 0

    return int(geometry_pred)


def dominant_label(buffer):
    counts = {}
    for s in buffer:
        counts[s] = counts.get(s, 0) + 1
    return max(counts.items(), key=lambda x: x[1])


def main():
    logging.basicConfig(filename="realtime_warnings.log", level=logging.WARNING)

    if EXPECTED_FEATURE_DIM is None:
        raise ValueError(
            "EXPECTED_FEATURE_DIM is None in constants.py. "
            "Set it to match the feature count produced by extract_mediapipe_features.py."
        )

    geometry_model = joblib.load(GEOMETRY_MODEL_PATH)
    yolo_model = YOLO(YOLO_MODEL_PATH)

    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path="pose_landmarker_lite.task"),
            num_poses=1,
        )
    )
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path="face_landmarker.task"),
            num_faces=1,
        )
    )

    frame_counter = 0
    phone_flag_cache = False
    state_buffer = deque(maxlen=SMOOTHING_WINDOW)

    last_yolo_boxes = []
    last_pose_landmarks = None
    last_face_landmarks = None

    display_state = 0
    display_state_label = LABELS[0]
    mean_ear, yaw_deg = 1.0, 0.0
    proba = np.zeros(NUM_CLASSES, dtype=np.float32)

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Engagement Monitor", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Engagement Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)            
            
    session_log_file = open(SESSION_LOG_PATH, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(session_log_file)
    log_writer.writerow(["timestamp", "state_label", "confidence", "ear", "yaw_deg"])

    state_colours = {
        0: (0, 200, 80),  # Engaged
        1: (0, 165, 255),  # Bored
        2: (0, 0, 220),  # Drowsy
        3: (200, 140, 0),  # LookingAway
        4: (160, 0, 200),  # UsingPhone
    }

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_counter += 1

            # --- Pipeline B: YOLO — run every N frames, cache result ---
            if frame_counter % YOLO_EVERY_N_FRAMES == 0:
                yolo_results = yolo_model(frame, verbose=False, conf=0.60)[0]
                phone_flag_cache = False

                for box in last_yolo_boxes:
                    conf = float(box.conf[0])
                    if conf > 0.75:   # increase threshold
                        phone_flag_cache = True
                        break
                last_yolo_boxes = yolo_results.boxes

                # ✅ Draw YOLO boxes EVERY frame
            for box in last_yolo_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 0, 200), 2)
                cv2.putText(
                    frame,
                    f"Phone {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (160, 0, 200),
                    2,
                )

            # --- Pipeline A: MediaPipe + geometry classifier — run every N frames ---
            geometry_pred = None
            if frame_counter % MEDIAPIPE_EVERY_N_FRAMES == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                result_pose = pose_landmarker.detect(mp_image)
                result_face = face_landmarker.detect(mp_image)

                if result_pose.pose_landmarks:
                    last_pose_landmarks = result_pose.pose_landmarks[0]

                if result_face.face_landmarks:
                    last_face_landmarks = result_face.face_landmarks[0]
                
                if last_pose_landmarks and last_face_landmarks:
                    features = extract_features(
                        last_pose_landmarks,
                        last_face_landmarks,
                        frame.shape[1],
                        frame.shape[0],
                    )

                    if len(features) == EXPECTED_FEATURE_DIM:
                        geometry_pred = int(geometry_model.predict([features])[0])

                        try:
                            gp = geometry_model.predict_proba([features])[0]
                            proba[:] = 0.0
                            proba[:len(gp)] = gp
                        except:
                            pass

                        mean_ear = float(features[9 + 4])
                        yaw_deg = float(features[9 + 4 + 1 + 1])
                
            if last_pose_landmarks:
                pose_proto = convert_to_landmark_list(last_pose_landmarks)
                mp_drawing.draw_landmarks(
                    frame,
                    pose_proto,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )


                
            raw_state = resolve_state(phone_flag_cache, geometry_pred, mean_ear, yaw_deg)
            if raw_state is not None:
                state_buffer.append(raw_state)

            if state_buffer:
                dominant, dom_count = dominant_label(state_buffer)
                if dom_count / len(state_buffer) >= SMOOTHING_CONFIDENCE:
                    if dominant != display_state:
                        display_state = int(dominant)
                        display_state_label = LABELS[display_state]
                        confidence = float(proba[display_state]) if display_state < len(proba) else 0.0
                        log_writer.writerow(
                            [
                                datetime.now().isoformat(timespec="seconds"),
                                display_state_label,
                                f"{confidence:.3f}",
                                f"{mean_ear:.3f}",
                                f"{yaw_deg:.1f}",
                            ]
                        )
                        session_log_file.flush()

            # --- Configuration & Styling ---
            alpha = 0.75  # Transparency (0 to 1)
            bg_color = (20, 20, 20)  # Sleek dark background
            text_color = (240, 240, 240)
            accent_color = state_colours.get(display_state, (200, 200, 200))

            # Coordinates for the small, compact box
            x, y, w, h = 15, 15, 210, 65  # Narrower and shorter
            conf_pct = int(float(proba[display_state]) * 100) if display_state < len(proba) else 0

            # 1. Create Transparent Overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # 2. Draw Accent Border (Left side stripe for a modern look)
            cv2.rectangle(frame, (x, y), (x + 3, y + h), accent_color, -1)

            # 3. State Label (Compact)
            cv2.putText(
                frame, 
                display_state_label.upper(), 
                (x + 12, y + 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                accent_color, 
                1, 
                cv2.LINE_AA
            )

            # 4. EAR Value (Small, right-aligned in box)
            cv2.putText(
                frame, 
                f"EAR: {mean_ear:.2f}", 
                (x + 140, y + 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, 
                (180, 180, 180), 
                1, 
                cv2.LINE_AA
            )

            # 5. Professional Progress Bar
            bar_x, bar_y, bar_w, bar_h = x + 12, y + 38, 150, 6
            fill_w = int((conf_pct / 100.0) * bar_w)

            # Bar background (Track)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            # Bar fill
            if fill_w > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), accent_color, -1)

            # Percentage Text
            cv2.putText(
                frame, 
                f"{conf_pct}%", 
                (bar_x + bar_w + 8, bar_y + 8), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, 
                text_color, 
                1, 
                cv2.LINE_AA
            )

            # 6. Phone Detection (Very compact alert)
            if phone_flag_cache:
                # Small red pill-shaped alert below the main box
                cv2.rectangle(frame, (x, y + h + 5), (x + w, y + h + 22), (0, 0, 180), -1)
                cv2.putText(
                    frame, 
                    "PHONE DETECTED", 
                    (x + 50, y + h + 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, 
                    (255, 255, 255), 
                    1, 
                    cv2.LINE_AA
                )

            cv2.imshow("Engagement Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        session_log_file.close()
        cv2.destroyAllWindows()
        print(f"Session log saved to {SESSION_LOG_PATH}")


if __name__ == "__main__":
    main()

