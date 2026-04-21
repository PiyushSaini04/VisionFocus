LABELS = {0: "Engaged", 1: "Bored", 2: "Drowsy", 3: "LookingAway", 4: "UsingPhone"}
NUM_CLASSES = 5

# MediaPipe Face Mesh EAR landmark indices (MediaPipe 468-point mesh)
# These are NOT the same as dlib's 68-point indices — do not mix them
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Upper-body Pose landmark indices (waist-up only)
# 0=nose  2=l-eye  5=r-eye  7=l-ear  8=r-ear
# 11=l-shoulder  12=r-shoulder  13=l-elbow  14=r-elbow
# 15=l-wrist     16=r-wrist
# Landmarks 17+ (hands, hips, knees, feet) are excluded entirely
UPPER_BODY_POSE_IDS = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]

# Face Mesh solvePnP reference point indices
FACE_2D_IDS = [1, 152, 263, 33, 61, 291]

# 3D canonical face model (cm) for solvePnP — matched to FACE_2D_IDS order
import numpy as np

FACE_3D_MODEL_PTS = np.array(
    [
        [0.0, 0.0, 0.0],  # nose tip (1)
        [0.0, -330.0, -65.0],  # chin (152)
        [-225.0, 170.0, -135.0],  # left eye outer corner (263)
        [225.0, 170.0, -135.0],  # right eye outer corner (33)
        [-150.0, -150.0, -125.0],  # left mouth corner (61)
        [150.0, -150.0, -125.0],  # right mouth corner (291)
    ],
    dtype=np.float64,
)

# Thresholds
EAR_DROWSY_THRESHOLD = 0.22  # eyes considered closed below this
HEAD_YAW_AWAY_DEG = 30.0  # head turned > 30° = looking away

# Feature vector length — set this after running extract_mediapipe_features.py
# and update here so train and inference scripts both validate against it
EXPECTED_FEATURE_DIM = 17 # e.g. 17 — fill in after first extraction run

# Temporal smoothing
SMOOTHING_WINDOW = 15  # frames in the rolling buffer
SMOOTHING_CONFIDENCE = 0.60  # fraction of buffer needed to commit to a state

# Inference cadence (frames between each model run)
YOLO_EVERY_N_FRAMES = 5
MEDIAPIPE_EVERY_N_FRAMES = 2

# Output paths
GEOMETRY_CSV_PATH = "data/student_engagement_data.csv"
GEOMETRY_MODEL_PATH = "models/catboost_engagement.pkl"
YOLO_MODEL_PATH = "models/phone_yolov8n/weights/best.pt"
SESSION_LOG_PATH = "session_log.csv"

