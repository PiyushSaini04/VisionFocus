## Posture Analysis and Engagement Monitoring System (Dual Pipeline)

### What this is
- **Pipeline A (Geometry)**: MediaPipe Tasks API (Pose + Face Landmarker) → scale-invariant features → classifier for labels **0–3**.
- **Pipeline B (Object Detection)**: Pascal VOC XML → YOLO format → YOLOv8n fine-tuned phone detector for label **4**.
- **Fusion (Inference)**: Both pipelines run in one webcam loop; deterministic conflict resolution + temporal smoothing.

### Label mapping (authoritative)
```
0 = Engaged
1 = Bored
2 = Drowsy
3 = LookingAway
4 = UsingPhone
```

### Setup
Install with pip (Python 3.10):
```bash
pip install -r requirements.txt
```

If `mediapipe==0.10.14` fails to install via pip on your system, use conda:
```bash
conda env create -f conda_env.yml
conda activate posture-engagement
```

### MediaPipe model files (required)
Download these `.task` files from the MediaPipe Model Zoo and place them in `project/` (same folder as scripts):
- `pose_landmarker_lite.task`
- `face_landmarker.task`

### Data layout
Put static training images here (classes 0–3 only):
- `data/images/Engaged/`
- `data/images/Bored/`
- `data/images/Drowsy/`
- `data/images/LookingAway/`

Phone images + Pascal VOC XML annotations:
- `data/images/Phone/` (images)
- `data/xml/` (matching `.xml`)

### Run the full training pipeline
From `project/`:
```bash
python extract_mediapipe_features.py
python convert_xml_to_yolo.py
python train_geometry_models.py
python train_yolo.py
```

Or on bash shells:
```bash
bash run_pipeline.sh
```

### Important: set `EXPECTED_FEATURE_DIM`
After running `extract_mediapipe_features.py`, set `EXPECTED_FEATURE_DIM` in `constants.py` to the printed feature count. Training and realtime inference will refuse to run if this is unset/mismatched.

### Start realtime monitoring
From `project/`:
```bash
python realtime_monitor.py
```
Press **q** to quit. A log is written to `session_log.csv`.

### Known limitations
- **Single-student scope**: the system assumes one student in frame. If multiple people appear, results are undefined. Crop to a single-student ROI in multi-person settings.
- **Static-to-webcam domain gap**: static training images rarely match webcam conditions. After deployment, collect 50–100 real webcam frames per class, append to your dataset, and retrain.
- **GPU strongly recommended**: on CPU, combined YOLO + MediaPipe latency typically reduces FPS. The cadence constants in `constants.py` are tuned for CPU; with CUDA, reduce them to 1.
- **EAR threshold is approximate**: `EAR_DROWSY_THRESHOLD=0.22` varies per-person (glasses/eye shape). Recalibrate using per-person EAR distributions.

