#!/bin/bash
set -e
echo "=== Step 1: Extract MediaPipe features (Pipeline A) ==="
python extract_mediapipe_features.py

echo "=== Step 2: Convert XML to YOLO format (Pipeline B) ==="
python convert_xml_to_yolo.py

echo "=== Step 3: Train geometry models ==="
python train_geometry_models.py

echo "=== Step 4: Fine-tune YOLOv8 ==="
python train_yolo.py

echo "=== All training complete. Run realtime_monitor.py to start. ==="

