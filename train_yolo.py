from ultralytics import YOLO

from constants import YOLO_MODEL_PATH


def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="data/phone_yolo/data.yaml",
        epochs=20,
        imgsz=640,
        batch=16,
        device="cpu",  # change to 0 for CUDA GPU
        project="models",
        name="phone_yolov8n",
        exist_ok=True,
        augment=True,
        mosaic=1.0,
        degrees=15,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    best = YOLO(YOLO_MODEL_PATH)
    metrics = best.val(data="data/phone_yolo/data.yaml", split="test")
    print(f"Test mAP50: {metrics.box.map50:.4f}")
    print(f"Test mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()

