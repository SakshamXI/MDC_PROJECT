"""Fine-tune a pretrained YOLO on WIDER FACE (run wider2yolo.py first).

  pip install ultralytics
  python train_yolo.py
"""
from ultralytics import YOLO

# ---------------- settings ----------------
DATA = "datasets/wider_yolo/data.yaml"
MODEL = "yolov8n.pt"
EPOCHS = 60
IMGSZ = 640
BATCH = 16
DEVICE = None          # None = auto, or 0 for first GPU, or "cpu"
# ------------------------------------------

if __name__ == "__main__":
    model = YOLO(MODEL)
    model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        patience=15,
        project="runs_face",
        name="yolo_face",
    )
    metrics = model.val()
    print("mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map)
    # best weights: runs_face/yolo_face/weights/best.pt
