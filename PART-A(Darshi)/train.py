"""
Trains a YOLOv8 model on the converted pneumonia dataset.
Run this after data_prep/train_val_split.py AND data_prep/make_dataset_yaml.py
(dataset.yaml must exist at /kaggle/working/dataset.yaml before this runs).
"""

from ultralytics import YOLO

DATASET_CONFIG = "/kaggle/working/dataset.yaml"

# nano is the smallest/fastest YOLOv8 model - good choice for a same-day
# baseline. bigger models (yolov8s.pt, yolov8m.pt) are more accurate but
# take longer to train - try those later once the pipeline is proven
BASE_MODEL = "yolov8n.pt"

EPOCHS = 15       # kept small today to finish training in time - raise to 50-100 later
IMAGE_SIZE = 640
BATCH_SIZE = 16


def main():
    model = YOLO(BASE_MODEL)

    model.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project="/kaggle/working/runs",
        name="pneumonia_baseline",
    )


if __name__ == "__main__":
    main()
