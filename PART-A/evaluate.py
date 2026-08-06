"""
Loads the trained model and checks how well it does on images it has
never seen before (the validation set). Run this after train.py.
"""

from ultralytics import YOLO

TRAINED_WEIGHTS = "/kaggle/working/runs/pneumonia_baseline/weights/best.pt"


def main():
    model = YOLO(TRAINED_WEIGHTS)

    # runs the model on the val set and computes precision, recall, and mAP
    metrics = model.val()

    print("precision:", metrics.box.p.mean())
    print("recall:", metrics.box.r.mean())
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()
