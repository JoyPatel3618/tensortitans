"""
Moves a slice of the converted images+labels into a val/ folder,
so we have unseen data to test the model on after training.

Run this after dicom_to_yolo.py.
"""

import os
import random
import shutil

TRAIN_IMG_DIR = "/kaggle/working/dataset/images/train/"
TRAIN_LBL_DIR = "/kaggle/working/dataset/labels/train/"
VAL_IMG_DIR = "/kaggle/working/dataset/images/val/"
VAL_LBL_DIR = "/kaggle/working/dataset/labels/val/"

VAL_FRACTION = 0.15  # 15% of images go to validation, 85% stay for training


def main():
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_LBL_DIR, exist_ok=True)

    all_images = os.listdir(TRAIN_IMG_DIR)

    # fixed seed so the split is the same every time we run this -
    # makes results reproducible instead of random each run
    random.seed(42)
    random.shuffle(all_images)

    val_count = int(VAL_FRACTION * len(all_images))
    val_images = all_images[:val_count]

    for img_name in val_images:
        base_name = img_name.replace(".png", "")
        shutil.move(TRAIN_IMG_DIR + img_name, VAL_IMG_DIR + img_name)
        shutil.move(TRAIN_LBL_DIR + base_name + ".txt", VAL_LBL_DIR + base_name + ".txt")

    print("train:", len(os.listdir(TRAIN_IMG_DIR)))
    print("val:", len(os.listdir(VAL_IMG_DIR)))


if __name__ == "__main__":
    main()
