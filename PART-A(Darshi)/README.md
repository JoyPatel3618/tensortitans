# Part A - Pneumonia Bounding Box Detection

**Owner:** Darshi
**Model:** YOLOv8n (Ultralytics)
**Dataset:** RSNA Pneumonia Detection Challenge (Kaggle)

## What this does
Takes a chest X-ray and draws a bounding box around any region that looks
like pneumonia, with a confidence score.

- Input: chest X-ray image (converted from DICOM to PNG)
- Output: bounding box coordinates + confidence score

## Why YOLOv8
Compared against Faster R-CNN and SSD. Faster R-CNN is generally more
accurate but much slower and heavier to train. YOLO trades
a small amount of accuracy for a big speed/resource win, and modern
versions like YOLOv8 have closed most of that accuracy gap. Prior work
training YOLOv3 on this exact RSNA dataset beat the official competition's
top leaderboard score, which was a strong signal that the YOLO family
works well on this specific problem. YOLOv8 specifically because it's the
current stable, actively maintained version, with an anchor-free detection
head (less manual tuning needed) and an easier training pipeline via the
`ultralytics` package.

## How to get the data
Uses the RSNA Pneumonia Detection Challenge dataset. It is NOT included in
this repo (too large for git, ~4GB).

To reproduce:
1. Join the competition on Kaggle: kaggle.com/c/rsna-pneumonia-detection-challenge
   (free, just accept the rules once)
2. Open/copy this repo's code into a new Kaggle Notebook created directly
   from the competition page - the dataset auto-attaches at
   `/kaggle/input/competitions/rsna-pneumonia-detection-challenge/`
3. Or, to get it on a local machine instead:
   ```bash
   pip install kaggle
   kaggle competitions download -c rsna-pneumonia-detection-challenge
   ```

## How to run (in a Kaggle notebook, GPU on)

Run in this exact order, every session (Kaggle wipes `/kaggle/working/`
each time the session restarts, so all 4 steps need a fresh run - just
re-running `train.py` alone after a restart will fail with a
FileNotFoundError):

1. `data_prep/dicom_to_yolo.py` - converts raw DICOM + CSV labels into
   YOLO's image + label format
2. `data_prep/train_val_split.py` - splits data into train/val so we can
   test on images the model hasn't seen
3. `data_prep/make_dataset_yaml.py` - (re)creates `dataset.yaml` at
   `/kaggle/working/dataset.yaml`, which `train.py` needs to find the data
4. `train.py` - trains the YOLOv8 model
5. `evaluate.py` - checks mAP/precision/recall on the validation set

Each script has its paths set for Kaggle's `/kaggle/input` and
`/kaggle/working` folders - copy the code into notebook cells (or `!python
script.py` if uploaded as files) and run in order. Easiest: use Kaggle's
"Run All" so you never accidentally skip a step.


## Status (Day 1 baseline)
See `results/metrics.md` for numbers and `results/sample_predictions/` for
example detections. This is a small-subset baseline (2000 images, 15
epochs) to prove the pipeline works end to end, not the final model.

## Next steps
- Train on the full dataset, more epochs
- Try YOLOv8s/m and compare
- Implement the RSNA-specific mAP metric (IoU averaged 0.4-0.75) for a
  fair leaderboard comparison
- Set up DVC remote storage for the dataset and model weights
