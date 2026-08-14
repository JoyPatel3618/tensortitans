# Part C - Multi-modal Diagnosis with Grad-CAM

**Status:** Partial Day 1 baseline - built to unblock progress while the
owner was busy with other priorities. Honest scope below.

## What's actually working (the visual half)
- A DenseNet-121 multi-label classifier that predicts probabilities for
  all 14 NIH pathologies from an X-ray image alone.
- Grad-CAM heatmap generation - shows which region of the X-ray drove a
  given prediction, for explainability.

- Input: chest X-ray image
- Output: per-pathology probability + heatmap overlay for a chosen pathology

## What's NOT done yet (the "multi-modal" half)
The problem statement calls for combining DenseNet visual features with
**ClinicalBERT text embeddings** from clinical notes, then fusing both for
the final prediction. That text-fusion step is not implemented - it needs
clinical text data paired with each image, which hasn't been sourced or
confirmed yet. Right now this is an image-only classifier, not yet
"multi-modal" in the way the problem statement describes.

## How it works
1. `prepare_labels.py` - converts the NIH dataset's raw label text
   (e.g. "Cardiomegaly|Effusion") into a proper multi-label table
2. `train_classifier.py` - trains DenseNet-121 to predict all 14
   pathology probabilities at once (multi-label, sigmoid output)
3. `gradcam.py` - generates a heatmap for any image + pathology,
   showing which region the model focused on

## How to run (locally, default paths)
```bash
pip install torch torchvision opencv-python
python prepare_labels.py
python train_classifier.py
python gradcam.py
```

On Kaggle: use the NIH dataset attached at `/kaggle/input/nih-chest-xrays/`
- swap the paths (commented KAGGLE paths are in each file).

## Next steps
- Source/confirm clinical text notes paired with images
- Set up ClinicalBERT text encoder
- Build the actual fusion layer combining DenseNet + ClinicalBERT features
- Scale training beyond the 1000-image, 5-epoch baseline
