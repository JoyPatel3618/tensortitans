# Part B - Embedding-based Similar Case Retrieval

**Owner:** Shreya
**Status:** Working baseline (Day 1) - built to unblock progress while Shreya
was busy with other priorities. She should review, adjust, and take over
from here.

## What this does
Given a new chest X-ray, finds visually similar past cases from the NIH
ChestX-ray14 dataset - powers the "Similar Cases" panel in the UI.

- Input: raw chest X-ray (any uploaded X-ray at runtime; NIH dataset images
  for building the index)
- Output: top-K similar case filenames + similarity distance

## How it works
1. `extract_embeddings.py` - runs a pretrained DenseNet-121 (ImageNet
   weights) over a batch of X-rays and saves a feature vector per image.
   This is a ONE-TIME step to build the searchable library.
2. `build_faiss_index.py` - puts all those vectors into a FAISS index for
   fast nearest-neighbor lookup.
3. `search_similar.py` - the runtime step: embeds ONE new query image and
   searches the index for the closest matches. This is what actually runs
   every time a user uploads an X-ray - it does not re-embed the whole
   dataset each time, only the new image.

## How to run (locally, default paths)
```bash
pip install faiss-cpu torch torchvision
python extract_embeddings.py
python build_faiss_index.py
python search_similar.py
```

On Kaggle: use the NIH ChestX-ray dataset attached at
`/kaggle/input/nih-chest-xrays/sample/images/` - swap the path in
`extract_embeddings.py` (commented KAGGLE path is right there).

## Day 1 status - what's real vs. what's a placeholder
- Real: full pipeline runs end to end on a small subset (1000 images) using
  generic ImageNet features.
- Placeholder / next steps (for Shreya to improve):
  - Swap the generic ImageNet-pretrained DenseNet for one fine-tuned on
    chest X-rays specifically (e.g. via `torchxrayvision`) - will give much
    more medically-relevant similarity than generic ImageNet features.
  - Scale from 1000 images to the full ~112,000 image dataset.
  - Add a proper evaluation of retrieval quality (e.g. do "similar" images
    actually share the same diagnosed pathology?).
  - Wire this into the actual UI panel.
