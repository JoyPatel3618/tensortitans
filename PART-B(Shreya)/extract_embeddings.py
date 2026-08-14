"""
Step 1 of Part B.

Loads chest X-ray images from the NIH ChestX-ray14 dataset and turns each
one into a feature vector (embedding) using a pretrained DenseNet-121.
These embeddings capture what an image "looks like" in a compact form -
similar-looking X-rays end up with similar vectors. We save all of them
so we can search through them later (build_faiss_index.py).

This is unsupervised - we are NOT training anything here, just using a
model that already knows how to see general image patterns.
"""

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import densenet121, DenseNet121_Weights
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent

# ---- LOCAL paths (default) ----
IMAGES_DIR = BASE_DIR / "data" / "images"

# ---- KAGGLE paths (use these instead if running in a Kaggle notebook) ----
# IMAGES_DIR = Path("/kaggle/input/nih-chest-xrays/sample/images/")

OUTPUT_EMBEDDINGS = BASE_DIR / "embeddings" / "embeddings.npy"
OUTPUT_FILENAMES = BASE_DIR / "embeddings" / "filenames.npy"

# how many images to embed for today's baseline - full NIH dataset is
# ~112,000 images, we use a small slice first to prove the pipeline works
NUM_IMAGES = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    # pretrained on ImageNet - good enough to capture general visual
    # similarity (shapes, textures, contrast patterns) without us having
    # to train anything ourselves today
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()  # remove the final classification layer -
    # we want the feature vector that comes right before it, not a class prediction
    model.eval()
    return model.to(device)


def main():
    OUTPUT_EMBEDDINGS.parent.mkdir(parents=True, exist_ok=True)

    model = load_model()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),  # X-rays are grayscale, DenseNet expects 3 channels
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_paths = list(IMAGES_DIR.glob("*.png"))[:NUM_IMAGES]

    all_embeddings = []
    all_filenames = []

    with torch.no_grad():  # no training happening, so no need to track gradients - faster
        for img_path in tqdm(image_paths, desc="extracting embeddings"):
            image = Image.open(img_path).convert("L")
            tensor = transform(image).unsqueeze(0).to(device)
            embedding = model(tensor).cpu().numpy().flatten()

            all_embeddings.append(embedding)
            all_filenames.append(img_path.name)

    np.save(OUTPUT_EMBEDDINGS, np.array(all_embeddings))
    np.save(OUTPUT_FILENAMES, np.array(all_filenames))

    print(f"done. saved {len(all_filenames)} embeddings to {OUTPUT_EMBEDDINGS}")


if __name__ == "__main__":
    main()
