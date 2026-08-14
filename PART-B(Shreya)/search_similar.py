"""
Step 3 of Part B - the actual "similar cases" feature.

Given one new X-ray, finds the K most visually similar X-rays from the
indexed dataset. This is the runtime step - unlike extract_embeddings.py,
this does NOT re-embed the whole dataset, only the one query image, then
searches the already-built index. That's what makes it fast enough to use
live in the app.
"""

from pathlib import Path

import faiss
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import densenet121, DenseNet121_Weights

BASE_DIR = Path(__file__).resolve().parent

EMBEDDINGS_DIR = BASE_DIR / "embeddings"
INDEX_PATH = EMBEDDINGS_DIR / "faiss_index.bin"
FILENAMES_PATH = EMBEDDINGS_DIR / "filenames.npy"

TOP_K = 5  # how many similar cases to return

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()
    model.eval()
    return model.to(device)


def embed_one_image(model, image_path):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("L")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor).cpu().numpy().astype("float32")
    return embedding


def find_similar(query_image_path):
    model = load_model()
    query_embedding = embed_one_image(model, query_image_path)

    index = faiss.read_index(str(INDEX_PATH))
    filenames = np.load(FILENAMES_PATH)

    distances, indices = index.search(query_embedding, TOP_K)

    results = [(filenames[i], float(d)) for i, d in zip(indices[0], distances[0])]
    return results


if __name__ == "__main__":
    # example usage - point this at any X-ray to test
    example_query = BASE_DIR / "data" / "images" / "example.png"
    matches = find_similar(example_query)

    print("most similar cases:")
    for filename, distance in matches:
        print(f"  {filename}  (distance: {distance:.2f})")
