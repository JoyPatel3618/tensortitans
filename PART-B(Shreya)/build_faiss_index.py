"""
Step 2 of Part B.

Takes the embeddings from extract_embeddings.py and builds a FAISS index -
basically a fast lookup structure that lets us find "which images have
embeddings closest to this one" without comparing against every single
image one by one (that would be too slow at scale).

Run this once, after extract_embeddings.py. The index itself gets reused
every time someone searches for similar cases.
"""

from pathlib import Path

import faiss
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

EMBEDDINGS_PATH = BASE_DIR / "embeddings" / "embeddings.npy"
INDEX_OUTPUT_PATH = BASE_DIR / "embeddings" / "faiss_index.bin"


def main():
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    embedding_dim = embeddings.shape[1]

    # IndexFlatL2 = simplest FAISS index, compares by straight-line
    # (Euclidean) distance between vectors - good enough for our scale
    # (thousands of images). Larger datasets would use a faster
    # approximate index instead.
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_OUTPUT_PATH))
    print(f"done. indexed {index.ntotal} embeddings, saved to {INDEX_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
