#!/usr/bin/env python3
"""
Generate 1000 synthetic .pkl files simulating BRCA Whole Slide Image
patch embeddings.

Each file contains a list of lists:
  - Outer list length: random between 90 and 110
  - Inner list length: 1536 (matching GigaPath embedding dimension)
  - Values: random floats in [0, 100) with 16 decimal places

Output directory: brca/embeddings/
Filenames: allembeddings1.pkl, allembeddings2.pkl, ..., allembeddings1000.pkl
"""

import os
import pickle
import numpy as np
from tqdm import tqdm

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "brca", "embeddings",
)
NUM_FILES = 1000
DIM = 1536
MIN_PATCHES = 90
MAX_PATCHES = 110
VALUE_MIN = 0.0
VALUE_MAX = 100.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    rng = np.random.default_rng(seed=42)

    for i in tqdm(range(1, NUM_FILES + 1), desc="Generating .pkl files"):
        fname = f"allembeddings{i}.pkl"
        fpath = os.path.join(OUT_DIR, fname)

        # Skip if already exists
        if os.path.isfile(fpath):
            continue

        n_patches = rng.integers(MIN_PATCHES, MAX_PATCHES + 1)  # [90, 110]

        # Generate random float64 values in [0, 100) — gives ~16 decimal places
        arr = rng.uniform(VALUE_MIN, VALUE_MAX, size=(n_patches, DIM))

        # Convert to list of lists (native Python) as per requirement
        data = arr.tolist()

        with open(fpath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    total = len([f for f in os.listdir(OUT_DIR) if f.endswith(".pkl")])
    print(f"\nDone. {total} .pkl files in {OUT_DIR}")

    # Spot-check
    with open(os.path.join(OUT_DIR, "allembeddings500.pkl"), "rb") as f:
        sample = pickle.load(f)
    print(f"Spot-check allembeddings500.pkl: "
          f"type={type(sample).__name__}, "
          f"outer_len={len(sample)}, "
          f"inner_len={len(sample[0])}, "
          f"sample_val={sample[0][0]}")


if __name__ == "__main__":
    main()
