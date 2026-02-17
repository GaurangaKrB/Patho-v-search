#!/usr/bin/env python3
"""
Generate realistic PANDA-like synthetic embeddings for feasibility testing.

Creates 50 slides in HDF5 format (.h5), matching the Prov-GigaPath PANDA
embedding structure. Embeddings are clustered by diagnosis so that similar
diagnoses produce similar search results — mimicking real data behavior.

Output: data/patch_embeddings/h5_files/<slide_id>.h5

Total size: ~5 MB (vs 32 GB for the real dataset).
"""

import os
import sys
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DIM = 1536
NUM_SLIDES = 50
OUT_DIR = Path("data/patch_embeddings/h5_files")

# PANDA ISUP grades — realistic diagnosis distribution
DIAGNOSES = {
    "ISUP_0_benign":     {"count": 12, "patches_range": (40, 120)},
    "ISUP_1_low_grade":  {"count": 10, "patches_range": (50, 150)},
    "ISUP_2_moderate":   {"count":  8, "patches_range": (60, 180)},
    "ISUP_3_high_grade": {"count":  8, "patches_range": (70, 200)},
    "ISUP_4_aggressive": {"count":  7, "patches_range": (80, 220)},
    "ISUP_5_critical":   {"count":  5, "patches_range": (90, 250)},
}

# Prov-GigaPath PANDA slide naming convention
PROVIDERS = ["radboud", "karolinska"]

# ---------------------------------------------------------------------------
# Embedding generation with clustering
# ---------------------------------------------------------------------------
def make_diagnosis_centroid(rng: np.random.RandomState, dim: int) -> np.ndarray:
    """Create a random unit vector as the 'centroid' for a diagnosis class."""
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def make_slide_embedding(
    centroid: np.ndarray,
    rng: np.random.RandomState,
    slide_spread: float = 0.3,
) -> np.ndarray:
    """Create a slide-level offset from the diagnosis centroid."""
    noise = rng.randn(len(centroid)).astype(np.float32) * slide_spread
    v = centroid + noise
    return v / np.linalg.norm(v)


def make_patch_embeddings(
    slide_vec: np.ndarray,
    n_patches: int,
    rng: np.random.RandomState,
    patch_spread: float = 0.5,
) -> np.ndarray:
    """Generate patch embeddings clustered around the slide vector."""
    noise = rng.randn(n_patches, len(slide_vec)).astype(np.float32) * patch_spread
    vecs = slide_vec[np.newaxis, :] + noise
    # L2-normalize each patch
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (vecs / norms).astype(np.float32)


def make_coords(n_patches: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate realistic tile coordinates (256px grid)."""
    # Simulate a grid of ~sqrt(n) x ~sqrt(n) tiles
    side = int(np.ceil(np.sqrt(n_patches)))
    all_coords = np.array([(x * 256, y * 256) for y in range(side) for x in range(side)])
    # Subsample to exactly n_patches
    idx = rng.choice(len(all_coords), size=n_patches, replace=False)
    return all_coords[idx].astype(np.int32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.RandomState(seed=42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Create centroids for each diagnosis class
    centroids = {}
    for diag in DIAGNOSES:
        centroids[diag] = make_diagnosis_centroid(rng, DIM)

    # Make centroids for adjacent ISUP grades slightly correlated
    diag_list = list(DIAGNOSES.keys())
    for i in range(1, len(diag_list)):
        # Blend 30% of the previous grade's centroid → adjacent grades are more similar
        centroids[diag_list[i]] = (
            0.7 * centroids[diag_list[i]] + 0.3 * centroids[diag_list[i - 1]]
        )
        centroids[diag_list[i]] /= np.linalg.norm(centroids[diag_list[i]])

    print(f"Generating {NUM_SLIDES} synthetic PANDA slides in {OUT_DIR}/")
    print(f"  Diagnosis classes: {len(DIAGNOSES)}")
    print()

    slide_idx = 0
    total_patches = 0
    total_bytes = 0

    for diag, cfg in DIAGNOSES.items():
        centroid = centroids[diag]
        for j in range(cfg["count"]):
            # Slide ID mimics PANDA naming
            provider = PROVIDERS[slide_idx % len(PROVIDERS)]
            slide_id = f"{provider}_{slide_idx:04d}"

            # Random patch count within the range for this diagnosis
            lo, hi = cfg["patches_range"]
            n_patches = rng.randint(lo, hi + 1)

            # Generate embeddings
            slide_vec = make_slide_embedding(centroid, rng)
            patch_vecs = make_patch_embeddings(slide_vec, n_patches, rng)
            coords = make_coords(n_patches, rng)

            # Write H5 file (matching Prov-GigaPath format)
            h5_path = OUT_DIR / f"{slide_id}.h5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("features", data=patch_vecs, dtype="float32")
                f.create_dataset("coords", data=coords, dtype="int32")
                # Store metadata as attributes
                f.attrs["slide_id"] = slide_id
                f.attrs["diagnosis"] = diag
                f.attrs["provider"] = provider
                f.attrs["isup_grade"] = int(diag.split("_")[1])
                f.attrs["n_patches"] = n_patches

            fsize = os.path.getsize(h5_path)
            total_bytes += fsize
            total_patches += n_patches

            print(f"  [{slide_idx+1:2d}/{NUM_SLIDES}] {slide_id}.h5  "
                  f"diag={diag:<20s}  patches={n_patches:4d}  "
                  f"size={fsize/1024:.0f} KB")

            slide_idx += 1

    print()
    print(f"Done: {NUM_SLIDES} slides, {total_patches} total patches")
    print(f"Total size: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
