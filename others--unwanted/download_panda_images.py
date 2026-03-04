#!/usr/bin/env python3
"""
Download PANDA WSI images from Kaggle, extract thumbnails and patch crops
at the coordinates stored in the corresponding .h5 embedding files.

Prerequisites
-------------
1. pip install kaggle tiffslide Pillow h5py numpy
2. Kaggle credentials at ~/.kaggle/kaggle.json:
     {"username": "yourname", "key": "abcdef1234567890"}
   Or set env vars: KAGGLE_USERNAME + KAGGLE_KEY
3. Accept PANDA competition rules:
   https://www.kaggle.com/c/prostate-cancer-grade-assessment/rules

Usage
-----
  # Test with 2 slides first:
  python3 scripts/download_panda_images.py --limit 2

  # Download images for all 50 slides (delete TIFFs after extraction):
  python3 scripts/download_panda_images.py

  # Extract 20 patches per slide, keep TIFFs:
  python3 scripts/download_panda_images.py --patches_per_slide 20 --keep_tiff
"""

import argparse
import csv
import glob
import os
import shutil
import sys
import tempfile
from typing import List

import h5py
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMPETITION = "prostate-cancer-grade-assessment"


# ---------------------------------------------------------------------------
# Kaggle download helpers  (classic kaggle package)
# ---------------------------------------------------------------------------
def _get_kaggle_api():
    """Initialise and return the classic Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("ERROR: 'kaggle' package not installed.  Run:")
        print("  pip install kaggle")
        sys.exit(1)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        print(f"ERROR: Kaggle authentication failed: {exc}")
        print("  Make sure ~/.kaggle/kaggle.json exists, or set")
        print("  KAGGLE_USERNAME and KAGGLE_KEY env vars.")
        sys.exit(1)
    return api


def download_train_csv(api, out_dir: str) -> str:
    """Download train.csv (ISUP labels) from the PANDA competition."""
    csv_path = os.path.join(out_dir, "train.csv")
    if os.path.isfile(csv_path):
        print(f"  train.csv already exists → {csv_path}")
        return csv_path

    print(f"  Downloading train.csv from Kaggle …")
    try:
        api.competition_download_file(
            COMPETITION, "train.csv", path=out_dir, quiet=False
        )
    except Exception as exc:
        print(f"  ⚠️  Could not download train.csv: {exc}")
        return ""

    if not os.path.isfile(csv_path):
        print(f"  WARNING: Could not find {csv_path} after download.")
        return ""
    print(f"  ✅ Saved {csv_path}")
    return csv_path


def download_single_tiff(api, slide_id: str, dest_dir: str) -> str:
    """Download one WSI TIFF from the Kaggle PANDA competition.

    Kaggle wraps individual competition files in ZIP archives,
    so we detect and extract them automatically.
    Returns the path to the extracted TIFF, or "" on failure.
    """
    import zipfile

    tiff_dest = os.path.join(dest_dir, f"{slide_id}.tiff")
    if os.path.isfile(tiff_dest):
        return tiff_dest

    try:
        api.competition_download_file(
            COMPETITION,
            f"train_images/{slide_id}.tiff",
            path=dest_dir,
            quiet=False,
        )
    except Exception as exc:
        err_str = str(exc)
        if "429" in err_str:
            print(f"  ⚠️  Download failed: {err_str}")
            return "RATE_LIMIT"
        print(f"  ⚠️  Download failed: {exc}")
        return ""

    # Kaggle wraps files in ZIP — check and extract
    if os.path.isfile(tiff_dest):
        # Check if the "tiff" file is actually a ZIP archive
        with open(tiff_dest, "rb") as f:
            magic = f.read(4)
        if magic == b"PK\x03\x04":  # ZIP magic bytes
            zip_tmp = tiff_dest + ".zip"
            os.rename(tiff_dest, zip_tmp)
            with zipfile.ZipFile(zip_tmp, "r") as zf:
                # Extract the TIFF from inside
                tiff_members = [m for m in zf.namelist() if m.endswith(".tiff")]
                if tiff_members:
                    with zf.open(tiff_members[0]) as src, open(tiff_dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            os.remove(zip_tmp)

    if not os.path.isfile(tiff_dest):
        # Check if it was nested in a subdirectory
        nested = os.path.join(dest_dir, "train_images", f"{slide_id}.tiff")
        if os.path.isfile(nested):
            shutil.move(nested, tiff_dest)

    return tiff_dest if os.path.isfile(tiff_dest) else ""


# ---------------------------------------------------------------------------
# WSI reading helpers (tiffslide — pure-Python, no system deps)
# ---------------------------------------------------------------------------
def open_slide(tiff_path: str):
    """Open a WSI using tiffslide."""
    try:
        from tiffslide import TiffSlide
    except ImportError:
        print("ERROR: tiffslide not installed.  Run:  pip install tiffslide")
        sys.exit(1)
    return TiffSlide(tiff_path)


def extract_thumbnail(slide, out_path: str, max_dim: int = 1024) -> None:
    """Save a low-resolution thumbnail of the whole slide."""
    thumb = slide.get_thumbnail((max_dim, max_dim))
    thumb.save(out_path)


def extract_patches(
    slide,
    coords: np.ndarray,
    out_dir: str,
    patch_size: int = 256,
    max_patches: int = 10,
) -> int:
    """Extract patch crops at the given (x, y) coordinates.

    Samples patches evenly across the slide (not just the first N).
    Returns the number of patches actually saved.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_total = len(coords)

    # Sample evenly across the coordinate list
    if n_total > max_patches:
        indices = np.linspace(0, n_total - 1, max_patches, dtype=int)
    else:
        indices = np.arange(n_total)

    saved = 0
    for idx in indices:
        x, y = int(coords[idx, 0]), int(coords[idx, 1])
        try:
            region = slide.read_region((x, y), 0, (patch_size, patch_size))
            # tiffslide returns RGBA; convert to RGB
            if region.mode == "RGBA":
                region = region.convert("RGB")
            fname = f"patch_{idx:04d}_x{x}_y{y}.png"
            region.save(os.path.join(out_dir, fname))
            saved += 1
        except Exception as exc:
            print(f"    ⚠️  Patch {idx} ({x},{y}): {exc}")
    return saved


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------
def load_labels(csv_path: str) -> dict:
    """Parse train.csv → {image_id: {isup_grade, gleason_score, provider}}."""
    labels = {}
    if not csv_path or not os.path.isfile(csv_path):
        return labels
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["image_id"]] = {
                "isup_grade": int(row.get("isup_grade", -1)),
                "gleason_score": row.get("gleason_score", ""),
                "data_provider": row.get("data_provider", ""),
            }
    return labels


def save_slide_labels(
    slide_ids: List[str], all_labels: dict, out_path: str
) -> None:
    """Write a CSV with labels for our specific slides."""
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slide_id", "isup_grade", "gleason_score", "data_provider"])
        for sid in sorted(slide_ids):
            info = all_labels.get(sid, {})
            writer.writerow([
                sid,
                info.get("isup_grade", ""),
                info.get("gleason_score", ""),
                info.get("data_provider", ""),
            ])
    print(f"  ✅ Slide labels → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download PANDA WSIs from Kaggle and extract patch images"
    )
    parser.add_argument(
        "--h5_dir", default="data/panda_real/h5_files",
        help="Directory with H5 embedding files (default: data/panda_real/h5_files)",
    )
    parser.add_argument(
        "--out_dir", default="data/panda_real/images",
        help="Where to save extracted images (default: data/panda_real/images)",
    )
    parser.add_argument(
        "--patches_per_slide", type=int, default=10,
        help="Max patch crops to extract per slide (default: 10)",
    )
    parser.add_argument(
        "--patch_size", type=int, default=256,
        help="Patch crop size in pixels (default: 256)",
    )
    parser.add_argument(
        "--thumb_size", type=int, default=1024,
        help="Max dimension for thumbnail (default: 1024)",
    )
    parser.add_argument(
        "--keep_tiff", action="store_true",
        help="Don't delete TIFFs after extraction (warning: ~500 MB each)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process first N slides (for testing)",
    )

    args = parser.parse_args()

    # ---- Discover slide IDs from H5 files ----
    h5_files = sorted(glob.glob(os.path.join(args.h5_dir, "*.h5")))
    if not h5_files:
        print(f"ERROR: No .h5 files found in {args.h5_dir}")
        sys.exit(1)

    slide_ids = [os.path.splitext(os.path.basename(f))[0] for f in h5_files]
    if args.limit:
        slide_ids = slide_ids[: args.limit]
    print(f"Found {len(slide_ids)} slide(s) to process in {args.h5_dir}")

    # ---- Create output dirs ----
    thumbs_dir = os.path.join(args.out_dir, "thumbnails")
    patches_dir = os.path.join(args.out_dir, "patches")
    os.makedirs(thumbs_dir, exist_ok=True)
    os.makedirs(patches_dir, exist_ok=True)

    # ---- Kaggle API ----
    api = _get_kaggle_api()

    # ---- Download train.csv for ISUP labels ----
    csv_path = download_train_csv(api, args.out_dir)
    all_labels = load_labels(csv_path)
    if all_labels:
        save_slide_labels(slide_ids, all_labels, os.path.join(args.out_dir, "slide_labels.csv"))

    # ---- Temp dir for TIFF downloads ----
    tiff_dir = os.path.join(args.out_dir, "tiffs") if args.keep_tiff else tempfile.mkdtemp(prefix="panda_tiff_")
    os.makedirs(tiff_dir, exist_ok=True)

    # ---- Process each slide ----
    success = 0
    failed = []

    for i, sid in enumerate(slide_ids):
        label_info = all_labels.get(sid, {})
        isup = label_info.get("isup_grade", "?")
        print(f"\n[{i + 1}/{len(slide_ids)}] {sid}  (ISUP grade: {isup})")

        # Check if already extracted
        slide_patches_dir = os.path.join(patches_dir, sid)
        h5_path = os.path.join(args.h5_dir, f"{sid}.h5")
        try:
            with h5py.File(h5_path, "r") as hf:
                coords_len = len(np.asarray(hf["coords"]))
        except Exception:
            coords_len = 0
            
        target_patches = min(coords_len, args.patches_per_slide)
        if os.path.isdir(slide_patches_dir):
            existing_pngs = len([f for f in os.listdir(slide_patches_dir) if f.endswith(".png")])
            if existing_pngs >= target_patches and target_patches > 0:
                print(f"  ⏭️  SKIPPED — already extracted {existing_pngs} patches")
                success += 1
                continue

        # 1. Download TIFF
        print(f"  📥 Downloading TIFF from Kaggle …")
        tiff_path = download_single_tiff(api, sid, tiff_dir)
        
        if tiff_path == "RATE_LIMIT":
            print("\n🚨 Kaggle API Rate Limit (429) hit! We have reached Kaggle's strict download quota.")
            print("Stopping further WSI downloads and proceeding to process the slides we successfully obtained.")
            break
            
        if not tiff_path:
            print(f"  ❌ SKIPPED — could not download")
            failed.append(sid)
            continue

        tiff_mb = os.path.getsize(tiff_path) / 1e6
        print(f"  📁 TIFF: {tiff_mb:.1f} MB")

        # 2. Open slide
        try:
            slide = open_slide(tiff_path)
        except Exception as exc:
            print(f"  ❌ SKIPPED — could not open TIFF: {exc}")
            failed.append(sid)
            if not args.keep_tiff:
                os.remove(tiff_path)
            continue

        # 3. Thumbnail
        thumb_path = os.path.join(thumbs_dir, f"{sid}_thumb.png")
        try:
            extract_thumbnail(slide, thumb_path, args.thumb_size)
            print(f"  🖼️  Thumbnail → {thumb_path}")
        except Exception as exc:
            print(f"  ⚠️  Thumbnail failed: {exc}")

        # 4. Extract patches at H5 coordinates
        h5_path = os.path.join(args.h5_dir, f"{sid}.h5")
        with h5py.File(h5_path, "r") as hf:
            coords = np.asarray(hf["coords"]).astype(np.int32)

        slide_patches_dir = os.path.join(patches_dir, sid)
        n_saved = extract_patches(
            slide, coords, slide_patches_dir,
            patch_size=args.patch_size,
            max_patches=args.patches_per_slide,
        )
        print(f"  🔬 Patches: {n_saved}/{min(len(coords), args.patches_per_slide)} "
              f"extracted (total coords: {len(coords)})")

        slide.close()

        # 5. Delete TIFF to save space
        if not args.keep_tiff:
            os.remove(tiff_path)
            print(f"  🗑️  Deleted TIFF ({tiff_mb:.0f} MB reclaimed)")

        success += 1

    # ---- Cleanup temp dir ----
    if not args.keep_tiff and os.path.isdir(tiff_dir):
        shutil.rmtree(tiff_dir, ignore_errors=True)

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"Done!  {success}/{len(slide_ids)} slides processed")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")
    print(f"\nOutput:")
    print(f"  Thumbnails : {thumbs_dir}/")
    print(f"  Patches    : {patches_dir}/")
    if csv_path:
        print(f"  Labels     : {os.path.join(args.out_dir, 'slide_labels.csv')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
