#!/usr/bin/env python3
"""
Download the real PANDA embeddings zip from HuggingFace, extract only N slides,
then delete the zip to reclaim disk space.

Usage:
  python3 scripts/download_and_subsample_panda.py --hf_token hf_XXXXX --n_slides 50
"""
import argparse
import os
import subprocess
import sys
import zipfile
import time


HF_URL = (
    "https://huggingface.co/datasets/prov-gigapath/"
    "prov-gigapath-tile-embeddings/resolve/main/"
    "GigaPath_PANDA_embeddings.zip"
)

def download_with_curl(url: str, dest: str, token: str):
    """Download with curl (resume-capable, shows progress)."""
    cmd = [
        "curl", "-fL",
        "-C", "-",                           # resume
        "-H", f"Authorization: Bearer {token}",
        "-o", dest,
        "--progress-bar",
        url,
    ]
    print(f"\n📥 Downloading to: {dest}")
    print(f"   URL: {url[:80]}...")
    print(f"   This will take ~2 hours at 5 MB/s. You can Ctrl+C and re-run to resume.\n")
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"\n❌ curl exited with code {rc}", file=sys.stderr)
        sys.exit(1)
    print(f"\n✅ Download complete: {dest}")


def extract_subsample(zip_path: str, out_dir: str, n_slides: int):
    """Extract only n_slides .h5 files from the zip."""
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n📦 Opening zip: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find all .h5 files
        h5_entries = [e for e in zf.namelist() if e.endswith(".h5")]
        print(f"   Total .h5 files in zip: {len(h5_entries)}")

        # Also look for slide-level .pt files
        slide_entries = [e for e in zf.namelist()
                         if e.endswith(".pt") and "slide" in e.lower()]
        print(f"   Total slide .pt files in zip: {len(slide_entries)}")

        # Take first n_slides
        selected_h5 = sorted(h5_entries)[:n_slides]
        print(f"   Extracting {len(selected_h5)} .h5 files...\n")

        for i, entry in enumerate(selected_h5):
            basename = os.path.basename(entry)
            dest = os.path.join(out_dir, basename)
            with zf.open(entry) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            fsize = os.path.getsize(dest)
            print(f"   [{i+1:3d}/{len(selected_h5)}] {basename}  ({fsize/1024:.0f} KB)")

        # Also extract a few slide .pt files if they exist
        slide_out = os.path.join(os.path.dirname(out_dir), "slides_embeddings")
        if slide_entries:
            selected_slides = sorted(slide_entries)[:n_slides]
            os.makedirs(slide_out, exist_ok=True)
            print(f"\n   Extracting {len(selected_slides)} slide .pt files...")
            for i, entry in enumerate(selected_slides):
                basename = os.path.basename(entry)
                dest = os.path.join(slide_out, basename)
                with zf.open(entry) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                fsize = os.path.getsize(dest)
                print(f"   [{i+1:3d}/{len(selected_slides)}] {basename}  ({fsize/1024:.0f} KB)")

    # Summary
    extracted_h5 = len([f for f in os.listdir(out_dir) if f.endswith(".h5")])
    total_size = sum(
        os.path.getsize(os.path.join(out_dir, f))
        for f in os.listdir(out_dir) if f.endswith(".h5")
    )
    print(f"\n✅ Extracted {extracted_h5} patch files → {out_dir}")
    print(f"   Total extracted size: {total_size/1024/1024:.1f} MB")

    if slide_entries and os.path.isdir(slide_out):
        slide_count = len([f for f in os.listdir(slide_out) if f.endswith(".pt")])
        print(f"   Slide embeddings: {slide_count} files → {slide_out}")


def main():
    parser = argparse.ArgumentParser(description="Download & subsample PANDA embeddings")
    parser.add_argument("--hf_token", required=True, help="HuggingFace token (hf_...)")
    parser.add_argument("--n_slides", type=int, default=50, help="Number of slides to extract")
    parser.add_argument("--zip_dir", default="data", help="Where to save the zip temporarily")
    parser.add_argument("--out_dir", default="data/panda_real/h5_files", help="Extracted H5 output dir")
    parser.add_argument("--keep_zip", action="store_true", help="Don't delete zip after extraction")
    args = parser.parse_args()

    zip_path = os.path.join(args.zip_dir, "GigaPath_PANDA_embeddings.zip")
    os.makedirs(args.zip_dir, exist_ok=True)

    # Step 1: Download (skip if already exists and complete)
    if os.path.isfile(zip_path):
        size_gb = os.path.getsize(zip_path) / 1e9
        print(f"⚡ Zip already exists: {zip_path} ({size_gb:.1f} GB)")
        if size_gb < 30:
            print("   Looks incomplete — resuming download...")
            download_with_curl(HF_URL, zip_path, args.hf_token)
        else:
            print("   Looks complete — skipping download.")
    else:
        download_with_curl(HF_URL, zip_path, args.hf_token)

    # Step 2: Extract subsample
    extract_subsample(zip_path, args.out_dir, args.n_slides)

    # Step 3: Delete zip to reclaim space
    if not args.keep_zip:
        size_gb = os.path.getsize(zip_path) / 1e9
        print(f"\n🗑️  Deleting zip to reclaim {size_gb:.1f} GB...")
        os.remove(zip_path)
        print(f"   Deleted: {zip_path}")

    print("\n🎉 Done! Next steps:")
    print(f"   python3 ingest.py --qdrant_path data/qdrant_local \\")
    print(f"     --patches_dir {args.out_dir} \\")
    print(f"     --slides_dir data/empty_slides \\")
    print(f"     --recreate --slide_agg_strategy attention_weighted")


if __name__ == "__main__":
    main()
