#!/usr/bin/env python3
import os
import sys
import fsspec

url = "https://huggingface.co/datasets/prov-gigapath/prov-gigapath-tile-embeddings/resolve/main/GigaPath_PANDA_embeddings.zip"
out_dir = "data/panda_real/h5_files"
n_slides = 50

def main():
    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in os.listdir(out_dir) if f.endswith(".h5")]
    print(f"Opening remote zip at {url[:60]}...")
    fs = fsspec.filesystem("zip", fo=url)

    files = fs.glob("GigaPath_PANDA_embeddings/h5_files/*.h5")
    print(f"Found {len(files)} .h5 files in remote zip.")

    selected = sorted(files)[:n_slides]
    
    needed = []
    for f in selected:
        basename = os.path.basename(f)
        if basename not in existing:
            needed.append(f)

    if not needed:
        print(f"All {n_slides} files already exist in {out_dir}.")
        return

    print(f"Need to extract {len(needed)} files (skipping {n_slides - len(needed)} existing)...")
    
    for i, fpath in enumerate(needed, 1):
        basename = os.path.basename(fpath)
        dest = os.path.join(out_dir, basename)
        print(f"[{i}/{len(needed)}] Extracting {basename} ...")
        
        with fs.open(fpath, "rb") as src, open(dest, "wb") as dst:
            dst.write(src.read())

    print("Done!")

if __name__ == "__main__":
    main()
