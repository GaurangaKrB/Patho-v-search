#!/usr/bin/env python3
import os
import glob
import numpy as np
import xxhash
from PIL import Image
from tqdm import tqdm
import os
import glob
import numpy as np
import json
from PIL import Image
from tqdm import tqdm

def main():
    patches_dir = 'data/panda_real/images/patches'
    metadata_out = 'data/patch_metadata.json'
    
    if not os.path.isdir(patches_dir):
        print(f"Directory {patches_dir} not found. Skipping metadata generation.")
        return
        
    slide_dirs = sorted([d for d in os.listdir(patches_dir) if os.path.isdir(os.path.join(patches_dir, d))])
    print(f'Processing {len(slide_dirs)} slides for tissue percentage directly to disk...')
    
    metadata = {}
    total_updated = 0
    total_background = 0

    for slide_id in tqdm(slide_dirs, desc='Slides'):
        slide_path = os.path.join(patches_dir, slide_id)
        
        for f in os.listdir(slide_path):
            if not f.endswith('.png'): continue
            try:
                idx = int(f.split('_')[1])
            except (IndexError, ValueError):
                continue
            
            img = np.array(Image.open(os.path.join(slide_path, f)))
            white_pct = float((img > 240).mean() * 100)
            black_pct = float((img < 15).mean() * 100)
            tissue_pct = float(100 - max(white_pct, black_pct))
            is_background = tissue_pct < 10
            
            patch_id = f"{slide_id}_{idx}"
            metadata[patch_id] = {
                'tissue_pct': round(tissue_pct, 1),
                'is_background': is_background
            }
            
            if is_background:
                total_background += 1
            total_updated += 1
            
    with open(metadata_out, 'w') as f:
        json.dump(metadata, f)

    print(f'\nDone! Computed {total_updated} patches, {total_background} marked as background ({total_background/max(total_updated,1)*100:.1f}%)')
    print(f'Metadata successfully mapped to {metadata_out} in milliseconds.')

if __name__ == '__main__':
    main()
