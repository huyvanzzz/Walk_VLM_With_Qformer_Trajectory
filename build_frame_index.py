#!/usr/bin/env python3
"""
Build frame index for WAD dataset
Usage: python scripts/build_frame_index.py
"""

import tarfile
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from collections import defaultdict
import pickle
import os

def main():
    print("="*80)
    print("BUILDING FRAME INDEX")
    print("="*80)
    
    LOCAL_DIR = "./wad_dataset"
    os.makedirs(LOCAL_DIR, exist_ok=True)
    INDEX_FILE = os.path.join(LOCAL_DIR, 'frame_index.pkl')
    
    # Check if already exists
    if os.path.exists(INDEX_FILE):
        print(f"\n✓ Frame index already exists at {INDEX_FILE}")
        response = input("Rebuild? (y/n): ")
        if response.lower() != 'y':
            return
    
    tar_files = [f"shard_{i:05d}.tar" for i in range(23)]
    print(f"\n[1] TAR shards: {len(tar_files)}")
    
    print("\n[2] Downloading shards...")
    shard_paths = []
    for shard_name in tqdm(tar_files):
        shard_path = hf_hub_download(
            repo_id="minhdang0901/WAD_Images_All_Size",
            filename=shard_name,
            repo_type="dataset"
        )
        shard_paths.append(shard_path)
    
    print(f"\n[3] Building index...")
    frame_index = defaultdict(dict)
    
    for shard_path in tqdm(shard_paths):
        with tarfile.open(shard_path, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(('.jpg', '.jpeg', '.png')):
                    parts = member.name.split('/')
                    
                    if len(parts) == 2:
                        frame_folder, img_filename = parts
                        
                        try:
                            frame_id = int(img_filename.split('.')[0])
                            frame_index[frame_folder][frame_id] = {
                                'shard': shard_path,
                                'tar_path': member.name,
                                'size': member.size
                            }
                        except ValueError:
                            continue
    
    print(f"\n[4] Saving index...")
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump(dict(frame_index), f)
    
    print(f"\n✓ Frame index built!")
    print(f"  Folders: {len(frame_index)}")
    print(f"  Saved to: {INDEX_FILE}")

if __name__ == '__main__':
    main()