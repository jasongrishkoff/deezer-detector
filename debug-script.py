#!/usr/bin/env python
"""
Debug the audio file paths to understand why AudioLoader can't find them
"""

import os
import sys
from glob import glob

# Add the current directory to the path
sys.path.insert(0, os.getcwd())

from loader.global_variables import *

def main():
    print("===== PATH CONFIGURATION =====")
    print(f"POS_DB_PATH: {POS_DB_PATH}")
    print(f"NEG_DB_PATH: {NEG_DB_PATH}")
    
    # Check for MP3 files in human_examples
    pos_files = glob(os.path.join(POS_DB_PATH, '*.mp3'))
    pos_recursive_files = glob(os.path.join(POS_DB_PATH, '**/*.mp3'), recursive=True)
    print(f"\n===== POSITIVE FILES =====")
    print(f"Found {len(pos_files)} MP3 files directly in {POS_DB_PATH}")
    print(f"Found {len(pos_recursive_files)} MP3 files recursively in {POS_DB_PATH}")
    if pos_files:
        print("First 3 files:")
        for f in pos_files[:3]:
            print(f"  {os.path.basename(f)}")
    
    # Check for MP3 files in negative folders
    print(f"\n===== NEGATIVE FILES =====")
    for codec in ['griffin256', 'griffin512']:
        neg_path = os.path.join(NEG_DB_PATH, codec)
        neg_files = glob(os.path.join(neg_path, '*.mp3'))
        neg_recursive_files = glob(os.path.join(neg_path, '**/*.mp3'), recursive=True)
        print(f"Found {len(neg_files)} MP3 files directly in {neg_path}")
        print(f"Found {len(neg_recursive_files)} MP3 files recursively in {neg_path}")
        if neg_files:
            print(f"First 3 files in {codec}:")
            for f in neg_files[:3]:
                print(f"  {os.path.basename(f)}")
    
    # Check split file
    print(f"\n===== SPLIT FILE =====")
    if os.path.exists(SPLIT_PATH):
        import numpy as np
        split_data = np.load(SPLIT_PATH, allow_pickle=True).item()
        print(f"Split file exists with keys: {list(split_data.keys())}")
        for key in split_data:
            filenames = split_data[key]
            print(f"{key}: {len(filenames)} files")
            
            found_pos = 0
            for fname in filenames[:10]:  # Check first 10
                if os.path.exists(os.path.join(POS_DB_PATH, fname)):
                    found_pos += 1
                elif glob(os.path.join(POS_DB_PATH, '**', fname), recursive=True):
                    found_pos += 1
            
            found_neg = 0
            for encoder in ['griffin256', 'griffin512']:
                encoder_found = 0
                for fname in filenames[:10]:  # Check first 10
                    if os.path.exists(os.path.join(NEG_DB_PATH, encoder, fname)):
                        encoder_found += 1
                    elif glob(os.path.join(NEG_DB_PATH, encoder, '**', fname), recursive=True):
                        encoder_found += 1
                print(f"  Found {encoder_found}/10 {key} files in {encoder}")
                found_neg = max(found_neg, encoder_found)
            
            print(f"  Found {found_pos}/10 {key} files in positive dir")
            print(f"  Found {found_neg}/10 {key} files in at least one negative dir")

if __name__ == "__main__":
    main()
