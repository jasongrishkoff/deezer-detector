"""
Extract bitrates from MP3 files and save to a numpy file
"""

import os
import sys
import numpy as np
from glob import glob
import subprocess
from tqdm import tqdm

# Add the parent directory to the path to import global_variables
sys.path.append(os.path.join(os.getcwd()))
from loader.global_variables import *

def get_mp3_bitrate(filename):
    """Get bitrate of an MP3 file using ffprobe"""
    try:
        cmd = [
            'ffprobe', 
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=bit_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            filename
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.stdout.strip():
            # Convert bit/s to kbit/s
            return int(result.stdout.strip()) // 1000
        else:
            # If ffprobe couldn't determine the bitrate
            return 320  # Default to 320 kbps
    except Exception as e:
        print(f"Error getting bitrate for {filename}: {e}")
        return 320

def extract_bitrates(sample_files=None):
    """
    Extract bitrates from MP3 files and save to a numpy file
    
    Args:
        sample_files: List of files to process. If None, process all files in POS_DB_PATH
    """
    print("Extracting bitrates from MP3 files")
    
    # Inside extract_bitrates.py
    if os.path.exists(SPLIT_PATH):
        split_dict = np.load(SPLIT_PATH, allow_pickle=True).item()
        # Get only the filenames in the split
        sample_files = []
        for key in split_dict:
            sample_files.extend([os.path.join(POS_DB_PATH, f) for f in split_dict[key]])
        files = sample_files
    else:
        files = glob(os.path.join(POS_DB_PATH, '**/*.mp3'), recursive=True)
    
    print(f"Processing {len(files)} files")
    
    # Extract bitrates
    bitrates = {}
    for file in tqdm(files):
        filename = os.path.basename(file)
        bitrates[filename] = get_mp3_bitrate(file)
    
    # Save bitrates
    os.makedirs(os.path.dirname(BR_PATH), exist_ok=True)
    np.save(BR_PATH, bitrates)
    
    print(f"Extracted bitrates for {len(bitrates)} files")
    print(f"Bitrates saved to {BR_PATH}")
    
    return bitrates

if __name__ == "__main__":
    extract_bitrates()
