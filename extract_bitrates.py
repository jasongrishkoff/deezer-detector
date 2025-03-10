import concurrent.futures
import os
import sys
import numpy as np
from glob import glob
import subprocess
from tqdm import tqdm

# Import paths
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
            return int(result.stdout.strip()) // 1000
        else:
            return 320  # Default to 320 kbps
    except Exception as e:
        print(f"Error getting bitrate for {filename}: {e}")
        return 320

def extract_bitrates(sample_files=None, workers=16):
    """Extract bitrates in parallel"""
    print("Extracting bitrates from MP3 files")
    
    if os.path.exists(SPLIT_PATH):
        split_dict = np.load(SPLIT_PATH, allow_pickle=True).item()
        sample_files = []
        for key in split_dict:
            sample_files.extend([os.path.join(POS_DB_PATH, f) for f in split_dict[key]])
        files = sample_files
    else:
        files = glob(os.path.join(POS_DB_PATH, '**/*.mp3'), recursive=True)
    
    print(f"Processing {len(files)} files with {workers} workers")
    
    # Process in parallel
    bitrates = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_file = {executor.submit(get_mp3_bitrate, file): file for file in files}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(files)):
            file = future_to_file[future]
            filename = os.path.basename(file)
            bitrates[filename] = future.result()
    
    # Save bitrates
    os.makedirs(os.path.dirname(BR_PATH), exist_ok=True)
    np.save(BR_PATH, bitrates)
    
    print(f"Extracted bitrates for {len(bitrates)} files")
    print(f"Bitrates saved to {BR_PATH}")
    
    return bitrates

if __name__ == "__main__":
    extract_bitrates()
