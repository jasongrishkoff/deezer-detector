import os
import numpy as np
from glob import glob

# Load the split file
SPLIT_PATH = "/workspace/deezer-detector/data/dataset_split.npy"
if os.path.exists(SPLIT_PATH):
    split_dict = np.load(SPLIT_PATH, allow_pickle=True).item()
    print("Split file exists with keys:", split_dict.keys())
    for key in split_dict:
        print(f"{key}: {len(split_dict[key])} files")
        if split_dict[key]:
            print(f"Example filenames in {key}:")
            for f in split_dict[key][:3]:
                print(f"  {f}")
    
    # Check if these files exist in your directories
    print("\nChecking if files exist in directories...")
    
    # Check in positive directory
    POS_DB_PATH = "/workspace/human_examples"
    for key in split_dict:
        found = 0
        for f in split_dict[key][:10]:  # Check first 10 files
            matches = glob(os.path.join(POS_DB_PATH, "**", f), recursive=True)
            if matches:
                found += 1
        print(f"Found {found}/10 {key} files in positive directory")
    
    # Check in negative directory
    NEG_DB_PATH = "/workspace/deezer-detector/fma_rebuilt"
    for encoder in ["griffin256", "griffin512"]:
        for key in split_dict:
            found = 0
            for f in split_dict[key][:10]:  # Check first 10 files
                matches = glob(os.path.join(NEG_DB_PATH, encoder, "**", f), recursive=True)
                if matches:
                    found += 1
            print(f"Found {found}/10 {key} files in negative directory for encoder {encoder}")
else:
    print("Split file does not exist at", SPLIT_PATH)
