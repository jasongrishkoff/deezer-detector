import os
import sys
from glob import glob

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

from loader.global_variables import *
from loader.audio import AudioLoader

def main():
    print("POS_DB_PATH:", POS_DB_PATH)
    print("NEG_DB_PATH:", NEG_DB_PATH)
    
    # Check positive files
    pos_files = glob(os.path.join(POS_DB_PATH, '**/*.mp3'), recursive=True)
    print(f"Found {len(pos_files)} positive files")
    if pos_files:
        print("Example positive files:")
        for f in pos_files[:3]:
            print(f"  {f}")
    
    # Check negative files
    neg_files = glob(os.path.join(NEG_DB_PATH, '**/**/*.mp3'), recursive=True)
    print(f"Found {len(neg_files)} negative files")
    if neg_files:
        print("Example negative files:")
        for f in neg_files[:3]:
            print(f"  {f}")
            
    # Print the expected directory structure
    if neg_files:
        first_file = neg_files[0]
        parts = first_file.split('/')
        encoder = parts[-3]  # This should be 'griffin256' or similar
        print(f"\nExpected encoder: {encoder}")
        
    print("\nTrying to create AudioLoader...")
    try:
        loader = AudioLoader(POS_DB_PATH, NEG_DB_PATH, {"batch_size": 32}, split_path=SPLIT_PATH)
        print("AudioLoader created successfully!")
        print(f"Found {len(loader.pos_list)} positive files and {len(loader.neg_list)} negative files")
        print(f"Encoders found: {loader.encoders}")
    except Exception as e:
        print(f"Error creating AudioLoader: {e}")
        
        # Try alternative paths
        print("\nTrying alternative paths...")
        try:
            # Use the actual structure we found
            alt_neg_path = os.path.dirname(os.path.dirname(neg_files[0]))
            print(f"Alternative NEG_DB_PATH: {alt_neg_path}")
            loader = AudioLoader(POS_DB_PATH, alt_neg_path, {"batch_size": 32}, split_path=SPLIT_PATH)
            print("AudioLoader created successfully with alternative path!")
            print(f"Found {len(loader.pos_list)} positive files and {len(loader.neg_list)} negative files")
            print(f"Encoders found: {loader.encoders}")
        except Exception as e:
            print(f"Error with alternative path: {e}")

if __name__ == "__main__":
    main()
