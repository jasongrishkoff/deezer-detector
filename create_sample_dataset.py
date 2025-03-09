"""
Create a sample dataset from your audio files and generate the train/val/test split
"""

import os
import sys
import numpy as np
from glob import glob
import random
import shutil

# Add the parent directory to the path to import global_variables
sys.path.append(os.path.join(os.getcwd()))
from loader.global_variables import *

def create_sample_dataset(sample_size=100, val_split=0.1, test_split=0.2):
    """
    Create a sample dataset and generate train/val/test split
    
    Args:
        sample_size: Number of audio files to sample from your dataset
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
    """
    print(f"Creating a sample dataset with {sample_size} files")
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
    
    # Get all audio files
    all_audio_files = glob(os.path.join(POS_DB_PATH, '**/*.mp3'), recursive=True)
    print(f"Found {len(all_audio_files)} audio files in {POS_DB_PATH}")
    
    # Take a random sample
    if len(all_audio_files) > sample_size:
        sampled_files = random.sample(all_audio_files, sample_size)
    else:
        sampled_files = all_audio_files
        print(f"Warning: Requested sample size {sample_size} is larger than available files ({len(all_audio_files)})")
    
    # Extract just the filenames (no path)
    sampled_filenames = [os.path.basename(file) for file in sampled_files]
    
    # Create train/val/test split
    np.random.seed(123)  # For reproducibility
    np.random.shuffle(sampled_filenames)
    
    val_idx = int(len(sampled_filenames) * (1 - val_split - test_split))
    test_idx = int(len(sampled_filenames) * (1 - test_split))
    
    train_files = sampled_filenames[:val_idx]
    val_files = sampled_filenames[val_idx:test_idx]
    test_files = sampled_filenames[test_idx:]
    
    # Create split dictionary
    split_dict = {
        'train': train_files,
        'validation': val_files,
        'test': test_files
    }
    
    # Save the split dictionary
    np.save(SPLIT_PATH, split_dict)
    
    print(f"Created dataset split with {len(train_files)} training, {len(val_files)} validation, and {len(test_files)} test files")
    print(f"Split dictionary saved to {SPLIT_PATH}")
    
    return sampled_files

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create a sample dataset and train/val/test split")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of audio files to sample")
    args = parser.parse_args()
    
    create_sample_dataset(args.sample_size)
