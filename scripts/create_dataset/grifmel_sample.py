import numpy as np
"""
Generate Griffin-Mel reconstructions for a sample of audio files
"""

import os
import sys
import argparse
from glob import glob

import torch

# Add the repo root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ae_models.griffinmel import GriffinMel
from ae_models.pipeline_sample import SamplePipeline
from loader.global_variables import *

def griffin_mel_sample(sample_size=100, device="cuda", gpu_id=0):
    # Set GPU if using CUDA
    if device == "cuda" and gpu_id >= 0:
        torch.cuda.set_device(gpu_id)
        print(f"Using GPU {gpu_id}")
    
    # Configuration
    griffin_conf = {
        "DEVICE": device,
        "DB_PATH": POS_DB_PATH,
        "OUT_DB": NEG_DB_PATH,
        "BR_PATH": BR_PATH,
        "DATA_SR": 44100,
        "SR": 44100,  # target sr
        "MIN_DURATION": 3,  # seconds
        "MAX_DURATION": 40,
        "VERBOSE": True,
    }
    
    # Get all MP3 paths from the split file or directly from the folder
    if os.path.exists(SPLIT_PATH):
        try:
            split_dict = np.load(SPLIT_PATH, allow_pickle=True).item()
            all_files = split_dict['train'] + split_dict['validation'] + split_dict['test']
            # Convert filenames to full paths
            all_mp3_paths = []
            for filename in all_files:
                # Find the file in the source directory (might be in subfolders)
                matches = glob(os.path.join(POS_DB_PATH, '**', filename), recursive=True)
                if matches:
                    all_mp3_paths.append(matches[0])
        except Exception as e:
            print(f"Error loading split file: {e}")
            all_mp3_paths = glob(os.path.join(POS_DB_PATH, '**/*.mp3'), recursive=True)
    else:
        all_mp3_paths = glob(os.path.join(POS_DB_PATH, '**/*.mp3'), recursive=True)
    
    print(f"Found {len(all_mp3_paths)} mp3 paths")
    
    # Load models
    riffusion_params = {
        "n_fft": int(400 / 1000 * griffin_conf["SR"]),  # /SR = 46ms
        "hop_fft": int(10 / 1000 * griffin_conf["SR"]),
        "win_fft": int(100 / 1000 * griffin_conf["SR"]),
        "griffin_iter": 32,
        "n_mels": 512,
    }
    
    riffusion_256 = dict(riffusion_params)
    riffusion_256["n_mels"] = 256
    
    print("Initializing Griffin-Mel models...")
    ae_griffinmel_512 = GriffinMel(riffusion_params, griffin_conf["SR"], griffin_conf["DEVICE"])
    ae_griffinmel_256 = GriffinMel(riffusion_256, griffin_conf["SR"], griffin_conf["DEVICE"])
    
    models = [ae_griffinmel_512, ae_griffinmel_256]
    out_name = ['griffin512', 'griffin256']
    
    # Run pipeline
    print("Starting audio processing pipeline...")
    pipeline = SamplePipeline(all_mp3_paths, griffin_conf)
    pipeline.run_loop(models, out_name, max_files=sample_size)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Griffin-Mel reconstructions for a sample of audio files")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of audio files to process")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()
    
    griffin_mel_sample(args.sample_size, args.device, args.gpu)
