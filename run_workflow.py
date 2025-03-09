"""
Run the complete workflow to detect AI-generated music
"""

import os
import sys
import argparse
import subprocess

def run_command(cmd, description):
    """Run a command and print output"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with return code {result.returncode}")
        return False
    
    print(f"\nSUCCESS: {description} completed")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run the complete AI music detection workflow")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of audio files to process")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--config", type=str, default="specnn_amplitude", help="Model configuration to use")
    parser.add_argument("--skip_dataset", action="store_true", help="Skip dataset creation (use if already done)")
    parser.add_argument("--skip_train", action="store_true", help="Skip training (use if already done)")
    args = parser.parse_args()
    
    # Create directories
    if not run_command(
        ["bash", "-c", "mkdir -p /workspace/deezer-detector/fma_rebuilt /workspace/deezer-detector/fma_codec /workspace/deezer-detector/weights /workspace/deezer-detector/results /workspace/deezer-detector/data"],
        "Creating directories"
    ):
        return
    
    # 1. Create sample dataset and split
    if not args.skip_dataset:
        if not run_command(
            [sys.executable, "create_sample_dataset.py", f"--sample_size={args.sample_size}"],
            "Creating sample dataset"
        ):
            return
            
        # 2. Extract bitrates
        if not run_command(
            [sys.executable, "extract_bitrates.py"],
            "Extracting bitrates"
        ):
            return
            
        # 3. Generate Griffin-Mel reconstructions
        if not run_command(
            [sys.executable, "scripts/create_dataset/grifmel_sample.py", f"--sample_size={args.sample_size}", f"--device={args.device}", f"--gpu={args.gpu}"],
            "Generating Griffin-Mel reconstructions"
        ):
            return
    
    # 4. Train model
    if not args.skip_train:
        if not run_command(
            [sys.executable, "scripts/train.py", f"--config={args.config}", f"--gpu={args.gpu}"],
            "Training model"
        ):
            return
    
    # 5. Evaluate model
    if not run_command(
        [sys.executable, "scripts/eval.py", f"--config={args.config}", f"--gpu={args.gpu}", "--steps=50"],
        "Evaluating model"
    ):
        return
    
    print("\n\nWorkflow completed successfully!")

if __name__ == "__main__":
    main()
