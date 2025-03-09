#!/usr/bin/env python

"""
Wrapper script for training to ensure the correct Python path
"""

import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the training script
from scripts.train import main

if __name__ == "__main__":
    main()
