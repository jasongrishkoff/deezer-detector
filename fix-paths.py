#!/usr/bin/env python
"""
Update global_variables.py to point to the correct file paths
"""

import os

# Path to global_variables.py
globals_path = "loader/global_variables.py"

# Read current content
with open(globals_path, 'r') as f:
    content = f.read()

# Create a backup
with open(globals_path + '.bak', 'w') as f:
    f.write(content)

# Update paths in content
updated_content = content.replace(
    "NEG_DB_PATH = HOME+\"/deezer-detector/temp_fake\"", 
    "NEG_DB_PATH = HOME+\"/deezer-detector/fma_rebuilt\""
)

# Write updated content
with open(globals_path, 'w') as f:
    f.write(updated_content)

print(f"Updated {globals_path}")
print("Created backup at {globals_path}.bak")
print("Changed NEG_DB_PATH to point to /workspace/deezer-detector/fma_rebuilt")
