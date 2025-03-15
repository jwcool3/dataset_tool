#!/usr/bin/env python3
"""
Setup script for Dataset Preparation Tool.
Creates the required directory structure and initializes empty __init__.py files.
"""

import os
import sys

def create_directory_structure():
    """Create the directory structure for the Dataset Preparation Tool."""
    # Define the directory structure
    directories = [
        "ui",
        "ui/tabs",
        "processors",
        "utils"
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create __init__.py if it doesn't exist
        init_file = os.path.join(directory, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write('"""{}"""'.format(directory.replace("/", " ").title()))
    
    print("Directory structure created successfully.")

if __name__ == "__main__":
    create_directory_structure()
    print("Setup completed. You can now run the application with 'python main.py'.")