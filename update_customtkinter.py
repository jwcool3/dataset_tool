#!/usr/bin/env python3
"""
Update script to replace CustomTkinter imports with compatibility layer.
"""

import os
import sys
import re
import shutil
from pathlib import Path

def backup_file(file_path):
    """Create a backup of the file with .bak extension."""
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")

def update_imports(file_path):
    """Update imports in the file to use the compatibility layer."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace direct customtkinter imports
    import_patterns = [
        (r'import\s+customtkinter\s+as\s+ctk', 'from ui.compatibility import ctk, get_widget'),
        (r'from\s+customtkinter\s+import\s+(.*)', 'from ui.compatibility import \\1'),
        (r'import\s+customtkinter', 'from ui.compatibility import ctk, get_widget')
    ]
    
    for pattern, replacement in import_patterns:
        content = re.sub(pattern, replacement, content)
    
    # Save updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def update_directory(directory):
    """Update all Python files in the directory and its subdirectories."""
    print(f"Updating files in {directory}...")
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}")
                backup_file(file_path)
                update_imports(file_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update_customtkinter.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)
    
    update_directory(directory)
    print("Done! All Python files have been updated.")