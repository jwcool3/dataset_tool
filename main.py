#!/usr/bin/env python3
"""
Dataset Preparation Tool - Main Entry Point
Launches the application and handles dependency checks.
"""

import tkinter as tk
from tkinter import messagebox
import sys
import traceback

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = {
        "opencv-python": "cv2",
        "pillow": "PIL",
        "numpy": "numpy"
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name.split(".")[0])
        except ImportError:
            missing_packages.append(package_name)
    
    return missing_packages

def main():
    """Main function to launch the application."""
    # Check for required dependencies
    missing_packages = check_dependencies()
    
    if missing_packages:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        error_message = "Missing required packages:\n"
        error_message += ", ".join(missing_packages)
        error_message += "\n\nPlease install the required packages using:\n"
        error_message += "pip install " + " ".join(missing_packages)
        
        messagebox.showerror("Missing Dependencies", error_message)
        root.destroy()
        return
    
    try:
        # Add the current directory to the path to ensure imports work correctly
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Only import the application after dependencies are checked
        from app import DatasetPreparationApp
        
        # Create and run the application
        app = DatasetPreparationApp()
        app.run()
        
    except Exception as e:
        # Handle any startup errors
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        error_message = f"Error starting application: {str(e)}\n\n"
        error_message += traceback.format_exc()
        
        messagebox.showerror("Startup Error", error_message)
        root.destroy()

if __name__ == "__main__":
    main()