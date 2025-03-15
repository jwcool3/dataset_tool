"""
Dataset Preparation Tool - Main Application
Initializes the UI and connects it with processing functionality.
"""

import tkinter as tk
import sys
import os

# Add the current directory to the path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import MainWindow

class DatasetPreparationApp:
    """Main application class that initializes and runs the application."""
    
    def __init__(self):
        """Initialize the application."""
        self.root = tk.Tk()
        self.root.title("Dataset Preparation Tool")
        
        # Set a minimum window size
        self.root.minsize(900, 700)
        
        # Add window icon if available
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass  # No icon available, continue without it
        
        # Center the window on screen
        window_width = 900
        window_height = 700
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Initialize the main window and UI
        self.main_window = MainWindow(self.root)
    
    def run(self):
        """Run the application main loop."""
        self.root.mainloop()