"""
Dialog windows for Dataset Preparation Tool
Contains about dialog, usage guide, and other popup windows.
"""

import tkinter as tk
from tkinter import ttk

class AboutDialog:
    """Dialog showing information about the application."""
    
    def __init__(self, parent):
        """
        Initialize the about dialog.
        
        Args:
            parent: Parent window
        """
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("About")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)  # Make dialog modal
        
        # Make the window pop to the front
        self.dialog.lift()
        self.dialog.attributes('-topmost', True)
        self.dialog.after_idle(self.dialog.attributes, '-topmost', False)
        
        # Create content
        self._create_content()
        
        # Wait for the dialog to close
        self.dialog.focus_set()
        self.dialog.grab_set()
    
    def _create_content(self):
        """Create the dialog content."""
        frame = ttk.Frame(self.dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        about_text = """Dataset Preparation Tool

Version 1.2

This tool helps prepare image datasets for AI model training by:
- Extracting frames from videos
- Cropping and resizing image-mask pairs
- Organizing and renaming files
- Converting image sequences to videos

Features:
- Advanced mask detection and processing
- Configurable processing pipeline
- Preview of processing effects
- Debug mode for visualizations
- Source resolution preservation option

Created with Python and Tkinter.
"""
        
        title_label = ttk.Label(frame, text="Dataset Preparation Tool", font=("Helvetica", 14, "bold"))
        title_label.pack(pady=(0, 10))
        
        text_widget = tk.Text(frame, wrap=tk.WORD, height=12, width=40)
        text_widget.pack(fill=tk.BOTH, expand=True, pady=5)
        text_widget.insert(tk.END, about_text)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        ok_button = ttk.Button(frame, text="OK", command=self.dialog.destroy)
        ok_button.pack(pady=10)


class UsageGuideDialog:
    """Dialog showing usage instructions for the application."""
    
    def __init__(self, parent):
        """
        Initialize the usage guide dialog.
        
        Args:
            parent: Parent window
        """
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Usage Guide")
        self.dialog.geometry("600x500")
        self.dialog.transient(parent)  # Make dialog modal
        
        # Make the window pop to the front
        self.dialog.lift()
        self.dialog.attributes('-topmost', True)
        self.dialog.after_idle(self.dialog.attributes, '-topmost', False)
        
        # Create content
        self._create_content()
        
        # Wait for the dialog to close
        self.dialog.focus_set()
        self.dialog.grab_set()
    
    def _create_content(self):
        """Create the dialog content."""
        frame = ttk.Frame(self.dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        guide_text = """Dataset Preparation Tool - Usage Guide:

1. Select Input Directory:
   - Choose a folder containing your source images, masks, or videos
   - The tool will automatically detect content types

2. Select Output Directory:
   - Choose where processed files will be saved
   - Subdirectories will be created for each processing step

3. Choose Processing Steps:
   - Extract frames: Convert videos to image sequences
   - Crop mask regions: Focus crops on the masked areas using advanced detection
   - Resize images: Scale images to target dimensions
   - Organize files: Rename and structure files
   - Convert to video: Create videos from image sequences

4. Configure Options:
   - Frame rate: How many frames to extract per second of video
   - Mask padding: Percentage of padding around detected mask regions
   - Use source resolution: Keep original video resolution instead of resizing
   - Output resolution: Target dimensions for resized images (when not using source resolution)
   - Naming pattern: Format for output filenames
   - Video FPS: Frame rate for generated videos
   - Debug mode: Save visualization images showing mask detection

5. Preview Processing:
   - See the effect of your settings on sample images
   - Helps verify that mask detection works correctly

6. Start Processing:
   - Run the selected steps on all files in the input directory

Tips:
- For mask detection, masks should be in a 'masks' subfolder
- The C/N organization option works when subfolders start with C or N
- Use debug mode to troubleshoot mask detection issues
- Save your configuration for reuse on similar datasets
- Processing pipeline will intelligently chain outputs between steps
"""
        
        title_label = ttk.Label(frame, text="Usage Guide", font=("Helvetica", 14, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Create a scrollable text widget
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        text_widget.insert(tk.END, guide_text)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        ok_button = ttk.Button(frame, text="OK", command=self.dialog.destroy)
        ok_button.pack(pady=10)