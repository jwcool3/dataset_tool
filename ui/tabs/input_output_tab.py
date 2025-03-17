"""
Input/Output Tab for Dataset Preparation Tool
Contains directory selection and processing pipeline options.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class InputOutputTab:
    """Tab for input/output directory selection and pipeline configuration."""
    
    def __init__(self, parent):
        """
        Initialize the input/output tab.
        
        Args:
            parent: Parent window containing shared variables and functions
        """
        self.parent = parent
        
        # Create a canvas with scrollbar for scrolling
        self.canvas = tk.Canvas(parent.notebook)
        self.scrollbar = ttk.Scrollbar(parent.notebook, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Create a frame inside the canvas
        self.frame = ttk.Frame(self.canvas, padding="10")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.frame, anchor="nw")
        
        # Pack the scrollbar and canvas
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Configure the canvas to update the scrollregion when the frame changes size
        self.frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Bind mousewheel scrolling
        self._bind_mousewheel(self.canvas)
        
        # Create the UI components
        self._create_directory_section()
        self._create_pipeline_section()
    
    def _on_frame_configure(self, event):
        """Update the scrollregion when the frame size changes."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        """Resize the frame when the canvas is resized."""
        width = event.width
        self.canvas.itemconfig(self.canvas_window, width=width)
    
    def _bind_mousewheel(self, widget):
        """Bind mousewheel to scroll the canvas."""
        def _on_mousewheel(event):
            widget.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # Bind for Windows/macOS
        widget.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Bind for Linux
        widget.bind_all("<Button-4>", lambda e: widget.yview_scroll(-1, "units"))
        widget.bind_all("<Button-5>", lambda e: widget.yview_scroll(1, "units"))
    
    def _create_directory_section(self):
        """Create the directory selection section."""
        io_frame = ttk.LabelFrame(self.frame, text="Directory Selection", padding="10")
        io_frame.pack(fill=tk.X, pady=5)
        
        # Input directory with dynamic label
        self.input_dir_label = ttk.Label(io_frame, text="Input Directory:")
        self.input_dir_label.grid(column=0, row=0, sticky=tk.W)
        ttk.Entry(io_frame, textvariable=self.parent.input_dir, width=50).grid(column=1, row=0, padx=5, sticky=tk.W)
        ttk.Button(io_frame, text="Browse...", command=self._browse_input_dir).grid(column=2, row=0, padx=5)
        
        # Output directory
        ttk.Label(io_frame, text="Output Directory:").grid(column=0, row=1, sticky=tk.W, pady=5)
        ttk.Entry(io_frame, textvariable=self.parent.output_dir, width=50).grid(column=1, row=1, padx=5, sticky=tk.W)
        ttk.Button(io_frame, text="Browse...", command=self._browse_output_dir).grid(column=2, row=1, padx=5)
        
        # Create the reinsertion note frame
        self.reinsertion_note_frame = ttk.Frame(io_frame, padding=(0, 5, 0, 0))
        self.reinsertion_note_frame.grid(column=0, row=2, columnspan=3, sticky=tk.W)
        self.reinsertion_note = ttk.Label(
            self.reinsertion_note_frame,
            text="Note: For Image Reinsertion, Input Directory = Cropped Images. Configure Source Directory in the Config tab.",
            foreground="blue",
            wraplength=600
        )
        self.reinsertion_note.pack(anchor=tk.W)
        
        # Initialize to hidden
        self.reinsertion_note_frame.grid_remove()
        
        # Register callback to update when reinsert option changes
        def update_reinsert_ui(*args):
            if self.parent.reinsert_crops_option.get():
                self.input_dir_label.config(text="Input Directory (Cropped Images):")
                self.reinsertion_note_frame.grid()
            else:
                self.input_dir_label.config(text="Input Directory:")
                self.reinsertion_note_frame.grid_remove()
        
        # Add the trace
        self.parent.reinsert_crops_option.trace_add("write", update_reinsert_ui)
        
        # Preview button
        ttk.Button(io_frame, text="Preview Processing", 
                command=self._preview_processing).grid(column=3, row=0, padx=5, pady=5)
        
        # Process and Cancel buttons
        self.process_button = ttk.Button(io_frame, text="Start Processing", 
                                        command=self.parent.start_processing)
        self.process_button.grid(column=3, row=1, padx=5, pady=5)
        
        self.cancel_button = ttk.Button(io_frame, text="Cancel", 
                                    command=self.parent.cancel_processing, state=tk.DISABLED)
        self.cancel_button.grid(column=4, row=1, padx=5, pady=5)
    
    def _create_pipeline_section(self):
        """Create the processing pipeline section with checkboxes."""
        pipeline_frame = ttk.LabelFrame(self.frame, text="Processing Pipeline", padding="10")
        pipeline_frame.pack(fill=tk.X, pady=5)
        
        # Processing options with improved layout
        processing_options = [
            ("Extract frames from videos", self.parent.extract_frames),
            ("Detect and crop mask regions", self.parent.crop_mask_regions),
            ("Expand mask regions", self.parent.expand_masks),  # Make sure this line is included
            ("Resize images and masks", self.parent.resize_images),
            ("Organize and rename files", self.parent.organize_files),
            ("Convert images to video", self.parent.convert_to_video),
            ("Add padding to make images square", self.parent.square_pad_images),
            ("Reinsert cropped images", self.parent.reinsert_crops_option)
        ]   
        
        # Create 2 columns of options
        for i, (text, var) in enumerate(processing_options):
            row = i % 3
            col = i // 3
            ttk.Checkbutton(pipeline_frame, text=text, variable=var).grid(
                column=col, row=row, sticky=tk.W, padx=10, pady=2
            )
        
        # Debug mode checkbox (separate for visibility)
        ttk.Checkbutton(pipeline_frame, text="Debug Mode (Save visualization images)", 
                       variable=self.parent.debug_mode).grid(
            column=0, row=3, columnspan=2, sticky=tk.W, padx=10, pady=5
        )
        
        # Add section for pipeline help/hint
        hint_frame = ttk.Frame(self.frame, padding="10")
        hint_frame.pack(fill=tk.X, pady=5)
        
        hint_text = "Hint: The processing pipeline executes steps in the order shown above. " + \
                   "Output from each step will be used as input for the next step."
        hint_label = ttk.Label(hint_frame, text=hint_text, foreground="gray", wraplength=600)
        hint_label.pack(anchor=tk.W)
    
    def _browse_input_dir(self):
        """Browse for an input directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.parent.input_dir.set(directory)
            
            # Auto-fill output directory with the input directory
            if not self.parent.output_dir.get():
                self.parent.output_dir.set(directory)
                
            # Try to load preview images
            self._load_preview_images()
            # Enable appropriate processing options based on content
            self._enable_appropriate_options(directory)

    def _browse_output_dir(self):
        """Browse for an output directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.parent.output_dir.set(directory)
    
    def _enable_appropriate_options(self, directory):
        """Enable processing options based on directory content."""
        # Check if directory exists
        if not os.path.isdir(directory):
            return
        
        # Look for video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        has_videos = False
        
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    has_videos = True
                    break
            if has_videos:
                break
        
        # Enable/disable video frame extraction
        if has_videos:
            self.parent.extract_frames.set(True)
        else:
            self.parent.extract_frames.set(False)
        
        # Check for image-mask pairs
        has_image_mask_pairs = False
        for root, dirs, files in os.walk(directory):
            # Skip if this is already a 'masks' directory
            if os.path.basename(root).lower() == "masks":
                continue
            
            # Check if there's a 'masks' subdirectory
            mask_dir = os.path.join(root, "masks")
            if os.path.isdir(mask_dir):
                # Check if there are matching images and masks
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        mask_path = os.path.join(mask_dir, file)
                        if os.path.exists(mask_path):
                            has_image_mask_pairs = True
                            break
            
            if has_image_mask_pairs:
                break
        
        # Enable/disable mask-related options
        if has_image_mask_pairs:
            self.parent.crop_mask_regions.set(True)
            self.parent.resize_images.set(True)
            self.parent.organize_files.set(True)
            # Enable mask expansion by default if we have mask pairs
            self.parent.expand_masks.set(True)
        else:
            # If no mask pairs but we have videos, enable resize and organize
            if has_videos:
                self.parent.resize_images.set(True)
                self.parent.organize_files.set(True)
            # Disable mask expansion if we have no masks
            self.parent.expand_masks.set(False)
        
        # Update status
        self.parent.status_label.config(text=f"Found: {'videos, ' if has_videos else ''}{'image-mask pairs' if has_image_mask_pairs else 'no image-mask pairs'}")
    
    def _load_preview_images(self):
        """Load sample images for preview."""
        # Use image_utils to find and load a sample image and mask
        from utils.image_utils import load_image_with_mask
        
        input_dir = self.parent.input_dir.get()
        image_path, mask_path = load_image_with_mask(input_dir)
        
        if image_path:
            self.parent.preview_tab.load_preview(image_path, mask_path)
        else:
            self.parent.status_label.config(text="No suitable images found for preview")
    
    def _preview_processing(self):
        """Preview the processing that would be applied."""
        # Select the Preview tab to show results
        self.parent.notebook.select(2)  # Index of the Preview tab
        
        # Check if we have an image to preview
        if self.parent.preview_image is None:
            messagebox.showinfo("Preview", "Please select an input directory with images first.")
            return
        
        # Generate the preview
        self.parent.preview_tab.generate_preview()