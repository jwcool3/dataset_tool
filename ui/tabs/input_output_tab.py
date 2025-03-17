"""
Input/Output Tab for Dataset Preparation Tool
Contains directory selection and processing pipeline options.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog

class InputOutputTab:
    """Tab for input/output directory selection and pipeline configuration."""
    
    def __init__(self, parent):
        """
        Initialize the input/output tab.
        
        Args:
            parent: Parent window containing shared variables and functions
        """
        self.parent = parent
        self.frame = ttk.Frame(parent.notebook, padding="10")
        
        # Create the UI components
        self._create_directory_section()
        self._create_pipeline_section()
    
    # Update to the _create_directory_section in ui/tabs/input_output_tab.py

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
# ...existing code...
    def _create_pipeline_section(self):
        """Create the processing pipeline section with checkboxes."""
        pipeline_frame = ttk.LabelFrame(self.frame, text="Processing Pipeline", padding="10")
        pipeline_frame.pack(fill=tk.X, pady=5)
        
        # Processing options with improved layout
        processing_options = [
            ("Extract frames from videos", self.parent.extract_frames),
            ("Detect and crop mask regions", self.parent.crop_mask_regions),
            ("Expand mask regions", self.parent.expand_masks),  # Ensure this exists
            ("Resize images and masks", self.parent.resize_images),
            ("Organize and rename files", self.parent.organize_files),
            ("Convert images to video", self.parent.convert_to_video),
            ("Add padding to make images square", self.parent.square_pad_images),
            ("Reinsert cropped images", self.parent.reinsert_crops_option)
        ]
        
        # Use a cleaner approach with fewer loops
        for i, (text, var) in enumerate(processing_options):
            row = i % 3  # 3 options per row
            col = i // 3
            checkbutton = ttk.Checkbutton(pipeline_frame, text=text, variable=var)
            checkbutton.grid(column=col, row=row, sticky=tk.W, padx=10, pady=5)
            
            # Highlight the mask expansion option to make it more noticeable
            if text == "Expand mask regions":
                checkbutton.configure(style="Accent.TCheckbutton")
        
        # Debug mode checkbox (separate for visibility)
        ttk.Checkbutton(pipeline_frame, text="Debug Mode (Save visualization images)", 
                    variable=self.parent.debug_mode).grid(
            column=0, row=3, columnspan=2, sticky=tk.W, padx=10, pady=5
        )
        
        # Add hint about standalone processing
        hint_frame = ttk.Frame(self.frame, padding="10")
        hint_frame.pack(fill=tk.X, pady=5)
        
        hint_text = ("Hint: The processing pipeline executes steps in the order shown above. " + 
                    "Each step can be run individually or as part of a sequence. " +
                    "For example, you can select only 'Expand mask regions' to process just the masks.")
        
        hint_label = ttk.Label(hint_frame, text=hint_text, foreground="gray", wraplength=600)
        hint_label.pack(anchor=tk.W)
    # ...existing code...
    
        # Add command to show/hide reinsertion note when the checkbox is toggled
        def on_reinsertion_toggle():
            if self.parent.reinsert_crops_option.get():
                self.reinsertion_note_frame.grid()  # Show the note
            else:
                self.reinsertion_note_frame.grid_remove()  # Hide the note
        
        # Create checkboxes for each option
        for i, (text, var) in enumerate(processing_options):
            row = i % 3
            col = i // 3
            cb = ttk.Checkbutton(pipeline_frame, text=text, variable=var)
            cb.grid(column=col, row=row, sticky=tk.W, padx=10, pady=2)
            
            # Add special handling for reinsertion option
            if text == "Reinsert cropped images":
                cb.config(command=on_reinsertion_toggle)

    
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
        else:
            # If no mask pairs but we have videos, enable resize and organize
            if has_videos:
                self.parent.resize_images.set(True)
                self.parent.organize_files.set(True)
        
        # Update status
        self.parent.status_label.config(text=f"Found: {'videos, ' if has_videos else ''}{'image-mask pairs' if has_image_mask_pairs else 'no image-mask pairs'}")
    
    def _load_preview_images(self):
        """Load sample images for preview."""
        # This functionality will be initialized here and completed in later stages
        from utils.image_utils import load_image_with_mask
        
        input_dir = self.parent.input_dir.get()
        image_path, mask_path = load_image_with_mask(input_dir)
        
        if image_path:
            self.parent.preview_tab.load_preview(image_path, mask_path)
    
    def _preview_processing(self):
        """Preview the processing that would be applied."""
        # Select the Preview tab to show results
        self.parent.notebook.select(2)  # Index of the Preview tab
        
        # The actual preview will be implemented later
        # This just changes to the preview tab for now
        if self.parent.preview_image is None:
            tk.messagebox.showinfo("Preview", "Please select an input directory with images first.")
            return
        
        self.parent.preview_tab.generate_preview()


    def _create_pipeline_section(self):
        """Create the processing pipeline section with checkboxes."""
        pipeline_frame = ttk.LabelFrame(self.frame, text="Processing Pipeline", padding="10")
        pipeline_frame.pack(fill=tk.X, pady=5)
        
        # Processing options with improved layout
        processing_options = [
            ("Extract frames from videos", self.parent.extract_frames),
            ("Detect and crop mask regions", self.parent.crop_mask_regions),
            ("Resize images and masks", self.parent.resize_images),
            ("Organize and rename files", self.parent.organize_files),
            ("Convert images to video", self.parent.convert_to_video),
            ("Add padding to make images square", self.parent.square_pad_images),
            ("Reinsert cropped images", self.parent.reinsert_crops_option)  # New option
        ]
        
        # Calculate layout - 3 rows, dynamic columns
        rows = 3
        cols = (len(processing_options) + rows - 1) // rows  # Ceiling division
        
        # Create grid of options
        for i, (text, var) in enumerate(processing_options):
            row = i % rows
            col = i // rows
            ttk.Checkbutton(pipeline_frame, text=text, variable=var).grid(
                column=col, row=row, sticky=tk.W, padx=10, pady=2
            )
        
        # Debug mode checkbox (separate for visibility)
        ttk.Checkbutton(pipeline_frame, text="Debug Mode (Save visualization images)", 
                    variable=self.parent.debug_mode).grid(
            column=0, row=rows, columnspan=cols, sticky=tk.W, padx=10, pady=5
        )

        def on_reinsert_toggle():
            """Called when the reinsertion option is toggled"""
            if self.parent.reinsert_crops_option.get():
                # Show a hint about the input directory
                if not hasattr(self, 'reinsert_hint_label'):
                    self.reinsert_hint_label = ttk.Label(
                        pipeline_frame, 
                        text="Note: Input Directory = Cropped Images, Source Directory = Original Images", 
                        foreground="blue"
                    )
                    self.reinsert_hint_label.grid(
                        column=0, row=4, columnspan=3, sticky=tk.W, padx=20, pady=5
                    )
            elif hasattr(self, 'reinsert_hint_label'):
                # Hide the hint
                self.reinsert_hint_label.grid_forget()

        # Add a command to the reinsert checkbox
        for i, (text, var) in enumerate(processing_options):
            if text == "Reinsert cropped images":
                reinsert_idx = i
                break
                
        # Find the reinsert checkbutton widget
        for widget in pipeline_frame.winfo_children():
            if isinstance(widget, ttk.Checkbutton) and widget.grid_info()['row'] == reinsert_idx % 3 and widget.grid_info()['column'] == reinsert_idx // 3:
                widget.configure(command=on_reinsert_toggle)
                break