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
        self.frame = ttk.Frame(parent.notebook, padding="10")
        
        # Create the UI components
        self._create_directory_section()
        self._create_pipeline_section()
    
    def _create_directory_section(self):
        """Create the directory selection section with clearer labeling."""
        io_frame = ttk.LabelFrame(self.frame, text="Directory Selection", padding="10")
        io_frame.pack(fill=tk.X, pady=5)
        
        # Standard mode frame and reinsertion mode frame (we'll toggle between them)
        self.standard_frame = ttk.Frame(io_frame)
        self.standard_frame.pack(fill=tk.X)
        
        self.reinsertion_frame = ttk.Frame(io_frame)
        # Initially don't pack this - we'll toggle it
        
        # STANDARD MODE UI
        # Input directory with standard label
        ttk.Label(self.standard_frame, text="Input Directory:").grid(column=0, row=0, sticky=tk.W)
        ttk.Entry(self.standard_frame, textvariable=self.parent.input_dir, width=50).grid(column=1, row=0, padx=5, sticky=tk.W)
        ttk.Button(self.standard_frame, text="Browse...", command=self._browse_input_dir).grid(column=2, row=0, padx=5)
        
        # Output directory
        ttk.Label(self.standard_frame, text="Output Directory:").grid(column=0, row=1, sticky=tk.W, pady=5)
        ttk.Entry(self.standard_frame, textvariable=self.parent.output_dir, width=50).grid(column=1, row=1, padx=5, sticky=tk.W)
        ttk.Button(self.standard_frame, text="Browse...", command=self._browse_output_dir).grid(column=2, row=1, padx=5)
        
        # REINSERTION MODE UI with clearer labels
        # Processed images directory (with better label)
        ttk.Label(self.reinsertion_frame, 
                 text="Processed Images Directory:", 
                 font=("Helvetica", 9, "bold")).grid(column=0, row=0, sticky=tk.W)
        ttk.Entry(self.reinsertion_frame, textvariable=self.parent.input_dir, width=50).grid(column=1, row=0, padx=5, sticky=tk.W)
        ttk.Button(self.reinsertion_frame, text="Browse...", command=self._browse_input_dir).grid(column=2, row=0, padx=5)
        
        # Output directory
        ttk.Label(self.reinsertion_frame, text="Output Directory:").grid(column=0, row=1, sticky=tk.W, pady=5)
        ttk.Entry(self.reinsertion_frame, textvariable=self.parent.output_dir, width=50).grid(column=1, row=1, padx=5, sticky=tk.W)
        ttk.Button(self.reinsertion_frame, text="Browse...", command=self._browse_output_dir).grid(column=2, row=1, padx=5)
        
        # Source (original) images directory
        ttk.Label(self.reinsertion_frame, 
                 text="Original Images Directory:", 
                 font=("Helvetica", 9, "bold"),
                 foreground="blue").grid(column=0, row=2, sticky=tk.W, pady=5)
        ttk.Entry(self.reinsertion_frame, textvariable=self.parent.source_images_dir, width=50).grid(column=1, row=2, padx=5, sticky=tk.W)
        ttk.Button(self.reinsertion_frame, text="Browse...", command=self._browse_source_dir).grid(column=2, row=2, padx=5)
        
        # Help text for reinsertion mode
        help_frame = ttk.Frame(self.reinsertion_frame, padding=(5, 5, 5, 5), relief="groove", borderwidth=1)
        help_frame.grid(column=0, row=3, columnspan=3, sticky=tk.W+tk.E, pady=5)
        
        help_text = """
REINSERTION MODE: 
- Processed Images = Directory with cropped/processed hair images + masks folder
- Original Images = Directory with your original uncropped images to insert hair into
- Both directories should have matching filenames or follow similar naming patterns
"""
        ttk.Label(
            help_frame,
            text=help_text,
            foreground="purple",
            font=("Helvetica", 9),
            justify="left"
        ).pack(anchor=tk.W, pady=5, padx=5)
        
        # Process and Cancel buttons
        button_frame = ttk.Frame(io_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Preview button
        ttk.Button(button_frame, text="Preview Processing", 
                  command=self._preview_processing).grid(column=0, row=0, padx=5, pady=5)
        
        self.process_button = ttk.Button(button_frame, text="Start Processing", 
                                        command=self.parent.start_processing)
        self.process_button.grid(column=1, row=0, padx=5, pady=5)
        
        self.cancel_button = ttk.Button(button_frame, text="Cancel", 
                                       command=self.parent.cancel_processing, state=tk.DISABLED)
        self.cancel_button.grid(column=2, row=0, padx=5, pady=5)
        
        # Directory verification button - new feature to check directory setup
        self.verify_button = ttk.Button(button_frame, text="Verify Directories", 
                                      command=self._verify_directories)
        self.verify_button.grid(column=3, row=0, padx=5, pady=5)

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
    
    def _browse_source_dir(self):
        """Browse for a source directory (original images)."""
        directory = filedialog.askdirectory(title="Select Original/Source Images Directory")
        if directory:
            self.parent.source_images_dir.set(directory)
    
    def _verify_directories(self):
        """Verify the directories are set up correctly for the selected mode."""
        if self.parent.reinsert_crops_option.get():
            # Reinsertion mode - need both processed and original directories
            processed_dir = self.parent.input_dir.get()
            original_dir = self.parent.source_images_dir.get()
            
            if not processed_dir or not os.path.isdir(processed_dir):
                messagebox.showerror("Directory Error", "Processed images directory is not set or invalid.")
                return False
                
            if not original_dir or not os.path.isdir(original_dir):
                messagebox.showerror("Directory Error", "Original images directory is not set or invalid.")
                return False
                
            # Check for processed images and masks
            has_processed_images = False
            has_masks = False
            processed_image_count = 0
            mask_count = 0
            
            # Check processed directory for images
            for root, dirs, files in os.walk(processed_dir):
                if os.path.basename(root).lower() == "masks":
                    # Count masks in masks folder
                    mask_count += sum(1 for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg')))
                    if mask_count > 0:
                        has_masks = True
                else:
                    # Count images in non-mask folders
                    processed_image_count += sum(1 for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg')))
                    if processed_image_count > 0:
                        has_processed_images = True
            
            # Check original directory for images
            original_image_count = 0
            for root, dirs, files in os.walk(original_dir):
                if os.path.basename(root).lower() != "masks":  # Skip mask directories
                    original_image_count += sum(1 for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg')))
            
            # Build verification report
            report = f"""Directory Verification for Reinsertion:

Processed Images Directory: {processed_dir}
- Found {processed_image_count} image files
- Found {mask_count} mask files

Original Images Directory: {original_dir}
- Found {original_image_count} image files

Status: {"✓ Ready for reinsertion" if has_processed_images and has_masks and original_image_count > 0 
         else "❌ Missing required files"}

Recommendations:
- Ensure processed directory has images and a 'masks' subfolder
- Ensure original directory contains the uncropped images
- Image filenames should match between directories
"""
            if has_processed_images and has_masks and original_image_count > 0:
                messagebox.showinfo("Directory Verification", report)
            else:
                messagebox.showerror("Directory Verification Failed", report)
                
        else:
            # Standard mode - just verify the input directory
            input_dir = self.parent.input_dir.get()
            
            if not input_dir or not os.path.isdir(input_dir):
                messagebox.showerror("Directory Error", "Input directory is not set or invalid.")
                return False
                
            # Check for videos or images
            has_videos = False
            has_images = False
            
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                        has_videos = True
                    elif file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        has_images = True
            
            if not (has_videos or has_images):
                messagebox.showerror("Directory Error", "Input directory does not contain any video or image files.")
                return False
                
            messagebox.showinfo("Directory Verification", 
                             f"Directory {input_dir} is valid and contains {'videos' if has_videos else ''}"
                             f"{' and ' if has_videos and has_images else ''}"
                             f"{'images' if has_images else ''}.")
            
        return True
    
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
            
            # Add command to the reinsert checkbutton
            if text == "Reinsert cropped images":
                checkbutton.configure(command=self._toggle_reinsert_mode)
        
        # Debug mode checkbox (separate for visibility)
        ttk.Checkbutton(pipeline_frame, text="Debug Mode (Save visualization images)", 
                      variable=self.parent.debug_mode).grid(
            column=0, row=3, columnspan=2, sticky=tk.W, padx=10, pady=5
        )

        ttk.Checkbutton(
            pipeline_frame, 
            text="Export cropped areas only (no reinsertion)", 
            variable=self.parent.export_cropped_only
        ).grid(column=2, row=3, sticky=tk.W, padx=10, pady=5)
        
        # Add hint about standalone processing
        hint_frame = ttk.Frame(self.frame, padding="10")
        hint_frame.pack(fill=tk.X, pady=5)
        
        hint_text = ("Hint: The processing pipeline executes steps in the order shown above. " + 
                    "Each step can be run individually or as part of a sequence. " +
                    "For example, you can select only 'Expand mask regions' to process just the masks.")
        
        hint_label = ttk.Label(hint_frame, text=hint_text, foreground="gray", wraplength=600)
        hint_label.pack(anchor=tk.W)
    
    def _toggle_reinsert_mode(self):
        """Toggle between standard and reinsertion modes for the UI."""
        if self.parent.reinsert_crops_option.get():
            # Switch to reinsertion mode UI
            self.standard_frame.pack_forget()
            self.reinsertion_frame.pack(fill=tk.X)
            
            # Make sure source directory is set (try to use output directory if not)
            if not self.parent.source_images_dir.get():
                # Try to use a reasonable default: parent directory of input
                input_dir = self.parent.input_dir.get()
                if input_dir:
                    parent_dir = os.path.dirname(input_dir)
                    self.parent.source_images_dir.set(parent_dir)
            
            # Check if both directories are set and display a reminder if needed
            if self.parent.input_dir.get() and self.parent.source_images_dir.get():
                if self.parent.input_dir.get() == self.parent.source_images_dir.get():
                    messagebox.showwarning("Directory Warning", 
                                         "The Processed Images directory and Original Images directory "
                                         "are currently set to the same location. This will not work for "
                                         "reinsertion. Please set different directories.")
            
        else:
            # Switch to standard mode UI
            self.reinsertion_frame.pack_forget()
            self.standard_frame.pack(fill=tk.X)
    
    def update_ui(self):
        """Update the UI state based on current settings."""
        # Make sure the directory frames are in the right mode
        self._toggle_reinsert_mode()