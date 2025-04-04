"""
Input/Output Tab for Dataset Preparation Tool
Contains directory selection and processing pipeline options.
"""

import os
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk

class InputOutputTab:
    """Tab for input/output directory selection and pipeline configuration."""
    
    def __init__(self, parent):
        """
        Initialize the input/output tab.
        
        Args:
            parent: Parent frame to contain this tab's content
        """
        # Store the parent frame reference
        self.parent_frame = parent
        
        # Get reference to the main window (needed for variables and methods)
        if hasattr(parent, 'master') and hasattr(parent.master, 'master'):
            self.main_window = parent.master.master  # tab_content_frame -> notebook_frame -> MainWindow
        else:
            # Fallback for testing/direct instantiation
            self.main_window = parent
        
        # Create the main frame with CustomTkinter
        self.frame = ctk.CTkFrame(parent)
        
        # Create the UI components with modern styling
        self._create_directory_section()
        self._create_pipeline_section()
    
    def _create_directory_section(self):
        """Create the directory selection section with modern UI."""
        # Create a card-like frame for directory selection
        io_frame = ctk.CTkFrame(self.frame, corner_radius=10)
        io_frame.pack(fill=tk.X, padx=20, pady=15)
        
        # Section header
        header = ctk.CTkLabel(io_frame, text="Directory Selection", font=ctk.CTkFont(size=16, weight="bold"))
        header.pack(anchor=tk.W, padx=15, pady=(15, 10))
        
        # Input directory row
        input_row = ctk.CTkFrame(io_frame, fg_color="transparent")
        input_row.pack(fill=tk.X, padx=15, pady=5)
        
        input_label = ctk.CTkLabel(input_row, text="Input Directory:", width=120, anchor="w")
        input_label.pack(side=tk.LEFT, padx=(0, 10))
        
        input_entry = ctk.CTkEntry(input_row, textvariable=self.main_window.input_dir, width=350)
        input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        input_button = ctk.CTkButton(
            input_row, 
            text="Browse...", 
            command=self._browse_input_dir,
            width=100,
            height=30
        )
        input_button.pack(side=tk.LEFT, padx=(0, 10))
        
        preview_button = ctk.CTkButton(
            input_row,
            text="Preview Processing",
            command=self._preview_processing,
            width=150,
            height=30,
            fg_color="#1E90FF",
            hover_color="#1670CD"
        )
        preview_button.pack(side=tk.LEFT)
        
        # Output directory row
        output_row = ctk.CTkFrame(io_frame, fg_color="transparent")
        output_row.pack(fill=tk.X, padx=15, pady=5)
        
        output_label = ctk.CTkLabel(output_row, text="Output Directory:", width=120, anchor="w")
        output_label.pack(side=tk.LEFT, padx=(0, 10))
        
        output_entry = ctk.CTkEntry(output_row, textvariable=self.main_window.output_dir, width=350)
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        output_button = ctk.CTkButton(
            output_row, 
            text="Browse...", 
            command=self._browse_output_dir,
            width=100,
            height=30
        )
        output_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Process and cancel buttons
        action_button = ctk.CTkButton(
            output_row,
            text="Start Processing",
            command=self.main_window.start_processing,
            width=150,
            height=30,
            fg_color="#4CAF50",
            hover_color="#388E3C"
        )
        action_button.pack(side=tk.LEFT)
        self.process_button = action_button
        
        # Create the cancel button (initially disabled)
        self.cancel_button = ctk.CTkButton(
            io_frame,
            text="Cancel",
            command=self.main_window.cancel_processing,
            width=100,
            height=30,
            fg_color="#F44336",
            hover_color="#D32F2F",
            state="disabled"
        )
        self.cancel_button.pack(side=tk.RIGHT, padx=15, pady=(0, 15))
        
        # Create the reinsertion note frame (initially hidden)
        self.reinsertion_note_frame = ctk.CTkFrame(
            io_frame, 
            corner_radius=5,
            fg_color="#FFF3E0", 
            border_width=1, 
            border_color="#FF9800"
        )
        self.reinsertion_note_frame.pack(fill=tk.X, padx=15, pady=(5, 15), ipady=5)
        
        self.reinsertion_note = ctk.CTkLabel(
            self.reinsertion_note_frame,
            text="REINSERTION MODE ACTIVE: Input Directory should contain your CROPPED IMAGES.\n"
                "Go to the Config tab to set the source directory that contains your ORIGINAL UNCROPPED IMAGES.",
            text_color="#E65100",
            font=ctk.CTkFont(weight="bold", size=12),
            wraplength=700
        )
        self.reinsertion_note.pack(padx=10, pady=5)
        
        # Initialize to hidden
        self.reinsertion_note_frame.pack_forget()
    
    def _browse_input_dir(self):
        """Browse for an input directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.main_window.input_dir.set(directory)
            
            # Auto-fill output directory with the input directory
            if not self.main_window.output_dir.get():
                self.main_window.output_dir.set(directory)
                
            # Try to load preview images
            self._load_preview_images()
            # Enable appropriate processing options based on content
            self._enable_appropriate_options(directory)

    def _browse_output_dir(self):
        """Browse for an output directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.main_window.output_dir.set(directory)
    
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
            self.main_window.extract_frames.set(True)
        else:
            self.main_window.extract_frames.set(False)
        
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
            self.main_window.crop_mask_regions.set(True)
            self.main_window.resize_images.set(True)
            self.main_window.organize_files.set(True)
        else:
            # If no mask pairs but we have videos, enable resize and organize
            if has_videos:
                self.main_window.resize_images.set(True)
                self.main_window.organize_files.set(True)
        
        # Update status with more modern UI feedback
        detected = []
        if has_videos:
            detected.append("videos")
        if has_image_mask_pairs:
            detected.append("image-mask pairs")
        
        if detected:
            status_text = f"Found: {', '.join(detected)}"
        else:
            status_text = "No processable content detected"
            
        self.main_window.status_label.configure(text=status_text)

    def _load_preview_images(self):
        """Load sample images for preview."""
        from utils.image_utils import load_image_with_mask
        
        input_dir = self.main_window.input_dir.get()
        image_path, mask_path = load_image_with_mask(input_dir)
        
        if image_path:
            self.main_window.preview_tab.load_preview(image_path, mask_path)
    
    def _preview_processing(self):
        """Preview the processing that would be applied."""
        # Select the Preview tab to show results
        self.main_window.tab_header.set("Preview")
        
        if self.main_window.preview_image is None:
            from tkinter import messagebox
            messagebox.showinfo("Preview", "Please select an input directory with images first.")
            return
        
        self.main_window.preview_tab.generate_preview()

    def _create_pipeline_section(self):
        """Create the processing pipeline section with modern checkboxes."""
        # Create a card-like frame for pipeline options
        pipeline_frame = ctk.CTkFrame(self.frame, corner_radius=10)
        pipeline_frame.pack(fill=tk.X, padx=20, pady=15)
        
        # Section header
        header = ctk.CTkLabel(pipeline_frame, text="Processing Pipeline", font=ctk.CTkFont(size=16, weight="bold"))
        header.pack(anchor=tk.W, padx=15, pady=(15, 10))
        
        # Create a frame for the checkboxes with grid layout
        options_frame = ctk.CTkFrame(pipeline_frame, fg_color="transparent")
        options_frame.pack(fill=tk.X, padx=15, pady=5)
        
        # Define processing options in a clearer grouping
        processing_options = [
            # Video Processing
            ("Extract frames from videos", self.main_window.extract_frames, 0, 0),
            ("Convert images to video", self.main_window.convert_to_video, 0, 1),
            
            # Image Processing
            ("Detect and crop mask regions", self.main_window.crop_mask_regions, 1, 0),
            ("Expand mask regions", self.main_window.expand_masks, 1, 1),
            
            # Image Transformations
            ("Resize images and masks", self.main_window.resize_images, 2, 0),
            ("Add padding to make images square", self.main_window.square_pad_images, 2, 1),
            
            # Output Operations
            ("Organize and rename files", self.main_window.organize_files, 3, 0),
            ("Reinsert cropped images", self.main_window.reinsert_crops_option, 3, 1),
            
            # Export Options
            ("Export cropped areas only (no reinsertion)", self.main_window.export_cropped_only, 4, 0),
            ("Debug Mode (Save visualization images)", self.main_window.debug_mode, 4, 1)
        ]
        
        # Create the checkboxes with improved styling
        for text, var, row, col in processing_options:
            checkbox = ctk.CTkCheckBox(
                options_frame, 
                text=text, 
                variable=var,
                corner_radius=4,
                border_width=2,
                checkbox_width=24,
                checkbox_height=24
            )
            checkbox.grid(row=row, column=col, sticky=tk.W, padx=15, pady=8)
            
            # Highlight expand mask regions option
            if text == "Expand mask regions":
                checkbox.configure(
                    fg_color="#5E35B1",
                    hover_color="#4527A0",
                    border_color="#7E57C2"
                )
            
            # Add special command to the reinsert checkbox
            if text == "Reinsert cropped images":
                checkbox.configure(command=self._on_reinsert_toggle)
        
        # Add hint about standalone processing with better styling
        hint_frame = ctk.CTkFrame(
            self.frame, 
            fg_color="#E3F2FD", 
            corner_radius=5,
            border_width=1,
            border_color="#BBDEFB"
        )
        hint_frame.pack(fill=tk.X, padx=20, pady=15)
        
        hint_icon = ctk.CTkLabel(
            hint_frame, 
            text="💡", 
            font=ctk.CTkFont(size=20),
            text_color="#1976D2"
        )
        hint_icon.pack(side=tk.LEFT, padx=(15, 5), pady=15)
        
        hint_text = ("The processing pipeline executes steps in the order shown above. " + 
                    "Each step can be run individually or as part of a sequence. " +
                    "For example, you can select only 'Expand mask regions' to process just the masks.")
        
        hint_label = ctk.CTkLabel(
            hint_frame, 
            text=hint_text, 
            text_color="#0D47A1",
            font=ctk.CTkFont(size=12),
            wraplength=600,
            justify="left"
        )
        hint_label.pack(side=tk.LEFT, padx=(0, 15), pady=15)

    def _on_reinsert_toggle(self):
        """Called when the reinsertion option is toggled"""
        if self.main_window.reinsert_crops_option.get():
            # Show the reinsertion note
            self.reinsertion_note_frame.pack(fill=tk.X, padx=15, pady=(5, 15), ipady=5)
        else:
            # Hide the reinsertion note
            self.reinsertion_note_frame.pack_forget()