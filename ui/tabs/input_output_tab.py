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
        self.reinsertion_note_frame = ttk.Frame(io_frame, padding=(5, 5, 5, 5), relief="groove", borderwidth=2)
        self.reinsertion_note_frame.grid(column=0, row=2, columnspan=3, sticky=tk.W+tk.E, pady=10)
        
        self.reinsertion_note = ttk.Label(
            self.reinsertion_note_frame,
            text="REINSERTION MODE ACTIVE: Input Directory should contain your CROPPED IMAGES.\n"
                "Go to the Config tab to set the source directory that contains your ORIGINAL UNCROPPED IMAGES.",
            foreground="red",
            font=("Helvetica", 9, "bold"),
            wraplength=600
        )
        self.reinsertion_note.pack(anchor=tk.W, pady=5)
        
        # Initialize to hidden
        self.reinsertion_note_frame.grid_remove()
        
        # Preview button - MAKE SURE THIS IS INCLUDED
        ttk.Button(io_frame, text="Preview Processing", 
                command=self._preview_processing).grid(column=3, row=0, padx=5, pady=5)
        
        # Process and Cancel buttons - MAKE SURE THESE ARE INCLUDED
        self.process_button = ttk.Button(io_frame, text="Start Processing", 
                                        command=self.parent.start_processing)
        self.process_button.grid(column=3, row=1, padx=5, pady=5)
        
        self.cancel_button = ttk.Button(io_frame, text="Cancel", 
                                    command=self.parent.cancel_processing, state=tk.DISABLED)
        self.cancel_button.grid(column=4, row=1, padx=5, pady=5)

    
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


# Update to input_output_tab.py to add Smart Hair Reinserter to processing pipeline

# Fix for the reinsert_idx variable error in input_output_tab.py

    def _create_pipeline_section(self):
        """Create the processing pipeline section with checkboxes."""
        pipeline_frame = ttk.LabelFrame(self.frame, text="Processing Pipeline", padding="10")
        pipeline_frame.pack(fill=tk.X, pady=5)
        
        # Processing options with improved layout
        processing_options = [
            ("Extract frames from videos", self.parent.extract_frames),
            ("Detect and crop mask regions", self.parent.crop_mask_regions),
            ("Expand mask regions", self.parent.expand_masks),
            ("Resize images and masks", self.parent.resize_images),
            ("Organize and rename files", self.parent.organize_files),
            ("Convert images to video", self.parent.convert_to_video),
            ("Add padding to make images square", self.parent.square_pad_images),
            ("Reinsert processed regions", self.parent.reinsert_crops_option)
        ]
        
        # Find the index of reinsert option before creating widgets
        reinsert_idx = None
        for i, (text, var) in enumerate(processing_options):
            if text == "Reinsert processed regions":
                reinsert_idx = i
                break
        
        # Create checkboxes in a grid layout
        for i, (text, var) in enumerate(processing_options):
            row = i % 3  # 3 options per row
            col = i // 3
            checkbutton = ttk.Checkbutton(pipeline_frame, text=text, variable=var)
            checkbutton.grid(column=col, row=row, sticky=tk.W, padx=10, pady=5)
        
        # Add the Smart Hair Reinserter option in a special frame
        hair_frame = ttk.Frame(pipeline_frame, padding=(5, 10, 5, 5), relief="groove", borderwidth=1)
        hair_frame.grid(column=0, row=3, columnspan=3, sticky=tk.W+tk.E, padx=10, pady=5)
        
        # Create heading
        ttk.Label(
            hair_frame,
            text="✨ Hair Replacement Mode ✨",
            font=("Helvetica", 10, "bold"),
            foreground="#8E44AD"  # Purple color for emphasis
        ).pack(anchor=tk.W, padx=5, pady=(0, 5))
        
        # Single checkbox for Smart Hair Reinserter
        self.hair_reinserter_cb = ttk.Checkbutton(
            hair_frame,
            text="Use intelligent hair processing (optimized for AI-generated hair)",
            variable=self.parent.use_smart_hair_reinserter,
            command=self._on_hair_reinserter_changed
        )
        self.hair_reinserter_cb.pack(anchor=tk.W, padx=20, pady=0)
        
        # Debug checkbox at the bottom
        ttk.Checkbutton(
            pipeline_frame, text="Debug Mode (Save visualization images)", 
            variable=self.parent.debug_mode
        ).grid(column=0, row=4, columnspan=2, sticky=tk.W, padx=10, pady=5)
        
        # Add hint about standalone processing
        hint_frame = ttk.Frame(self.frame, padding="10")
        hint_frame.pack(fill=tk.X, pady=5)
        
        hint_text = ("Hint: The processing pipeline executes steps in the order shown above. " + 
                    "Each step can be run individually or as part of a sequence. " +
                    "For example, you can select only 'Expand mask regions' to process just the masks.")
        
        hint_label = ttk.Label(hint_frame, text=hint_text, foreground="gray", wraplength=600)
        hint_label.pack(anchor=tk.W)

        # Function to handle when reinsertion is toggled
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
                        column=0, row=5, columnspan=3, sticky=tk.W, padx=20, pady=5
                    )
            elif hasattr(self, 'reinsert_hint_label'):
                # Hide the hint
                self.reinsert_hint_label.grid_forget()

        # Update the state of hair reinserter
        self._update_hair_reinserter_state()
        
        # Find and configure the reinsertion checkbox widget
        if reinsert_idx is not None:
            # Calculate grid position
            row = reinsert_idx % 3
            col = reinsert_idx // 3
            
            # Find the widget at this position
            for widget in pipeline_frame.winfo_children():
                if (isinstance(widget, ttk.Checkbutton) and 
                    widget.grid_info().get('row') == row and 
                    widget.grid_info().get('column') == col):
                    widget.configure(command=on_reinsert_toggle)
                    break
    # Add handlers for checkbox state management
    def _on_processing_option_changed(self, changed_var):
        """Handle changes to processing options."""
        self._update_hair_reinserter_state()
        
        # If 'Reinsert processed regions' is checked, show the hint
        if changed_var == self.parent.reinsert_crops_option and changed_var.get():
            self._show_reinsertion_hint()
        elif changed_var == self.parent.reinsert_crops_option and not changed_var.get():
            self._hide_reinsertion_hint()



    def _show_reinsertion_hint(self):
        """Show hint about reinsertion directories."""
        if not hasattr(self, 'reinsert_hint_frame'):
            self.reinsert_hint_frame = ttk.Frame(self.frame, padding=5, relief="groove")
            self.reinsert_hint_frame.pack(fill=tk.X, pady=5, padx=10)
            
            ttk.Label(
                self.reinsert_hint_frame,
                text="IMPORTANT: For reinsertion, your Input Directory should contain PROCESSED images with masks.\n"
                    "In the Configuration tab, set the Source Directory to your ORIGINAL images.",
                foreground="blue",
                font=("Helvetica", 9, "bold"),
                wraplength=600
            ).pack(pady=5)
        
    def _hide_reinsertion_hint(self):
        """Hide the reinsertion hint."""
        if hasattr(self, 'reinsert_hint_frame'):
            self.reinsert_hint_frame.pack_forget()


    def _update_hair_reinserter_state(self):
        """Update the state of the hair reinserter checkbox based on reinsertion option."""
        if hasattr(self, 'hair_reinserter_cb'):
            if self.parent.reinsert_crops_option.get():
                self.hair_reinserter_cb.configure(state="normal")
            else:
                self.hair_reinserter_cb.configure(state="disabled")
                # Uncheck hair reinserter if reinsertion is disabled
                self.parent.use_smart_hair_reinserter.set(False)

    def _on_hair_reinserter_changed(self):
        """Handle changes to the hair reinserter option."""
        if self.parent.use_smart_hair_reinserter.get():
            # When enabling hair reinserter, switch to Config tab
            self.parent.notebook.select(1)  # Config tab
            
            # Enable relevant settings for hair processing
            self.parent.reinsert_handle_different_masks.set(True)
            
            # Show a message
            self.parent.status_label.config(
                text="Hair Replacement Mode activated. Adjust settings in the Configuration tab."
            )

    def _highlight_hair_options(self):
        """Highlight Smart Hair Reinserter options in the Config tab when enabled."""
        # Check if Smart Hair Reinserter is enabled
        if not self.parent.use_smart_hair_reinserter.get():
            return
        
        # Show a notification to user
        self.parent.status_label.config(
            text="Smart Hair Reinserter enabled! Check Configuration tab for hair-specific settings."
        )
        
        # Switch to the Config tab
        self.parent.notebook.select(1)  # Config tab
        
        try:
            # Just switching to the Config tab is enough if we can't highlight anything
            # No need to highlight specific frames that might not exist
            pass
        except Exception as e:
            print(f"Error during tab switching: {e}")