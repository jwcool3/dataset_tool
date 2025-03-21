import tkinter as tk
from tkinter import ttk, filedialog

class ConfigTab:
    """Tab for configuring processing options with dynamic UI based on selected processing steps."""
    
# Look for this code in your ConfigTab class's __init__ method
# and modify it to not reference hair_reinserter_frame

    def __init__(self, parent):
        """
        Initialize the configuration tab with adaptive UI.
        
        Args:
            parent: Parent window containing shared variables and functions
        """
        # Store the parent reference
        self.parent = parent
        self.root = parent.root
        
        # Create the main frame that will be added to the notebook
        self.frame = ttk.Frame(parent.notebook)
        
        # Initialize storage for sections and UI elements
        self.sections = {}
        self.ui_elements = {}
        
        # Create main scrollable frame
        self._create_scrollable_frame()
        
        # Create all UI sections (initially hidden)
        self._create_all_ui_sections()
        
        # Connect processing checkboxes to UI visibility
        self._connect_processing_steps()
        
        # Create the "Show All Settings" option at the top
        self._create_show_all_option()
        
        # Update UI based on current selections
        self.update_ui_visibility()
    def _create_scrollable_frame(self):
        """Create the main scrollable frame for content."""
        # Create a canvas with scrollbar for the main content
        self.canvas = tk.Canvas(self.frame)
        self.scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        
        # Pack scrollbar and canvas
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure canvas to use scrollbar
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Create a frame inside the canvas for content
        self.content_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")
        
        # Configure canvas behavior
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.content_frame.bind("<Configure>", self._on_frame_configure)
        
        # Bind mouse wheel for scrolling
        self._bind_mousewheel()
    
    def _on_canvas_resize(self, event):
        """Update the inner frame width when canvas resizes."""
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def _on_frame_configure(self, event):
        """Update scroll region when the inner frame changes size."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _bind_mousewheel(self):
        """Bind mousewheel to scrolling."""
        def _on_mousewheel(event):
            # Cross-platform mouse wheel handling
            if event.delta:
                # Windows/macOS
                self.canvas.yview_scroll(int(-1 * (event.delta/120)), "units")
            else:
                # Linux
                if event.num == 4:
                    self.canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    self.canvas.yview_scroll(1, "units")
        
        # Bind events
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.canvas.bind_all("<Button-4>", _on_mousewheel)
        self.canvas.bind_all("<Button-5>", _on_mousewheel)
    
    def _create_show_all_option(self):
        """Create the option to show all settings regardless of processing steps."""
        # Add an info bar at the top
        info_frame = ttk.Frame(self.content_frame, relief=tk.GROOVE, borderwidth=1)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create "show all settings" variable
        self.show_all_settings = tk.BooleanVar(value=False)
        
        # Create checkbox
        show_all_cb = ttk.Checkbutton(
            info_frame,
            text="Show all settings (Advanced mode)",
            variable=self.show_all_settings,
            command=self.update_ui_visibility
        )
        show_all_cb.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Add help text
        help_text = ttk.Label(
            info_frame,
            text="By default, only settings related to your selected processing steps are shown.",
            font=("Helvetica", 9),
            foreground="#555555"
        )
        help_text.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Add a separator
        ttk.Separator(self.content_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=5)
        
        # Add currently selected processes display
        self.active_processes_frame = ttk.LabelFrame(self.content_frame, text="Active Processing Steps")
        self.active_processes_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # This will be filled dynamically in update_ui_visibility
        self.active_processes_label = ttk.Label(self.active_processes_frame, text="None selected")
        self.active_processes_label.pack(fill=tk.X, padx=10, pady=5)
    
    def _connect_processing_steps(self):
        """Connect processing step variables to UI updates."""
        # Define all processing steps to monitor
        processing_steps = [
            self.parent.extract_frames,
            self.parent.crop_mask_regions,
            self.parent.expand_masks,
            self.parent.resize_images,
            self.parent.organize_files,
            self.parent.convert_to_video,
            self.parent.square_pad_images,
            self.parent.reinsert_crops_option
        ]
        
        # Connect each variable to update UI when changed
        for var in processing_steps:
            var.trace_add("write", lambda *args: self.update_ui_visibility())
    
    def _create_all_ui_sections(self):
        """Create all possible UI sections (initially hidden)."""
        # Create basic/core sections (always visible)
        self._create_core_settings()
        
        # Create processing-specific sections
        self._create_frame_extraction_section()
        self._create_mask_detection_section()
        self._create_mask_expansion_section()
        self._create_resolution_section()
        self._create_square_padding_section()
        self._create_organization_section()
        self._create_video_conversion_section()
        self._create_reinsertion_section()
        self._create_debug_section()
        
        # Initially hide all sections
        for section_id in self.sections:
            self._hide_section(section_id)
    
    def update_ui_visibility(self):
        """Update which UI sections are visible based on processing selections."""
        # First, determine which sections should be visible
        visible_sections = set()
        
        # Always show core settings
        visible_sections.add("core_settings")
        visible_sections.add("general_debug")
        
        # Build a mapping of which processing steps show which sections
        section_mapping = {
            # Extract frames selected
            self.parent.extract_frames.get(): ["frame_extraction"],
            
            # Crop mask regions selected
            self.parent.crop_mask_regions.get(): ["mask_detection"],
            
            # Expand masks selected
            self.parent.expand_masks.get(): ["mask_expansion"],
            
            # Resize images selected
            self.parent.resize_images.get(): ["resolution"],
            
            # Square pad images selected
            self.parent.square_pad_images.get(): ["square_padding"],
            
            # Organize files selected
            self.parent.organize_files.get(): ["organization"],
            
            # Convert to video selected
            self.parent.convert_to_video.get(): ["video_conversion"],
            
            # Reinsert crops selected
            self.parent.reinsert_crops_option.get(): ["reinsertion"]
        }
        
        # Populate visible sections based on selected processing steps
        for is_selected, section_ids in section_mapping.items():
            if is_selected:
                for section_id in section_ids:
                    visible_sections.add(section_id)
        
        # If "show all settings" is enabled, show everything regardless
        if self.show_all_settings.get():
            visible_sections = set(self.sections.keys())
        
        # Update visibility of each section
        for section_id, section_info in self.sections.items():
            if section_id in visible_sections:
                self._show_section(section_id)
            else:
                self._hide_section(section_id)
        
        # Update the active processes label
        active_steps = []
        if self.parent.extract_frames.get():
            active_steps.append("Extract Frames")
        if self.parent.crop_mask_regions.get():
            active_steps.append("Crop Mask Regions")
        if self.parent.expand_masks.get():
            active_steps.append("Expand Masks")
        if self.parent.resize_images.get():
            active_steps.append("Resize Images")
        if self.parent.square_pad_images.get():
            active_steps.append("Square Padding")
        if self.parent.organize_files.get():
            active_steps.append("Organize Files")
        if self.parent.convert_to_video.get():
            active_steps.append("Convert to Video")
        if self.parent.reinsert_crops_option.get():
            active_steps.append("Reinsert Crops")
        
        if active_steps:
            self.active_processes_label.config(text=", ".join(active_steps))
        else:
            self.active_processes_label.config(text="None selected")
        
        # Force update of scroll region
        self.content_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _show_section(self, section_id):
        """Show a section by its ID."""
        if section_id in self.sections:
            section = self.sections[section_id]
            
            # Check if this section was previously hidden
            was_hidden = not section["visible"]
            
            # Show the section frame
            section["frame"].pack(fill=tk.X, padx=10, pady=5)
            section["visible"] = True
            
            # If it was previously hidden, add a highlight effect
            if was_hidden:
                try:
                    # Try to get original background safely
                    try:
                        orig_bg = section["frame"].cget("background")
                    except tk.TclError:
                        # Use a default color if background property is not available
                        orig_bg = "#f0f0f0"  # Default light gray
                    
                    # Flash with a light yellow background to indicate it's newly visible
                    try:
                        section["frame"].configure(background="#FFFFC0")  # Light yellow
                        
                        # Reset back to original after a delay
                        self.frame.after(1500, lambda: section["frame"].configure(background=orig_bg))
                    except tk.TclError:
                        # If configure fails, just skip the highlighting
                        pass
                except Exception:
                    # If any error occurs, just show the section without highlighting
                    pass
    def _hide_section(self, section_id):
        """Hide a section by its ID."""
        if section_id in self.sections:
            section = self.sections[section_id]
            section["frame"].pack_forget()
            section["visible"] = False
    
    def _create_section(self, title, section_id, default_expanded=True):
        """Create a section with toggle capability and store it."""
        # Create a labeled frame for the section
        section_frame = ttk.LabelFrame(self.content_frame, text=title, padding=10)
        
        # Create a variable to track expanded/collapsed state
        expanded = tk.BooleanVar(value=default_expanded)
        
        # Create a frame to hold the content
        content_frame = ttk.Frame(section_frame)
        
        # Function to toggle expand/collapse
        def toggle_section():
            if expanded.get():
                content_frame.pack(fill=tk.X, padx=5, pady=5)
                toggle_button.configure(text="▼ " + title)
            else:
                content_frame.pack_forget()
                toggle_button.configure(text="► " + title)
        
        # Create toggle button
        toggle_button = ttk.Button(
            section_frame,
            text=("▼ " if default_expanded else "► ") + title,
            command=lambda: [expanded.set(not expanded.get()), toggle_section()]
        )
        toggle_button.pack(anchor=tk.W, padx=5, pady=2)
        
        # Show content based on default state
        if default_expanded:
            content_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Store the section
        self.sections[section_id] = {
            "frame": section_frame,
            "content": content_frame,
            "expanded": expanded,
            "toggle": toggle_button,
            "visible": False,  # Start with not visible
        }
        
        return content_frame
    
    def _create_core_settings(self):
        """Create the core settings section (always visible)."""
        content = self._create_section("Core Settings", "core_settings")
        
        # Create a frame for mask padding
        padding_frame = ttk.Frame(content)
        padding_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(padding_frame, text="Mask padding (%):").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(
            padding_frame,
            from_=0,
            to=100,
            increment=5,
            textvariable=self.parent.fill_ratio,
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        # Use source resolution
        ttk.Checkbutton(
            content,
            text="Use source resolution",
            variable=self.parent.use_source_resolution,
            command=self._toggle_resolution_controls
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Resolution
        resolution_frame = ttk.Frame(content)
        resolution_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(resolution_frame, text="Output resolution:").pack(side=tk.LEFT, padx=5)
        
        width_frame = ttk.Frame(resolution_frame)
        width_frame.pack(side=tk.LEFT, padx=5)
        
        self.width_spinbox = ttk.Spinbox(
            width_frame,
            from_=64,
            to=2048,
            increment=64,
            textvariable=self.parent.output_width,
            width=6
        )
        self.width_spinbox.pack(side=tk.LEFT)
        
        ttk.Label(width_frame, text="x").pack(side=tk.LEFT, padx=2)
        
        self.height_spinbox = ttk.Spinbox(
            width_frame,
            from_=64,
            to=2048,
            increment=64,
            textvariable=self.parent.output_height,
            width=6
        )
        self.height_spinbox.pack(side=tk.LEFT)
    
    def _create_frame_extraction_section(self):
        """Create frame extraction settings section."""
        content = self._create_section("Frame Extraction", "frame_extraction")
        
        # Frame rate
        frame_rate_frame = ttk.Frame(content)
        frame_rate_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(frame_rate_frame, text="Frame extraction rate (fps):").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(
            frame_rate_frame,
            from_=0.1,
            to=30.0,
            increment=0.1,
            textvariable=self.parent.frame_rate,
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        # Mask video settings
        ttk.Checkbutton(
            content,
            text="Use mask video for all source videos",
            variable=self.parent.use_mask_video,
            command=self._toggle_mask_video_controls
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        mask_path_frame = ttk.Frame(content)
        mask_path_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(mask_path_frame, text="Mask video:").pack(side=tk.LEFT, padx=5)
        self.mask_video_entry = ttk.Entry(
            mask_path_frame,
            textvariable=self.parent.mask_video_path,
            width=40
        )
        self.mask_video_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.mask_video_button = ttk.Button(
            mask_path_frame,
            text="Browse...",
            command=self._browse_mask_video
        )
        self.mask_video_button.pack(side=tk.RIGHT, padx=5)
        
        # Help text
        ttk.Label(
            content,
            text="This will extract frames from the mask video and copy them to the masks subfolder of each source video.",
            wraplength=600
        ).pack(fill=tk.X, padx=5, pady=5)
    
    def _create_mask_detection_section(self):
        """Create mask detection settings section."""
        content = self._create_section("Mask Detection & Cropping", "mask_detection")
        
        # Add information about mask detection
        info_frame = ttk.Frame(content, padding=5, relief="groove")
        info_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(
            info_frame,
            text="This will detect regions in the mask and crop the corresponding areas in the source image.",
            wraplength=600
        ).pack(pady=5)
    
    def _create_mask_expansion_section(self):
        """Create mask expansion settings section."""
        content = self._create_section("Mask Expansion", "mask_expansion")
        
        # Make the label more eye-catching
        header_label = ttk.Label(content, text="Mask Expansion Settings", font=("Helvetica", 10, "bold"))
        header_label.pack(anchor=tk.W, padx=5, pady=(0, 10))
        
        # Iterations control
        iterations_frame = ttk.Frame(content)
        iterations_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(iterations_frame, text="Dilation Iterations:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(
            iterations_frame,
            from_=1,
            to=50,
            increment=1,
            textvariable=self.parent.mask_expand_iterations,
            width=5
        ).pack(side=tk.LEFT, padx=5)
        
        # Kernel size control
        kernel_frame = ttk.Frame(content)
        kernel_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(kernel_frame, text="Kernel Size:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(
            kernel_frame,
            from_=3,
            to=21,
            increment=2,  # Only odd numbers make sense for kernel size
            textvariable=self.parent.mask_expand_kernel_size,
            width=5
        ).pack(side=tk.LEFT, padx=5)
        
        # Preserve directory structure option
        ttk.Checkbutton(
            content,
            text="Preserve directory structure",
            variable=self.parent.mask_expand_preserve_structure
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Help text with more visible styling
        help_frame = ttk.Frame(content, padding=(5, 10, 5, 5), relief="groove", borderwidth=1)
        help_frame.pack(fill=tk.X, padx=5, pady=10)
        
        help_text = ("Dilates mask regions to make them larger. Higher iteration values create larger expansions. "
                    "Kernel size controls the shape of the expansion (odd numbers only).")
        ttk.Label(help_frame, text=help_text, wraplength=600).pack(padx=5, pady=5)
    
    def _create_resolution_section(self):
        """Create resolution settings section."""
        content = self._create_section("Image Resizing Options", "resolution")
        
        # Conditional resize option
        ttk.Checkbutton(
            content,
            text="Only resize if image is larger than:",
            variable=self.parent.resize_if_larger,
            command=self._toggle_conditional_resize_controls
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        resize_frame = ttk.Frame(content)
        resize_frame.pack(fill=tk.X, pady=5, padx=20)
        
        ttk.Label(resize_frame, text="Max Width:").pack(side=tk.LEFT)
        self.max_width_spinbox = ttk.Spinbox(
            resize_frame,
            from_=64,
            to=4096,
            increment=64,
            textvariable=self.parent.max_width,
            width=6
        )
        self.max_width_spinbox.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(resize_frame, text="Max Height:").pack(side=tk.LEFT, padx=5)
        self.max_height_spinbox = ttk.Spinbox(
            resize_frame,
            from_=64,
            to=4096,
            increment=64,
            textvariable=self.parent.max_height,
            width=6
        )
        self.max_height_spinbox.pack(side=tk.LEFT, padx=2)
    
    def _create_square_padding_section(self):
        """Create square padding settings section."""
        content = self._create_section("Square Padding Options", "square_padding")
        
        # Padding color
        padding_frame = ttk.Frame(content)
        padding_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(padding_frame, text="Padding Color:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(
            padding_frame,
            textvariable=self.parent.padding_color,
            values=["black", "white", "gray"],
            width=10,
            state="readonly"
        ).pack(side=tk.LEFT, padx=5)
        
        # Source resolution for padding
        ttk.Checkbutton(
            content,
            text="Use source resolution for padding",
            variable=self.parent.use_source_resolution_padding,
            command=self._toggle_square_padding_controls
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Target size
        target_frame = ttk.Frame(content)
        target_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(target_frame, text="Target Size:").pack(side=tk.LEFT, padx=5)
        self.square_target_spinbox = ttk.Spinbox(
            target_frame,
            from_=64,
            to=2048,
            increment=64,
            textvariable=self.parent.square_target_size,
            width=6
        )
        self.square_target_spinbox.pack(side=tk.LEFT, padx=5)
    
    def _create_organization_section(self):
        """Create file organization settings section."""
        content = self._create_section("File Organization", "organization")
        
        # Naming pattern
        naming_frame = ttk.Frame(content)
        naming_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(naming_frame, text="Naming pattern:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(
            naming_frame,
            textvariable=self.parent.naming_pattern,
            width=20
        ).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Pattern help
        ttk.Label(
            content,
            text="Use {index} for sequential numbering. Example: image_{index:04d}.png"
        ).pack(anchor=tk.W, padx=5, pady=5)
    
    def _create_video_conversion_section(self):
        """Create video conversion settings section."""
        content = self._create_section("Video Conversion", "video_conversion")
        
        # Video FPS
        fps_frame = ttk.Frame(content)
        fps_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(fps_frame, text="Video output FPS:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(
            fps_frame,
            from_=1,
            to=60,
            increment=1,
            textvariable=self.parent.video_fps,
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        # Help text
        ttk.Label(
            content,
            text="This will convert processed image sequences into video files with the specified frame rate.",
            wraplength=600
        ).pack(anchor=tk.W, padx=5, pady=5)
    
# This is a patch to add Smart Hair Reinserter settings to the Config Tab

# Add this method to the ConfigTab class by updating _create_reinsertion_section:


    def _create_reinsertion_section(self):
        """Create the UI for crop reinsertion settings."""
        content = self._create_section("Reinsertion Settings", "reinsertion")
        
        # Basic Options frame
        basic_frame = ttk.LabelFrame(content, text="Basic Options", padding=5)
        basic_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Add mask-only option
        ttk.Checkbutton(
            basic_frame,
            text="Only reinsert masked regions (preserve rest of image)",
            variable=self.parent.reinsert_mask_only
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Add hair-specific options directly in basic frame instead of a separate frame
        if hasattr(self.parent, 'hair_color_correction'):
            ttk.Checkbutton(
                basic_frame,
                text="Enable automatic color correction for hair",
                variable=self.parent.hair_color_correction
            ).pack(anchor=tk.W, padx=5, pady=5)
        
        if hasattr(self.parent, 'hair_top_alignment'):
            ttk.Checkbutton(
                basic_frame,
                text="Prioritize top alignment for hair",
                variable=self.parent.hair_top_alignment
            ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Vertical Alignment frame
        vertical_frame = ttk.LabelFrame(content, text="Vertical Alignment", padding=5)
        vertical_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Vertical Alignment slider
        if hasattr(self.parent, 'vertical_alignment_bias'):
            ttk.Label(vertical_frame, text="Vertical Position:").pack(side=tk.LEFT, padx=5)
            slider = ttk.Scale(
                vertical_frame,
                from_=-50,
                to=50,
                orient=tk.HORIZONTAL,
                variable=self.parent.vertical_alignment_bias,
                length=200
            )
            slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        


        ttk.Label(
            source_frame,
            text="Select the directory containing the ORIGINAL UNCROPPED images:",
            font=("Helvetica", 9)
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        source_dir_frame = ttk.Frame(source_frame)
        source_dir_frame.pack(fill=tk.X, pady=5)
        
        ttk.Entry(
            source_dir_frame,
            textvariable=self.parent.source_images_dir,
            width=40
        ).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        ttk.Button(
            source_dir_frame,
            text="Browse...",
            command=self._browse_source_dir
        ).pack(side=tk.RIGHT, padx=5)
        
        # Important guidance note
        reminder_frame = ttk.Frame(content, padding=5, relief="groove")
        reminder_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Label(
            reminder_frame,
            text="INPUT DIRECTORY (set in the Input/Output tab): Your PROCESSED/CROPPED images.\n"
                "SOURCE DIRECTORY (set above): Your ORIGINAL UNCROPPED images.",
            foreground="blue",
            font=("Helvetica", 9, "bold"),
            wraplength=600
        ).pack(pady=5)
        
        # Hair-specific settings
        hair_frame = ttk.LabelFrame(content, text="Hair Replacement Settings", padding=5)
        hair_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Hair presets
        preset_frame = ttk.Frame(hair_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(preset_frame, text="Quick Presets:").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            preset_frame, 
            text="Natural Hair",
            command=lambda: self._apply_hair_preset("natural")
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            preset_frame, 
            text="Anime Hair",
            command=lambda: self._apply_hair_preset("anime")
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            preset_frame, 
            text="Updo/Ponytail",
            command=lambda: self._apply_hair_preset("updo")
        ).pack(side=tk.LEFT, padx=5)
        
        # Vertical Alignment Bias slider
        vertical_bias_frame = ttk.Frame(hair_frame)
        vertical_bias_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(vertical_bias_frame, text="Vertical Position:").pack(side=tk.LEFT, padx=5)
        vertical_bias_slider = ttk.Scale(
            vertical_bias_frame,
            from_=-50,
            to=50,
            orient=tk.HORIZONTAL,
            variable=self.parent.vertical_alignment_bias,
            length=200
        )
        vertical_bias_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Label to show current bias value
        self.vertical_bias_label = ttk.Label(vertical_bias_frame, text="0")
        self.vertical_bias_label.pack(side=tk.LEFT, padx=5)
        
        # Update label when slider moves
        def update_vertical_bias_label(*args):
            bias = self.parent.vertical_alignment_bias.get()
            direction = "up" if bias < 0 else "down" if bias > 0 else "center"
            strength = abs(bias)
            if strength == 0:
                self.vertical_bias_label.config(text="Centered")
            else:
                self.vertical_bias_label.config(text=f"{direction} {strength}")
        
        self.parent.vertical_alignment_bias.trace_add("write", update_vertical_bias_label)
        update_vertical_bias_label()  # Initial update
        
        # Soft Edge Width slider
        edge_frame = ttk.Frame(hair_frame)
        edge_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(edge_frame, text="Edge Softness:").pack(side=tk.LEFT, padx=5)
        edge_slider = ttk.Scale(
            edge_frame,
            from_=0,
            to=30,
            orient=tk.HORIZONTAL,
            variable=self.parent.soft_edge_width,
            length=200
        )
        edge_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Label to show current value
        self.edge_label = ttk.Label(edge_frame, text="15")
        self.edge_label.pack(side=tk.LEFT, padx=5)
        
        # Update label when slider moves
        def update_edge_label(*args):
            width = self.parent.soft_edge_width.get()
            if width < 10:
                desc = "Sharp"
            elif width < 20:
                desc = "Natural" 
            else:
                desc = "Very Soft"
            self.edge_label.config(text=f"{width} ({desc})")
        
        self.parent.soft_edge_width.trace_add("write", update_edge_label)
        update_edge_label()  # Initial update
        
        # Color correction checkbox
        ttk.Checkbutton(
            hair_frame,
            text="Enable automatic color correction for hair",
            variable=self.parent.hair_color_correction
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Advanced settings in a collapsible section
        self.advanced_frame = self._create_collapsible_section(
            content, 
            "Advanced Settings", 
            "advanced_reinsertion",
            False  # Start collapsed
        )
        
        # Options that used to be separate checkboxes
        ttk.Checkbutton(
            self.advanced_frame,
            text="Only reinsert masked regions (preserve rest of image)",
            variable=self.parent.reinsert_mask_only
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Hidden - enable by default and keep out of UI
        self.parent.reinsert_handle_different_masks.set(True)
        
        # Method selection (with simplified, clearer labels)
        method_frame = ttk.Frame(self.advanced_frame)
        method_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(method_frame, text="Alignment Strategy:").pack(side=tk.LEFT, padx=5)
        
        method_combo = ttk.Combobox(
            method_frame,
            textvariable=self.parent.reinsert_alignment_method,
            values=["centroid", "landmarks", "bbox"],
            width=15,
            state="readonly"
        )
        method_combo.pack(side=tk.LEFT, padx=5)
        
        # Map the method names to more user-friendly descriptions
        method_descriptions = {
            "centroid": "Center of mass (balanced)",
            "landmarks": "Feature matching (detailed)",
            "bbox": "Bounding box (simpler)"
        }
        
        # Update description when method changes
        method_description_label = ttk.Label(method_frame, text="", foreground="gray")
        method_description_label.pack(side=tk.LEFT, padx=5)
        
        def update_method_description(*args):
            method = self.parent.reinsert_alignment_method.get()
            if method in method_descriptions:
                method_description_label.config(text=method_descriptions[method])
        
        self.parent.reinsert_alignment_method.trace_add("write", update_method_description)
        update_method_description()  # Initial update
        
        # Tips section at the bottom
        tips_frame = ttk.LabelFrame(content, text="Quick Tips", padding=5)
        tips_frame.pack(fill=tk.X, pady=5, padx=5)
        
        tips_text = (
            "• Negative vertical position moves hair up, positive moves it down\n"
            "• Increase edge softness for more gradual transitions\n"
            "• Use the 'Natural Hair' preset as a starting point\n"
            "• Preview Processing to test settings before running"
        )
        
        ttk.Label(
            tips_frame,
            text=tips_text,
            wraplength=600
        ).pack(padx=5, pady=5, anchor=tk.W)
        # Soft Edge Frame
        soft_edge_frame = ttk.LabelFrame(content, text="Soft Edge Settings", padding=5)
        soft_edge_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Feather Pixels Slider
        ttk.Label(soft_edge_frame, text="Soft Edge Width:").pack(side=tk.LEFT, padx=5)
        feather_slider = ttk.Scale(
            soft_edge_frame,
            from_=0,
            to=30,
            orient=tk.HORIZONTAL,
            variable=self.parent.soft_edge_width,
            length=200
        )
        feather_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Label to show current feather value
        self.feather_label = ttk.Label(soft_edge_frame, text="15")
        self.feather_label.pack(side=tk.LEFT, padx=5)
        
        # Update label when slider moves
        def update_feather_label(*args):
            width = self.parent.soft_edge_width.get()
            self.feather_label.config(text=f"{width:.0f}")
        
        self.parent.soft_edge_width.trace_add("write", update_feather_label)
        
        # Source directory
        source_frame = ttk.LabelFrame(content, text="Original Uncropped Images Directory", padding=5)
        source_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(
            source_frame,
            text="Select the directory containing the ORIGINAL UNCROPPED images:",
            font=("Helvetica", 9, "bold")
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        source_dir_frame = ttk.Frame(source_frame)
        source_dir_frame.pack(fill=tk.X, pady=5)
        
        ttk.Entry(
            source_dir_frame,
            textvariable=self.parent.source_images_dir,
            width=40
        ).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        ttk.Button(
            source_dir_frame,
            text="Browse...",
            command=self._browse_source_dir
        ).pack(side=tk.RIGHT, padx=5)
        
        # Important note
        self.note_frame = ttk.Frame(content, padding=5, relief="groove")
        self.note_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(
            self.note_frame,
            text="IMPORTANT: The Input Directory (set in the Input/Output tab) should contain your CROPPED IMAGES.\n"
                "The directory above should contain your ORIGINAL UNCROPPED IMAGES.",
            foreground="blue",
            font=("Helvetica", 9, "bold"),
            wraplength=600
        ).pack(pady=5)
        
        # Advanced settings frame 
        self.advanced_frame = ttk.LabelFrame(content, text="Advanced Mask Alignment Options", padding=5)
        self.advanced_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Alignment method selection
        alignment_frame = ttk.Frame(self.advanced_frame)
        alignment_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(alignment_frame, text="Alignment Method:").pack(side=tk.LEFT, padx=5)
        
        alignment_combo = ttk.Combobox(
            alignment_frame,
            textvariable=self.parent.reinsert_alignment_method,
            values=["none", "centroid", "bbox", "landmarks", "contour", "iou"],
            width=15,
            state="readonly"
        )
        alignment_combo.pack(side=tk.LEFT, padx=5)
        
        # Blend mode selection
        blend_frame = ttk.Frame(self.advanced_frame)
        blend_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(blend_frame, text="Blend Mode:").pack(side=tk.LEFT, padx=5)
        
        blend_combo = ttk.Combobox(
            blend_frame,
            textvariable=self.parent.reinsert_blend_mode,
            values=["alpha", "poisson", "feathered"],
            width=15,
            state="readonly"
        )
        blend_combo.pack(side=tk.LEFT, padx=5)
        
        # Blend extent slider
        extent_frame = ttk.Frame(self.advanced_frame)
        extent_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(extent_frame, text="Blend Extent:").pack(side=tk.LEFT, padx=5)
        
        extent_slider = ttk.Scale(
            extent_frame,
            from_=0,
            to=20,
            orient=tk.HORIZONTAL,
            variable=self.parent.reinsert_blend_extent,
            length=200
        )
        extent_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Label to show value
        self.extent_value_label = ttk.Label(extent_frame, text=f"{self.parent.reinsert_blend_extent.get()} px")
        self.extent_value_label.pack(side=tk.LEFT, padx=5)
        
        # Update label when slider is moved
        def update_extent_label(*args):
            self.extent_value_label.config(text=f"{self.parent.reinsert_blend_extent.get()} px")
        
        self.parent.reinsert_blend_extent.trace_add("write", update_extent_label)
        
        # Preserve edges option
        ttk.Checkbutton(
            self.advanced_frame,
            text="Preserve original image edges",
            variable=self.parent.reinsert_preserve_edges
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Explanation section
        explanation_frame = ttk.LabelFrame(content, text="Tips for Hair Reinsertion", padding=5)
        explanation_frame.pack(fill=tk.X, pady=5, padx=5)
        
        explanation_text = (
            "• Use Smart Hair Reinserter for best results with hair replacement\n"
            "• Positive vertical bias values move hair downward, negative values move hair upward\n"
            "• Increase soft edge width for more gradual blending between original and processed hair\n"
            "• For best results, ensure hair masks don't include the face or other features\n"
            "• Try different preset options if default alignment doesn't work well"
        )
        
        ttk.Label(
            explanation_frame,
            text=explanation_text,
            wraplength=600
        ).pack(padx=5, pady=5, anchor=tk.W)
        
        # Initially hide the advanced options if different masks handling is disabled
        if not self.parent.reinsert_handle_different_masks.get():
            self.advanced_frame.pack_forget()
        
        # Initially hide the hair reinserter options if not enabled
        if not self.parent.use_smart_hair_reinserter.get():
            self.hair_reinserter_frame.pack_forget()


    # Add this method to toggle hair reinserter controls
    def _toggle_hair_reinserter_controls(self):
        """Show or hide Smart Hair Reinserter controls based on the checkbox state."""
        if hasattr(self, 'hair_reinserter_frame'):
            if self.parent.use_smart_hair_reinserter.get():
                # Show the hair reinserter controls
                try:
                    self.hair_reinserter_frame.pack(fill=tk.X, pady=5, padx=5, after=self.note_frame)
                except:
                    self.hair_reinserter_frame.pack(fill=tk.X, pady=5, padx=5)
            else:
                # Hide the hair reinserter controls
                self.hair_reinserter_frame.pack_forget()

    # Add this method for hair presets
    def _apply_hair_preset(self, preset_type):
        """Apply predefined settings for different hair types."""
        if preset_type == "natural":
            # Natural hair settings
            self.parent.vertical_alignment_bias.set(10)
            self.parent.soft_edge_width.set(15)
            self.parent.reinsert_blend_mode.set("feathered")
            self.parent.hair_color_correction.set(True)
            self.parent.hair_top_alignment.set(True)
        elif preset_type == "anime":
            # Anime hair settings - usually needs more prominent edges
            self.parent.vertical_alignment_bias.set(5)
            self.parent.soft_edge_width.set(8)
            self.parent.reinsert_blend_mode.set("alpha")
            self.parent.hair_color_correction.set(True)
            self.parent.hair_top_alignment.set(True)
        elif preset_type == "updo":
            # Updo/ponytail settings - higher alignment for vertical hairstyles
            self.parent.vertical_alignment_bias.set(-20)
            self.parent.soft_edge_width.set(12)
            self.parent.reinsert_blend_mode.set("feathered")
            self.parent.hair_color_correction.set(True)
            self.parent.hair_top_alignment.set(True)
    def _toggle_mask_alignment_controls(self):
        """Show or hide advanced mask alignment controls based on the checkbox state."""
        if hasattr(self, 'advanced_frame'):
            if self.parent.reinsert_handle_different_masks.get():
                # Pack it after the note_frame if available, otherwise just pack it normally
                try:
                    self.advanced_frame.pack(fill=tk.X, pady=5, padx=5, after=self.note_frame)
                except:
                    self.advanced_frame.pack(fill=tk.X, pady=5, padx=5)
            else:
                self.advanced_frame.pack_forget()


    
    def _create_debug_section(self):
        """Create debug options section."""
        content = self._create_section("Debug Options", "general_debug", default_expanded=False)
        
        ttk.Checkbutton(
            content,
            text="Debug Mode (Save visualization images)",
            variable=self.parent.debug_mode
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Help text
        help_frame = ttk.Frame(content, padding=5, relief="groove")
        help_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(
            help_frame,
            text="Debug mode saves additional images showing the processing steps, bounding boxes, and other visual aids to help troubleshoot issues.",
            wraplength=600
        ).pack(pady=5)
    
    # Control toggle methods
    def _toggle_resolution_controls(self):
        """Enable or disable resolution controls based on checkbox state."""
        if not hasattr(self, 'width_spinbox') or not hasattr(self, 'height_spinbox'):
            return
            
        if self.parent.use_source_resolution.get():
            self.width_spinbox.configure(state="disabled")
            self.height_spinbox.configure(state="disabled")
        else:
            self.width_spinbox.configure(state="normal")
            self.height_spinbox.configure(state="normal")
    
    def _toggle_conditional_resize_controls(self):
        """Enable or disable conditional resize controls based on checkbox state."""
        if not hasattr(self, 'max_width_spinbox') or not hasattr(self, 'max_height_spinbox'):
            return
            
        if self.parent.resize_if_larger.get():
            self.max_width_spinbox.configure(state="normal")
            self.max_height_spinbox.configure(state="normal")
        else:
            self.max_width_spinbox.configure(state="disabled")
            self.max_height_spinbox.configure(state="disabled")
    
    def _toggle_mask_video_controls(self):
        """Enable or disable mask video controls based on checkbox state."""
        if not hasattr(self, 'mask_video_entry') or not hasattr(self, 'mask_video_button'):
            return
            
        if self.parent.use_mask_video.get():
            self.mask_video_entry.configure(state="normal")
            self.mask_video_button.configure(state="normal")
        else:
            self.mask_video_entry.configure(state="disabled")
            self.mask_video_button.configure(state="disabled")
    
    def _toggle_square_padding_controls(self):
        """Enable or disable square padding controls based on checkbox state."""
        if not hasattr(self, 'square_target_spinbox'):
            return
            
        if self.parent.use_source_resolution_padding.get():
            self.square_target_spinbox.configure(state="disabled")
        else:
            self.square_target_spinbox.configure(state="normal")
    
    def _toggle_portrait_crop_controls(self):
        """Enable or disable portrait crop controls based on checkbox state."""
        # Find the crop position combobox
        for widget in self.content_frame.winfo_children():
            if isinstance(widget, ttk.Combobox) and widget.cget("textvariable") == str(self.parent.portrait_crop_position):
                if self.parent.portrait_crop_enabled.get():
                    widget.configure(state="readonly")
                else:
                    widget.configure(state="disabled")
                break
    
    # File dialogs
    def _browse_source_dir(self):
        """Browse for source directory."""
        from tkinter import filedialog
        directory = filedialog.askdirectory()
        if directory:
            self.parent.source_images_dir.set(directory)
    
    def _browse_mask_video(self):
        """Browse for a mask video file."""
        from tkinter import filedialog
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(filetypes=filetypes, title="Select Mask Video")
        if filepath:
            self.parent.mask_video_path.set(filepath)
    
    def refresh(self):
        """
        Refresh the UI state based on current variable values.
        Call this method when loading a configuration to update UI.
        """
        # Toggle controls based on current state
        self._toggle_resolution_controls()
        self._toggle_mask_video_controls()
        self._toggle_conditional_resize_controls()
        self._toggle_square_padding_controls()
        
        # Update UI visibility based on current settings
        self.update_ui_visibility()

    def add_tooltip(self, widget, text):
        """Add tooltip to a widget."""
        def enter(event):
            x = y = 0
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            
            # Create a toplevel window
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            
            label = ttk.Label(
                self.tooltip, 
                text=text, 
                wraplength=400,
                background="#ffffcc", 
                relief="solid", 
                borderwidth=1,
                padding=5
            )
            label.pack()
            
        def leave(event):
            if hasattr(self, "tooltip"):
                self.tooltip.destroy()
                
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def highlight_section(self, section_id):
        """Highlight a section to draw attention to it."""
        if section_id in self.sections:
            section = self.sections[section_id]
            frame = section["frame"]
            
            try:
                # Try to get original background safely
                try:
                    orig_bg = frame.cget("background")
                except tk.TclError:
                    # Use a default color if background property is not available
                    orig_bg = "#f0f0f0"  # Default light gray
                
                # Highlight with a light blue background
                try:
                    frame.configure(background="#e0f0ff")
                    
                    # Flash effect: alternate between highlight color and original
                    def flash_effect(count=0):
                        if count >= 6:  # 3 flashes
                            try:
                                frame.configure(background=orig_bg)
                            except tk.TclError:
                                pass
                            return
                        
                        try:
                            if count % 2 == 0:
                                frame.configure(background="#e0f0ff")
                            else:
                                frame.configure(background=orig_bg)
                        except tk.TclError:
                            pass
                        
                        self.frame.after(500, lambda: flash_effect(count + 1))
                    
                    flash_effect()
                except tk.TclError:
                    # If configure fails, just skip the highlighting
                    pass
            except Exception:
                # If any error occurs, just continue without highlighting
                pass

    def create_workflow_overview(self):
        """Create a visual workflow overview of enabled processing steps."""
        # Create a new top-level window
        workflow_window = tk.Toplevel(self.parent.root)
        workflow_window.title("Processing Workflow Overview")
        workflow_window.geometry("600x400")
        
        # Create a canvas for drawing the workflow
        canvas = tk.Canvas(workflow_window, bg="white")
        canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Define processing steps and their dependencies
        workflow_steps = [
            {"id": "extract_frames", "name": "Extract Frames", "depends_on": []},
            {"id": "crop_mask_regions", "name": "Crop Mask Regions", "depends_on": []},
            {"id": "expand_masks", "name": "Expand Masks", "depends_on": ["crop_mask_regions"]},
            {"id": "resize_images", "name": "Resize Images", "depends_on": []},
            {"id": "square_pad_images", "name": "Square Padding", "depends_on": []},
            {"id": "organize_files", "name": "Organize Files", "depends_on": []},
            {"id": "convert_to_video", "name": "Convert to Video", "depends_on": []},
            {"id": "reinsert_crops_option", "name": "Reinsert Crops", "depends_on": ["crop_mask_regions"]}
        ]
        
        # Filter to only show enabled steps
        active_steps = []
        for step in workflow_steps:
            step_var = getattr(self.parent, step["id"])
            if step_var.get():
                active_steps.append(step)
        
        if not active_steps:
            canvas.create_text(300, 200, text="No processing steps selected", font=("Helvetica", 14))
            return
        
        # Draw the workflow steps
        step_width = 150
        step_height = 40
        x_spacing = 180
        y_spacing = 60
        
        # Position steps
        positions = {}
        current_x = 50
        current_y = 50
        
        for i, step in enumerate(active_steps):
            # Simple layout: steps in a row
            x = current_x
            y = current_y
            
            # Create a rounded rectangle for the step
            rect_id = canvas.create_rectangle(
                x, y, x + step_width, y + step_height,
                fill="#e0f0ff", outline="#0078d7", width=2,
                tags=step["id"]
            )
            
            # Add step name
            text_id = canvas.create_text(
                x + step_width/2, y + step_height/2,
                text=step["name"],
                font=("Helvetica", 10, "bold"),
                tags=step["id"]
            )
            
            # Store position
            positions[step["id"]] = {
                "x": x, "y": y, "width": step_width, "height": step_height
            }
            
            # Update position for next step
            current_x += x_spacing
            if current_x > 500:  # Start a new row
                current_x = 50
                current_y += y_spacing
        
        # Draw connections between steps
        for step in active_steps:
            if not step["depends_on"]:
                continue
                
            for dep_id in step["depends_on"]:
                if dep_id not in positions:
                    continue  # Skip if dependency is not active
                    
                from_pos = positions[dep_id]
                to_pos = positions[step["id"]]
                
                # Draw arrow from dependency to current step
                canvas.create_line(
                    from_pos["x"] + from_pos["width"], from_pos["y"] + from_pos["height"]/2,
                    to_pos["x"], to_pos["y"] + to_pos["height"]/2,
                    arrow=tk.LAST, width=2, fill="#0078d7"
                )


    # Helper method to create collapsible sections
    def _create_collapsible_section(self, parent, title, section_id, default_expanded=True):
        """Create a collapsible section that can be expanded or hidden."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Create a variable to track expanded state
        is_expanded = tk.BooleanVar(value=default_expanded)
        
        # Create header with toggle button
        header_frame = ttk.Frame(frame)
        header_frame.pack(fill=tk.X)
        
        toggle_text = "▼ " if default_expanded else "► "
        toggle_button = ttk.Button(
            header_frame,
            text=toggle_text + title,
            style="Toolbutton",  # Use toolbutton style for a flatter appearance
            width=20
        )
        toggle_button.pack(side=tk.LEFT, anchor=tk.W)
        
        # Create content frame
        content_frame = ttk.Frame(frame, padding=(15, 5, 5, 5))
        if default_expanded:
            content_frame.pack(fill=tk.X, pady=5)
        
        # Toggle function
        def toggle_section():
            if is_expanded.get():
                # Collapse
                content_frame.pack_forget()
                toggle_button.config(text="► " + title)
                is_expanded.set(False)
            else:
                # Expand
                content_frame.pack(fill=tk.X, pady=5)
                toggle_button.config(text="▼ " + title)
                is_expanded.set(True)
        
        toggle_button.config(command=toggle_section)
        
        return content_frame
