import tkinter as tk
from tkinter import ttk, filedialog

class ConfigTab:
    """Tab for configuring processing options with dynamic UI based on selected processing steps."""
    
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
    
    def _create_reinsertion_section(self):
        """Create crop reinsertion settings section."""
        content = self._create_section("Crop Reinsertion", "reinsertion")
        
        # Add mask-only option
        ttk.Checkbutton(
            content,
            text="Use mask-only reinsertion (only reinsert masked regions)",
            variable=self.parent.reinsert_mask_only
        ).pack(anchor=tk.W, padx=5, pady=5)
        
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
        note_frame = ttk.Frame(content, padding=5, relief="groove")
        note_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(
            note_frame,
            text="IMPORTANT: The Input Directory (set in the Input/Output tab) should contain your CROPPED IMAGES.\n"
                 "The directory above should contain your ORIGINAL UNCROPPED IMAGES.",
            foreground="blue",
            font=("Helvetica", 9, "bold"),
            wraplength=600
        ).pack(pady=5)
        
        # Padding control
        padding_frame = ttk.Frame(content)
        padding_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(padding_frame, text="Reinsertion padding (%):").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(
            padding_frame,
            from_=0,
            to=50,
            increment=5,
            textvariable=self.parent.reinsert_padding,
            width=5
        ).pack(side=tk.LEFT)
        
        # Positioning
        pos_frame = ttk.LabelFrame(content, text="Positioning", padding=5)
        pos_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(
            pos_frame,
            text="Auto-center (try to find original position)",
            variable=self.parent.use_center_position
        ).pack(anchor=tk.W, padx=5, pady=5)
    
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

