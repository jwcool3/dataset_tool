import tkinter as tk
from tkinter import ttk, filedialog

class ConfigTab:
    """Tab for configuring processing options with improved UI organization."""
    
    def __init__(self, parent):
        """
        Initialize the configuration tab.
        
        Args:
            parent: Parent window containing shared variables and functions
        """
        # Store the parent reference
        self.parent = parent
        self.root = parent.root
        
        # Create the main frame that will be added to the notebook
        self.frame = ttk.Frame(parent.notebook)
        
        # Initialize dictionaries for tracking UI elements
        self.sections = {}
        self.ui_elements = {}


        # Create a notebook for sub-tabs within the configuration tab
        self.sub_notebook = ttk.Notebook(self.frame)
        self.sub_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create the three sub-tabs
        self.basic_tab = ttk.Frame(self.sub_notebook)
        self.processing_tab = ttk.Frame(self.sub_notebook)
        self.advanced_tab = ttk.Frame(self.sub_notebook)
        
        # Add the tabs to the notebook
        self.sub_notebook.add(self.basic_tab, text="Basic Setup")
        self.sub_notebook.add(self.processing_tab, text="Processing")
        self.sub_notebook.add(self.advanced_tab, text="Advanced")
        
        # Create scrollable frames for each tab
        self.basic_scroll = self._create_scrollable_frame(self.basic_tab)
        self.processing_scroll = self._create_scrollable_frame(self.processing_tab)
        self.advanced_scroll = self._create_scrollable_frame(self.advanced_tab)
        
        # Initialize the UI components in each tab
        self._init_basic_tab()
        self._init_processing_tab()
        self._init_advanced_tab()
        
        # Track the collapsible sections
        self.sections = {}
        
        # Store references to UI elements
        self.ui_elements = {}
    
    def _create_scrollable_frame(self, parent):
        """Create a scrollable frame and return the content frame."""
        # Create canvas and scrollbar
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        
        # Configure canvas
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Create content frame inside canvas
        content_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        
        # Pack scrollbar and canvas
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mousewheel for scrolling
        self._bind_mousewheel(canvas)
        
        return content_frame
    
    def _bind_mousewheel(self, widget):
        """Bind mousewheel to scrolling."""
        def _on_mousewheel(event):
            # Cross-platform handling
            if event.delta:
                widget.yview_scroll(int(-1 * (event.delta/120)), "units")
            else:
                if event.num == 4:
                    widget.yview_scroll(-1, "units")
                elif event.num == 5:
                    widget.yview_scroll(1, "units")
        
        widget.bind_all("<MouseWheel>", _on_mousewheel)
        widget.bind_all("<Button-4>", _on_mousewheel)
        widget.bind_all("<Button-5>", _on_mousewheel)
    
    def _create_collapsible_section(self, parent, title, default_expanded=True):
        """Create a collapsible section with a toggle button."""
        # Create a frame for the section
        section_frame = ttk.LabelFrame(parent, text=title, padding=10)
        section_frame.pack(fill=tk.X, padx=10, pady=5, expand=False)
        
        # Create a frame for the content that can be hidden
        content_frame = ttk.Frame(section_frame)
        
        # Create toggle button with arrow indicators
        toggle_var = tk.BooleanVar(value=default_expanded)
        
        def toggle_section():
            if toggle_var.get():
                content_frame.pack(fill=tk.X, padx=5, pady=5, expand=True)
                toggle_button.configure(text="▼ " + title)
            else:
                content_frame.pack_forget()
                toggle_button.configure(text="► " + title)
        
        toggle_button = ttk.Button(
            section_frame, 
            text=("▼ " if default_expanded else "► ") + title,
            command=lambda: [toggle_var.set(not toggle_var.get()), toggle_section()]
        )
        toggle_button.pack(anchor=tk.W, padx=5, pady=2)
        
        # Show or hide content based on default state
        if default_expanded:
            content_frame.pack(fill=tk.X, padx=5, pady=5, expand=True)
        
        # Store the section in our dictionary
        self.sections[title] = {
            "frame": section_frame,
            "content": content_frame,
            "toggle_button": toggle_button,
            "toggle_var": toggle_var
        }
        
        return content_frame
    
    def _init_basic_tab(self):
        """Initialize the Basic Setup tab."""
        # Source & Output Settings
        source_content = self._create_collapsible_section(
            self.basic_scroll, 
            "Source & Output Settings", 
            default_expanded=True
        )
        
        # Frame extraction rate
        extraction_frame = ttk.Frame(source_content)
        extraction_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(extraction_frame, text="Frame extraction rate (fps):").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(
            extraction_frame, 
            from_=0.1, 
            to=30.0, 
            increment=0.1, 
            textvariable=self.parent.frame_rate, 
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        # Video output FPS
        video_fps_frame = ttk.Frame(source_content)
        video_fps_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(video_fps_frame, text="Video output FPS:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(
            video_fps_frame, 
            from_=1, 
            to=60, 
            increment=1, 
            textvariable=self.parent.video_fps, 
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        # Mask padding
        mask_padding_frame = ttk.Frame(source_content)
        mask_padding_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(mask_padding_frame, text="Mask padding (%):").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(
            mask_padding_frame, 
            from_=0, 
            to=100, 
            increment=5, 
            textvariable=self.parent.fill_ratio, 
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        # Resolution Settings
        resolution_content = self._create_collapsible_section(
            self.basic_scroll, 
            "Resolution Settings", 
            default_expanded=True
        )
        
        # Use source resolution
        ttk.Checkbutton(
            resolution_content,
            text="Use source resolution",
            variable=self.parent.use_source_resolution,
            command=self._toggle_resolution_controls
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Output resolution
        resolution_frame = ttk.Frame(resolution_content)
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
        
        # Conditional resize
        resize_frame = ttk.Frame(resolution_content)
        resize_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(
            resize_frame,
            text="Only resize if image is larger than:",
            variable=self.parent.resize_if_larger,
            command=self._toggle_conditional_resize_controls
        ).pack(side=tk.LEFT, padx=5)
        
        size_frame = ttk.Frame(resize_frame)
        size_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(size_frame, text="Max Width:").pack(side=tk.LEFT)
        self.max_width_spinbox = ttk.Spinbox(
            size_frame,
            from_=64,
            to=4096,
            increment=64,
            textvariable=self.parent.max_width,
            width=6
        )
        self.max_width_spinbox.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(size_frame, text="Max Height:").pack(side=tk.LEFT, padx=5)
        self.max_height_spinbox = ttk.Spinbox(
            size_frame,
            from_=64,
            to=4096,
            increment=64,
            textvariable=self.parent.max_height,
            width=6
        )
        self.max_height_spinbox.pack(side=tk.LEFT, padx=2)
        
        # Naming & Organization
        naming_content = self._create_collapsible_section(
            self.basic_scroll, 
            "Naming & Organization", 
            default_expanded=False
        )
        
        # Naming pattern
        naming_frame = ttk.Frame(naming_content)
        naming_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(naming_frame, text="Naming pattern:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(
            naming_frame,
            textvariable=self.parent.naming_pattern,
            width=20
        ).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Pattern help
        ttk.Label(
            naming_content,
            text="Use {index} for sequential numbering. Example: image_{index:04d}.png"
        ).pack(anchor=tk.W, padx=5, pady=5)
    
    def _init_processing_tab(self):
        """Initialize the Processing tab."""
        # Frame Extraction
        extraction_content = self._create_collapsible_section(
            self.processing_scroll, 
            "Frame Extraction", 
            default_expanded=False
        )
        
        ttk.Checkbutton(
            extraction_content,
            text="Extract frames from videos",
            variable=self.parent.extract_frames
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Mask video settings
        mask_video_frame = ttk.Frame(extraction_content)
        mask_video_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(
            mask_video_frame,
            text="Use mask video for all source videos",
            variable=self.parent.use_mask_video,
            command=self._toggle_mask_video_controls
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        mask_path_frame = ttk.Frame(extraction_content)
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
        
        mask_info = ttk.Label(
            extraction_content,
            text="This will extract frames from the mask video and copy them to the masks subfolder of each source video.",
            wraplength=600
        )
        mask_info.pack(fill=tk.X, padx=5, pady=5)
        
        # Mask Processing
        mask_content = self._create_collapsible_section(
            self.processing_scroll, 
            "Mask Processing", 
            default_expanded=False
        )
        
        ttk.Checkbutton(
            mask_content,
            text="Detect and crop mask regions",
            variable=self.parent.crop_mask_regions
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Mask expansion
        expansion_frame = ttk.Frame(mask_content)
        expansion_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(
            expansion_frame,
            text="Expand mask regions",
            variable=self.parent.expand_masks
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(expansion_frame, text="Iterations:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Spinbox(
            expansion_frame,
            from_=1,
            to=50,
            increment=1,
            textvariable=self.parent.mask_expand_iterations,
            width=5
        ).pack(side=tk.LEFT)
        
        ttk.Label(expansion_frame, text="Kernel Size:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Spinbox(
            expansion_frame,
            from_=3,
            to=21,
            increment=2,
            textvariable=self.parent.mask_expand_kernel_size,
            width=5
        ).pack(side=tk.LEFT)
        
        ttk.Checkbutton(
            mask_content,
            text="Preserve directory structure when expanding masks",
            variable=self.parent.mask_expand_preserve_structure
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Image Transformations
        transform_content = self._create_collapsible_section(
            self.processing_scroll, 
            "Image Transformations", 
            default_expanded=False
        )
        
        ttk.Checkbutton(
            transform_content,
            text="Resize images and masks",
            variable=self.parent.resize_images
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Square padding
        padding_frame = ttk.Frame(transform_content)
        padding_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(
            padding_frame,
            text="Add padding to make images square",
            variable=self.parent.square_pad_images
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        padding_settings = ttk.Frame(transform_content)
        padding_settings.pack(fill=tk.X, pady=5, padx=20)
        
        ttk.Label(padding_settings, text="Padding Color:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(
            padding_settings,
            textvariable=self.parent.padding_color,
            values=["black", "white", "gray"],
            width=10,
            state="readonly"
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Checkbutton(
            padding_settings,
            text="Use source resolution for padding",
            variable=self.parent.use_source_resolution_padding,
            command=self._toggle_square_padding_controls
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        target_frame = ttk.Frame(padding_settings)
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
        
        # Portrait crop
        ttk.Checkbutton(
            transform_content,
            text="Crop portrait photos to make them square",
            variable=self.parent.portrait_crop_enabled,
            command=self._toggle_portrait_crop_controls
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        portrait_frame = ttk.Frame(transform_content)
        portrait_frame.pack(fill=tk.X, pady=5, padx=20)
        
        ttk.Label(portrait_frame, text="Crop Position:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(
            portrait_frame,
            textvariable=self.parent.portrait_crop_position,
            values=["top", "center", "bottom"],
            width=10,
            state="readonly"
        ).pack(side=tk.LEFT, padx=5)
        
        # Video Conversion
        video_content = self._create_collapsible_section(
            self.processing_scroll, 
            "Video Conversion", 
            default_expanded=False
        )
        
        ttk.Checkbutton(
            video_content,
            text="Convert images to video",
            variable=self.parent.convert_to_video
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        ttk.Checkbutton(
            video_content,
            text="Organize and rename files",
            variable=self.parent.organize_files
        ).pack(anchor=tk.W, padx=5, pady=5)
    
    def _init_advanced_tab(self):
        """Initialize the Advanced tab."""
        # Crop Reinsertion
        reinsertion_content = self._create_collapsible_section(
            self.advanced_scroll, 
            "Crop Reinsertion", 
            default_expanded=False
        )
        
        ttk.Checkbutton(
            reinsertion_content,
            text="Reinsert cropped images",
            variable=self.parent.reinsert_crops_option
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Add mask-only option
        ttk.Checkbutton(
            reinsertion_content,
            text="Use mask-only reinsertion (only reinsert masked regions)",
            variable=self.parent.reinsert_mask_only
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Source directory
        source_frame = ttk.LabelFrame(reinsertion_content, text="Original Uncropped Images Directory", padding=5)
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
        note_frame = ttk.Frame(reinsertion_content, padding=5, relief="groove")
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
        padding_frame = ttk.Frame(reinsertion_content)
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
        pos_frame = ttk.LabelFrame(reinsertion_content, text="Positioning", padding=5)
        pos_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(
            pos_frame,
            text="Auto-center (try to find original position)",
            variable=self.parent.use_center_position
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Debug Options
        debug_content = self._create_collapsible_section(
            self.advanced_scroll, 
            "Debug Options", 
            default_expanded=False
        )
        
        ttk.Checkbutton(
            debug_content,
            text="Debug Mode (Save visualization images)",
            variable=self.parent.debug_mode
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Help text
        help_frame = ttk.Frame(debug_content, padding=5, relief="groove")
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
        for widget in self.frame.winfo_children():
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
        self._toggle_portrait_crop_controls()