"""
Configuration Tab for Dataset Preparation Tool
Contains all the configuration options for the processing pipeline.
"""

import tkinter as tk
from tkinter import ttk

class ConfigTab:
    """Tab for configuring processing options."""
    
# ...existing code...
    def __init__(self, parent):
        """
        Initialize the configuration tab.
        
        Args:
            parent: Parent window containing shared variables and functions
        """
        # Store the parent reference
        self.parent = parent
        
        # Create the main frame that will be added to the notebook
        self.frame = ttk.Frame(parent.notebook)
        
        # Create a canvas and scrollbar inside this main frame
        self.canvas = tk.Canvas(self.frame)
        self.scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack the scrollbar and canvas within the main frame
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Create a content frame inside the canvas for actual content
        self.content_frame = ttk.Frame(self.canvas, padding="10")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")
        
        # Configure the canvas to update the scrollregion
        self.content_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Initialize attributes that will be set later
        self.width_spinbox = None
        self.height_spinbox = None
        self.mask_video_entry = None
        self.mask_video_button = None
        self.max_width_spinbox = None
        self.max_height_spinbox = None
        self.square_target_spinbox = None
        self.resolution_frame = None
        
        # Create the UI components for each configuration section
        self._create_general_config()
        self._create_mask_video_config()
        
        # Add this mask expansion section early in the tab order for visibility
        self._create_mask_expand_section()
        
        self._create_conditional_resize()
        self._create_square_padding()
        self._create_portrait_crop()
        self._create_crop_reinsertion()
        
        # Bind mousewheel scrolling for better usability
        self._bind_mousewheel(self.canvas)
    # ...existing code...
    
    def _create_general_config(self):
        """Create the general configuration options section."""
        config_frame = ttk.LabelFrame(self.frame, text="Configuration Options", padding="10")
        config_frame.pack(fill=tk.X, pady=5)
        
        # Create a grid layout for configuration options
        ttk.Label(config_frame, text="Frame extraction rate (fps):").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(config_frame, from_=0.1, to=30.0, increment=0.1, textvariable=self.parent.frame_rate, width=10).grid(
            column=1, row=0, padx=5, sticky=tk.W)
        
        ttk.Label(config_frame, text="Mask padding (%):").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(config_frame, from_=0, to=100, increment=5, textvariable=self.parent.fill_ratio, width=10).grid(
            column=1, row=1, padx=5, sticky=tk.W)
        
        ttk.Label(config_frame, text="Video output FPS:").grid(column=2, row=0, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(config_frame, from_=1, to=60, increment=1, textvariable=self.parent.video_fps, width=10).grid(
            column=3, row=0, padx=5, sticky=tk.W)
        
        # Add source resolution checkbox
        ttk.Checkbutton(
            config_frame, 
            text="Use source resolution", 
            variable=self.parent.use_source_resolution, 
            command=self._toggle_resolution_controls
        ).grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
        
        # Output resolution controls
        ttk.Label(config_frame, text="Output resolution:").grid(column=0, row=3, sticky=tk.W, padx=5, pady=5)
        


        # Custom resolution frame
        self.resolution_frame = ttk.Frame(config_frame)
        self.resolution_frame.grid(column=1, row=3, sticky=tk.W)
        
        self.width_spinbox = ttk.Spinbox(
            self.resolution_frame, 
            from_=64, 
            to=2048, 
            increment=64, 
            textvariable=self.parent.output_width, 
            width=6
        )
        self.width_spinbox.pack(side=tk.LEFT)
        
        ttk.Label(self.resolution_frame, text="x").pack(side=tk.LEFT, padx=2)
        
        self.height_spinbox = ttk.Spinbox(
            self.resolution_frame, 
            from_=64, 
            to=2048, 
            increment=64, 
            textvariable=self.parent.output_height, 
            width=6
        )
        self.height_spinbox.pack(side=tk.LEFT)
        
        ttk.Label(config_frame, text="Naming pattern:").grid(column=2, row=1, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(config_frame, textvariable=self.parent.naming_pattern, width=20).grid(column=3, row=1, padx=5, sticky=tk.W)
        
        # Add a description for naming pattern
        ttk.Label(config_frame, text="Use {index} for sequential numbering").grid(
            column=2, row=2, columnspan=2, sticky=tk.W, padx=5)
        
        # Set initial state of the resolution controls
        self._toggle_resolution_controls()
    

    def _create_crop_reinsertion(self):
        """Create the crop reinsertion options section."""
        # Create the main frame for crop reinsertion options
        reinsertion_frame = ttk.LabelFrame(self.frame, text="Crop Reinsertion", padding="10")
        reinsertion_frame.pack(fill=tk.X, pady=5)
        
        # Source images directory - make it very clear what this is
        source_frame = ttk.LabelFrame(reinsertion_frame, text="Original Uncropped Images Directory", padding=5)
        source_frame.pack(fill=tk.X, pady=5)

        ttk.Label(source_frame, 
                text="Select the directory containing the ORIGINAL UNCROPPED images:",
                font=("Helvetica", 9, "bold")).pack(anchor=tk.W, padx=5, pady=5)

        source_dir_frame = ttk.Frame(source_frame)
        source_dir_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(source_dir_frame, textvariable=self.parent.source_images_dir, width=40).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(source_dir_frame, text="Browse...", command=self._browse_source_dir).pack(side=tk.RIGHT, padx=5)
        
        # Add a direct clarification about Input Directory
        input_reminder = ttk.Frame(reinsertion_frame, padding=5, relief="groove")
        input_reminder.pack(fill=tk.X, pady=10)
        
        ttk.Label(input_reminder, 
                text="IMPORTANT: The Input Directory (set in the Input/Output tab) should contain your CROPPED IMAGES.\n"
                    "Go to the Config tab to set the source directory that contains your ORIGINAL UNCROPPED IMAGES.",
                foreground="blue",
                font=("Helvetica", 9, "bold"),
                wraplength=600).pack(pady=5)
        
        # Add mask-only option
        ttk.Checkbutton(
            reinsertion_frame, 
            text="Use mask-only reinsertion (only reinsert masked regions)", 
            variable=self.parent.reinsert_mask_only
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Rest of your existing code for matching method, padding, etc.
        
    def _create_mask_video_config(self):
        """Create the mask video options section."""
        mask_video_frame = ttk.LabelFrame(self.frame, text="Mask Video Options", padding="10")
        mask_video_frame.pack(fill=tk.X, pady=5)

        # Add checkbox to enable mask video
        ttk.Checkbutton(
            mask_video_frame,
            text="Use mask video for all source videos",
            variable=self.parent.use_mask_video,
            command=self._toggle_mask_video_controls
        ).grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)

        # Add mask video path entry and browse button
        ttk.Label(mask_video_frame, text="Mask video:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        self.mask_video_entry = ttk.Entry(mask_video_frame, textvariable=self.parent.mask_video_path, width=50)
        self.mask_video_entry.grid(column=1, row=1, padx=5, sticky=tk.W)
        self.mask_video_button = ttk.Button(mask_video_frame, text="Browse...", command=self._browse_mask_video)
        self.mask_video_button.grid(column=2, row=1, padx=5)

        # Add help text
        ttk.Label(
            mask_video_frame, 
            text="This will extract frames from the mask video and copy them to the masks subfolder of each source video.",
            wraplength=600
        ).grid(column=0, row=2, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Initialize all UI states - do this at the end after all UI components are created
        self.root_after_id = self.frame.after(100, self._initialize_ui_states)
    

    def _on_frame_configure(self, event):
        """Update the scrollregion of the canvas when the frame is resized."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    def _on_canvas_configure(self, event):
        """Update the canvas window to match the frame size."""
        self.canvas.itemconfigure(self.canvas_window, width=event.width)
    def _bind_mousewheel(self, widget):
        """Bind mousewheel scrolling to the canvas."""
        widget.bind_all("<MouseWheel>", lambda event: widget.yview_scroll(int(-1*(event.delta/120)), "units"))
        widget.bind_all("<Button-4>", lambda event: widget.yview_scroll(-1, "units"))
        widget.bind_all("<Button-5>", lambda event: widget.yview_scroll(1, "units"))




# ...existing code...
    def _create_mask_expand_section(self):
        """Create the mask expansion options section."""
        print("Creating mask expansion section")  # Debug print
        
        # Important: Use content_frame instead of frame for consistent layout
        expand_frame = ttk.LabelFrame(self.content_frame, text="Mask Expansion Options", padding="10")
        expand_frame.pack(fill=tk.X, pady=5)
        
        # Make the label more eye-catching
        header_label = ttk.Label(expand_frame, text="Mask Expansion Settings", font=("Helvetica", 10, "bold"))
        header_label.grid(column=0, row=0, columnspan=2, sticky=tk.W, padx=5, pady=(0, 10))
        
        # Iterations control
        ttk.Label(expand_frame, text="Dilation Iterations:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        iterations_spinbox = ttk.Spinbox(
            expand_frame, 
            from_=1, 
            to=50, 
            increment=1, 
            textvariable=self.parent.mask_expand_iterations, 
            width=5
        )
        iterations_spinbox.grid(column=1, row=1, padx=5, sticky=tk.W)
        
        # Kernel size control
        ttk.Label(expand_frame, text="Kernel Size:").grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
        kernel_spinbox = ttk.Spinbox(
            expand_frame, 
            from_=3, 
            to=21, 
            increment=2,  # Only odd numbers make sense for kernel size
            textvariable=self.parent.mask_expand_kernel_size, 
            width=5
        )
        kernel_spinbox.grid(column=1, row=2, padx=5, sticky=tk.W)
        
        # Preserve directory structure option
        ttk.Checkbutton(
            expand_frame, 
            text="Preserve directory structure", 
            variable=self.parent.mask_expand_preserve_structure
        ).grid(column=0, row=3, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Help text with more visible styling
        help_frame = ttk.Frame(expand_frame, padding=(5, 10, 5, 5), relief="groove", borderwidth=1)
        help_frame.grid(column=0, row=4, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=10)
        
        help_text = ("Dilates mask regions to make them larger. Higher iteration values create larger expansions. "
                    "Kernel size controls the shape of the expansion (odd numbers only).")
        ttk.Label(help_frame, text=help_text, wraplength=600).pack(padx=5, pady=5)

    def _create_conditional_resize(self):
        """Create the conditional resize options section."""
        resize_frame = ttk.LabelFrame(self.frame, text="Conditional Resize Options", padding="10")
        resize_frame.pack(fill=tk.X, pady=5)
        
        # Checkbox to enable conditional resize
        ttk.Checkbutton(
            resize_frame, 
            text="Only resize if image is larger than:", 
            variable=self.parent.resize_if_larger,
            command=self._toggle_conditional_resize_controls
        ).grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        
        # Max width and height spinboxes
        ttk.Label(resize_frame, text="Max Width:").grid(column=0, row=1, sticky=tk.W, padx=25, pady=2)
        self.max_width_spinbox = ttk.Spinbox(
            resize_frame, 
            from_=64, 
            to=4096, 
            increment=64, 
            textvariable=self.parent.max_width, 
            width=6
        )
        self.max_width_spinbox.grid(column=1, row=1, padx=5, sticky=tk.W)
        
        ttk.Label(resize_frame, text="Max Height:").grid(column=2, row=1, sticky=tk.W, padx=5, pady=2)
        self.max_height_spinbox = ttk.Spinbox(
            resize_frame, 
            from_=64, 
            to=4096, 
            increment=64, 
            textvariable=self.parent.max_height, 
            width=6
        )
        self.max_height_spinbox.grid(column=3, row=1, padx=5, sticky=tk.W)
        
        # Initially disable/enable controls based on the checkbox state
        self._toggle_conditional_resize_controls()
    
    def _create_square_padding(self):
        """Create the square padding options section."""
        padding_frame = ttk.LabelFrame(self.frame, text="Square Padding Options", padding="10")
        padding_frame.pack(fill=tk.X, pady=5)
        
        # Color options
        ttk.Label(padding_frame, text="Padding Color:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        color_combo = ttk.Combobox(
            padding_frame, 
            textvariable=self.parent.padding_color, 
            values=["black", "white", "gray"],
            width=10,
            state="readonly"
        )
        color_combo.grid(column=1, row=0, padx=5, sticky=tk.W)
        
        # Use source resolution checkbox
        ttk.Checkbutton(
            padding_frame, 
            text="Use source resolution for padding", 
            variable=self.parent.use_source_resolution_padding,
            command=self._toggle_square_padding_controls
        ).grid(column=0, row=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Target size
        ttk.Label(padding_frame, text="Target Size:").grid(column=0, row=2, sticky=tk.W, padx=25, pady=2)
        self.square_target_spinbox = ttk.Spinbox(
            padding_frame, 
            from_=64, 
            to=2048, 
            increment=64, 
            textvariable=self.parent.square_target_size, 
            width=6
        )
        self.square_target_spinbox.grid(column=1, row=2, padx=5, sticky=tk.W)
        
        # Initialize control states
        self._toggle_square_padding_controls()
    
    def _create_portrait_crop(self):
        """Create the portrait crop options section."""
        portrait_frame = ttk.LabelFrame(self.frame, text="Portrait Photo Handling", padding="10")
        portrait_frame.pack(fill=tk.X, pady=5)
        
        # Checkbox to enable portrait cropping
        ttk.Checkbutton(
            portrait_frame, 
            text="Crop portrait photos to make them square", 
            variable=self.parent.portrait_crop_enabled,
            command=self._toggle_portrait_crop_controls
        ).grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        
        # Crop position combobox
        ttk.Label(portrait_frame, text="Crop Position:").grid(column=0, row=1, sticky=tk.W, padx=25, pady=2)
        position_combo = ttk.Combobox(
            portrait_frame, 
            textvariable=self.parent.portrait_crop_position, 
            values=["top", "center", "bottom"],
            width=10,
            state="disabled"
        )
        position_combo.grid(column=1, row=1, padx=5, sticky=tk.W)
        
        # Help text
        ttk.Label(
            portrait_frame, 
            text="This will crop portrait (tall) photos to make them square. Useful for maintaining aspect ratio consistency.",
            wraplength=600
        ).grid(column=0, row=2, columnspan=3, sticky=tk.W, padx=5, pady=5)
    
    def _initialize_ui_states(self):
        """Initialize all UI states after all UI components have been created."""
        self._toggle_resolution_controls()
        self._toggle_mask_video_controls()
        self._toggle_conditional_resize_controls()
        self._toggle_square_padding_controls()
        self._toggle_portrait_crop_controls()
    
    def _toggle_resolution_controls(self):
        """Enable or disable resolution controls based on use_source_resolution checkbox."""
        if not hasattr(self, 'width_spinbox') or not hasattr(self, 'height_spinbox'):
            # These attributes might not exist yet during initialization
            return
            
        if self.parent.use_source_resolution.get():
            self.width_spinbox.configure(state="disabled")
            self.height_spinbox.configure(state="disabled")
        else:
            self.width_spinbox.configure(state="normal")
            self.height_spinbox.configure(state="normal")
    
    def _toggle_mask_video_controls(self):
        """Enable or disable mask video controls based on the checkbox state."""
        if not hasattr(self, 'mask_video_entry') or not hasattr(self, 'mask_video_button'):
            # These attributes might not exist yet during initialization
            return
            
        if self.parent.use_mask_video.get():
            self.mask_video_entry.configure(state="normal")
            self.mask_video_button.configure(state="normal")
        else:
            self.mask_video_entry.configure(state="disabled")
            self.mask_video_button.configure(state="disabled")
    
    def _toggle_conditional_resize_controls(self):
        """Enable or disable conditional resize controls based on checkbox state."""
        if not hasattr(self, 'max_width_spinbox') or not hasattr(self, 'max_height_spinbox'):
            # These attributes might not exist yet during initialization
            return
            
        if self.parent.resize_if_larger.get():
            self.max_width_spinbox.configure(state="normal")
            self.max_height_spinbox.configure(state="normal")
        else:
            self.max_width_spinbox.configure(state="disabled")
            self.max_height_spinbox.configure(state="disabled")
    
    def _toggle_square_padding_controls(self):
        """Enable or disable square padding target size controls."""
        if not hasattr(self, 'square_target_spinbox'):
            # This attribute might not exist yet during initialization
            return
            
        if self.parent.use_source_resolution_padding.get():
            self.square_target_spinbox.configure(state="disabled")
        else:
            self.square_target_spinbox.configure(state="normal")
    
    def _toggle_portrait_crop_controls(self):
        """Enable or disable portrait crop controls based on checkbox state."""
        # Find and update the crop position combobox
        for widget in self.frame.winfo_children():
            if isinstance(widget, ttk.LabelFrame) and widget.cget("text") == "Portrait Photo Handling":
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Combobox):
                        if self.parent.portrait_crop_enabled.get():
                            child.configure(state="readonly")
                        else:
                            child.configure(state="disabled")
                # At least one matching frame was found
                return
                
        # If we're here, the needed widgets might not be created yet
    
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

    def _create_crop_reinsertion(self):
        """Create the crop reinsertion options section."""
        # Create the main frame for crop reinsertion options
        reinsertion_frame = ttk.LabelFrame(self.frame, text="Crop Reinsertion", padding="10")
        reinsertion_frame.pack(fill=tk.X, pady=5)
        
        # Source images directory - make it very clear what this is
        source_frame = ttk.LabelFrame(reinsertion_frame, text="Original Uncropped Images Directory", padding=5)
        source_frame.pack(fill=tk.X, pady=5)

        ttk.Label(source_frame, 
                text="Select the directory containing the ORIGINAL UNCROPPED images:",
                font=("Helvetica", 9, "bold")).pack(anchor=tk.W, padx=5, pady=5)

        source_dir_frame = ttk.Frame(source_frame)
        source_dir_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(source_dir_frame, textvariable=self.parent.source_images_dir, width=40).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(source_dir_frame, text="Browse...", command=self._browse_source_dir).pack(side=tk.RIGHT, padx=5)
        
        # Add a direct clarification about Input Directory
        input_reminder = ttk.Frame(reinsertion_frame, padding=5, relief="groove")
        input_reminder.pack(fill=tk.X, pady=10)
        
        ttk.Label(input_reminder, 
                text="IMPORTANT: The Input Directory (set in the Input/Output tab) should contain your CROPPED images.",
                foreground="blue",
                font=("Helvetica", 9, "bold"),
                wraplength=600).pack(pady=5)
        

    def _browse_source_dir(self):
        """Browse for the source images directory."""
        from tkinter import filedialog
        directory = filedialog.askdirectory()
        if directory:
            self.parent.source_images_dir.set(directory)

    def _browse_original_dir(self):
        """Browse for original images directory."""
        from tkinter import filedialog
        directory = filedialog.askdirectory()
        if directory:
            self.parent.original_images_dir.set(directory)

    def _browse_original_image(self):
        """Browse for original image file."""
        from tkinter import filedialog
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(filetypes=filetypes, title="Select Original Image")
        if filepath:
            self.parent.selected_original_image.set(filepath)