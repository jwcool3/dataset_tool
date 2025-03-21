"""
Simplified Reinsertion UI for Dataset Preparation Tool
Provides a streamlined interface for reinsertion options.
"""

import tkinter as tk
from tkinter import ttk, filedialog
import os

class SimplifiedReinsertionUI:
    """UI component for simplified reinsertion controls."""
    
    def __init__(self, parent_frame, app):
        """
        Initialize the simplified reinsertion UI.
        
        Args:
            parent_frame: Parent frame to contain this UI component
            app: The main application with shared variables
        """
        self.app = app
        self.frame = ttk.LabelFrame(parent_frame, text="Image Reinsertion", padding=10)
        
        # Create the UI components
        self._create_ui()
    
    def _create_ui(self):
        """Create the simplified UI components."""
        # Create a more organized layout with tabs for different reinsertion types
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.basic_tab = ttk.Frame(self.notebook, padding=10)
        self.hair_tab = ttk.Frame(self.notebook, padding=10)
        self.advanced_tab = ttk.Frame(self.notebook, padding=10)
        
        self.notebook.add(self.basic_tab, text="Basic")
        self.notebook.add(self.hair_tab, text="Hair")
        self.notebook.add(self.advanced_tab, text="Advanced")
        
        # Basic tab content
        self._create_basic_tab()
        
        # Hair tab content
        self._create_hair_tab()
        
        # Advanced tab content
        self._create_advanced_tab()
        
        # Source directory selection (common to all tabs)
        self._create_source_directory_section()
    
    def _create_basic_tab(self):
        """Create the basic reinsertion options tab."""
        # Enable reinsertion checkbox
        ttk.Checkbutton(
            self.basic_tab,
            text="Enable image reinsertion",
            variable=self.app.reinsert_crops_option,
            command=self._on_reinsertion_toggle
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Simple explanation
        ttk.Label(
            self.basic_tab,
            text="Reinserts cropped/processed images back into the original images.",
            wraplength=400
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Mask-only option
        ttk.Checkbutton(
            self.basic_tab,
            text="Mask-only reinsertion (only replace masked areas)",
            variable=self.app.reinsert_mask_only
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Enhanced reinsertion option
        ttk.Checkbutton(
            self.basic_tab,
            text="Enhanced reinsertion (better handles different resolutions)",
            variable=self.app.use_enhanced_reinserter
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Edge softness slider
        softness_frame = ttk.Frame(self.basic_tab)
        softness_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(softness_frame, text="Edge Softness:").pack(side=tk.LEFT, padx=5)
        
        edge_slider = ttk.Scale(
            softness_frame,
            from_=0,
            to=30,
            orient=tk.HORIZONTAL,
            variable=self.app.soft_edge_width,
            length=200
        )
        edge_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Label to show current softness value
        self.edge_label = ttk.Label(softness_frame, text="15")
        self.edge_label.pack(side=tk.LEFT, padx=5)
        
        # Update label when slider moves
        def update_edge_label(*args):
            width = self.app.soft_edge_width.get()
            self.edge_label.config(text=f"{width:.0f}")
        
        self.app.soft_edge_width.trace_add("write", update_edge_label)
    
    def _create_hair_tab(self):
        """Create the hair reinsertion tab."""
        # Enable Smart Hair Reinserter
        ttk.Checkbutton(
            self.hair_tab,
            text="Use Smart Hair Reinserter (optimized for hair replacement)",
            variable=self.app.use_smart_hair_reinserter,
            command=self._on_hair_reinserter_toggle
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Simple explanation
        ttk.Label(
            self.hair_tab,
            text="Specialized for hair replacement with intelligent alignment and blending.",
            wraplength=400
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Hair color correction
        ttk.Checkbutton(
            self.hair_tab,
            text="Enable automatic color correction for hair",
            variable=self.app.hair_color_correction
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Hair alignment toggle
        ttk.Checkbutton(
            self.hair_tab,
            text="Prioritize top alignment for hair (recommended)",
            variable=self.app.hair_top_alignment
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Vertical Alignment Bias slider
        vertical_frame = ttk.Frame(self.hair_tab)
        vertical_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(vertical_frame, text="Vertical Position:").pack(side=tk.LEFT, padx=5)
        
        vertical_slider = ttk.Scale(
            vertical_frame,
            from_=-50,
            to=50,
            orient=tk.HORIZONTAL,
            variable=self.app.vertical_alignment_bias,
            length=200
        )
        vertical_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Add explanatory labels at min, center, and max
        ttk.Label(vertical_frame, text="Up").pack(side=tk.LEFT)
        self.vertical_value = ttk.Label(vertical_frame, text="0")
        self.vertical_value.pack(side=tk.LEFT, padx=5)
        ttk.Label(vertical_frame, text="Down").pack(side=tk.LEFT)
        
        # Update value label when slider moves
        def update_vertical_label(*args):
            value = self.app.vertical_alignment_bias.get()
            self.vertical_value.config(text=f"{value:.0f}")
        
        self.app.vertical_alignment_bias.trace_add("write", update_vertical_label)
        
        # Edge softness slider for hair
        softness_frame = ttk.Frame(self.hair_tab)
        softness_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(softness_frame, text="Edge Softness:").pack(side=tk.LEFT, padx=5)
        
        hair_edge_slider = ttk.Scale(
            softness_frame,
            from_=0,
            to=30,
            orient=tk.HORIZONTAL,
            variable=self.app.soft_edge_width,
            length=200
        )
        hair_edge_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Label for sharp-soft ends
        ttk.Label(softness_frame, text="Sharp").pack(side=tk.LEFT)
        self.hair_edge_label = ttk.Label(softness_frame, text="15")
        self.hair_edge_label.pack(side=tk.LEFT, padx=5)
        ttk.Label(softness_frame, text="Soft").pack(side=tk.LEFT)
        
        # Update label when slider moves
        def update_hair_edge_label(*args):
            width = self.app.soft_edge_width.get()
            self.hair_edge_label.config(text=f"{width:.0f}")
        
        self.app.soft_edge_width.trace_add("write", update_hair_edge_label)
        
        # Hair presets
        preset_frame = ttk.LabelFrame(self.hair_tab, text="Quick Presets", padding=5)
        preset_frame.pack(fill=tk.X, pady=10, padx=5)
        
        preset_buttons = ttk.Frame(preset_frame)
        preset_buttons.pack(fill=tk.X, pady=5)
        
        # Create preset buttons with colors
        self._create_preset_button(
            preset_buttons, "Natural Hair", "natural", 
            "#9A7B4F", lambda: self._apply_hair_preset("natural")
        ).pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        self._create_preset_button(
            preset_buttons, "Anime Hair", "anime", 
            "#6A5ACD", lambda: self._apply_hair_preset("anime")
        ).pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        self._create_preset_button(
            preset_buttons, "Updo/Ponytail", "updo", 
            "#8E44AD", lambda: self._apply_hair_preset("updo")
        ).pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
    
    def _create_advanced_tab(self):
        """Create the advanced reinsertion options tab."""
        # Handle different masks option
        ttk.Checkbutton(
            self.advanced_tab,
            text="Handle different masks between source and processed images",
            variable=self.app.reinsert_handle_different_masks,
            command=self._on_different_masks_toggle
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Alignment method selection
        alignment_frame = ttk.Frame(self.advanced_tab)
        alignment_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(alignment_frame, text="Alignment Method:").pack(side=tk.LEFT, padx=5)
        
        ttk.Combobox(
            alignment_frame,
            textvariable=self.app.reinsert_alignment_method,
            values=["none", "centroid", "bbox", "landmarks", "contour", "iou"],
            width=15,
            state="readonly"
        ).pack(side=tk.LEFT, padx=5)
        
        # Blend mode selection
        blend_frame = ttk.Frame(self.advanced_tab)
        blend_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(blend_frame, text="Blend Mode:").pack(side=tk.LEFT, padx=5)
        
        ttk.Combobox(
            blend_frame,
            textvariable=self.app.reinsert_blend_mode,
            values=["alpha", "poisson", "feathered"],
            width=15,
            state="readonly"
        ).pack(side=tk.LEFT, padx=5)
        
        # Blend extent slider
        extent_frame = ttk.Frame(self.advanced_tab)
        extent_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(extent_frame, text="Blend Extent:").pack(side=tk.LEFT, padx=5)
        
        ttk.Scale(
            extent_frame,
            from_=0,
            to=20,
            orient=tk.HORIZONTAL,
            variable=self.app.reinsert_blend_extent,
            length=200
        ).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Debug mode option
        ttk.Checkbutton(
            self.advanced_tab,
            text="Debug Mode (Save visualization images)",
            variable=self.app.debug_mode
        ).pack(anchor=tk.W, padx=5, pady=10)
    
    def _create_source_directory_section(self):
        """Create the source directory selection section (common to all methods)."""
        source_frame = ttk.LabelFrame(self.frame, text="Original Images Directory", padding=10)
        source_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Important note
        important_note = ttk.Label(
            source_frame,
            text="IMPORTANT: Select the directory containing your ORIGINAL UNCROPPED images.\nThe Input Directory should contain your PROCESSED or CROPPED images.",
            foreground="blue",
            font=("Helvetica", 9, "bold"),
            wraplength=500
        )
        important_note.pack(fill=tk.X, pady=5)
        
        # Directory selection
        dir_frame = ttk.Frame(source_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(dir_frame, text="Source Directory:").pack(side=tk.LEFT, padx=5)
        
        ttk.Entry(
            dir_frame,
            textvariable=self.app.source_images_dir,
            width=40
        ).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(
            dir_frame,
            text="Browse...",
            command=self._browse_source_dir
        ).pack(side=tk.RIGHT, padx=5)
    
    def _browse_source_dir(self):
        """Browse for source directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.app.source_images_dir.set(directory)
    
    def _create_preset_button(self, parent, text, preset_type, color, command):
        """Create a styled preset button."""
        button_frame = ttk.Frame(parent)
        
        # Create the button
        button = ttk.Button(
            button_frame,
            text=text,
            command=command
        )
        button.pack(fill=tk.BOTH, expand=True)
        
        return button_frame
    
    def _apply_hair_preset(self, preset_type):
        """Apply predefined settings for different hair types."""
        if preset_type == "natural":
            # Natural hair settings
            self.app.vertical_alignment_bias.set(10)
            self.app.soft_edge_width.set(15)
            self.app.reinsert_blend_mode.set("feathered")
            self.app.hair_color_correction.set(True)
            self.app.hair_top_alignment.set(True)
        elif preset_type == "anime":
            # Anime hair settings - usually needs more prominent edges
            self.app.vertical_alignment_bias.set(5)
            self.app.soft_edge_width.set(8)
            self.app.reinsert_blend_mode.set("alpha")
            self.app.hair_color_correction.set(True)
            self.app.hair_top_alignment.set(True)
        elif preset_type == "updo":
            # Updo/ponytail settings - higher alignment for vertical hairstyles
            self.app.vertical_alignment_bias.set(-20)
            self.app.soft_edge_width.set(12)
            self.app.reinsert_blend_mode.set("feathered")
            self.app.hair_color_correction.set(True)
            self.app.hair_top_alignment.set(True)
    
    def _on_reinsertion_toggle(self):
        """Called when reinsertion option is toggled."""
        # Enable/disable relevant UI components
        if self.app.reinsert_crops_option.get():
            # If reinsertion is enabled, ensure smart hair is disabled by default
            if not self.app.use_smart_hair_reinserter.get() and not self.app.use_enhanced_reinserter.get():
                self.app.use_enhanced_reinserter.set(True)
        else:
            # If reinsertion is disabled, disable other options
            self.app.use_smart_hair_reinserter.set(False)
            self.app.use_enhanced_reinserter.set(False)
    
    def _on_hair_reinserter_toggle(self):
        """Called when Smart Hair Reinserter is toggled."""
        if self.app.use_smart_hair_reinserter.get():
            # Enable reinsertion and disable enhanced reinserter
            self.app.reinsert_crops_option.set(True)
            self.app.use_enhanced_reinserter.set(False)
            
            # Select the Hair tab
            self.notebook.select(self.hair_tab)
    
    def _on_different_masks_toggle(self):
        """Called when different masks option is toggled."""
        # This would enable/disable advanced alignment options if needed
        pass
    
    def show(self):
        """Show the frame."""
        self.frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
    def hide(self):
        """Hide the frame."""
        self.frame.pack_forget()
    
    def update_ui(self):
        """Update UI state based on variable values."""
        # Select the appropriate tab based on which reinsertion method is active
        if self.app.use_smart_hair_reinserter.get():
            self.notebook.select(self.hair_tab)
        elif self.app.use_enhanced_reinserter.get() or self.app.reinsert_handle_different_masks.get():
            self.notebook.select(self.advanced_tab)
        else:
            self.notebook.select(self.basic_tab)