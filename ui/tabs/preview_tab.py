"""
Preview Tab for Dataset Preparation Tool
Contains preview canvas for image and mask visualization.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np

class PreviewTab:
    """Tab for previewing processing effects on images."""
    
    def __init__(self, parent):
        """
        Initialize the preview tab.
        
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
        
        # Create the preview section
        self._create_preview_section()
        
        # Storage for tk image objects (to prevent garbage collection)
        self.tk_img = None
        self.tk_mask = None
        self.tk_before_img = None
        self.tk_before_mask = None
        self.tk_after_img = None
        self.tk_after_mask = None
    
    def _on_frame_configure(self, event):
        """Update the scrollregion when the frame size changes."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        """Resize the frame when the canvas is resized."""
        width = event.width
        self.canvas.itemconfig(self.canvas_window, width=width)
    
    def _create_preview_section(self):
        """Create the image preview section."""
        preview_frame = ttk.LabelFrame(self.frame, text="Image Preview", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.preview_canvas = tk.Canvas(preview_frame, bg="#2c2c2c", width=800, height=400)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bind canvas resize event
        self.preview_canvas.bind("<Configure>", self._on_canvas_resize)
    
    def _on_canvas_resize(self, event):
        """Handle canvas resize event."""
        # Only update if we have an image
        if self.parent.preview_image is not None:
            # Check if we have a mask
            has_mask = self.parent.preview_mask is not None
            self.update_preview(show_mask=has_mask)
    
    def load_preview(self, image_path, mask_path=None):
        """
        Load image and mask for preview.
        
        Args:
            image_path: Path to the image file
            mask_path: Path to the mask file (optional)
        """
        if image_path:
            try:
                self.parent.preview_image = cv2.imread(image_path)
                
                if mask_path:
                    self.parent.preview_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    self.update_preview(show_mask=True)
                else:
                    self.parent.preview_mask = None
                    self.update_preview(show_mask=False)
                
                # Update status
                if mask_path:
                    self.parent.status_label.config(text=f"Loaded preview image and mask")
                else:
                    self.parent.status_label.config(text=f"Loaded preview image (no mask found)")
                
            except Exception as e:
                self.parent.status_label.config(text=f"Error loading preview: {str(e)}")
                messagebox.showerror("Preview Error", f"Failed to load preview images: {str(e)}")
    
    def update_preview(self, show_mask=False):
        """
        Update the preview canvas with the loaded image and/or mask.
        
        Args:
            show_mask: Whether to show the mask alongside the image
        """
        if self.parent.preview_image is None:
            return
        
        # Resize image for display
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        # Ensure valid dimensions
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 400
            canvas_height = 200
        
        # Determine layout based on whether we have a mask
        if show_mask and self.parent.preview_mask is not None:
            # Display image and mask side by side
            display_width = canvas_width // 2
            display_height = canvas_height
            
            # Scale image
            img_h, img_w = self.parent.preview_image.shape[:2]
            scale = min(display_width / img_w, display_height / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            
            # Resize image and mask for display
            display_img = cv2.resize(self.parent.preview_image, (new_w, new_h))
            display_mask = cv2.resize(self.parent.preview_mask, (new_w, new_h))
            
            # Convert to PIL format for display
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(display_img)
            
            # For mask, create a colored version for better visibility
            colored_mask = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            colored_mask[..., 1] = display_mask  # Set green channel
            pil_mask = Image.fromarray(colored_mask)
            
            # Create photo images
            self.tk_img = ImageTk.PhotoImage(pil_img)
            self.tk_mask = ImageTk.PhotoImage(pil_mask)
            
            # Clear canvas and display images
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(new_w//2, canvas_height//2, image=self.tk_img)
            self.preview_canvas.create_image(canvas_width - new_w//2, canvas_height//2, image=self.tk_mask)
            
            # Add labels
            self.preview_canvas.create_text(new_w//2, 10, text="Original Image", fill="white", anchor=tk.N)
            self.preview_canvas.create_text(canvas_width - new_w//2, 10, text="Mask", fill="white", anchor=tk.N)
        else:
            # Display only the image
            img_h, img_w = self.parent.preview_image.shape[:2]
            scale = min(canvas_width / img_w, canvas_height / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            
            display_img = cv2.resize(self.parent.preview_image, (new_w, new_h))
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(display_img)
            
            self.tk_img = ImageTk.PhotoImage(pil_img)
            
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(canvas_width//2, canvas_height//2, image=self.tk_img)
            self.preview_canvas.create_text(canvas_width//2, 10, text="Original Image", fill="white", anchor=tk.N)
    
    def generate_preview(self):
        """Generate a preview of the processing that would be applied."""
        if self.parent.preview_image is None or self.parent.preview_mask is None:
            messagebox.showinfo("Preview", "Please select an input directory with image and mask files first.")
            return
        
        # For now, we'll just show the before/after in a simplified way
        # In a full implementation, this would apply actual processing
        
        # Create copies of the image and mask for processing
        image_copy = self.parent.preview_image.copy()
        mask_copy = self.parent.preview_mask.copy()
        
        # Example processing: Add a border to the image to simulate processing
        processed_image = cv2.copyMakeBorder(image_copy, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 255])
        processed_mask = cv2.copyMakeBorder(mask_copy, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        
        # Show the before/after preview
        self._show_before_after(image_copy, mask_copy, processed_image, processed_mask)
    
    def _show_before_after(self, before_img, before_mask, after_img, after_mask):
        """
        Show a before/after comparison of the processing.
        
        Args:
            before_img: Original image
            before_mask: Original mask
            after_img: Processed image
            after_mask: Processed mask
        """
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        # Ensure valid dimensions
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 400
            canvas_height = 200
        
        # Use a 2x2 grid layout for before/after comparison
        grid_width = canvas_width // 2
        grid_height = canvas_height // 2
        
        # Scale images to fit in the grid
        before_h, before_w = before_img.shape[:2]
        after_h, after_w = after_img.shape[:2]
        
        scale_before = min(grid_width / before_w, grid_height / before_h) * 0.9  # 90% to leave margin
        scale_after = min(grid_width / after_w, grid_height / after_h) * 0.9
        
        new_before_w, new_before_h = int(before_w * scale_before), int(before_h * scale_before)
        new_after_w, new_after_h = int(after_w * scale_after), int(after_h * scale_after)
        
        # Resize images for display
        display_before_img = cv2.resize(before_img, (new_before_w, new_before_h))
        display_before_mask = cv2.resize(before_mask, (new_before_w, new_before_h))
        display_after_img = cv2.resize(after_img, (new_after_w, new_after_h))
        display_after_mask = cv2.resize(after_mask, (new_after_w, new_after_h))
        
        # Convert to PIL format for display
        display_before_img = cv2.cvtColor(display_before_img, cv2.COLOR_BGR2RGB)
        display_after_img = cv2.cvtColor(display_after_img, cv2.COLOR_BGR2RGB)
        
        # Create colored masks for better visibility
        before_colored_mask = np.zeros((new_before_h, new_before_w, 3), dtype=np.uint8)
        before_colored_mask[..., 1] = display_before_mask  # Use mask for green channel
        
        after_colored_mask = np.zeros((new_after_h, new_after_w, 3), dtype=np.uint8)
        after_colored_mask[..., 1] = display_after_mask  # Use mask for green channel
        
        # Convert to PIL images
        pil_before_img = Image.fromarray(display_before_img)
        pil_before_mask = Image.fromarray(before_colored_mask)
        pil_after_img = Image.fromarray(display_after_img)
        pil_after_mask = Image.fromarray(after_colored_mask)
        
        # Create photo images for Tkinter
        self.tk_before_img = ImageTk.PhotoImage(pil_before_img)
        self.tk_before_mask = ImageTk.PhotoImage(pil_before_mask)
        self.tk_after_img = ImageTk.PhotoImage(pil_after_img)
        self.tk_after_mask = ImageTk.PhotoImage(pil_after_mask)
        
        # Clear canvas
        self.preview_canvas.delete("all")
        
        # Position images on canvas
        # Top left: Before Image
        self.preview_canvas.create_image(grid_width//2, grid_height//2, image=self.tk_before_img)
        self.preview_canvas.create_text(grid_width//2, 10, text="Before: Image", fill="white", anchor=tk.N)
        
        # Top right: Before Mask
        self.preview_canvas.create_image(grid_width + grid_width//2, grid_height//2, image=self.tk_before_mask)
        self.preview_canvas.create_text(grid_width + grid_width//2, 10, text="Before: Mask", fill="white", anchor=tk.N)
        
        # Bottom left: After Image
        self.preview_canvas.create_image(grid_width//2, grid_height + grid_height//2, image=self.tk_after_img)
        self.preview_canvas.create_text(grid_width//2, grid_height + 10, text="After: Image", fill="white", anchor=tk.N)
        
        # Bottom right: After Mask
        self.preview_canvas.create_image(grid_width + grid_width//2, grid_height + grid_height//2, image=self.tk_after_mask)
        self.preview_canvas.create_text(grid_width + grid_width//2, grid_height + 10, text="After: Mask", fill="white", anchor=tk.N)