"""
Gallery Tab for Dataset Preparation Tool
Displays multiple versions of images from different subfolders for comparison.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import shutil

class GalleryTab:
    """Tab for viewing and comparing multiple versions of the same image."""
    
    def __init__(self, parent):
        """
        Initialize the gallery tab.
        
        Args:
            parent: Parent window containing shared variables and functions
        """
        self.parent = parent
        self.frame = ttk.Frame(parent.notebook, padding="10")
        
        # State variables
        self.current_image_index = 0
        self.images_data = []  # List of dicts with path and thumbnail info
        self.thumbnails = []  # Store references to prevent garbage collection
        self.selected_labels = []  # Store references to selected image labels
        self.version_labels = []  # Store references to version labels
        self.version_frames = []  # Store references to version frames
        
        # Create the UI components
        self._create_control_section()
        self._create_gallery_section()
    
    def _create_control_section(self):
        """Create the control panel for the gallery view."""
        control_frame = ttk.LabelFrame(self.frame, text="Gallery Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        # Create a grid layout for controls
        grid_frame = ttk.Frame(control_frame)
        grid_frame.pack(fill=tk.X)
        
        # Add refresh button
        self.refresh_button = ttk.Button(grid_frame, text="Refresh Gallery", command=self.load_gallery)
        self.refresh_button.grid(column=0, row=0, padx=5, pady=5, sticky=tk.W)
        
        # Add navigation buttons
        nav_frame = ttk.Frame(grid_frame)
        nav_frame.grid(column=1, row=0, padx=5, pady=5)
        
        self.prev_button = ttk.Button(nav_frame, text="← Previous", command=self._show_previous_image)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.image_counter_label = ttk.Label(nav_frame, text="Image 0/0")
        self.image_counter_label.pack(side=tk.LEFT, padx=10)
        
        self.next_button = ttk.Button(nav_frame, text="Next →", command=self._show_next_image)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # Add delete button
        self.delete_button = ttk.Button(grid_frame, text="Delete Selected", command=self._delete_selected_images)
        self.delete_button.grid(column=2, row=0, padx=5, pady=5, sticky=tk.E)
        
        # Add a separator
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Add info section
        info_frame = ttk.Frame(control_frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        self.info_label = ttk.Label(info_frame, text="Select a folder with image versions to begin.")
        self.info_label.pack(side=tk.LEFT, padx=5)
    
    def _create_gallery_section(self):
        """Create the gallery view section."""
        # Main gallery frame with scrollbar
        self.gallery_outer_frame = ttk.Frame(self.frame)
        self.gallery_outer_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create a canvas for scrolling
        self.gallery_canvas = tk.Canvas(self.gallery_outer_frame, bg="#f0f0f0")
        self.gallery_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        self.scrollbar = ttk.Scrollbar(self.gallery_outer_frame, orient=tk.VERTICAL, command=self.gallery_canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.gallery_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Frame inside canvas for content
        self.gallery_content = ttk.Frame(self.gallery_canvas)
        self.gallery_canvas_window = self.gallery_canvas.create_window((0, 0), window=self.gallery_content, anchor=tk.NW)
        
        # Configure scrolling
        self.gallery_content.bind("<Configure>", self._on_frame_configure)
        self.gallery_canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Bind mousewheel scrolling
        self.gallery_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
    
    def _on_frame_configure(self, event):
        """Update scroll region when the inner frame changes size."""
        self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        """Resize the inner frame when the canvas changes size."""
        canvas_width = event.width
        self.gallery_canvas.itemconfig(self.gallery_canvas_window, width=canvas_width)
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling."""
        self.gallery_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def load_gallery(self):
        """Load images from input directory and refresh the gallery view."""
        input_dir = self.parent.input_dir.get()
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("Error", "Please select a valid input directory first.")
            return
        
        # Reset state
        self.current_image_index = 0
        self.images_data = []
        
        # Clear existing gallery
        for widget in self.gallery_content.winfo_children():
            widget.destroy()
        
        # Get all immediate subdirectories
        try:
            subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith('.')]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read input directory: {str(e)}")
            return
        
        if not subdirs:
            self.info_label.config(text="No subdirectories found in input folder.")
            return
        
        # Find all image files in each subdirectory
        all_images = {}
        for subdir in subdirs:
            subdir_path = os.path.join(input_dir, subdir)
            image_files = []
            
            # First check for images directly in this subdirectory
            direct_images = [f for f in os.listdir(subdir_path) 
                           if os.path.isfile(os.path.join(subdir_path, f)) and 
                           f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in direct_images:
                image_files.append({
                    'path': os.path.join(subdir_path, img_file),
                    'filename': img_file,
                    'subdir': subdir,
                    'is_mask': False
                })
            
            # Then check for a masks subdirectory
            masks_dir = os.path.join(subdir_path, "masks")
            if os.path.isdir(masks_dir):
                mask_files = [f for f in os.listdir(masks_dir) 
                             if os.path.isfile(os.path.join(masks_dir, f)) and 
                             f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for mask_file in mask_files:
                    image_files.append({
                        'path': os.path.join(masks_dir, mask_file),
                        'filename': mask_file,
                        'subdir': subdir,
                        'is_mask': True
                    })
            
            all_images[subdir] = image_files
        
        # Group images by filename across directories
        image_groups = {}
        for subdir, images in all_images.items():
            for img_info in images:
                filename = img_info['filename']
                if filename not in image_groups:
                    image_groups[filename] = []
                image_groups[filename].append(img_info)
        
        # Filter out images that don't appear in multiple directories
        for filename, img_group in image_groups.items():
            # Only count unique subdirectories, not masks
            subdirs_with_this_image = set(img['subdir'] for img in img_group)
            if len(subdirs_with_this_image) > 1:
                self.images_data.append({
                    'filename': filename,
                    'versions': img_group
                })
        
        # Sort by filename
        self.images_data.sort(key=lambda x: x['filename'])
        
        # Update counter
        self._update_counter()
        
        if not self.images_data:
            self.info_label.config(text="No matching images found across subdirectories.")
            return
        
        # Show the first image
        self._show_current_image()
    
    def _show_current_image(self):
        """Display the current image and its versions in the gallery."""
        if not self.images_data:
            return
        
        # Clear existing gallery
        for widget in self.gallery_content.winfo_children():
            widget.destroy()
        
        # Reset references
        self.thumbnails = []
        self.selected_labels = []
        self.version_labels = []
        self.version_frames = []
        
        # Get current image data
        image_data = self.images_data[self.current_image_index]
        filename = image_data['filename']
        versions = image_data['versions']
        
        # Update info label
        self.info_label.config(text=f"Viewing {filename}")
        
        # Create title
        title_frame = ttk.Frame(self.gallery_content)
        title_frame.pack(fill=tk.X, pady=(10, 5))
        
        title_label = ttk.Label(title_frame, text=f"Image: {filename}", font=("Helvetica", 12, "bold"))
        title_label.pack()
        
        # Group versions by subdirectory
        versions_by_subdir = {}
        for version in versions:
            subdir = version['subdir']
            if subdir not in versions_by_subdir:
                versions_by_subdir[subdir] = []
            versions_by_subdir[subdir].append(version)
        
        # Display each version grouped by subdirectory
        row = 0
        for subdir, version_list in versions_by_subdir.items():
            # Create subdirectory frame
            subdir_frame = ttk.LabelFrame(self.gallery_content, text=f"Folder: {subdir}")
            subdir_frame.pack(fill=tk.X, pady=10, padx=5)
            
            # Create a grid to display images
            grid_frame = ttk.Frame(subdir_frame)
            grid_frame.pack(fill=tk.X, pady=5, padx=5)
            
            # Display each image in this subdirectory
            col = 0
            for version in version_list:
                self._add_image_to_grid(grid_frame, version, row, col)
                col += 1
            
            row += 1
    
    def _add_image_to_grid(self, parent_frame, image_info, row, col):
        """Add an image thumbnail to the grid."""
        # Create a frame for this image
        version_frame = ttk.Frame(parent_frame, padding=5)
        version_frame.grid(row=row, column=col, padx=10, pady=5)
        self.version_frames.append(version_frame)
        
        # Load and display the image
        try:
            # Load image with OpenCV for processing
            img = cv2.imread(image_info['path'])
            if img is None:
                raise ValueError(f"Failed to load image: {image_info['path']}")
            
            # Convert to RGB for PIL
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize for thumbnail
            max_size = 200
            h, w = img_rgb.shape[:2]
            scale = min(max_size / w, max_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            thumbnail = cv2.resize(img_rgb, (new_w, new_h))
            
            # Convert to PIL and then to ImageTk
            pil_img = Image.fromarray(thumbnail)
            tk_img = ImageTk.PhotoImage(pil_img)
            
            # Store reference to prevent garbage collection
            self.thumbnails.append(tk_img)
            
            # Display image
            img_label = ttk.Label(version_frame, image=tk_img)
            img_label.pack(pady=(0, 5))
            img_label.bind("<Button-1>", lambda e, path=image_info['path']: self._toggle_selection(e.widget))
            
            # Add image info labels
            type_text = "Mask" if image_info['is_mask'] else "Image"
            info_text = f"{type_text} | {w}x{h}"
            
            info_label = ttk.Label(version_frame, text=info_text)
            info_label.pack()
            
            # Add path label
            path_label = ttk.Label(version_frame, text=image_info['path'], font=("Helvetica", 7))
            path_label.pack()
            self.version_labels.append(path_label)
            
            # Add selection checkbox
            var = tk.BooleanVar(value=False)
            image_info['select_var'] = var  # Store reference in the image info dict
            
            select_check = ttk.Checkbutton(version_frame, text="Select", variable=var)
            select_check.pack(pady=(5, 0))
            self.selected_labels.append((select_check, image_info))
            
        except Exception as e:
            print(f"Error displaying image: {str(e)}")
            error_label = ttk.Label(version_frame, text=f"Error: {str(e)}")
            error_label.pack()
    
    def _toggle_selection(self, widget):
        """Toggle selection when an image is clicked."""
        for check, image_info in self.selected_labels:
            if check.winfo_parent() == widget.winfo_parent():
                current_val = image_info['select_var'].get()
                image_info['select_var'].set(not current_val)
                break
    
    def _show_next_image(self):
        """Show the next image in the gallery."""
        if not self.images_data:
            return
        
        self.current_image_index = (self.current_image_index + 1) % len(self.images_data)
        self._update_counter()
        self._show_current_image()
    
    def _show_previous_image(self):
        """Show the previous image in the gallery."""
        if not self.images_data:
            return
        
        self.current_image_index = (self.current_image_index - 1) % len(self.images_data)
        self._update_counter()
        self._show_current_image()
    
    def _update_counter(self):
        """Update the image counter label."""
        total = len(self.images_data)
        current = self.current_image_index + 1 if total > 0 else 0
        self.image_counter_label.config(text=f"Image {current}/{total}")
    
    def _delete_selected_images(self):
        """Delete the selected images."""
        if not self.images_data:
            return
        
        selected_paths = []
        for check, image_info in self.selected_labels:
            if image_info['select_var'].get():
                selected_paths.append(image_info['path'])
        
        if not selected_paths:
            messagebox.showinfo("Info", "No images selected for deletion.")
            return
        
        # Confirm deletion
        confirm = messagebox.askyesno("Confirm Deletion", 
            f"Are you sure you want to delete {len(selected_paths)} selected images? This cannot be undone.")
        
        if not confirm:
            return
        
        # Delete selected files
        deleted_count = 0
        failed_count = 0
        for path in selected_paths:
            try:
                # Delete the file
                os.remove(path)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {path}: {str(e)}")
                failed_count += 1
        
        # Show result
        if failed_count > 0:
            messagebox.showwarning("Deletion Results", 
                f"Deleted {deleted_count} images. Failed to delete {failed_count} images.")
        else:
            messagebox.showinfo("Deletion Complete", f"Successfully deleted {deleted_count} images.")
        
        # Refresh gallery
        self.load_gallery()