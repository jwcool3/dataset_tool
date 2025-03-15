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
        
        # Create view mode selection
        self.view_mode_frame = ttk.Frame(self.gallery_outer_frame)
        self.view_mode_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.view_mode = tk.StringVar(value="single")
        ttk.Radiobutton(self.view_mode_frame, text="Single Image View", 
                       value="single", variable=self.view_mode, 
                       command=self._switch_view_mode).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(self.view_mode_frame, text="Gallery Overview", 
                       value="overview", variable=self.view_mode, 
                       command=self._switch_view_mode).pack(side=tk.LEFT, padx=5)
        
        # Create a paned window for dynamic resizing
        self.gallery_paned = ttk.PanedWindow(self.gallery_outer_frame, orient=tk.VERTICAL)
        self.gallery_paned.pack(fill=tk.BOTH, expand=True)
        
        # Create a canvas for scrolling
        self.gallery_canvas_frame = ttk.Frame(self.gallery_paned)
        self.gallery_paned.add(self.gallery_canvas_frame, weight=1)
        
        self.gallery_canvas = tk.Canvas(self.gallery_canvas_frame, bg="#f0f0f0")
        self.gallery_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        self.scrollbar = ttk.Scrollbar(self.gallery_canvas_frame, orient=tk.VERTICAL, command=self.gallery_canvas.yview)
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
        
        # Group images by filename across directories (base name without extension)
        image_groups = {}
        for subdir, images in all_images.items():
            for img_info in images:
                # Strip extension to group by base filename
                base_filename = os.path.splitext(img_info['filename'])[0]
                
                if base_filename not in image_groups:
                    image_groups[base_filename] = []
                image_groups[base_filename].append(img_info)
        
        # Filter and process image groups
        for base_filename, img_group in image_groups.items():
            # Separate source images and masks
            source_images = [img for img in img_group if not img['is_mask']]
            
            # Only include images that appear in multiple directories
            source_subdirs = set(img['subdir'] for img in source_images)
            
            if len(source_subdirs) > 1:
                # Get the common display filename
                display_name = base_filename + os.path.splitext(source_images[0]['filename'])[1] if source_images else base_filename
                
                self.images_data.append({
                    'filename': display_name,
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
        
        # Filter out masks and source images
        source_versions = [v for v in versions if not v['is_mask']]
        mask_versions = [v for v in versions if v['is_mask']]
        
        # Display source images in a grid
        sources_frame = ttk.LabelFrame(self.gallery_content, text="Source Images")
        sources_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)
        
        # Get available width and calculate number of columns
        window_width = self.gallery_canvas.winfo_width()
        
        # Calculate ideal thumbnail size based on window width
        # We want to fill the area while maintaining some margins
        padding_per_item = 30  # Total horizontal padding per thumbnail cell
        
        # Number of columns: try to fit at least 2, at most 5
        num_cols = min(5, max(2, len(source_versions)))
        
        # Calculate optimal thumbnail width to fill space
        optimal_width = (window_width - (padding_per_item * num_cols)) // num_cols
        
        # Create frame for the grid
        grid_outer_frame = ttk.Frame(sources_frame)
        grid_outer_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        grid_frame = ttk.Frame(grid_outer_frame)
        grid_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        # Make grid cells expandable
        for i in range(num_cols):
            grid_frame.columnconfigure(i, weight=1)
        
        # Display source images in a responsive grid layout
        for i, version in enumerate(source_versions):
            col = i % num_cols
            row = i // num_cols
            
            # Pass optimal thumbnail size based on available space
            self._add_image_to_grid(grid_frame, version, row, col, optimal_width)
        
        # Add mask toggle if masks exist
        if mask_versions:
            mask_control_frame = ttk.Frame(self.gallery_content)
            mask_control_frame.pack(fill=tk.X, pady=5, padx=5)
            
            # Create variable for mask toggle
            self.show_mask_var = tk.BooleanVar(value=False)
            
            # Create toggle button
            mask_toggle = ttk.Checkbutton(mask_control_frame, 
                                         text="Show Associated Mask", 
                                         variable=self.show_mask_var,
                                         command=self._toggle_mask_view)
            mask_toggle.pack(side=tk.LEFT, padx=5)
            
            # Create frame for mask (initially empty)
            self.mask_frame = ttk.LabelFrame(self.gallery_content, text="Mask")
            self.mask_frame.pack(fill=tk.X, pady=10, padx=5, expand=False)
            
            # Store the mask version for later use
            self.current_mask_version = mask_versions[0] if mask_versions else None
    
    def _add_image_to_grid(self, parent_frame, image_info, row, col, optimal_size=None):
        """Add an image thumbnail to the grid."""
        # Create a frame for this image
        version_frame = ttk.Frame(parent_frame, padding=5)
        version_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        self.version_frames.append(version_frame)
        
        # Load and display the image
        try:
            # Load image with OpenCV for processing
            img = cv2.imread(image_info['path'])
            if img is None:
                raise ValueError(f"Failed to load image: {image_info['path']}")
            
            # Convert to RGB for PIL
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get appropriate thumbnail size
            if optimal_size:
                # Use the provided optimal size
                max_size = optimal_size
            else:
                # Calculate based on window width if not provided
                window_width = self.gallery_canvas.winfo_width()
                if window_width < 800:
                    max_size = 180
                elif window_width < 1200:
                    max_size = 240
                else:
                    max_size = 300
            
            # Resize for thumbnail while preserving aspect ratio
            h, w = img_rgb.shape[:2]
            scale = min(max_size / w, max_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            thumbnail = cv2.resize(img_rgb, (new_w, new_h))
            
            # Convert to PIL and then to ImageTk
            pil_img = Image.fromarray(thumbnail)
            tk_img = ImageTk.PhotoImage(pil_img)
            
            # Store reference to prevent garbage collection
            self.thumbnails.append(tk_img)
            
            # Create a frame with a border to highlight the image
            img_frame = ttk.Frame(version_frame, borderwidth=2, relief="solid")
            img_frame.pack(pady=(0, 5), fill=tk.BOTH, expand=True)
            
            # Display image
            img_label = ttk.Label(img_frame, image=tk_img)
            img_label.pack(fill=tk.BOTH, expand=True)
            img_label.bind("<Button-1>", lambda e, path=image_info['path']: self._toggle_selection(e.widget))
            
            # Add image info labels in a compact format
            info_frame = ttk.Frame(version_frame)
            info_frame.pack(fill=tk.X, expand=True)
            
            # Add dimension info
            type_text = "Mask" if image_info['is_mask'] else "Image"
            info_text = f"{type_text} | {w}x{h}"
            
            info_label = ttk.Label(info_frame, text=info_text, font=("Helvetica", 9))
            info_label.pack()
            
            # Add folder label
            folder_label = ttk.Label(info_frame, text=f"Folder: {image_info['subdir']}", font=("Helvetica", 9))
            folder_label.pack()
            
            # Show shorter path instead of full path
            short_path = os.path.basename(os.path.dirname(image_info['path']))
            path_label = ttk.Label(info_frame, text=f"{short_path}/{os.path.basename(image_info['path'])}", 
                                  font=("Helvetica", 7))
            path_label.pack()
            self.version_labels.append(path_label)
            
            # Add selection checkbox in a separate frame
            select_frame = ttk.Frame(version_frame)
            select_frame.pack(fill=tk.X, expand=True, pady=(5, 0))
            
            var = tk.BooleanVar(value=False)
            image_info['select_var'] = var  # Store reference in the image info dict
            
            select_check = ttk.Checkbutton(select_frame, text="Select", variable=var)
            select_check.pack(pady=(0, 0))
            self.selected_labels.append((select_check, image_info))
            
        except Exception as e:
            print(f"Error displaying image: {str(e)}")
            error_label = ttk.Label(version_frame, text=f"Error: {str(e)}")
            error_label.pack()
    
    def _update_layout(self, event=None):
        """Update the layout when the window is resized."""
        # Only respond to main window resize events, not child widget events
        if event and event.widget != self.parent.root:
            return
            
        # If we're in single image view, update the grid layout
        if hasattr(self, 'view_mode') and self.view_mode.get() == "single":
            current_idx = self.current_image_index
            self._show_current_image()
        elif hasattr(self, 'view_mode') and self.view_mode.get() == "overview":
            self._show_gallery_overview()
    
    def _adjust_thumbnail_size(self):
        """Adjust thumbnail size based on window width."""
        window_width = self.parent.root.winfo_width()
        if window_width < 800:
            return 150  # Small window
        elif window_width < 1200:
            return 180  # Medium window
        else:
            return 220  # Large window
    
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
        
        self.view_mode.set("single")  # Ensure we're in single image view
        self.current_image_index = (self.current_image_index + 1) % len(self.images_data)
        self._update_counter()
        self._show_current_image()
    
    def _show_previous_image(self):
        """Show the previous image in the gallery."""
        if not self.images_data:
            return
        
        self.view_mode.set("single")  # Ensure we're in single image view
        self.current_image_index = (self.current_image_index - 1) % len(self.images_data)
        self._update_counter()
        self._show_current_image()
    
    def _switch_view_mode(self):
        """Switch between single image view and gallery overview."""
        if self.view_mode.get() == "single":
            # Switch to single image view
            if hasattr(self, 'current_image_index'):
                self._show_current_image()
        else:
            # Switch to gallery overview
            self._show_gallery_overview()

    def _show_gallery_overview(self):
        """Display a gallery overview of all image groups."""
        # Clear existing gallery
        for widget in self.gallery_content.winfo_children():
            widget.destroy()
        
        # Reset references
        self.thumbnails = []
        
        if not self.images_data:
            self.info_label.config(text="No images to display. Please load images first.")
            return
        
        # Create a title
        title_frame = ttk.Frame(self.gallery_content)
        title_frame.pack(fill=tk.X, pady=(10, 15))
        
        title_label = ttk.Label(title_frame, text="Gallery Overview", font=("Helvetica", 14, "bold"))
        title_label.pack()
        
        # Create a scalable frame for the gallery grid
        overview_frame = ttk.Frame(self.gallery_content)
        overview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Calculate optimal grid layout based on canvas width
        canvas_width = self.gallery_canvas.winfo_width()
        
        # Calculate thumbnail size and columns
        padding_per_item = 25  # Total horizontal padding per thumbnail
        
        # Min 2 columns, max based on width (1 column per 200px of width)
        num_cols = max(2, min(6, canvas_width // 220))
        
        # Calculate optimal thumbnail size to fill space
        optimal_width = (canvas_width - (padding_per_item * num_cols)) // num_cols
        optimal_width = max(160, min(300, optimal_width))  # Constrain between 160-300px
        
        # Configure the grid columns to be equal
        for i in range(num_cols):
            overview_frame.columnconfigure(i, weight=1)
        
        # Add each image group as a thumbnail
        for i, image_group in enumerate(self.images_data):
            row, col = divmod(i, num_cols)
            self._add_overview_thumbnail(overview_frame, image_group, i, row, col, optimal_width)
    
    def _add_overview_thumbnail(self, parent_frame, image_group, index, row, col, optimal_width=180):
        """Add a thumbnail for the overview gallery."""
        # Create frame for this thumbnail
        thumb_frame = ttk.Frame(parent_frame, padding=5)
        thumb_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        
        # Get a representative image (first source image)
        source_images = [v for v in image_group['versions'] if not v['is_mask']]
        if not source_images:
            return
        
        rep_image = source_images[0]
        
        try:
            # Load the image
            img = cv2.imread(rep_image['path'])
            if img is None:
                raise ValueError(f"Failed to load image: {rep_image['path']}")
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create thumbnail
            h, w = img_rgb.shape[:2]
            scale = min(optimal_width / w, optimal_width / h)
            new_w, new_h = int(w * scale), int(h * scale)
            thumbnail = cv2.resize(img_rgb, (new_w, new_h))
            
            # Convert to PIL and ImageTk
            pil_img = Image.fromarray(thumbnail)
            tk_img = ImageTk.PhotoImage(pil_img)
            self.thumbnails.append(tk_img)
            
            # Display image with a border to indicate selection
            img_frame = ttk.Frame(thumb_frame, borderwidth=2, relief="solid")
            img_frame.pack(pady=(0, 5), fill=tk.BOTH, expand=True)
            
            img_label = ttk.Label(img_frame, image=tk_img)
            img_label.pack(fill=tk.BOTH, expand=True)
            
            # Make clickable to view this image
            img_label.bind("<Button-1>", lambda e, idx=index: self._view_image_from_overview(idx))
            
            # Add filename label
            filename_label = ttk.Label(thumb_frame, text=image_group['filename'], font=("Helvetica", 9))
            filename_label.pack()
            
            # Add count of versions
            count_label = ttk.Label(thumb_frame, 
                                  text=f"{len(source_images)} versions", 
                                  font=("Helvetica", 8))
            count_label.pack()
            
        except Exception as e:
            error_label = ttk.Label(thumb_frame, text=f"Error: {str(e)}")
            error_label.pack()
            
    def _view_image_from_overview(self, index):
        """Switch to single image view for the selected index."""
        self.current_image_index = index
        self.view_mode.set("single")
        self._show_current_image()
        self._update_counter()
    
    def _toggle_mask_view(self):
        """Toggle display of the mask image."""
        if not hasattr(self, 'current_mask_version') or not self.current_mask_version:
            return
        
        # Clear the mask frame
        for widget in self.mask_frame.winfo_children():
            widget.destroy()
        
        # If toggled on, display the mask
        if self.show_mask_var.get() and self.current_mask_version:
            mask_grid = ttk.Frame(self.mask_frame)
            mask_grid.pack(fill=tk.X, pady=5, padx=5)
            
            # Add the mask image to the grid
            self._add_image_to_grid(mask_grid, self.current_mask_version, 0, 0)
    
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