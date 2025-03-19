"""
Dataset Explorer for Dataset Preparation Tool
UI component for browsing and managing datasets.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import datetime
from PIL import Image, ImageTk
import json



class DatasetExplorer:
    """UI component for browsing and managing datasets."""
    
    def __init__(self, parent, registry):
        """
        Initialize the dataset explorer UI.
        
        Args:
            parent: Parent frame for the UI
            registry: DatasetRegistry instance
        """
        self.parent = parent
        self.registry = registry
        self.selected_dataset = None
        
        # Create UI components
        self._create_layout()
        self._create_toolbar()
        self._create_tree_view()
        self._create_details_panel()
        
        # Load datasets
        self.refresh_datasets()
    
    def _create_layout(self):
        """Create the main layout."""
        # Main frame using PanedWindow for resizable sections
        self.paned_window = ttk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left frame for tree view
        self.left_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.left_frame, weight=1)
        
        # Right frame for details and operations
        self.right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_frame, weight=2)
    
    def _create_toolbar(self):
        """Create the toolbar with actions."""
        # Toolbar frame
        self.toolbar = ttk.Frame(self.left_frame)
        self.toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        # Add button
        self.add_btn = ttk.Button(self.toolbar, text="Add Dataset", command=self._add_dataset)
        self.add_btn.pack(side=tk.LEFT, padx=2)
        
        # Refresh button
        self.refresh_btn = ttk.Button(self.toolbar, text="Refresh", command=self.refresh_datasets)
        self.refresh_btn.pack(side=tk.LEFT, padx=2)
        
        # Filter control
        self.filter_frame = ttk.Frame(self.left_frame)
        self.filter_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        ttk.Label(self.filter_frame, text="Filter:").pack(side=tk.LEFT, padx=2)
        
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(self.filter_frame, textvariable=self.search_var, width=20)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        self.search_btn = ttk.Button(self.filter_frame, text="Search", command=self.refresh_datasets)
        self.search_btn.pack(side=tk.LEFT, padx=2)
        
        # Bind Enter key to search
        self.search_entry.bind("<Return>", lambda e: self.refresh_datasets())
    
    def _create_tree_view(self):
        """Create the tree view for datasets."""
        # Create frame for tree with scrollbar
        tree_frame = ttk.Frame(self.left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create scrollbar
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create tree view
        self.tree = ttk.Treeview(tree_frame, columns=("id", "files", "date"), show="tree headings")
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Configure scrollbar
        tree_scroll.config(command=self.tree.yview)
        self.tree.config(yscrollcommand=tree_scroll.set)
        
        # Configure columns
        self.tree.column("id", width=50, stretch=False)
        self.tree.column("files", width=70, stretch=False)
        self.tree.column("date", width=100, stretch=False)
        
        self.tree.heading("id", text="ID")
        self.tree.heading("files", text="Files")
        self.tree.heading("date", text="Modified")
        
        # Bind selection event
        self.tree.bind("<<TreeviewSelect>>", self._on_dataset_select)
    
    def _create_details_panel(self):
        """Create the details panel for selected dataset."""
        # Details frame
        self.details_frame = ttk.LabelFrame(self.right_frame, text="Dataset Details")
        self.details_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create scrollable frame for details
        self.details_canvas = tk.Canvas(self.details_frame)
        details_scrollbar = ttk.Scrollbar(self.details_frame, orient=tk.VERTICAL, command=self.details_canvas.yview)
        
        self.details_canvas.configure(yscrollcommand=details_scrollbar.set)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.details_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame inside canvas for content
        self.details_content = ttk.Frame(self.details_canvas)
        self.details_canvas_window = self.details_canvas.create_window((0, 0), window=self.details_content, anchor="nw")
        
        # Configure canvas
        self.details_content.bind("<Configure>", self._on_details_configure)
        self.details_canvas.bind("<Configure>", self._on_details_canvas_resize)
        
        # Details sections (empty initially, will be populated on dataset selection)
        self.info_frame = ttk.Frame(self.details_content)
        self.info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.actions_frame = ttk.LabelFrame(self.details_content, text="Actions")
        self.actions_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.preview_frame = ttk.LabelFrame(self.details_content, text="Preview")
        self.preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def _on_details_configure(self, event):
        """Handle resize of details content."""
        self.details_canvas.configure(scrollregion=self.details_canvas.bbox("all"))
    
    def _on_details_canvas_resize(self, event):
        """Handle resize of details canvas."""
        self.details_canvas.itemconfig(self.details_canvas_window, width=event.width)
    
    def refresh_datasets(self):
        """Refresh the dataset tree view."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Get all root datasets (no parent)
        search_term = self.search_var.get() if hasattr(self, 'search_var') else None
        datasets = self.registry.get_datasets(parent_id=None, search_term=search_term)
        
        # Add each dataset to the tree
        for dataset in datasets:
            self._add_dataset_to_tree(dataset)
    
    def _add_dataset_to_tree(self, dataset, parent=""):
        """
        Add a dataset to the tree view.
        
        Args:
            dataset: Dataset information
            parent: Parent item ID in the tree view
        """
        # Format modified date
        try:
            modified_date = datetime.datetime.fromisoformat(dataset['modified_date'])
            date_display = modified_date.strftime("%Y-%m-%d")
        except:
            date_display = "Unknown"
        
        # Create tree item
        item_id = self.tree.insert(
            parent, "end", 
            text=dataset['name'],
            values=(dataset['id'], dataset['file_count'], date_display),
            tags=("dataset",)
        )
        
        # Load child datasets
        children = self.registry.get_datasets(parent_id=dataset['id'])
        for child in children:
            self._add_dataset_to_tree(child, item_id)
        
        return item_id
    
    def _on_dataset_select(self, event):
        """Handle dataset selection in the tree view."""
        selected_items = self.tree.selection()
        if not selected_items:
            self._clear_details()
            self.selected_dataset = None
            return
        
        # Get dataset ID from the selected item
        item_id = selected_items[0]
        dataset_id = self.tree.item(item_id, "values")[0]
        
        # Load dataset details
        dataset = self.registry.get_dataset(int(dataset_id))
        if not dataset:
            self._clear_details()
            self.selected_dataset = None
            return
        
        # Store selected dataset
        self.selected_dataset = dataset
        
        # Update details panel
        self._update_details()
    
    def _clear_details(self):
        """Clear the details panel."""
        # Clear info section
        for widget in self.info_frame.winfo_children():
            widget.destroy()
        
        # Clear actions section
        for widget in self.actions_frame.winfo_children():
            widget.destroy()
        
        # Clear preview section
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
    
    def _update_details(self):
        """Update the details panel with selected dataset info."""
        if not self.selected_dataset:
            return
            
        # Clear existing widgets
        self._clear_details()
        
        dataset = self.selected_dataset
        
        # Info section
        ttk.Label(self.info_frame, text=f"Name: {dataset['name']}", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=2)
        ttk.Label(self.info_frame, text=f"Path: {dataset['path']}").pack(anchor=tk.W, pady=2)
        ttk.Label(self.info_frame, text=f"Category: {dataset['category'] or 'None'}").pack(anchor=tk.W, pady=2)
        ttk.Label(self.info_frame, text=f"Files: {dataset['file_count']} total, {dataset['image_count']} images, {dataset['video_count']} videos").pack(anchor=tk.W, pady=2)
        
        # Format dates
        try:
            created = datetime.datetime.fromisoformat(dataset['created_date']).strftime("%Y-%m-%d %H:%M")
            modified = datetime.datetime.fromisoformat(dataset['modified_date']).strftime("%Y-%m-%d %H:%M")
            ttk.Label(self.info_frame, text=f"Created: {created}").pack(anchor=tk.W, pady=2)
            ttk.Label(self.info_frame, text=f"Modified: {modified}").pack(anchor=tk.W, pady=2)
        except:
            pass
        
        # Description
        desc_frame = ttk.LabelFrame(self.info_frame, text="Description")
        desc_frame.pack(fill=tk.X, pady=5, expand=True)
        
        desc_text = tk.Text(desc_frame, height=3, width=40, wrap=tk.WORD)
        desc_text.pack(fill=tk.X, expand=True, padx=5, pady=5)
        desc_text.insert("1.0", dataset['description'] or "No description")
        desc_text.config(state=tk.DISABLED)
        
        # Tags
        tags = self.registry.get_tags(dataset['id'])
        tags_frame = ttk.LabelFrame(self.info_frame, text="Tags")
        tags_frame.pack(fill=tk.X, pady=5, expand=True)
        
        if tags:
            tags_text = ", ".join(tags)
            ttk.Label(tags_frame, text=tags_text, wraplength=400).pack(padx=5, pady=5, anchor=tk.W)
        else:
            ttk.Label(tags_frame, text="No tags").pack(padx=5, pady=5, anchor=tk.W)
        
        # Tag management
        tag_manage_frame = ttk.Frame(tags_frame)
        tag_manage_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.new_tag_var = tk.StringVar()
        tag_entry = ttk.Entry(tag_manage_frame, textvariable=self.new_tag_var, width=15)
        tag_entry.pack(side=tk.LEFT, padx=2)
        
        add_tag_btn = ttk.Button(tag_manage_frame, text="Add Tag", 
                                command=lambda: self._add_tag(self.new_tag_var.get()))
        add_tag_btn.pack(side=tk.LEFT, padx=2)
        
        # Actions section
        edit_btn = ttk.Button(self.actions_frame, text="Edit Dataset", command=self._edit_dataset)
        edit_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        delete_btn = ttk.Button(self.actions_frame, text="Delete Dataset", command=self._delete_dataset)
        delete_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        open_btn = ttk.Button(self.actions_frame, text="Open in Explorer", command=self._open_dataset_location)
        open_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Process button to connect with main processing pipeline
        process_btn = ttk.Button(self.actions_frame, text="Process Dataset", command=self._process_dataset)
        process_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Preview section - show sample images or stats
        self._load_preview()
    
    def _load_preview(self):
        """Load a preview of dataset content."""
        if not self.selected_dataset:
            return
            
        path = self.selected_dataset['path']
        
        # Create a notebook for different preview types
        preview_notebook = ttk.Notebook(self.preview_frame)
        preview_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Thumbnail tab
        thumb_frame = ttk.Frame(preview_notebook)
        preview_notebook.add(thumb_frame, text="Thumbnails")
        
        # Stats tab
        stats_frame = ttk.Frame(preview_notebook)
        preview_notebook.add(stats_frame, text="Statistics")
        
        # History tab
        history_frame = ttk.Frame(preview_notebook)
        preview_notebook.add(history_frame, text="History")
        
        # Load thumbnails
        self._load_thumbnails(thumb_frame, path)
        
        # Load stats
        self._load_statistics(stats_frame, path)
        
        # Load processing history
        self._load_history(history_frame)
    
    def _load_thumbnails(self, parent, path):
        """Load thumbnail previews of dataset images."""
        # Create a canvas with scrollbar for thumbnails
        thumb_canvas = tk.Canvas(parent)
        thumb_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=thumb_canvas.yview)
        
        thumb_canvas.configure(yscrollcommand=thumb_scrollbar.set)
        thumb_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        thumb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame inside canvas for thumbnails
        thumb_frame = ttk.Frame(thumb_canvas)
        thumb_window = thumb_canvas.create_window((0, 0), window=thumb_frame, anchor="nw")
        
        # Configure canvas
        thumb_frame.bind("<Configure>", lambda e: thumb_canvas.configure(scrollregion=thumb_canvas.bbox("all")))
        thumb_canvas.bind("<Configure>", lambda e: thumb_canvas.itemconfig(thumb_window, width=e.width))
        
        # Find image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(self._find_files(path, ext, max_count=50))  # Limit to 50 files for performance
        
        if not image_files:
            ttk.Label(thumb_frame, text="No images found in this dataset").pack(padx=20, pady=20)
            return
        
        # Determine grid layout
        thumb_size = 150
        canvas_width = thumb_canvas.winfo_width()
        if canvas_width < 100:  # If canvas not yet sized, use a reasonable default
            canvas_width = 600
        
        cols = max(1, canvas_width // (thumb_size + 10))
        
        # Create thumbnail grid
        self.thumb_images = []  # Store references to prevent garbage collection
        
        for i, img_path in enumerate(image_files[:20]):  # Limit to 20 thumbnails
            # Calculate grid position
            row, col = divmod(i, cols)
            
            # Create frame for this thumbnail
            item_frame = ttk.Frame(thumb_frame, width=thumb_size, height=thumb_size)
            item_frame.grid(row=row, column=col, padx=5, pady=5)
            item_frame.pack_propagate(False)  # Force fixed size
            
            try:
                # Load and resize image
                img = Image.open(img_path)
                img.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
                
                # Create PhotoImage
                tk_img = ImageTk.PhotoImage(img)
                self.thumb_images.append(tk_img)  # Store reference
                
                # Display image
                img_label = ttk.Label(item_frame, image=tk_img)
                img_label.pack(fill=tk.BOTH, expand=True)
                
                # Add tooltip with filename
                self._add_tooltip(img_label, os.path.basename(img_path))
                
            except Exception as e:
                # Show error placeholder
                error_label = ttk.Label(item_frame, text="Error loading image")
                error_label.pack(fill=tk.BOTH, expand=True)
                print(f"Error loading thumbnail: {str(e)}")
    
    def _load_statistics(self, parent, path):
        """Load and display dataset statistics."""
        # Create frame for statistics
        stats_frame = ttk.Frame(parent)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Calculate statistics
        stats = self._calculate_statistics(path)
        
        # Display stats in a grid
        row = 0
        
        # File counts
        ttk.Label(stats_frame, text="File Statistics:", font=("Helvetica", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row += 1
        
        ttk.Label(stats_frame, text="Total Files:").grid(row=row, column=0, sticky="w", padx=5)
        ttk.Label(stats_frame, text=str(stats['file_count'])).grid(row=row, column=1, sticky="w", padx=5)
        row += 1
        
        ttk.Label(stats_frame, text="Images:").grid(row=row, column=0, sticky="w", padx=5)
        ttk.Label(stats_frame, text=str(stats['image_count'])).grid(row=row, column=1, sticky="w", padx=5)
        row += 1
        
        ttk.Label(stats_frame, text="Videos:").grid(row=row, column=0, sticky="w", padx=5)
        ttk.Label(stats_frame, text=str(stats['video_count'])).grid(row=row, column=1, sticky="w", padx=5)
        row += 1
        
        # Folder structure
        ttk.Label(stats_frame, text="Folder Structure:", font=("Helvetica", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 5))
        row += 1
        
        ttk.Label(stats_frame, text="Subfolders:").grid(row=row, column=0, sticky="w", padx=5)
        ttk.Label(stats_frame, text=str(stats['subfolder_count'])).grid(row=row, column=1, sticky="w", padx=5)
        row += 1
        
        ttk.Label(stats_frame, text="Max Depth:").grid(row=row, column=0, sticky="w", padx=5)
        ttk.Label(stats_frame, text=str(stats['max_depth'])).grid(row=row, column=1, sticky="w", padx=5)
        row += 1
        
        # Image statistics if applicable
        if stats['image_count'] > 0:
            ttk.Label(stats_frame, text="Image Statistics:", font=("Helvetica", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 5))
            row += 1
            
            if 'avg_width' in stats:
                ttk.Label(stats_frame, text="Avg. Dimensions:").grid(row=row, column=0, sticky="w", padx=5)
                ttk.Label(stats_frame, text=f"{stats['avg_width']:.0f} × {stats['avg_height']:.0f}").grid(row=row, column=1, sticky="w", padx=5)
                row += 1
            
            if 'unique_resolutions' in stats:
                ttk.Label(stats_frame, text="Unique Resolutions:").grid(row=row, column=0, sticky="w", padx=5)
                ttk.Label(stats_frame, text=str(stats['unique_resolutions'])).grid(row=row, column=1, sticky="w", padx=5)
                row += 1
            
            # Show file formats
            if stats['formats']:
                ttk.Label(stats_frame, text="File Formats:").grid(row=row, column=0, sticky="w", padx=5)
                formats_str = ", ".join(f"{fmt} ({count})" for fmt, count in stats['formats'].items())
                ttk.Label(stats_frame, text=formats_str, wraplength=300).grid(row=row, column=1, sticky="w", padx=5)
                row += 1
    
    def _load_history(self, parent):
        """Load and display processing history."""
        # Create frame for history
        history_frame = ttk.Frame(parent)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Get processing history
        history = []
        if self.selected_dataset and self.selected_dataset.get('processing_history'):
            try:
                history = json.loads(self.selected_dataset['processing_history'])
            except:
                history = []
        
        if not history:
            ttk.Label(history_frame, text="No processing history available").pack(padx=10, pady=10)
            return
        
        # Create a tree view for history
        columns = ("timestamp", "operation", "params")
        history_tree = ttk.Treeview(history_frame, columns=columns, show="headings")
        history_tree.pack(fill=tk.BOTH, expand=True)
        
        # Configure columns
        history_tree.heading("timestamp", text="Timestamp")
        history_tree.heading("operation", text="Operation")
        history_tree.heading("params", text="Parameters")
        
        history_tree.column("timestamp", width=150)
        history_tree.column("operation", width=150)
        history_tree.column("params", width=300)
        
        # Add history entries
        for entry in reversed(history):  # Show most recent first
            # Format timestamp
            timestamp = entry.get('timestamp', '')
            try:
                dt = datetime.datetime.fromisoformat(timestamp)
                timestamp_display = dt.strftime("%Y-%m-%d %H:%M")
            except:
                timestamp_display = timestamp
            
            # Format parameters
            params = entry.get('params', {})
            if isinstance(params, dict):
                params_display = ", ".join(f"{k}={v}" for k, v in params.items())
            else:
                params_display = str(params)
            
            # Add to tree
            history_tree.insert("", "end", values=(
                timestamp_display,
                entry.get('operation', ''),
                params_display
            ))
    
    def _calculate_statistics(self, path):
        """Calculate statistics for a dataset."""
        stats = {
            'file_count': 0,
            'image_count': 0,
            'video_count': 0,
            'subfolder_count': 0,
            'max_depth': 0,
            'formats': {}
        }
        
        if not os.path.isdir(path):
            return stats
            
        # Collect image dimensions for sampling
        widths = []
        heights = []
        resolutions = set()
        
        # Define file types
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
        
        # Track deepest folder level
        base_depth = path.count(os.sep)
        max_depth = 0
        
        # Scan directory
        for root, dirs, files in os.walk(path):
            # Calculate depth
            current_depth = root.count(os.sep) - base_depth
            max_depth = max(max_depth, current_depth)
            
            # Count subfolders (excluding hidden ones)
            valid_dirs = [d for d in dirs if not d.startswith('.')]
            stats['subfolder_count'] += len(valid_dirs)
            
            # Process files
            for file in files:
                stats['file_count'] += 1
                
                # Get file extension
                ext = os.path.splitext(file)[1].lower()
                
                # Update format counts
                if ext not in stats['formats']:
                    stats['formats'][ext] = 0
                stats['formats'][ext] += 1
                
                # Categorize file
                if ext in image_extensions:
                    stats['image_count'] += 1
                    
                    # Try to get image dimensions (sample first 100 images)
                    if len(widths) < 100:
                        try:
                            img_path = os.path.join(root, file)
                            img = Image.open(img_path)
                            w, h = img.size
                            widths.append(w)
                            heights.append(h)
                            resolutions.add((w, h))
                            img.close()
                        except:
                            pass
                            
                elif ext in video_extensions:
                    stats['video_count'] += 1
        
        # Calculate image statistics if applicable
        if widths and heights:
            stats['avg_width'] = sum(widths) / len(widths)
            stats['avg_height'] = sum(heights) / len(heights)
            stats['unique_resolutions'] = len(resolutions)
        
        stats['max_depth'] = max_depth
        
        return stats
    
    def _find_files(self, directory, extension, max_count=None):
        """Find files with a specific extension in a directory tree."""
        found_files = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(extension.lower()):
                    found_files.append(os.path.join(root, file))
                    if max_count and len(found_files) >= max_count:
                        return found_files
        
        return found_files
    
    def _add_tooltip(self, widget, text):
        """Add a tooltip to a widget."""
        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            
            # Create toplevel window
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            
            label = ttk.Label(self.tooltip, text=text, wraplength=250,
                           background="#ffffe0", relief="solid", borderwidth=1, padding=5)
            label.pack()
            
        def leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()
                
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)
    
    def _add_dataset(self):
        """Show dialog to add a new dataset."""
        # Create dialog
        dialog = tk.Toplevel(self.parent)
        dialog.title("Add Dataset")
        dialog.geometry("500x400")
        dialog.transient(self.parent)
        dialog.grab_set()
        
        # Create form
        form_frame = ttk.Frame(dialog, padding=10)
        form_frame.pack(fill=tk.BOTH, expand=True)
        
        # Dataset name
        ttk.Label(form_frame, text="Dataset Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        name_var = tk.StringVar()
        name_entry = ttk.Entry(form_frame, textvariable=name_var, width=30)
        name_entry.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Dataset path
        ttk.Label(form_frame, text="Dataset Path:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        path_frame = ttk.Frame(form_frame)
        path_frame.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        path_var = tk.StringVar()
        path_entry = ttk.Entry(path_frame, textvariable=path_var, width=30)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(path_frame, text="Browse...", 
                              command=lambda: path_var.set(filedialog.askdirectory()))
        browse_btn.pack(side=tk.RIGHT, padx=5)
        
        # Category
        ttk.Label(form_frame, text="Category:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        category_var = tk.StringVar()
        category_combo = ttk.Combobox(form_frame, textvariable=category_var, width=30)
        category_combo.grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        category_combo['values'] = ["Training", "Validation", "Testing", "Raw", "Processed"]
        
        # Description
        ttk.Label(form_frame, text="Description:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        description_text = tk.Text(form_frame, height=5, width=30)
        description_text.grid(row=3, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Parent dataset selection
        ttk.Label(form_frame, text="Parent Dataset:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Get all datasets for the dropdown
        datasets = self.registry.get_datasets()
        dataset_names = ["None"] + [f"{d['id']}: {d['name']}" for d in datasets]
        
        parent_var = tk.StringVar(value="None")
        parent_combo = ttk.Combobox(form_frame, textvariable=parent_var, width=30)
        parent_combo.grid(row=4, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        parent_combo['values'] = dataset_names
        
        # Tags
        ttk.Label(form_frame, text="Tags:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        tags_var = tk.StringVar()
        tags_entry = ttk.Entry(form_frame, textvariable=tags_var, width=30)
        tags_entry.grid(row=5, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        ttk.Label(form_frame, text="Separate tags with commas").grid(row=6, column=1, sticky=tk.W, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(form_frame)
        button_frame.grid(row=7, column=0, columnspan=2, pady=20)
        
        cancel_btn = ttk.Button(button_frame, text="Cancel", command=dialog.destroy)
        cancel_btn.pack(side=tk.RIGHT, padx=5)
        
        create_btn = ttk.Button(button_frame, text="Create Dataset", 
                              command=lambda: self._create_dataset(
                                  name_var.get(),
                                  path_var.get(),
                                  description_text.get("1.0", tk.END).strip(),
                                  category_var.get(),
                                  parent_var.get(),
                                  tags_var.get(),
                                  dialog
                              ))
        create_btn.pack(side=tk.RIGHT, padx=5)
        
        # Configure grid column weights
        form_frame.columnconfigure(1, weight=1)
    
    def _create_dataset(self, name, path, description, category, parent, tags, dialog):
        """Create a new dataset from form data."""
        # Validate inputs
        if not name or not name.strip():
            messagebox.showerror("Error", "Dataset name is required")
            return
        
        if not path or not os.path.isdir(path):
            messagebox.showerror("Error", "Please select a valid directory")
            return
        
        # Parse parent dataset ID
        parent_id = None
        if parent and parent != "None":
            try:
                parent_id = int(parent.split(":")[0])
            except:
                pass
        
        # Create the dataset
        try:
            dataset_id = self.registry.add_dataset(
                name=name.strip(),
                path=path,
                description=description,
                category=category,
                parent_id=parent_id
            )
            
            # Add tags if provided
            if tags:
                tag_list = [t.strip() for t in tags.split(",") if t.strip()]
                for tag in tag_list:
                    self.registry.add_tag(dataset_id, tag)
            
            # Close dialog and refresh
            dialog.destroy()
            self.refresh_datasets()
            
            # Select the new dataset
            self._select_dataset_by_id(dataset_id)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create dataset: {str(e)}")
    
    def _select_dataset_by_id(self, dataset_id):
        """Find and select a dataset in the tree by ID."""
        # Look through all items to find the matching dataset
        def find_item(item=""):
            # Check this item
            if item:
                values = self.tree.item(item, "values")
                if values and values[0] == str(dataset_id):
                    return item
                
                # Check children
                for child in self.tree.get_children(item):
                    result = find_item(child)
                    if result:
                        return result
            else:
                # Start with top-level items
                for top_item in self.tree.get_children():
                    result = find_item(top_item)
                    if result:
                        return result
            
            return None
        
        # Find the item
        item = find_item()
        if item:
            # Select and show the item
            self.tree.selection_set(item)
            self.tree.see(item)
            self._on_dataset_select(None)  # Update details panel
    
    def _edit_dataset(self):
        """Show dialog to edit the selected dataset."""
        if not self.selected_dataset:
            return
            
        dataset = self.selected_dataset
        
        # Create dialog
        dialog = tk.Toplevel(self.parent)
        dialog.title(f"Edit Dataset: {dataset['name']}")
        dialog.geometry("500x400")
        dialog.transient(self.parent)
        dialog.grab_set()
        
        # Create form
        form_frame = ttk.Frame(dialog, padding=10)
        form_frame.pack(fill=tk.BOTH, expand=True)
        
        # Dataset name
        ttk.Label(form_frame, text="Dataset Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        name_var = tk.StringVar(value=dataset['name'])
        name_entry = ttk.Entry(form_frame, textvariable=name_var, width=30)
        name_entry.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Category
        ttk.Label(form_frame, text="Category:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        category_var = tk.StringVar(value=dataset['category'] or "")
        category_combo = ttk.Combobox(form_frame, textvariable=category_var, width=30)
        category_combo.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        category_combo['values'] = ["Training", "Validation", "Testing", "Raw", "Processed"]
        
        # Description
        ttk.Label(form_frame, text="Description:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        description_text = tk.Text(form_frame, height=5, width=30)
        description_text.grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        description_text.insert("1.0", dataset['description'] or "")
        
        # Tags
        ttk.Label(form_frame, text="Tags:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        current_tags = self.registry.get_tags(dataset['id'])
        tags_var = tk.StringVar(value=", ".join(current_tags))
        tags_entry = ttk.Entry(form_frame, textvariable=tags_var, width=30)
        tags_entry.grid(row=3, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        ttk.Label(form_frame, text="Separate tags with commas").grid(row=4, column=1, sticky=tk.W, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(form_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        cancel_btn = ttk.Button(button_frame, text="Cancel", command=dialog.destroy)
        cancel_btn.pack(side=tk.RIGHT, padx=5)
        
        update_btn = ttk.Button(button_frame, text="Update Dataset", 
                              command=lambda: self._update_dataset(
                                  dataset['id'],
                                  name_var.get(),
                                  description_text.get("1.0", tk.END).strip(),
                                  category_var.get(),
                                  tags_var.get(),
                                  dialog
                              ))
        update_btn.pack(side=tk.RIGHT, padx=5)
        
        # Configure grid column weights
        form_frame.columnconfigure(1, weight=1)
    
    def _update_dataset(self, dataset_id, name, description, category, tags, dialog):
        """Update dataset with new values."""
        # Validate inputs
        if not name or not name.strip():
            messagebox.showerror("Error", "Dataset name is required")
            return
        
        # Update the dataset
        try:
            success = self.registry.update_dataset(
                dataset_id,
                name=name.strip(),
                description=description,
                category=category
            )
            
            if not success:
                messagebox.showerror("Error", "Failed to update dataset")
                return
            
            # Update tags
            current_tags = self.registry.get_tags(dataset_id)
            new_tags = [t.strip() for t in tags.split(",") if t.strip()]
            
            # Remove deleted tags
            for tag in current_tags:
                if tag not in new_tags:
                    self.registry.remove_tag(dataset_id, tag)
            
            # Add new tags
            for tag in new_tags:
                if tag not in current_tags:
                    self.registry.add_tag(dataset_id, tag)
            
            # Close dialog and refresh
            dialog.destroy()
            self.refresh_datasets()
            
            # Reselect the dataset
            self._select_dataset_by_id(dataset_id)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update dataset: {str(e)}")
    
    def _delete_dataset(self):
        """Delete the selected dataset."""
        if not self.selected_dataset:
            return
            
        dataset = self.selected_dataset
        
        # Confirm deletion
        result = messagebox.askquestion(
            "Confirm Deletion",
            f"Are you sure you want to delete the dataset '{dataset['name']}'?\n\nThis will only remove it from the registry, not delete the actual files.",
            icon='warning'
        )
        
        if result != 'yes':
            return
        
        # Offer to delete files too
        delete_files = False
        if os.path.exists(dataset['path']):
            file_result = messagebox.askquestion(
                "Delete Files",
                f"Do you also want to delete all files in '{dataset['path']}'?\n\nThis action cannot be undone.",
                icon='warning'
            )
            delete_files = (file_result == 'yes')
        
        # Delete the dataset
        try:
            success = self.registry.delete_dataset(dataset['id'], delete_files=delete_files)
            
            if success:
                messagebox.showinfo("Success", "Dataset deleted successfully")
                self.refresh_datasets()
                self.selected_dataset = None
                self._clear_details()
            else:
                messagebox.showerror("Error", "Failed to delete dataset")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error deleting dataset: {str(e)}")
    
    def _open_dataset_location(self):
        """Open the dataset folder in the file explorer."""
        if not self.selected_dataset:
            return
            
        path = self.selected_dataset['path']
        if not os.path.exists(path):
            messagebox.showerror("Error", f"Path does not exist: {path}")
            return
            
        # Open folder in file explorer (platform-specific)
        import subprocess
        import platform
        
        try:
            if platform.system() == 'Windows':
                subprocess.Popen(['explorer', path])
            elif platform.system() == 'Darwin':  # macOS
                subprocess.Popen(['open', path])
            else:  # Linux
                subprocess.Popen(['xdg-open', path])
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open folder: {str(e)}")
    
    def _process_dataset(self):
        """Send the dataset to the main processing pipeline."""
        if not self.selected_dataset:
            return
            
        try:
            # Set the input directory in the main application
            self.parent.app.input_dir.set(self.selected_dataset['path'])
            
            # Switch to the Input/Output tab
            self.parent.app.notebook.select(0)  # Assuming Input/Output is the first tab
            
            # Show a confirmation message
            messagebox.showinfo(
                "Dataset Selected",
                f"Dataset '{self.selected_dataset['name']}' has been set as the input directory.\n\n"
                "Configure your processing options and click 'Start Processing'."
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set dataset for processing: {str(e)}")
    
    def _add_tag(self, tag):
        """Add a tag to the selected dataset."""
        if not self.selected_dataset or not tag.strip():
            return
            
        # Add the tag
        success = self.registry.add_tag(self.selected_dataset['id'], tag.strip())
        
        if success:
            # Clear the tag entry
            self.new_tag_var.set("")
            
            # Refresh the dataset details
            self._update_details()
        else:
            messagebox.showerror("Error", f"Failed to add tag '{tag}'")