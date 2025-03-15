"""
Outlier Group Detection for Dataset Preparation Tool
Identifies groups of images with significant outliers.
"""
import threading
import tkinter as tk
import os
import numpy as np
import cv2
from utils.image_comparison import ImageComparison

class OutlierGroupDetector:
    """Class to identify groups containing significant outliers."""
    
    def __init__(self, gallery_tab, threshold=0.4):
        """
        Initialize the outlier group detector.
        
        Args:
            gallery_tab: The gallery tab containing image groups
            threshold: Outlier threshold (higher = more restrictive)
        """
        self.gallery_tab = gallery_tab
        self.threshold = threshold
        self.comparator = ImageComparison()
        
        # Storage for results
        self.outlier_groups = []
        self.current_outlier_group_index = 0
    
    def find_outlier_groups(self, method="histogram_correlation", min_group_size=3):
        """
        Analyze all image groups to find those with significant outliers.
        
        Args:
            method: Comparison method to use
            min_group_size: Minimum number of images in a group to analyze
            
        Returns:
            list: Groups with outliers, sorted by severity
        """
        self.outlier_groups = []
        
        # Get all image groups from the gallery tab
        if not hasattr(self.gallery_tab, 'images_data') or not self.gallery_tab.images_data:
            return []
        
        # Process each group
        for group_index, image_data in enumerate(self.gallery_tab.images_data):
            # Get versions (excluding masks)
            versions = image_data['versions']
            source_versions = [v for v in versions if not v['is_mask']]
            
            # Skip groups that are too small
            if len(source_versions) < min_group_size:
                continue
            
            # Get image paths
            image_paths = [v['path'] for v in source_versions]
            
            # Analyze this group
            group_result = self._analyze_group(
                group_index, 
                image_data['filename'], 
                image_paths,
                method
            )
            
            # Add to results if outliers found
            if group_result and group_result['outliers']:
                self.outlier_groups.append(group_result)
        
        # Sort groups by max outlier score (descending)
        self.outlier_groups.sort(key=lambda x: x['max_outlier_score'], reverse=True)
        
        # Reset current index
        self.current_outlier_group_index = 0
        
        return self.outlier_groups
        
    def _analyze_group(self, group_index, group_name, image_paths, method):
        """
        Analyze a single group for outliers.
        
        Args:
            group_index: Index of the group in the gallery
            group_name: Name of the image group
            image_paths: List of image paths in the group
            method: Comparison method to use
            
        Returns:
            dict: Group analysis results with outlier information
        """
        # Compare images within this group
        result = self.comparator.compare_images(image_paths, method=method)
        
        if not result.get("success", False):
            return None
        
        # Get outlier scores
        outlier_scores = result.get("outlier_scores", [])
        paths = result.get("paths", [])
        
        # Calculate group statistics
        mean_score = np.mean(outlier_scores)
        std_score = np.std(outlier_scores)
        
        # Identify outliers (score > mean + threshold * std)
        outlier_threshold = mean_score + (self.threshold * std_score)
        outliers = []
        
        for i, score in enumerate(outlier_scores):
            if score > outlier_threshold:
                # Get folder context for display
                path = paths[i]
                folder = os.path.basename(os.path.dirname(path))
                filename = os.path.basename(path)
                
                outliers.append({
                    'path': path,
                    'score': score,
                    'display_name': f"{folder}/{filename}",
                    'index': i
                })
        
        # Sort outliers by score (descending)
        outliers.sort(key=lambda x: x['score'], reverse=True)
                
        # Create group result only if outliers found
        if outliers:
            return {
                'group_index': group_index,
                'group_name': group_name,
                'outliers': outliers,
                'mean_score': mean_score,
                'std_score': std_score,
                'threshold': outlier_threshold,
                'max_outlier_score': outliers[0]['score'] if outliers else 0,
                'method': method,
                'paths': paths
            }
        
        return None
    
    def get_next_outlier_group(self):
        """
        Get the next group with outliers.
        
        Returns:
            dict: Next outlier group information
        """
        if not self.outlier_groups:
            return None
        
        # Get current group
        group = self.outlier_groups[self.current_outlier_group_index]
        
        # Increment index (with wraparound)
        self.current_outlier_group_index = (self.current_outlier_group_index + 1) % len(self.outlier_groups)
        
        return group
    
    def get_previous_outlier_group(self):
        """
        Get the previous group with outliers.
        
        Returns:
            dict: Previous outlier group information
        """
        if not self.outlier_groups:
            return None
        
        # Decrement index (with wraparound)
        self.current_outlier_group_index = (self.current_outlier_group_index - 1) % len(self.outlier_groups)
        
        # Get current group
        group = self.outlier_groups[self.current_outlier_group_index]
        
        return group


class OutlierReviewDialog:
    """Dialog for reviewing image groups with outliers."""
    
    def __init__(self, parent, detector):
        """
        Initialize the outlier review dialog.
        
        Args:
            parent: Parent window
            detector: OutlierGroupDetector instance
        """
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk
        
        self.parent = parent
        self.detector = detector
        
        # Dialog setup
        self.dialog = tk.Toplevel(parent.root)
        self.dialog.title("Outlier Group Review")
        self.dialog.geometry("900x700")
        self.dialog.minsize(800, 600)
        self.dialog.transient(parent.root)
        
        # Make the window pop to the front
        self.dialog.lift()
        self.dialog.attributes('-topmost', True)
        self.dialog.after_idle(self.dialog.attributes, '-topmost', False)
        
        # Image references
        self.thumbnails = []
        self.current_group = None
        
        # Create the UI
        self._create_content()
        
        # Start with first group
        self._load_current_group()
        
        # Wait for the dialog to close
        self.dialog.focus_set()
        self.dialog.grab_set()
    
    def _create_content(self):
        """Create the dialog UI."""
        import tkinter as tk
        from tkinter import ttk
        
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header section
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Navigation controls
        nav_frame = ttk.Frame(header_frame)
        nav_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.prev_btn = ttk.Button(nav_frame, text="← Previous Group", command=self._show_previous_group)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.group_label = ttk.Label(nav_frame, text="Group 0/0", font=("Helvetica", 10))
        self.group_label.pack(side=tk.LEFT, padx=10)
        
        self.next_btn = ttk.Button(nav_frame, text="Next Group →", command=self._show_next_group)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        # Group information
        self.group_info_frame = ttk.LabelFrame(main_frame, text="Group Information", padding="10")
        self.group_info_frame.pack(fill=tk.X, pady=5)
        
        self.group_name_label = ttk.Label(self.group_info_frame, text="Group: ", font=("Helvetica", 10, "bold"))
        self.group_name_label.pack(anchor=tk.W)
        
        self.outlier_stats_label = ttk.Label(self.group_info_frame, text="")
        self.outlier_stats_label.pack(anchor=tk.W, pady=5)
        
        # View button
        view_btn = ttk.Button(self.group_info_frame, text="View Group in Gallery", command=self._view_in_gallery)
        view_btn.pack(anchor=tk.W, pady=5)
        
        # Compare button
        compare_btn = ttk.Button(self.group_info_frame, text="Compare All in Group", command=self._compare_group)
        compare_btn.pack(anchor=tk.W, pady=5)
        
        # Create notebook for outliers and group views
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Outliers tab
        self.outliers_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.outliers_frame, text="Outliers")
        
        # Add scrollable frame for outliers
        outlier_scroll_frame = ttk.Frame(self.outliers_frame)
        outlier_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        self.outlier_canvas = tk.Canvas(outlier_scroll_frame)
        self.outlier_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        outlier_scrollbar = ttk.Scrollbar(outlier_scroll_frame, orient=tk.VERTICAL, command=self.outlier_canvas.yview)
        outlier_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.outlier_canvas.configure(yscrollcommand=outlier_scrollbar.set)
        
        # Frame inside canvas for outlier content
        self.outlier_content = ttk.Frame(self.outlier_canvas)
        outlier_window = self.outlier_canvas.create_window((0, 0), window=self.outlier_content, anchor=tk.NW)
        
        # Configure scrolling
        self.outlier_content.bind("<Configure>", 
                                lambda e: self.outlier_canvas.configure(scrollregion=self.outlier_canvas.bbox("all")))
        self.outlier_canvas.bind("<Configure>", 
                               lambda e: self.outlier_canvas.itemconfig(outlier_window, width=e.width))
        
        # Group overview tab
        self.overview_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.overview_frame, text="Group Overview")
        
        # Overview content
        self.overview_label = ttk.Label(self.overview_frame)
        self.overview_label.pack(fill=tk.BOTH, expand=True)
        
        # Close button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        settings_btn = ttk.Button(button_frame, text="Detection Settings", command=self._show_settings)
        settings_btn.pack(side=tk.LEFT, padx=5)
        
        close_button = ttk.Button(button_frame, text="Close", command=self.dialog.destroy)
        close_button.pack(side=tk.RIGHT, padx=5)
    
    def _load_current_group(self):
        """Load and display the current outlier group."""
        self.current_group = self.detector.get_next_outlier_group()
        self._display_group()
    
    def _show_next_group(self):
        """Show the next outlier group."""
        self.current_group = self.detector.get_next_outlier_group()
        self._display_group()
    
    def _show_previous_group(self):
        """Show the previous outlier group."""
        self.current_group = self.detector.get_previous_outlier_group()
        self._display_group()
    
    def _display_group(self):
        """Display the current group's information and outliers."""
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk
        import cv2
        
        if not self.current_group:
            # No groups with outliers found
            self.group_name_label.config(text="No outlier groups found")
            self.outlier_stats_label.config(text="Try adjusting the detection threshold.")
            self.group_label.config(text="Group 0/0")
            
            # Clear outlier content
            for widget in self.outlier_content.winfo_children():
                widget.destroy()
                
            return
        
        # Update navigation label
        total_groups = len(self.detector.outlier_groups)
        current_index = self.detector.current_outlier_group_index
        # Adjust for showing the previous group due to how the index is managed
        if current_index == 0:
            display_index = total_groups
        else:
            display_index = current_index
            
        self.group_label.config(text=f"Group {display_index}/{total_groups}")
        
        # Update group information
        group_name = self.current_group['group_name']
        self.group_name_label.config(text=f"Group: {group_name}")
        
        # Update stats
        stats_text = (
            f"Method: {self.current_group['method']}\n"
            f"Found {len(self.current_group['outliers'])} outliers\n"
            f"Mean score: {self.current_group['mean_score']:.4f}\n"
            f"Threshold: {self.current_group['threshold']:.4f}"
        )
        self.outlier_stats_label.config(text=stats_text)
        
        # Clear existing content
        for widget in self.outlier_content.winfo_children():
            widget.destroy()
        
        # Reset thumbnails
        self.thumbnails = []
        
        # Display outliers
        outliers = self.current_group['outliers']
        
        for i, outlier in enumerate(outliers):
            # Create frame for this outlier
            outlier_frame = ttk.Frame(self.outlier_content, padding=10)
            outlier_frame.pack(fill=tk.X, pady=5)
            
            # Add separator between outliers
            if i > 0:
                ttk.Separator(self.outlier_content, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
            
            try:
                # Load image
                img = cv2.imread(outlier['path'])
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize for display
                    h, w = img_rgb.shape[:2]
                    scale = min(250 / w, 250 / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    display_img = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    # Convert to PhotoImage
                    pil_img = Image.fromarray(display_img)
                    tk_img = ImageTk.PhotoImage(pil_img)
                    self.thumbnails.append(tk_img)  # Keep reference
                    
                    # Display image and info side by side
                    img_label = ttk.Label(outlier_frame, image=tk_img)
                    img_label.grid(column=0, row=0, rowspan=4, padx=10, pady=5)
                    
                    # Add info
                    ttk.Label(
                        outlier_frame, 
                        text=f"Outlier #{i+1}: {outlier['display_name']}", 
                        font=("Helvetica", 10, "bold")
                    ).grid(column=1, row=0, sticky=tk.W, padx=10)
                    
                    ttk.Label(
                        outlier_frame, 
                        text=f"Outlier Score: {outlier['score']:.4f}"
                    ).grid(column=1, row=1, sticky=tk.W, padx=10)
                    
                    ttk.Label(
                        outlier_frame, 
                        text=f"Dimensions: {w}x{h}"
                    ).grid(column=1, row=2, sticky=tk.W, padx=10)
                    
                    # Add buttons
                    button_frame = ttk.Frame(outlier_frame)
                    button_frame.grid(column=1, row=3, sticky=tk.W, padx=10, pady=5)
                    
                    ttk.Button(
                        button_frame, 
                        text="Compare with Others", 
                        command=lambda p=outlier['path']: self._compare_with_others(p)
                    ).pack(side=tk.LEFT, padx=5)
                    
                    ttk.Button(
                        button_frame, 
                        text="Open in Gallery", 
                        command=lambda idx=self.current_group['group_index']: self._view_in_gallery(idx)
                    ).pack(side=tk.LEFT, padx=5)
                    
                else:
                    ttk.Label(
                        outlier_frame,
                        text=f"Error loading image: {outlier['display_name']}"
                    ).pack(pady=10)
                    
            except Exception as e:
                ttk.Label(
                    outlier_frame,
                    text=f"Error processing {outlier['display_name']}: {str(e)}"
                ).pack(pady=10)
        
        # Generate and display overview visualization
        self._generate_overview()
    
    def _generate_overview(self):
        """Generate and display an overview visualization of the group."""
        from PIL import Image, ImageTk
        
        if not self.current_group:
            return
            
        # Create visualization
        try:
            # Get all image paths
            image_paths = self.current_group['paths']
            
            # Load images
            images = []
            for path in image_paths:
                img = cv2.imread(path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img_rgb)
            
            # Get comparison results
            result = self.detector.comparator.compare_images(
                image_paths, 
                method=self.current_group['method']
            )
            
            if result.get("success", False):
                # Generate overview visualization
                vis = self.detector.comparator.generate_comparison_visualization(
                    images,
                    result["distance_matrix"],
                    result["outlier_scores"],
                    image_paths
                )
                
                # Resize for display if needed
                canvas_width = self.overview_frame.winfo_width()
                canvas_height = self.overview_frame.winfo_height()
                
                # Ensure we have reasonable minimum dimensions
                if canvas_width < 100:
                    canvas_width = 800
                if canvas_height < 100:
                    canvas_height = 500
                
                # Resize if image is too large
                if vis.width > canvas_width or vis.height > canvas_height:
                    scale = min(canvas_width / vis.width, canvas_height / vis.height)
                    new_size = (int(vis.width * scale), int(vis.height * scale))
                    vis = vis.resize(new_size, Image.LANCZOS)
                
                # Convert to PhotoImage
                self.overview_img = ImageTk.PhotoImage(vis)
                
                # Display on the label
                self.overview_label.config(image=self.overview_img)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error generating overview: {str(e)}")
    
    def _view_in_gallery(self, group_index=None):
        """Navigate to the current group in the gallery tab."""
        if group_index is None and self.current_group:
            group_index = self.current_group['group_index']
            
        if group_index is not None:
            # Switch to gallery tab
            self.parent.notebook.select(3)  # Assuming gallery tab is index 3
            
            # Go to the specified group
            gallery_tab = self.parent.gallery_tab
            gallery_tab.current_image_index = group_index
            gallery_tab.view_mode.set("single")  # Switch to single view mode
            gallery_tab._update_counter()
            gallery_tab._show_current_image()
    
    def _compare_group(self):
        """Open the comparison dialog for all images in the current group."""
        if not self.current_group:
            return
            
        # Get all image paths
        image_paths = self.current_group['paths']
        
        # Import and create ComparisonDialog
        from ui.comparison_dialog import ComparisonDialog
        ComparisonDialog(self.parent, image_paths)
    
    def _compare_with_others(self, outlier_path):
        """Compare the outlier with other images in the group."""
        if not self.current_group:
            return
            
        # Get all image paths and ensure the outlier is first
        image_paths = self.current_group['paths'].copy()
        
        # Move outlier to front of list
        if outlier_path in image_paths:
            image_paths.remove(outlier_path)
        image_paths.insert(0, outlier_path)
        
        # Import and create ComparisonDialog
        from ui.comparison_dialog import ComparisonDialog
        ComparisonDialog(self.parent, image_paths)
    
    def _show_settings(self):
        """Show settings dialog for outlier detection."""
        import tkinter as tk
        from tkinter import ttk
        
        # Create settings dialog
        settings_dialog = tk.Toplevel(self.dialog)
        settings_dialog.title("Outlier Detection Settings")
        settings_dialog.geometry("400x300")
        settings_dialog.transient(self.dialog)
        settings_dialog.grab_set()
        
        # Create settings frame
        settings_frame = ttk.Frame(settings_dialog, padding=20)
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # Threshold setting
        ttk.Label(settings_frame, text="Outlier Threshold:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=10)
        
        threshold_var = tk.DoubleVar(value=self.detector.threshold)
        threshold_scale = ttk.Scale(
            settings_frame, 
            from_=0.1, 
            to=2.0, 
            orient=tk.HORIZONTAL, 
            variable=threshold_var, 
            length=200
        )
        threshold_scale.grid(column=1, row=0, sticky=tk.W, padx=5, pady=10)
        
        threshold_label = ttk.Label(settings_frame, text=f"{threshold_var.get():.2f}")
        threshold_label.grid(column=2, row=0, sticky=tk.W, padx=5, pady=10)
        
        # Update threshold label when scale changes
        def update_threshold_label(*args):
            threshold_label.config(text=f"{threshold_var.get():.2f}")
        
        threshold_var.trace_add("write", update_threshold_label)
        
        # Method selection
        ttk.Label(settings_frame, text="Comparison Method:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=10)
        
        method_var = tk.StringVar(value="histogram_correlation")
        method_combo = ttk.Combobox(settings_frame, textvariable=method_var, width=25)
        method_combo['values'] = [
            "histogram_correlation",
            "histogram_chi",
            "histogram_intersection",
            "histogram_bhattacharyya",
            "ssim",
            "mse"
        ]
        method_combo.grid(column=1, row=1, columnspan=2, sticky=tk.W, padx=5, pady=10)
        method_combo.state(['readonly'])
        
        # Minimum group size
        ttk.Label(settings_frame, text="Minimum Group Size:").grid(column=0, row=2, sticky=tk.W, padx=5, pady=10)
        
        min_size_var = tk.IntVar(value=3)
        min_size_spin = ttk.Spinbox(settings_frame, from_=2, to=20, textvariable=min_size_var, width=5)
        min_size_spin.grid(column=1, row=2, sticky=tk.W, padx=5, pady=10)
        
        # Add explanation
        explanation = ttk.Label(
            settings_frame, 
            text="Threshold: Higher values detect only more extreme outliers.\n"
                 "Lower values will find more potential outliers.",
            wraplength=380
        )
        explanation.grid(column=0, row=3, columnspan=3, sticky=tk.W, padx=5, pady=10)
        
        # Add buttons
        button_frame = ttk.Frame(settings_dialog)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            button_frame, 
            text="Cancel", 
            command=settings_dialog.destroy
        ).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Apply & Rescan", 
            command=lambda: self._apply_settings(
                threshold_var.get(),
                method_var.get(),
                min_size_var.get(),
                settings_dialog
            )
        ).pack(side=tk.RIGHT, padx=5)
    
    def _apply_settings(self, threshold, method, min_size, dialog):
        """Apply new settings and rescan for outliers."""
        # Update detector settings
        self.detector.threshold = threshold
        
        # Close settings dialog
        dialog.destroy()
        
        # Rescan with new settings
        self.detector.find_outlier_groups(method=method, min_group_size=min_size)
        
        # Reset to first group
        self.current_group = self.detector.get_next_outlier_group()
        self._display_group()


def add_outlier_detection_button(parent):
    """
    Add outlier detection button to the gallery controls.
    
    Args:
        parent: The main window
    """
    import tkinter as tk
    from tkinter import ttk
    
    # Check if gallery tab is initialized
    if not hasattr(parent, 'gallery_tab'):
        return
        
    gallery_tab = parent.gallery_tab
    
    # Find control frame
    control_frame = None
    for child in gallery_tab.frame.winfo_children():
        if isinstance(child, ttk.LabelFrame) and "Controls" in child.cget("text"):
            control_frame = child
            break
    
    if not control_frame:
        return
    
    # Find buttons frame
    buttons_frame = None
    for child in control_frame.winfo_children():
        if isinstance(child, ttk.Frame) and hasattr(child, 'winfo_children'):
            for widget in child.winfo_children():
                if isinstance(widget, ttk.Button) and "Compare" in widget.cget("text"):
                    buttons_frame = child
                    break
    
    if not buttons_frame:
        return
    
    # Create detector instance
    detector = OutlierGroupDetector(gallery_tab)
    
    # Create and add outlier scan button
    outlier_btn = ttk.Button(
        buttons_frame, 
        text="Find Outlier Groups", 
        command=lambda: run_outlier_scan(parent, detector)
    )
    outlier_btn.pack(side=tk.LEFT, padx=5)


def run_outlier_scan(parent, detector):
    """
    Run the outlier scan and show results dialog.
    
    Args:
        parent: The main window
        detector: OutlierGroupDetector instance
    """
    import tkinter as tk
    from tkinter import ttk, messagebox
    
    # Show progress dialog
    progress_dialog = tk.Toplevel(parent.root)
    progress_dialog.title("Scanning for Outliers")
    progress_dialog.geometry("400x150")
    progress_dialog.transient(parent.root)
    progress_dialog.grab_set()
    
    # Add progress bar
    ttk.Label(
        progress_dialog, 
        text="Scanning image groups for outliers...",
        font=("Helvetica", 10, "bold")
    ).pack(padx=20, pady=10)
    
    progress_var = tk.DoubleVar(value=0)
    progress_bar = ttk.Progressbar(
        progress_dialog, 
        variable=progress_var, 
        maximum=100, 
        length=350
    )
    progress_bar.pack(padx=20, pady=10)
    
    status_label = ttk.Label(progress_dialog, text="Processing...")
    status_label.pack(padx=20, pady=5)
    
    # Run analysis in a separate thread
    def scan_thread():
        try:
            # Get gallery tab data
            if not hasattr(parent.gallery_tab, 'images_data') or not parent.gallery_tab.images_data:
                progress_dialog.after(0, progress_dialog.destroy)
                progress_dialog.after(10, lambda: messagebox.showinfo(
                    "No Images",
                    "No image groups found. Please load images first."
                ))
                return
            
            # Update progress
            progress_dialog.after(0, lambda: progress_var.set(10))
            progress_dialog.after(0, lambda: status_label.config(
                text=f"Analyzing {len(parent.gallery_tab.images_data)} image groups..."
            ))
            
            # Run outlier detection
            results = detector.find_outlier_groups()
            
            # Update progress
            progress_dialog.after(0, lambda: progress_var.set(90))
            
            # Process results
            if results:
                # Close progress dialog
                progress_dialog.after(0, progress_dialog.destroy)
                
                # Show results dialog
                progress_dialog.after(10, lambda: OutlierReviewDialog(parent, detector))
            else:
                # No outliers found
                progress_dialog.after(0, progress_dialog.destroy)
                progress_dialog.after(10, lambda: messagebox.showinfo(
                    "No Outliers Found",
                    "No significant outliers were found in any image group.\n\n" +
                    "Try lowering the detection threshold to find more potential outliers."
                ))
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            progress_dialog.after(0, progress_dialog.destroy)
            progress_dialog.after(10, lambda: messagebox.showerror(
                "Error",
                f"Error scanning for outliers: {str(e)}"
            ))
    
    # Start scanning thread
    threading.Thread(target=scan_thread, daemon=True).start()


def add_outlier_detection_menu(parent):
    """
    Add outlier detection options to the application menu.
    
    Args:
        parent: The main window
    """
    # Get the menu bar
    menu_bar = parent.menu_bar
    
    # Check if there's already a Tools menu
    tools_menu = None
    for i in range(menu_bar.index('end') + 1):
        if menu_bar.type(i) == 'cascade' and menu_bar.entrycget(i, 'label') == 'Tools':
            tools_menu = menu_bar.nametowidget(menu_bar.entrycget(i, 'menu'))
            break
    
    # Create Tools menu if it doesn't exist
    if tools_menu is None:
        tools_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Tools", menu=tools_menu)
    
    # Create detector instance
    detector = OutlierGroupDetector(parent.gallery_tab)
    
    # Add outlier detection option
    tools_menu.add_separator()
    tools_menu.add_command(
        label="Find Outlier Groups...", 
        command=lambda: run_outlier_scan(parent, detector)
    )
    tools_menu.add_command(
        label="Outlier Detection Settings...", 
        command=lambda: show_settings_dialog(parent, detector)
    )


def show_settings_dialog(parent, detector):
    """
    Show settings dialog for outlier detection.
    
    Args:
        parent: The main window
        detector: OutlierGroupDetector instance
    """
    import tkinter as tk
    from tkinter import ttk
    
    # Create settings dialog
    settings_dialog = tk.Toplevel(parent.root)
    settings_dialog.title("Outlier Detection Settings")
    settings_dialog.geometry("400x300")
    settings_dialog.transient(parent.root)
    settings_dialog.grab_set()
    
    # Create settings frame
    settings_frame = ttk.Frame(settings_dialog, padding=20)
    settings_frame.pack(fill=tk.BOTH, expand=True)
    
    # Threshold setting
    ttk.Label(settings_frame, text="Outlier Threshold:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=10)
    
    threshold_var = tk.DoubleVar(value=detector.threshold)
    threshold_scale = ttk.Scale(
        settings_frame, 
        from_=0.1, 
        to=2.0, 
        orient=tk.HORIZONTAL, 
        variable=threshold_var, 
        length=200
    )
    threshold_scale.grid(column=1, row=0, sticky=tk.W, padx=5, pady=10)
    
    threshold_label = ttk.Label(settings_frame, text=f"{threshold_var.get():.2f}")
    threshold_label.grid(column=2, row=0, sticky=tk.W, padx=5, pady=10)
    
    # Update threshold label when scale changes
    def update_threshold_label(*args):
        threshold_label.config(text=f"{threshold_var.get():.2f}")
    
    threshold_var.trace_add("write", update_threshold_label)
    
    # Method selection
    ttk.Label(settings_frame, text="Comparison Method:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=10)
    
    method_var = tk.StringVar(value="histogram_correlation")
    method_combo = ttk.Combobox(settings_frame, textvariable=method_var, width=25)
    method_combo['values'] = [
        "histogram_correlation",
        "histogram_chi",
        "histogram_intersection",
        "histogram_bhattacharyya",
        "ssim",
        "mse"
    ]
    method_combo.grid(column=1, row=1, columnspan=2, sticky=tk.W, padx=5, pady=10)
    method_combo.state(['readonly'])
    
    # Minimum group size
    ttk.Label(settings_frame, text="Minimum Group Size:").grid(column=0, row=2, sticky=tk.W, padx=5, pady=10)
    
    min_size_var = tk.IntVar(value=3)
    min_size_spin = ttk.Spinbox(settings_frame, from_=2, to=20, textvariable=min_size_var, width=5)
    min_size_spin.grid(column=1, row=2, sticky=tk.W, padx=5, pady=10)
    
    # Add explanation
    explanation = ttk.Label(
        settings_frame, 
        text="Threshold: Higher values detect only more extreme outliers.\n"
             "Lower values will find more potential outliers.",
        wraplength=380
    )
    explanation.grid(column=0, row=3, columnspan=3, sticky=tk.W, padx=5, pady=10)
    
    # Add buttons
    button_frame = ttk.Frame(settings_dialog)
    button_frame.pack(fill=tk.X, pady=10)
    
    ttk.Button(
        button_frame, 
        text="Cancel", 
        command=settings_dialog.destroy
    ).pack(side=tk.RIGHT, padx=5)
    
    # Save settings but don't run scan
    ttk.Button(
        button_frame, 
        text="Save Settings", 
        command=lambda: save_settings(
            detector,
            threshold_var.get(),
            settings_dialog
        )
    ).pack(side=tk.RIGHT, padx=5)
    
    # Save and run scan
    ttk.Button(
        button_frame, 
        text="Save & Scan Now", 
        command=lambda: save_and_scan(
            parent,
            detector,
            threshold_var.get(),
            method_var.get(),
            min_size_var.get(),
            settings_dialog
        )
    ).pack(side=tk.RIGHT, padx=5)


def save_settings(detector, threshold, dialog):
    """
    Save detector settings without running scan.
    
    Args:
        detector: OutlierGroupDetector instance
        threshold: Outlier threshold value
        dialog: Settings dialog to close
    """
    # Update detector settings
    detector.threshold = threshold
    
    # Close settings dialog
    dialog.destroy()


def save_and_scan(parent, detector, threshold, method, min_size, dialog):
    """
    Save settings and run outlier scan.
    
    Args:
        parent: The main window
        detector: OutlierGroupDetector instance
        threshold: Outlier threshold value
        method: Comparison method
        min_size: Minimum group size
        dialog: Settings dialog to close
    """
    # Update detector settings
    detector.threshold = threshold
    
    # Close settings dialog
    dialog.destroy()
    
    # Run scan
    run_outlier_scan(parent, detector)


def integrate_outlier_detection(parent):
    """
    Integrate outlier detection functionality into the application.
    
    Args:
        parent: The main window
    """
    # Add detection button to gallery tab
    add_outlier_detection_button(parent)
    
    # Add menu options
    add_outlier_detection_menu(parent)
    
    # Create detector instance to be used by both the button and menu
    detector = OutlierGroupDetector(parent.gallery_tab)
    
    # Store detector instance on the parent for later access
    parent.outlier_detector = detector
    
    # Add help information about outlier detection
    add_outlier_detection_help(parent)


def add_outlier_detection_help(parent):
    """
    Add help information about outlier detection.
    
    Args:
        parent: The main window
    """
    # Check if Help menu exists
    help_menu = None
    menu_bar = parent.menu_bar
    
    for i in range(menu_bar.index('end') + 1):
        if menu_bar.type(i) == 'cascade' and menu_bar.entrycget(i, 'label') == 'Help':
            help_menu = menu_bar.nametowidget(menu_bar.entrycget(i, 'menu'))
            break
    
    if help_menu:
        # Add help command for outlier detection
        help_menu.add_command(
            label="Outlier Detection Help",
            command=lambda: show_outlier_help(parent)
        )


def show_outlier_help(parent):
    """
    Show help information about outlier detection.
    
    Args:
        parent: The main window
    """
    import tkinter as tk
    from tkinter import ttk
    
    # Create help dialog
    help_dialog = tk.Toplevel(parent.root)
    help_dialog.title("Outlier Detection Help")
    help_dialog.geometry("600x500")
    help_dialog.transient(parent.root)
    
    # Make dialog modal
    help_dialog.grab_set()
    
    # Add scrollable text widget
    help_frame = ttk.Frame(help_dialog, padding=10)
    help_frame.pack(fill=tk.BOTH, expand=True)
    
    # Create text widget with scrollbar
    text_frame = ttk.Frame(help_frame)
    text_frame.pack(fill=tk.BOTH, expand=True)
    
    text = tk.Text(text_frame, wrap=tk.WORD, padx=10, pady=10)
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    scrollbar = ttk.Scrollbar(text_frame, command=text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text.configure(yscrollcommand=scrollbar.set)
    
    # Insert help text
    help_text = """Outlier Detection Tool

The Outlier Detection tool analyzes your image groups to find images that significantly differ from others in the same group. This helps you quickly identify problematic or inconsistent images across your dataset.

Using Outlier Detection:

1. Click the "Find Outlier Groups" button in the Gallery tab, or select
   Tools > Find Outlier Groups... from the menu

2. Wait while the tool scans all image groups
   - Each group is analyzed to identify statistical outliers
   - Groups containing significant outliers are collected for review

3. In the Outlier Review dialog:
   - Navigate through detected outlier groups with Previous/Next buttons
   - View outlier details in the Outliers tab
   - See a visual analysis of the entire group in the Group Overview tab
   - Use the View Group in Gallery button to see the group in context
   - Use the Compare buttons to visualize differences

Outlier Detection Methods:

1. histogram_correlation: Analyzes color distribution similarity
   - Good for detecting color/tone differences
   - Fast and effective for most cases

2. histogram_chi, histogram_intersection, histogram_bhattacharyya:
   - Alternative histogram-based methods
   - Each has different sensitivity to various types of differences

3. ssim: Structural Similarity Index
   - Considers image structure, not just colors
   - Better matches human perception of similarity
   - Good for detecting blur, compression artifacts, etc.

4. mse: Mean Squared Error
   - Direct pixel-by-pixel comparison
   - Very sensitive to small changes and alignment

Adjusting Detection Settings:

1. Access settings via Tools > Outlier Detection Settings...

2. Outlier Threshold: Controls detection sensitivity
   - Higher values (>1.0): Only detect extreme outliers
   - Medium values (0.4-1.0): Balanced detection
   - Lower values (<0.4): Detect subtle differences

3. Minimum Group Size: Ignore groups with fewer images than this value

Tips:

- Start with a medium threshold value (0.4-0.6) and adjust based on results
- Different comparison methods are sensitive to different types of differences
- Review each outlier carefully - not all outliers are problematic
- Use the comparison tools to understand exactly how outliers differ

The Outlier Detection tool is valuable for:
- Quality control of processed images
- Finding corrupted or problematic images
- Ensuring consistency across your dataset
- Identifying artifacts from processing steps
"""

    text.insert(tk.END, help_text)
    text.config(state=tk.DISABLED)  # Make read-only
    
    # Add close button
    close_btn = ttk.Button(help_frame, text="Close", command=help_dialog.destroy)
    close_btn.pack(pady=10)