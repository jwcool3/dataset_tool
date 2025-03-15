"""
Gallery Integration Component for Dataset Preparation Tool
Adds comparison functionality to the Gallery tab.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import threading
from ui.comparison_dialog import ComparisonDialog

class GalleryComparisonIntegrator:
    """Integrates image comparison functionality with the Gallery tab."""
    
    def __init__(self, gallery_tab):
        """
        Initialize the gallery comparison integrator.
        
        Args:
            gallery_tab: The gallery tab to integrate with
        """
        self.gallery_tab = gallery_tab
        self.parent = gallery_tab.parent
        
        # Add compare button to gallery controls
        self._add_comparison_controls()
    
    def _add_comparison_controls(self):
        """Add comparison controls to the gallery UI."""
        # Check if the gallery has already been initialized
        if not hasattr(self.gallery_tab, 'refresh_button'):
            # Wait for the gallery tab to be initialized
            self.gallery_tab.frame.after(500, self._add_comparison_controls)
            return
            
        # Get the control frame
        control_frame = None
        for child in self.gallery_tab.frame.winfo_children():
            if isinstance(child, ttk.LabelFrame) and "Gallery Controls" in child.cget("text"):
                control_frame = child
                break
        
        if not control_frame:
            return
            
        # Find the buttons frame or create one
        buttons_frame = None
        for child in control_frame.winfo_children():
            if isinstance(child, ttk.Frame):
                # Look for the delete button as a clue
                for widget in child.winfo_children():
                    if isinstance(widget, ttk.Button) and "Delete" in widget.cget("text"):
                        buttons_frame = child
                        break
        
        if not buttons_frame:
            # Create a new frame for buttons
            grid_frame = None
            for child in control_frame.winfo_children():
                if isinstance(child, ttk.Frame):
                    grid_frame = child
                    break
                    
            if grid_frame:
                buttons_frame = ttk.Frame(grid_frame)
                buttons_frame.grid(column=2, row=0, padx=5, pady=5, sticky=tk.E)
        
        # Add the compare button if it doesn't exist
        compare_button_exists = False
        for widget in buttons_frame.winfo_children():
            if isinstance(widget, ttk.Button) and "Compare" in widget.cget("text"):
                compare_button_exists = True
                break
                
        if not compare_button_exists and buttons_frame:
            self.compare_button = ttk.Button(
                buttons_frame, 
                text="Compare Images", 
                command=self._compare_current_group
            )
            self.compare_button.pack(side=tk.LEFT, padx=5)
        
        # Add advanced comparison submenu
        self._add_advanced_comparison_menu()
    
    def _add_advanced_comparison_menu(self):
        """Add advanced comparison options to the menu."""
        # Get the menu bar
        menu_bar = self.parent.menu_bar
        
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
        
        # Add comparison submenu
        comparison_menu = tk.Menu(tools_menu, tearoff=0)
        tools_menu.add_cascade(label="Image Comparison", menu=comparison_menu)
        
        # Add comparison options
        comparison_menu.add_command(
            label="Compare Current Image Group", 
            command=self._compare_current_group
        )
        comparison_menu.add_command(
            label="Find Outliers in Current Group", 
            command=self._find_outliers
        )
        comparison_menu.add_command(
            label="Batch Compare All Groups", 
            command=self._batch_compare
        )
        comparison_menu.add_separator()
        comparison_menu.add_command(
            label="Comparison Settings", 
            command=self._show_comparison_settings
        )
    
    def _compare_current_group(self):
        """Compare images in the current group using the comparison dialog."""
        # Get current image group from gallery tab
        if not hasattr(self.gallery_tab, 'images_data') or not self.gallery_tab.images_data:
            messagebox.showinfo("No Images", "Please load images first.")
            return
        
        # Get current image group
        image_data = self.gallery_tab.images_data[self.gallery_tab.current_image_index]
        versions = image_data['versions']
        
        # Filter to only include source images (not masks)
        source_versions = [v for v in versions if not v['is_mask']]
        
        if len(source_versions) < 2:
            messagebox.showinfo(
                "Not Enough Images", 
                "Need at least 2 images in this group for comparison."
            )
            return
        
        # Get paths of all source images in this group
        image_paths = [v['path'] for v in source_versions]
        
        # Launch comparison dialog
        ComparisonDialog(self.parent, image_paths)
    
    def _find_outliers(self):
        """Find outliers in all image groups."""
        if not hasattr(self.gallery_tab, 'images_data') or not self.gallery_tab.images_data:
            messagebox.showinfo("No Images", "Please load images first.")
            return
        
        # Create a dialog to configure outlier detection
        settings_dialog = tk.Toplevel(self.parent.root)
        settings_dialog.title("Outlier Detection Settings")
        settings_dialog.geometry("400x300")
        settings_dialog.transient(self.parent.root)
        settings_dialog.grab_set()
        
        # Add settings
        settings_frame = ttk.Frame(settings_dialog, padding=20)
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # Method
        ttk.Label(settings_frame, text="Comparison Method:").grid(
            column=0, row=0, sticky=tk.W, padx=5, pady=5
        )
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
        method_combo.grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)
        method_combo.state(['readonly'])
        
        # Threshold
        ttk.Label(settings_frame, text="Outlier Threshold:").grid(
            column=0, row=1, sticky=tk.W, padx=5, pady=5
        )
        threshold_var = tk.DoubleVar(value=0.3)
        threshold_scale = ttk.Scale(
            settings_frame, 
            from_=0.1, 
            to=0.5, 
            orient=tk.HORIZONTAL, 
            variable=threshold_var, 
            length=200
        )
        threshold_scale.grid(column=1, row=1, sticky=tk.W, padx=5, pady=5)
        
        threshold_label = ttk.Label(settings_frame, text="0.30")
        threshold_label.grid(column=2, row=1, sticky=tk.W, padx=5, pady=5)
        
        # Update threshold label when scale changes
        def update_threshold_label(*args):
            threshold_label.config(text=f"{threshold_var.get():.2f}")
        
        threshold_var.trace_add("write", update_threshold_label)
        
        # Scope
        ttk.Label(settings_frame, text="Scope:").grid(
            column=0, row=2, sticky=tk.W, padx=5, pady=5
        )
        scope_var = tk.StringVar(value="current")
        
        ttk.Radiobutton(
            settings_frame, 
            text="Current Image Group Only",
            variable=scope_var, 
            value="current"
        ).grid(column=1, row=2, sticky=tk.W, padx=5, pady=2)
        
        ttk.Radiobutton(
            settings_frame, 
            text="All Image Groups",
            variable=scope_var, 
            value="all"
        ).grid(column=1, row=3, sticky=tk.W, padx=5, pady=2)
        
        # Add explanation text
        explanation = ttk.Label(
            settings_frame, 
            text="Higher threshold values will be more selective, finding only the most different images. "
                 "Lower values will find more potential outliers.",
            wraplength=350
        )
        explanation.grid(column=0, row=4, columnspan=3, sticky=tk.W, padx=5, pady=10)
        
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
            text="Find Outliers", 
            command=lambda: self._run_outlier_detection(
                method_var.get(), 
                threshold_var.get(), 
                scope_var.get(), 
                settings_dialog
            )
        ).pack(side=tk.RIGHT, padx=5)
    
    def _run_outlier_detection(self, method, threshold, scope, dialog):
        """
        Run outlier detection with the given parameters.
        
        Args:
            method: Comparison method to use
            threshold: Outlier threshold
            scope: Scope of analysis ('current' or 'all')
            dialog: Settings dialog to close when done
        """
        dialog.destroy()
        
        # Import ImageComparison
        from utils.image_comparison import ImageComparison
        comparator = ImageComparison()
        
        # Get image paths to analyze
        if scope == 'current':
            # Get current image group
            image_data = self.gallery_tab.images_data[self.gallery_tab.current_image_index]
            versions = image_data['versions']
            source_versions = [v for v in versions if not v['is_mask']]
            image_paths = [v['path'] for v in source_versions]
        else:
            # Get all image groups
            image_paths = []
            for image_data in self.gallery_tab.images_data:
                versions = image_data['versions']
                source_versions = [v for v in versions if not v['is_mask']]
                image_paths.extend([v['path'] for v in source_versions])
        
        if len(image_paths) < 3:  # Need at least 3 images for meaningful outlier detection
            messagebox.showinfo(
                "Not Enough Images", 
                "Need at least 3 images for outlier detection."
            )
            return
        
        # Show progress dialog
        progress_dialog = tk.Toplevel(self.parent.root)
        progress_dialog.title("Finding Outliers")
        progress_dialog.geometry("400x150")
        progress_dialog.transient(self.parent.root)
        progress_dialog.grab_set()
        
        # Add progress bar
        ttk.Label(
            progress_dialog, 
            text="Analyzing images to find outliers...",
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
        def run_analysis():
            try:
                # Update progress
                progress_dialog.after(0, lambda: status_label.config(
                    text=f"Comparing {len(image_paths)} images using {method}..."
                ))
                progress_dialog.after(0, lambda: progress_var.set(10))
                
                # Run outlier detection
                result = comparator.find_outliers(image_paths, threshold, method)
                
                # Update progress
                progress_dialog.after(0, lambda: progress_var.set(90))
                
                # Process results
                if result.get("success", False):
                    outlier_paths = result.get("outlier_paths", [])
                    outlier_scores = result.get("outlier_scores", [])
                    
                    if outlier_paths:
                        # Close progress dialog
                        progress_dialog.after(0, progress_dialog.destroy)
                        
                        # Show results dialog
                        self._show_outlier_results(outlier_paths, outlier_scores, method, threshold)
                    else:
                        # No outliers found
                        progress_dialog.after(0, progress_dialog.destroy)
                        progress_dialog.after(10, lambda: messagebox.showinfo(
                            "No Outliers Found",
                            f"No outliers were found with threshold {threshold:.2f}.\n\n" +
                            "Try lowering the threshold to find more potential outliers."
                        ))
                else:
                    # Error occurred
                    error_msg = result.get("message", "Unknown error")
                    progress_dialog.after(0, progress_dialog.destroy)
                    progress_dialog.after(10, lambda: messagebox.showerror(
                        "Error",
                        f"Error finding outliers: {error_msg}"
                    ))
            
            except Exception as e:
                import traceback
                traceback.print_exc()
                progress_dialog.after(0, progress_dialog.destroy)
                progress_dialog.after(10, lambda: messagebox.showerror(
                    "Error",
                    f"Error finding outliers: {str(e)}"
                ))
        
        # Start analysis thread
        threading.Thread(target=run_analysis, daemon=True).start()
    
    def _show_outlier_results(self, outlier_paths, outlier_scores, method, threshold):
        """
        Show results of outlier detection.
        
        Args:
            outlier_paths: List of paths to outlier images
            outlier_scores: List of outlier scores
            method: Comparison method used
            threshold: Outlier threshold used
        """
        # Create results dialog
        results_dialog = tk.Toplevel(self.parent.root)
        results_dialog.title("Outlier Detection Results")
        results_dialog.geometry("800x600")
        results_dialog.minsize(600, 400)
        
        # Make dialog modal
        results_dialog.transient(self.parent.root)
        results_dialog.grab_set()
        
        # Main frame
        main_frame = ttk.Frame(results_dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            header_frame, 
            text=f"Found {len(outlier_paths)} Potential Outliers",
            font=("Helvetica", 12, "bold")
        ).pack(side=tk.LEFT)
        
        # Add info about method and threshold
        info_text = f"Method: {method} | Threshold: {threshold:.2f}"
        ttk.Label(header_frame, text=info_text).pack(side=tk.RIGHT)
        
        # Add explanation
        explanation = ttk.Label(
            main_frame, 
            text="The following images were identified as potential outliers based on their "
                 "difference from other images in the set. Higher scores indicate more difference.",
            wraplength=780
        )
        explanation.pack(fill=tk.X, pady=(0, 10))
        
        # Create scrollable frame for results
        scroll_frame = ttk.Frame(main_frame)
        scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(scroll_frame)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(scroll_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Frame inside canvas for content
        content_frame = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=content_frame, anchor=tk.NW)
        
        # Configure scrolling
        content_frame.bind("<Configure>", 
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", 
                   lambda e: canvas.itemconfig(canvas_window, width=e.width))
        
        # Function to bind mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Store references to thumbnail images
        thumbnails = []
        
        # Add each outlier to the results
        import cv2
        from PIL import Image, ImageTk
        
        # Create a separator after each item (not before the first one)
        for i, (path, score) in enumerate(zip(outlier_paths, outlier_scores)):
            # Create frame for this outlier
            item_frame = ttk.Frame(content_frame, padding=10)
            item_frame.pack(fill=tk.X, pady=5)
            
            # Add separator before all items except the first
            if i > 0:
                ttk.Separator(content_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)
            
            try:
                # Load the image
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize for display
                    h, w = img.shape[:2]
                    scale = min(200 / w, 200 / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    display_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    # Convert to PhotoImage
                    pil_img = Image.fromarray(display_img)
                    tk_img = ImageTk.PhotoImage(pil_img)
                    thumbnails.append(tk_img)  # Keep reference
                    
                    # Create grid layout
                    img_label = ttk.Label(item_frame, image=tk_img)
                    img_label.grid(row=0, column=0, rowspan=3, padx=(0, 10))
                    
                    # Add info
                    filename = os.path.basename(path)
                    ttk.Label(
                        item_frame, 
                        text=f"File: {filename}",
                        font=("Helvetica", 10, "bold")
                    ).grid(row=0, column=1, sticky=tk.W)
                    
                    ttk.Label(
                        item_frame, 
                        text=f"Outlier Score: {score:.4f}"
                    ).grid(row=1, column=1, sticky=tk.W)
                    
                    ttk.Label(
                        item_frame, 
                        text=f"Dimensions: {w}x{h}"
                    ).grid(row=2, column=1, sticky=tk.W)
                    
                    # Add view button
                    view_btn = ttk.Button(
                        item_frame, 
                        text="View in Gallery", 
                        command=lambda p=path: self._view_in_gallery(p)
                    )
                    view_btn.grid(row=0, column=2, rowspan=3, padx=10)
                    
                    # Add compare button
                    compare_btn = ttk.Button(
                        item_frame, 
                        text="Compare...", 
                        command=lambda p=path: self._compare_with_outlier(p)
                    )
                    compare_btn.grid(row=0, column=3, rowspan=3, padx=10)
                    
                else:
                    ttk.Label(
                        item_frame, 
                        text=f"Error loading image: {os.path.basename(path)}"
                    ).pack()
                
            except Exception as e:
                ttk.Label(
                    item_frame, 
                    text=f"Error displaying {os.path.basename(path)}: {str(e)}"
                ).pack()
        
        # Add buttons at the bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        export_btn = ttk.Button(
            button_frame, 
            text="Export Results", 
            command=lambda: self._export_outlier_results(outlier_paths, outlier_scores, method, threshold)
        )
        export_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Close", 
            command=results_dialog.destroy
        ).pack(side=tk.RIGHT, padx=5)
    
    def _view_in_gallery(self, path):
        """
        Find the image in the gallery and navigate to it.
        
        Args:
            path: Path to the image to view
        """
        # Find the image in the gallery data
        for i, image_group in enumerate(self.gallery_tab.images_data):
            for version in image_group['versions']:
                if version['path'] == path:
                    # Switch to this image
                    self.gallery_tab.current_image_index = i
                    self.gallery_tab._update_counter()
                    self.gallery_tab.view_mode.set("large")  # Use large preview mode
                    self.gallery_tab._show_large_preview()
                    return
        
        # Image not found in gallery
        messagebox.showinfo(
            "Image Not Found",
            "This image could not be found in the current gallery view."
        )
    
    def _compare_with_outlier(self, path):
        """
        Launch a comparison dialog to compare this outlier with other images.
        
        Args:
            path: Path to the outlier image
        """
        # Get all image paths from the current scope
        all_images = []
        for image_data in self.gallery_tab.images_data:
            versions = image_data['versions']
            source_versions = [v for v in versions if not v['is_mask']]
            all_images.extend([v['path'] for v in source_versions])
        
        # Ensure the outlier is first in the list
        if path in all_images:
            all_images.remove(path)
        
        all_images.insert(0, path)
        
        # Launch comparison dialog
        ComparisonDialog(self.parent, all_images)
    
    def _export_outlier_results(self, outlier_paths, outlier_scores, method, threshold):
        """
        Export outlier detection results to a CSV file.
        
        Args:
            outlier_paths: List of paths to outlier images
            outlier_scores: List of outlier scores
            method: Comparison method used
            threshold: Outlier threshold used
        """
        # Ask for file location
        import tkinter.filedialog as filedialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")],
            title="Export Outlier Results"
        )
        
        if not file_path:
            return
        
        try:
            # Create CSV file
            import csv
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow([
                    "Rank", "Path", "Filename", "Outlier Score", 
                    "Comparison Method", "Threshold"
                ])
                
                # Write outlier data
                for i, (path, score) in enumerate(zip(outlier_paths, outlier_scores)):
                    writer.writerow([
                        i+1, path, os.path.basename(path), score, method, threshold
                    ])
            
            messagebox.showinfo(
                "Export Complete",
                f"Outlier detection results exported to {file_path}"
            )
            
        except Exception as e:
            messagebox.showerror(
                "Export Error",
                f"Error exporting results: {str(e)}"
            )
    
    def _batch_compare(self):
        """Run batch comparison on all image groups."""
        if not hasattr(self.gallery_tab, 'images_data') or not self.gallery_tab.images_data:
            messagebox.showinfo("No Images", "Please load images first.")
            return
        
        # Create settings dialog
        settings_dialog = tk.Toplevel(self.parent.root)
        settings_dialog.title("Batch Comparison Settings")
        settings_dialog.geometry("400x250")
        settings_dialog.transient(self.parent.root)
        settings_dialog.grab_set()
        
        # Add settings
        settings_frame = ttk.Frame(settings_dialog, padding=20)
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # Method
        ttk.Label(settings_frame, text="Comparison Method:").grid(
            column=0, row=0, sticky=tk.W, padx=5, pady=5
        )
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
        method_combo.grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)
        method_combo.state(['readonly'])
        
        # Mode
        ttk.Label(settings_frame, text="Analysis Mode:").grid(
            column=0, row=1, sticky=tk.W, padx=5, pady=5
        )
        mode_var = tk.StringVar(value="outliers")
        
        ttk.Radiobutton(
            settings_frame, 
            text="Find Outliers in Each Group",
            variable=mode_var, 
            value="outliers"
        ).grid(column=1, row=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Radiobutton(
            settings_frame, 
            text="Generate Similarity Report",
            variable=mode_var, 
            value="report"
        ).grid(column=1, row=2, sticky=tk.W, padx=5, pady=2)
        
        # Add explanation text
        explanation = ttk.Label(
            settings_frame, 
            text="Batch comparison will analyze all image groups in the gallery. "
                 "The results will be saved to a file of your choice.",
            wraplength=350
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
            text="Run Batch Analysis", 
            command=lambda: self._run_batch_analysis(
                method_var.get(), 
                mode_var.get(), 
                settings_dialog
            )
        ).pack(side=tk.RIGHT, padx=5)
    
    def _run_batch_analysis(self, method, mode, dialog):
        """
        Run batch analysis on all image groups.
        
        Args:
            method: Comparison method to use
            mode: Analysis mode ('outliers' or 'report')
            dialog: Settings dialog to close when done
        """
        dialog.destroy()
        
        # Collect all image groups
        image_groups = []
        for image_data in self.gallery_tab.images_data:
            group_name = image_data['filename']
            versions = image_data['versions']
            source_versions = [v for v in versions if not v['is_mask']]
            
            if len(source_versions) >= 2:  # Need at least 2 images to compare
                image_paths = [v['path'] for v in source_versions]
                image_groups.append({
                    'name': group_name,
                    'paths': image_paths
                })
        
        if not image_groups:
            messagebox.showinfo(
                "No Valid Groups",
                "No valid image groups found for comparison."
            )
            return
        
        # Ask for output file
        import tkinter.filedialog as filedialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Batch Analysis Results"
        )
        
        if not file_path:
            return
        
        # Show progress dialog
        progress_dialog = tk.Toplevel(self.parent.root)
        progress_dialog.title("Batch Analysis")
        progress_dialog.geometry("400x150")
        progress_dialog.transient(self.parent.root)
        progress_dialog.grab_set()
        
        # Add progress bar
        ttk.Label(
            progress_dialog, 
            text="Analyzing image groups...",
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
        
        # Import ImageComparison
        from utils.image_comparison import ImageComparison
        comparator = ImageComparison()
        
        # Run analysis in a separate thread
        def run_batch_analysis():
            try:
                # Prepare results list
                results = []
                
                # Process each group
                total_groups = len(image_groups)
                for i, group in enumerate(image_groups):
                    # Update progress
                    group_progress = i / total_groups * 100
                    progress_dialog.after(0, lambda: progress_var.set(group_progress))
                    progress_dialog.after(0, lambda: status_label.config(
                        text=f"Processing group {i+1}/{total_groups}: {group['name']}"
                    ))
                    
                    # Run comparison
                    if mode == 'outliers':
                        # Find outliers in this group
                        result = comparator.find_outliers(group['paths'], threshold=0.3, method=method)
                        
                        if result.get("success", False):
                            outlier_paths = result.get("outlier_paths", [])
                            outlier_scores = result.get("outlier_scores", [])
                            
                            # Store results
                            group_result = {
                                'group_name': group['name'],
                                'total_images': len(group['paths']),
                                'outliers': [{
                                    'path': path,
                                    'filename': os.path.basename(path),
                                    'score': score
                                } for path, score in zip(outlier_paths, outlier_scores)]
                            }
                            
                            results.append(group_result)
                    else:
                        # Generate similarity report
                        result = comparator.compare_images(group['paths'], method=method)
                        
                        if result.get("success", False):
                            # Get average similarity for this group
                            distance_matrix = result.get("distance_matrix", [])
                            if len(distance_matrix) > 0:
                                # Calculate average distance (excluding self-comparisons)
                                n = len(distance_matrix)
                                total_distance = 0
                                count = 0
                                
                                for i in range(n):
                                    for j in range(n):
                                        if i != j:  # Exclude self-comparisons
                                            total_distance += distance_matrix[i][j]
                                            count += 1
                                
                                avg_distance = total_distance / count if count > 0 else 0
                                avg_similarity = 1.0 - avg_distance
                                
                                # Store results
                                group_result = {
                                    'group_name': group['name'],
                                    'total_images': len(group['paths']),
                                    'avg_similarity': avg_similarity,
                                    'most_different_image': os.path.basename(result['ranked_paths'][0])
                                        if result['ranked_paths'] else "None"
                                }
                                
                                results.append(group_result)
                
                # Write results to file
                progress_dialog.after(0, lambda: progress_var.set(90))
                progress_dialog.after(0, lambda: status_label.config(text="Writing results to file..."))
                
                import csv
                with open(file_path, 'w', newline='') as csvfile:
                    if mode == 'outliers':
                        # Write outlier results
                        writer = csv.writer(csvfile)
                        writer.writerow([
                            "Group", "Total Images", "Outlier Filename", 
                            "Outlier Score", "Outlier Path"
                        ])
                        
                        for group_result in results:
                            group_name = group_result['group_name']
                            total_images = group_result['total_images']
                            
                            if group_result['outliers']:
                                for outlier in group_result['outliers']:
                                    writer.writerow([
                                        group_name,
                                        total_images,
                                        outlier['filename'],
                                        outlier['score'],
                                        outlier['path']
                                    ])
                            else:
                                writer.writerow([
                                    group_name,
                                    total_images,
                                    "No outliers found",
                                    "",
                                    ""
                                ])
                    else:
                        # Write similarity report
                        writer = csv.writer(csvfile)
                        writer.writerow([
                            "Group", "Total Images", "Average Similarity", 
                            "Most Different Image", "Method"
                        ])
                        
                        for group_result in results:
                            writer.writerow([
                                group_result['group_name'],
                                group_result['total_images'],
                                f"{group_result['avg_similarity']:.4f}",
                                group_result['most_different_image'],
                                method
                            ])
                
                # Done
                progress_dialog.after(0, lambda: progress_var.set(100))
                progress_dialog.after(0, progress_dialog.destroy)
                progress_dialog.after(10, lambda: messagebox.showinfo(
                    "Batch Analysis Complete",
                    f"Analysis completed successfully.\n\nResults saved to: {file_path}"
                ))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                progress_dialog.after(0, progress_dialog.destroy)
                progress_dialog.after(10, lambda: messagebox.showerror(
                    "Error",
                    f"Error during batch analysis: {str(e)}"
                ))
        
        # Start analysis thread
        threading.Thread(target=run_batch_analysis, daemon=True).start()
    
    def _show_comparison_settings(self):
        """Show comparison settings dialog."""
        settings_dialog = tk.Toplevel(self.parent.root)
        settings_dialog.title("Comparison Settings")
        settings_dialog.geometry("450x300")
        settings_dialog.transient(self.parent.root)
        settings_dialog.grab_set()
        
        # Add settings
        settings_frame = ttk.Frame(settings_dialog, padding=20)
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add help text and descriptions of methods
        help_text = (
            "Image Comparison Methods:\n\n"
            "• histogram_correlation: Compares color distribution similarity (higher = more similar)\n"
            "• histogram_chi: Chi-square test for color histograms (sensitive to small differences)\n"
            "• histogram_intersection: Measures overlap between color histograms\n"
            "• histogram_bhattacharyya: Statistical distance between distributions\n"
            "• ssim: Structural Similarity Index (considers image structure)\n"
            "• mse: Mean Squared Error (pixel-by-pixel difference)\n\n"
            "The best method to use depends on the types of differences you're looking for."
        )
        
        text_widget = tk.Text(settings_frame, wrap=tk.WORD, height=12, width=50)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Add close button
        button_frame = ttk.Frame(settings_dialog)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            button_frame, 
            text="Close", 
            command=settings_dialog.destroy
        ).pack(side=tk.RIGHT, padx=5)