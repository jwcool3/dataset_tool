"""
Image Comparison Dialog for Dataset Preparation Tool
Provides UI for comparing images and identifying outliers.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from utils.image_comparison import ImageComparison

class ComparisonDialog:
    """Dialog for comparing images and finding outliers in a group."""
    
    def __init__(self, parent, image_paths):
        """
        Initialize the comparison dialog.
        
        Args:
            parent: Parent window
            image_paths: List of paths to images to compare
        """
        self.parent = parent
        self.image_paths = image_paths
        self.comparator = ImageComparison()
        
        # Selected images for detailed comparison
        self.selected_image1 = None
        self.selected_image2 = None
        
        # Image references to prevent garbage collection
        self.visualization_img = None
        self.outlier_images = []
        self.comparison_img = None
        
        # Dialog setup
        self.dialog = tk.Toplevel(parent.root)
        self.dialog.title("Image Comparison")
        self.dialog.geometry("1000x800")
        self.dialog.minsize(800, 600)
        self.dialog.transient(parent.root)  # Make dialog modal
        
        # Make the window pop to the front
        self.dialog.lift()
        self.dialog.attributes('-topmost', True)
        self.dialog.after_idle(self.dialog.attributes, '-topmost', False)
        
        # Create content
        self._create_content()
        
        # Wait for the dialog to close
        self.dialog.focus_set()
        self.dialog.grab_set()
    
    def _create_content(self):
        """Create the dialog content."""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create control panel
        control_frame = ttk.LabelFrame(main_frame, text="Comparison Settings", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        # Grid for controls
        controls_grid = ttk.Frame(control_frame)
        controls_grid.pack(fill=tk.X, pady=5)
        
        # Comparison method selection
        ttk.Label(controls_grid, text="Method:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        self.method_var = tk.StringVar(value="histogram_correlation")
        method_combo = ttk.Combobox(controls_grid, textvariable=self.method_var, width=25)
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
        
        # Color space selection
        ttk.Label(controls_grid, text="Color Space:").grid(column=2, row=0, sticky=tk.W, padx=5, pady=5)
        self.color_space_var = tk.StringVar(value="rgb")
        color_space_combo = ttk.Combobox(controls_grid, textvariable=self.color_space_var, width=10)
        color_space_combo['values'] = ["rgb", "hsv", "lab"]
        color_space_combo.grid(column=3, row=0, sticky=tk.W, padx=5, pady=5)
        color_space_combo.state(['readonly'])
        
        # Resize option
        self.resize_var = tk.BooleanVar(value=True)
        resize_check = ttk.Checkbutton(controls_grid, text="Resize images before comparison", 
                                      variable=self.resize_var)
        resize_check.grid(column=0, row=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Resize dimensions
        ttk.Label(controls_grid, text="Resize dimensions:").grid(column=2, row=1, sticky=tk.W, padx=5, pady=5)
        size_frame = ttk.Frame(controls_grid)
        size_frame.grid(column=3, row=1, sticky=tk.W, padx=5, pady=5)
        
        self.width_var = tk.IntVar(value=256)
        width_spinbox = ttk.Spinbox(size_frame, from_=32, to=1024, increment=32, 
                                  textvariable=self.width_var, width=5)
        width_spinbox.pack(side=tk.LEFT)
        
        ttk.Label(size_frame, text="x").pack(side=tk.LEFT, padx=2)
        
        self.height_var = tk.IntVar(value=256)
        height_spinbox = ttk.Spinbox(size_frame, from_=32, to=1024, increment=32, 
                                   textvariable=self.height_var, width=5)
        height_spinbox.pack(side=tk.LEFT)
        
        # Compare button
        self.compare_button = ttk.Button(controls_grid, text="Compare Images", command=self._run_comparison)
        self.compare_button.grid(column=4, row=0, rowspan=2, sticky=tk.E, padx=20, pady=5)
        
        # Create a progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready to compare {} images".format(len(self.image_paths)))
        self.status_label.pack(anchor=tk.W, pady=5)
        
        # Create notebook for results
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Overview tab
        self.overview_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.overview_frame, text="Overview")
        
        # Create a canvas for visualization
        self.viz_frame = ttk.LabelFrame(self.overview_frame, text="Similarity Analysis")
        self.viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.viz_label = ttk.Label(self.viz_frame)
        self.viz_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Outliers tab
        self.outliers_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.outliers_frame, text="Outliers")
        
        # Create vertical layout for outliers
        self.outlier_header = ttk.Label(self.outliers_frame, text="Images ranked by difference (most different first)", 
                                      font=("Helvetica", 12, "bold"))
        self.outlier_header.pack(fill=tk.X, pady=10)
        
        # Create scrollable frame for outliers
        outlier_scroll_frame = ttk.Frame(self.outliers_frame)
        outlier_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        outlier_canvas = tk.Canvas(outlier_scroll_frame)
        outlier_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        outlier_scrollbar = ttk.Scrollbar(outlier_scroll_frame, orient=tk.VERTICAL, command=outlier_canvas.yview)
        outlier_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        outlier_canvas.configure(yscrollcommand=outlier_scrollbar.set)
        
        # Frame inside canvas for outlier content
        self.outlier_content = ttk.Frame(outlier_canvas)
        outlier_window = outlier_canvas.create_window((0, 0), window=self.outlier_content, anchor=tk.NW)
        
        # Configure scrolling
        self.outlier_content.bind("<Configure>", 
                                lambda e: outlier_canvas.configure(scrollregion=outlier_canvas.bbox("all")))
        outlier_canvas.bind("<Configure>", 
                          lambda e: outlier_canvas.itemconfig(outlier_window, width=e.width))
        
        # Comparison tab
        self.comparison_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.comparison_frame, text="Detailed Comparison")
        
        # Create header for comparison
        comparison_header = ttk.Label(self.comparison_frame, 
                                    text="Select two images to compare in detail", 
                                    font=("Helvetica", 12, "bold"))
        comparison_header.pack(fill=tk.X, pady=10)
        
        # Create selection controls
        selection_frame = ttk.Frame(self.comparison_frame)
        selection_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(selection_frame, text="Image 1:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        self.image1_var = tk.StringVar()
        self.image1_combo = ttk.Combobox(selection_frame, textvariable=self.image1_var, width=40)
        self.image1_combo.grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(selection_frame, text="Image 2:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        self.image2_var = tk.StringVar()
        self.image2_combo = ttk.Combobox(selection_frame, textvariable=self.image2_var, width=40)
        self.image2_combo.grid(column=1, row=1, sticky=tk.W, padx=5, pady=5)
        
        compare_btn = ttk.Button(selection_frame, text="Compare Selected", command=self._compare_selected)
        compare_btn.grid(column=2, row=0, rowspan=2, sticky=tk.E, padx=5, pady=5)
        
        # Frame for the comparison visualization
        self.detail_frame = ttk.LabelFrame(self.comparison_frame, text="Comparison Details")
        self.detail_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.detail_label = ttk.Label(self.detail_frame)
        self.detail_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Fill image combo boxes with paths
        self._fill_image_combos()
        
        # Close button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        close_button = ttk.Button(button_frame, text="Close", command=self.dialog.destroy)
        close_button.pack(side=tk.RIGHT, padx=5)
    
    def _fill_image_combos(self):
        """Fill the image selection combo boxes with file paths."""
        # Extract filenames for display
        filenames = [os.path.basename(path) for path in self.image_paths]
        
        # Configure comboboxes
        self.image1_combo['values'] = filenames
        self.image2_combo['values'] = filenames
        
        # Set initial values if possible
        if filenames:
            self.image1_var.set(filenames[0])
            if len(filenames) > 1:
                self.image2_var.set(filenames[1])
            else:
                self.image2_var.set(filenames[0])
    
    def _run_comparison(self):
        """Run the image comparison on a separate thread."""
        # Disable the compare button while processing
        self.compare_button.configure(state="disabled")
        self.progress_var.set(0)
        self.status_label.config(text="Starting comparison...")
        
        # Clear previous results
        self.viz_label.config(image="")
        for widget in self.outlier_content.winfo_children():
            widget.destroy()
        
        # Start the comparison in a separate thread
        thread = threading.Thread(target=self._compare_thread)
        thread.daemon = True
        thread.start()
    
    def _compare_thread(self):
        """Run the comparison process in a background thread."""
        try:
            # Update progress
            self._update_progress(10, "Loading images...")
            
            # Get comparison parameters
            method = self.method_var.get()
            color_space = self.color_space_var.get()
            resize = self.resize_var.get()
            resize_dim = (self.width_var.get(), self.height_var.get())
            
            # Run comparison
            self._update_progress(20, "Comparing images...")
            result = self.comparator.compare_images(
                self.image_paths, 
                method=method, 
                resize=resize, 
                resize_dim=resize_dim, 
                color_space=color_space
            )
            
            if not result.get("success", False):
                self._update_progress(0, f"Error: {result.get('message', 'Unknown error')}")
                self.compare_button.configure(state="normal")
                return
            
            # Store results
            self.comparison_result = result
            
            # Load images for visualization
            self._update_progress(50, "Generating visualization...")
            
            # Create visualization
            self._update_progress(70, "Creating visualization...")
            self._create_visualization(result)
            
            # Create outlier display
            self._update_progress(90, "Displaying outliers...")
            self._display_outliers(result)
            
            # Done
            self._update_progress(100, f"Comparison complete. Found {len(result['ranked_paths'])} images with {method}.")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._update_progress(0, f"Error during comparison: {str(e)}")
        
        finally:
            # Re-enable the compare button
            self.compare_button.configure(state="normal")
    
    def _update_progress(self, value, message):
        """Update the progress bar and status message from any thread."""
        self.dialog.after(0, lambda: self._update_ui(value, message))
    
    def _update_ui(self, value, message):
        """Update UI elements (must be called from the main thread)."""
        self.progress_var.set(value)
        self.status_label.config(text=message)
    
    def _create_visualization(self, result):
        """Create and display the similarity visualization."""
        try:
            # Get data from result
            distance_matrix = result.get("distance_matrix")
            outlier_scores = result.get("outlier_scores")
            paths = result.get("paths")
            
            if distance_matrix is None or outlier_scores is None:
                return
            
            # Generate visualization image
            images = []
            for path in paths:
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
            
            # Generate visualization
            viz_img = self.comparator.generate_comparison_visualization(images, distance_matrix, outlier_scores, paths)
            
            # Convert to PhotoImage
            self.visualization_img = ImageTk.PhotoImage(viz_img)
            
            # Display on the label
            self.viz_label.config(image=self.visualization_img)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_label.config(text=f"Error creating visualization: {str(e)}")
    
    def _display_outliers(self, result):
        """Display the outlier images ranked by dissimilarity."""
        try:
            # Clear previous outliers
            for widget in self.outlier_content.winfo_children():
                widget.destroy()
            
            # Reset image references
            self.outlier_images = []
            
            # Get ranked paths
            ranked_paths = result.get("ranked_paths", [])
            outlier_scores = result.get("outlier_scores", [])
            sorted_indices = result.get("sorted_indices", [])
            
            if not ranked_paths:
                ttk.Label(self.outlier_content, text="No outliers found.").pack(pady=20)
                return
            
            # Create a frame for each outlier
            for i, path in enumerate(ranked_paths):
                if i >= len(sorted_indices):
                    continue
                    
                score_idx = sorted_indices[i]
                if score_idx >= len(outlier_scores):
                    continue
                    
                score = outlier_scores[score_idx]
                
                # Create frame for this outlier
                outlier_frame = ttk.Frame(self.outlier_content, padding=5)
                outlier_frame.pack(fill=tk.X, pady=10, padx=10)
                
                # Add a separator above (except for the first one)
                if i > 0:
                    ttk.Separator(self.outlier_content, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5, padx=10)
                
                # Load the image for display
                try:
                    img = cv2.imread(path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize for display
                    h, w = img.shape[:2]
                    scale = min(300 / w, 300 / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    display_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    # Convert to PhotoImage
                    pil_img = Image.fromarray(display_img)
                    tk_img = ImageTk.PhotoImage(pil_img)
                    self.outlier_images.append(tk_img)  # Keep reference
                    
                    # Display image and info side by side
                    img_label = ttk.Label(outlier_frame, image=tk_img)
                    img_label.grid(column=0, row=0, rowspan=3, padx=10, pady=5)
                    
                    # Add info labels
                    rank_label = ttk.Label(outlier_frame, text=f"Rank #{i+1} (Outlier Score: {score:.4f})", 
                                        font=("Helvetica", 10, "bold"))
                    rank_label.grid(column=1, row=0, sticky=tk.W, padx=10)
                    
                    path_label = ttk.Label(outlier_frame, text=f"File: {os.path.basename(path)}")
                    path_label.grid(column=1, row=1, sticky=tk.W, padx=10)
                    
                    size_label = ttk.Label(outlier_frame, text=f"Dimensions: {w}x{h}")
                    size_label.grid(column=1, row=2, sticky=tk.W, padx=10)
                    
                except Exception as e:
                    ttk.Label(outlier_frame, text=f"Error loading image: {str(e)}").pack()
            
            # Switch to the outliers tab
            self.notebook.select(1)  # Index 1 is the outliers tab
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_label.config(text=f"Error displaying outliers: {str(e)}")
    
    def _compare_selected(self):
            """Compare the two selected images in detail."""
            try:
                # Get the selected image indices
                image1_name = self.image1_var.get()
                image2_name = self.image2_var.get()
                
                # Find the corresponding file paths
                image1_idx = -1
                image2_idx = -1
                
                for i, path in enumerate(self.image_paths):
                    if os.path.basename(path) == image1_name:
                        image1_idx = i
                    if os.path.basename(path) == image2_name:
                        image2_idx = i
                
                if image1_idx == -1 or image2_idx == -1:
                    self.status_label.config(text="Error: Selected images not found.")
                    return
                
                image1_path = self.image_paths[image1_idx]
                image2_path = self.image_paths[image2_idx]
                
                # Load the images
                img1 = cv2.imread(image1_path)
                img2 = cv2.imread(image2_path)
                
                if img1 is None or img2 is None:
                    self.status_label.config(text="Error loading selected images.")
                    return
                
                # Convert to RGB
                img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                
                # Generate comparison visualization
                comparison = self.comparator.visualize_differences(img1_rgb, img2_rgb)
                
                # Calculate thumbnail size to fit in the window
                h, w = comparison.shape[:2]
                detail_width = self.detail_frame.winfo_width() - 40  # Subtract padding
                detail_height = self.detail_frame.winfo_height() - 40
                
                # Ensure we have reasonable minimum dimensions
                if detail_width < 100:
                    detail_width = 800
                if detail_height < 100:
                    detail_height = 400
                    
                # Calculate scaling
                scale = min(detail_width / w, detail_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                # Resize for display
                if scale < 1.0:
                    display_img = cv2.resize(comparison, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    display_img = comparison
                
                # Convert to PhotoImage
                pil_img = Image.fromarray(display_img)
                self.comparison_img = ImageTk.PhotoImage(pil_img)
                
                # Display on the label
                self.detail_label.config(image=self.comparison_img)
                
                # Switch to the comparison tab
                self.notebook.select(2)  # Index 2 is the detailed comparison tab
                
                # Update status
                self.status_label.config(text=f"Comparing {os.path.basename(image1_path)} and {os.path.basename(image2_path)}")
                
                # Calculate a similarity score between the two images
                if hasattr(self, 'comparison_result') and self.comparison_result:
                    distance_matrix = self.comparison_result.get("distance_matrix")
                    paths = self.comparison_result.get("paths")
                    
                    # Find the indices in the distance matrix
                    matrix_idx1 = -1
                    matrix_idx2 = -1
                    
                    for i, path in enumerate(paths):
                        if path == image1_path:
                            matrix_idx1 = i
                        if path == image2_path:
                            matrix_idx2 = i
                    
                    if matrix_idx1 >= 0 and matrix_idx2 >= 0 and matrix_idx1 < len(distance_matrix) and matrix_idx2 < len(distance_matrix[0]):
                        # Get the similarity score
                        similarity_score = 1.0 - distance_matrix[matrix_idx1][matrix_idx2]
                        
                        # Update the status with similarity information
                        self.status_label.config(
                            text=f"Comparing {os.path.basename(image1_path)} and {os.path.basename(image2_path)} - " +
                                f"Similarity: {similarity_score:.4f}"
                        )
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.status_label.config(text=f"Error in detailed comparison: {str(e)}")