"""
Dataset Manager Tab for Dataset Preparation Tool
Main UI tab component for dataset management.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

from utils.dataset_manager.registry import DatasetRegistry
from utils.dataset_manager.explorer import DatasetExplorer
from utils.dataset_manager.operations import DatasetOperations
from utils.dataset_manager.analyzer import DatasetAnalyzer

class DatasetManagerTab:
    """Dataset Manager tab for the main application."""
    
    def __init__(self, app):
        """
        Initialize the Dataset Manager tab.
        
        Args:
            app: The main application
        """
        self.app = app
        
        # Create registry
        self.registry = DatasetRegistry(app)
        
        # Create operations handler
        self.operations = DatasetOperations(self.registry)
        
        # Create analyzer
        self.analyzer = DatasetAnalyzer(self.registry)
        
        # Create the tab
        self.frame = ttk.Frame(app.notebook)
        
        # Create the UI components
        self._create_ui()
    
    def _create_ui(self):
        """Create the UI elements for the tab."""
        # Use a PanedWindow for resizable sections
        self.main_pane = ttk.PanedWindow(self.frame, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create the dataset explorer panel
        self.explorer = DatasetExplorer(self.main_pane, self.registry)
        self.main_pane.add(self.explorer.paned_window, weight=1)  # Use explorer.paned_window for the frame
        
        # Create operations panel on the right
        self.operations_frame = ttk.Frame(self.main_pane)
        self.main_pane.add(self.operations_frame, weight=1)
        
        # Create operations UI
        self._create_operations_ui()
    
    def _create_operations_ui(self):
        """Create the UI for dataset operations."""
        # Tab control for different operation categories
        self.ops_notebook = ttk.Notebook(self.operations_frame)
        self.ops_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Split tab
        split_frame = ttk.Frame(self.ops_notebook, padding=10)
        self.ops_notebook.add(split_frame, text="Split Dataset")
        self._create_split_ui(split_frame)
        
        # Merge tab
        merge_frame = ttk.Frame(self.ops_notebook, padding=10)
        self.ops_notebook.add(merge_frame, text="Merge Datasets")
        self._create_merge_ui(merge_frame)
        
        # Filter tab
        filter_frame = ttk.Frame(self.ops_notebook, padding=10)
        self.ops_notebook.add(filter_frame, text="Filter Dataset")
        self._create_filter_ui(filter_frame)
        
        # Export tab
        export_frame = ttk.Frame(self.ops_notebook, padding=10)
        self.ops_notebook.add(export_frame, text="Export Dataset")
        self._create_export_ui(export_frame)
        
        # Analysis tab
        analysis_frame = ttk.Frame(self.ops_notebook, padding=10)
        self.ops_notebook.add(analysis_frame, text="Analysis")
        self._create_analysis_ui(analysis_frame)
    
    def _create_split_ui(self, parent):
        """Create the UI for dataset splitting."""
        # Title and description
        ttk.Label(parent, text="Split Dataset", font=("Helvetica", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Label(
            parent,
            text="Split a dataset into training, validation, and test sets with configurable ratios.",
            wraplength=400
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # Form for split settings
        form_frame = ttk.Frame(parent)
        form_frame.pack(fill=tk.X, pady=10)
        
        # Split ratios
        ratio_frame = ttk.LabelFrame(form_frame, text="Split Ratios")
        ratio_frame.pack(fill=tk.X, pady=5)
        
        # Train ratio
        train_frame = ttk.Frame(ratio_frame)
        train_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(train_frame, text="Training:").pack(side=tk.LEFT, padx=5)
        self.train_ratio = tk.DoubleVar(value=0.7)
        train_scale = ttk.Scale(train_frame, from_=0.1, to=0.9, variable=self.train_ratio, 
                              length=200, command=self._update_ratio_labels)
        train_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.train_label = ttk.Label(train_frame, text="70%")
        self.train_label.pack(side=tk.LEFT, padx=5)
        
        # Validation ratio
        val_frame = ttk.Frame(ratio_frame)
        val_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(val_frame, text="Validation:").pack(side=tk.LEFT, padx=5)
        self.val_ratio = tk.DoubleVar(value=0.15)
        val_scale = ttk.Scale(val_frame, from_=0.0, to=0.5, variable=self.val_ratio, 
                            length=200, command=self._update_ratio_labels)
        val_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.val_label = ttk.Label(val_frame, text="15%")
        self.val_label.pack(side=tk.LEFT, padx=5)
        
        # Test ratio
        test_frame = ttk.Frame(ratio_frame)
        test_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(test_frame, text="Test:").pack(side=tk.LEFT, padx=5)
        self.test_ratio = tk.DoubleVar(value=0.15)
        test_scale = ttk.Scale(test_frame, from_=0.0, to=0.5, variable=self.test_ratio, 
                             length=200, command=self._update_ratio_labels)
        test_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.test_label = ttk.Label(test_frame, text="15%")
        self.test_label.pack(side=tk.LEFT, padx=5)
        
        # Options
        options_frame = ttk.LabelFrame(form_frame, text="Options")
        options_frame.pack(fill=tk.X, pady=10)
        
        # Random seed
        seed_frame = ttk.Frame(options_frame)
        seed_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(seed_frame, text="Random Seed:").pack(side=tk.LEFT, padx=5)
        self.random_seed = tk.IntVar(value=42)
        seed_spin = ttk.Spinbox(seed_frame, from_=0, to=9999, textvariable=self.random_seed, width=10)
        seed_spin.pack(side=tk.LEFT, padx=5)
        
        # Stratification option
        stratify_frame = ttk.Frame(options_frame)
        stratify_frame.pack(fill=tk.X, pady=5)
        
        self.stratify = tk.BooleanVar(value=False)
        stratify_check = ttk.Checkbutton(stratify_frame, text="Stratified Split (maintain class distribution)", 
                                       variable=self.stratify)
        stratify_check.pack(anchor=tk.W, padx=5)
        
        # Button to execute split
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=tk.X, pady=20)
        
        split_btn = ttk.Button(action_frame, text="Split Selected Dataset", command=self._split_dataset)
        split_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.split_status = ttk.Label(action_frame, text="")
        self.split_status.pack(side=tk.LEFT, padx=10)
    
    def _update_ratio_labels(self, *args):
        """Update the ratio labels and ensure they sum to 1.0."""
        train = self.train_ratio.get()
        val = self.val_ratio.get()
        test = self.test_ratio.get()
        
        # Normalize to sum to 1.0
        total = train + val + test
        
        if total > 0:
            factor = 1.0 / total
            train_norm = train * factor
            val_norm = val * factor
            test_norm = test * factor
            
            # Update values silently (without triggering another update)
            self.train_ratio.set(train_norm)
            self.val_ratio.set(val_norm)
            self.test_ratio.set(test_norm)
        
        # Update labels
        self.train_label.config(text=f"{train_norm*100:.1f}%")
        self.val_label.config(text=f"{val_norm*100:.1f}%")
        self.test_label.config(text=f"{test_norm*100:.1f}%")
    
    def _split_dataset(self):
        """Split the selected dataset."""
        # Get selected dataset
        if not hasattr(self.explorer, 'selected_dataset') or not self.explorer.selected_dataset:
            messagebox.showerror("Error", "Please select a dataset to split.")
            return
            
        dataset_id = self.explorer.selected_dataset['id']
        
        # Get split parameters
        train_ratio = self.train_ratio.get()
        val_ratio = self.val_ratio.get()
        test_ratio = self.test_ratio.get()
        random_seed = self.random_seed.get()
        stratify = self.stratify.get()
        
        # Update status
        self.split_status.config(text="Splitting dataset...")
        
        # Perform split in a separate thread
        def split_thread():
            try:
                result = self.operations.split_dataset(
                    dataset_id, 
                    train_ratio, 
                    val_ratio, 
                    test_ratio, 
                    random_seed, 
                    stratify_by=stratify
                )
                
                if result:
                    self.split_status.config(text="Split completed successfully.")
                    self.explorer.refresh_datasets()
                else:
                    self.split_status.config(text="Split failed.")
            except Exception as e:
                self.split_status.config(text=f"Error: {str(e)}")
                messagebox.showerror("Split Error", str(e))
        
        threading.Thread(target=split_thread, daemon=True).start()
    
    def _create_merge_ui(self, parent):
        """Create the UI for dataset merging."""
        # Title and description
        ttk.Label(parent, text="Merge Datasets", font=("Helvetica", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Label(
            parent,
            text="Combine multiple datasets into a single dataset.",
            wraplength=400
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # Frame for dataset selection
        select_frame = ttk.LabelFrame(parent, text="Select Datasets to Merge")
        select_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create a listbox with checkboxes for dataset selection
        datasets_frame = ttk.Frame(select_frame)
        datasets_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollable listbox
        scrollbar = ttk.Scrollbar(datasets_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.merge_listbox = tk.Listbox(datasets_frame, selectmode=tk.MULTIPLE)
        self.merge_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbar
        scrollbar.config(command=self.merge_listbox.yview)
        self.merge_listbox.config(yscrollcommand=scrollbar.set)
        
        # Refresh button
        refresh_btn = ttk.Button(select_frame, text="Refresh Dataset List", 
                                command=self._refresh_merge_datasets)
        refresh_btn.pack(anchor=tk.W, padx=5, pady=5)
        
        # Options for merging
        options_frame = ttk.LabelFrame(parent, text="Merge Options")
        options_frame.pack(fill=tk.X, pady=10)
        
        # Name for merged dataset
        name_frame = ttk.Frame(options_frame)
        name_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(name_frame, text="Name for Merged Dataset:").pack(side=tk.LEFT, padx=5)
        self.merged_name = tk.StringVar()
        name_entry = ttk.Entry(name_frame, textvariable=self.merged_name, width=30)
        name_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Merge method
        method_frame = ttk.Frame(options_frame)
        method_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(method_frame, text="Merge Method:").pack(side=tk.LEFT, padx=5)
        self.merge_method = tk.StringVar(value="copy")
        method_combo = ttk.Combobox(method_frame, textvariable=self.merge_method, width=20)
        method_combo['values'] = ["copy", "link"]
        method_combo.pack(side=tk.LEFT, padx=5)
        method_combo.state(['readonly'])
        
        # Explanation of methods
        explain_frame = ttk.Frame(options_frame)
        explain_frame.pack(fill=tk.X, pady=5)
        
        explanation = (
            "• copy: Create a new dataset with copies of all files (uses more disk space)\n"
            "• link: Create a new dataset with links to original files (saves space but requires original datasets)"
        )
        ttk.Label(explain_frame, text=explanation, wraplength=400).pack(padx=5, pady=5)
        
        # Button to execute merge
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=tk.X, pady=20)
        
        merge_btn = ttk.Button(action_frame, text="Merge Selected Datasets", command=self._merge_datasets)
        merge_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.merge_status = ttk.Label(action_frame, text="")
        self.merge_status.pack(side=tk.LEFT, padx=10)
        
        # Load datasets initially
        self._refresh_merge_datasets()
    
    def _refresh_merge_datasets(self):
        """Refresh the list of datasets for merging."""
        # Clear listbox
        self.merge_listbox.delete(0, tk.END)
        
        # Get all datasets
        datasets = self.registry.get_datasets()
        
        # Add to listbox
        for dataset in datasets:
            display_text = f"{dataset['id']}: {dataset['name']}"
            self.merge_listbox.insert(tk.END, display_text)
    
    def _merge_datasets(self):
        """Merge the selected datasets."""
        # Get selected datasets
        selected_indices = self.merge_listbox.curselection()
        if not selected_indices or len(selected_indices) < 2:
            messagebox.showerror("Error", "Please select at least two datasets to merge.")
            return
        
        # Get dataset IDs
        dataset_ids = []
        for idx in selected_indices:
            # Parse ID from the text "ID: Name"
            text = self.merge_listbox.get(idx)
            try:
                dataset_id = int(text.split(":")[0])
                dataset_ids.append(dataset_id)
            except:
                pass
        
        if len(dataset_ids) < 2:
            messagebox.showerror("Error", "Please select at least two valid datasets to merge.")
            return
        
        # Get merge options
        merged_name = self.merged_name.get()
        if not merged_name:
            merged_name = f"Merged_{len(dataset_ids)}_datasets"
            
        merge_method = self.merge_method.get()
        
        # Update status
        self.merge_status.config(text="Merging datasets...")
        
        # Perform merge in a separate thread
        def merge_thread():
            try:
                result = self.operations.merge_datasets(
                    dataset_ids,
                    merged_name=merged_name,
                    merge_method=merge_method
                )
                
                if result:
                    self.merge_status.config(text="Merge completed successfully.")
                    self.explorer.refresh_datasets()
                else:
                    self.merge_status.config(text="Merge failed.")
            except Exception as e:
                self.merge_status.config(text=f"Error: {str(e)}")
                messagebox.showerror("Merge Error", str(e))
        
        threading.Thread(target=merge_thread, daemon=True).start()
    
    def _create_filter_ui(self, parent):
        """Create the UI for dataset filtering."""
        # Title and description
        ttk.Label(parent, text="Filter Dataset", font=("Helvetica", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Label(
            parent,
            text="Create a new dataset by filtering an existing one based on criteria.",
            wraplength=400
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # Filter criteria frame
        filter_frame = ttk.LabelFrame(parent, text="Filter Criteria")
        filter_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # File type filter
        type_frame = ttk.Frame(filter_frame)
        type_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(type_frame, text="File Type:").pack(side=tk.LEFT, padx=5)
        self.filter_extension = tk.StringVar(value="all")
        type_combo = ttk.Combobox(type_frame, textvariable=self.filter_extension, width=20)
        type_combo['values'] = ["all", ".jpg", ".jpeg", ".png", ".bmp"]
        type_combo.pack(side=tk.LEFT, padx=5)
        type_combo.state(['readonly'])
        
        # Image size filter
        size_frame = ttk.LabelFrame(filter_frame, text="Image Size Filters")
        size_frame.pack(fill=tk.X, pady=10)
        
        # Min width
        min_width_frame = ttk.Frame(size_frame)
        min_width_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(min_width_frame, text="Minimum Width:").pack(side=tk.LEFT, padx=5)
        self.min_width = tk.IntVar(value=0)
        min_width_spin = ttk.Spinbox(min_width_frame, from_=0, to=10000, textvariable=self.min_width, width=10)
        min_width_spin.pack(side=tk.LEFT, padx=5)
        
        # Min height
        min_height_frame = ttk.Frame(size_frame)
        min_height_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(min_height_frame, text="Minimum Height:").pack(side=tk.LEFT, padx=5)
        self.min_height = tk.IntVar(value=0)
        min_height_spin = ttk.Spinbox(min_height_frame, from_=0, to=10000, textvariable=self.min_height, width=10)
        min_height_spin.pack(side=tk.LEFT, padx=5)
        
        # Max width
        max_width_frame = ttk.Frame(size_frame)
        max_width_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(max_width_frame, text="Maximum Width:").pack(side=tk.LEFT, padx=5)
        self.max_width = tk.IntVar(value=0)  # 0 means no limit
        max_width_spin = ttk.Spinbox(max_width_frame, from_=0, to=10000, textvariable=self.max_width, width=10)
        max_width_spin.pack(side=tk.LEFT, padx=5)
        
        # Max height
        max_height_frame = ttk.Frame(size_frame)
        max_height_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(max_height_frame, text="Maximum Height:").pack(side=tk.LEFT, padx=5)
        self.max_height = tk.IntVar(value=0)  # 0 means no limit
        max_height_spin = ttk.Spinbox(max_height_frame, from_=0, to=10000, textvariable=self.max_height, width=10)
        max_height_spin.pack(side=tk.LEFT, padx=5)
        
        # Aspect ratio filter
        aspect_frame = ttk.LabelFrame(filter_frame, text="Aspect Ratio Filters")
        aspect_frame.pack(fill=tk.X, pady=10)
        
        # Min aspect ratio
        min_aspect_frame = ttk.Frame(aspect_frame)
        min_aspect_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(min_aspect_frame, text="Minimum Aspect Ratio (w/h):").pack(side=tk.LEFT, padx=5)
        self.min_aspect = tk.DoubleVar(value=0.0)
        min_aspect_spin = ttk.Spinbox(min_aspect_frame, from_=0.0, to=10.0, increment=0.1, 
                                     textvariable=self.min_aspect, width=10)
        min_aspect_spin.pack(side=tk.LEFT, padx=5)
        
        # Max aspect ratio
        max_aspect_frame = ttk.Frame(aspect_frame)
        max_aspect_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(max_aspect_frame, text="Maximum Aspect Ratio (w/h):").pack(side=tk.LEFT, padx=5)
        self.max_aspect = tk.DoubleVar(value=0.0)  # 0 means no limit
        max_aspect_spin = ttk.Spinbox(max_aspect_frame, from_=0.0, to=10.0, increment=0.1, 
                                     textvariable=self.max_aspect, width=10)
        max_aspect_spin.pack(side=tk.LEFT, padx=5)
        
        # Output options
        output_frame = ttk.LabelFrame(parent, text="Output Options")
        output_frame.pack(fill=tk.X, pady=10)
        
        # Output name
        name_frame = ttk.Frame(output_frame)
        name_frame.pack(fill=tk.X, pady=5)
        
ttk.Label(name_frame, text="Name for Filtered Dataset:").pack(side=tk.LEFT, padx=5)
        self.filtered_name = tk.StringVar()
        name_entry = ttk.Entry(name_frame, textvariable=self.filtered_name, width=30)
        name_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Button to execute filter
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=tk.X, pady=20)
        
        filter_btn = ttk.Button(action_frame, text="Filter Selected Dataset", command=self._filter_dataset)
        filter_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.filter_status = ttk.Label(action_frame, text="")
        self.filter_status.pack(side=tk.LEFT, padx=10)
    
    def _filter_dataset(self):
        """Filter the selected dataset."""
        # Get selected dataset
        if not hasattr(self.explorer, 'selected_dataset') or not self.explorer.selected_dataset:
            messagebox.showerror("Error", "Please select a dataset to filter.")
            return
            
        dataset_id = self.explorer.selected_dataset['id']
        
        # Build filter criteria
        criteria = {}
        
        # File extension
        if self.filter_extension.get() != "all":
            criteria["extension"] = self.filter_extension.get()
        
        # Image dimensions
        if self.min_width.get() > 0:
            criteria["min_width"] = self.min_width.get()
        
        if self.min_height.get() > 0:
            criteria["min_height"] = self.min_height.get()
        
        if self.max_width.get() > 0:
            criteria["max_width"] = self.max_width.get()
        
        if self.max_height.get() > 0:
            criteria["max_height"] = self.max_height.get()
        
        # Aspect ratio
        if self.min_aspect.get() > 0:
            criteria["min_aspect_ratio"] = self.min_aspect.get()
        
        if self.max_aspect.get() > 0:
            criteria["max_aspect_ratio"] = self.max_aspect.get()
        
        # Validate criteria
        if not criteria:
            messagebox.showwarning("Warning", "No filter criteria specified. The filtered dataset will be the same as the original.")
            return
        
        # Get output name
        output_name = self.filtered_name.get()
        
        # Update status
        self.filter_status.config(text="Filtering dataset...")
        
        # Perform filter in a separate thread
        def filter_thread():
            try:
                result = self.operations.filter_dataset(
                    dataset_id,
                    criteria,
                    output_name=output_name
                )
                
                if result:
                    self.filter_status.config(text="Filter completed successfully.")
                    self.explorer.refresh_datasets()
                else:
                    self.filter_status.config(text="Filter failed or no files matched the criteria.")
            except Exception as e:
                self.filter_status.config(text=f"Error: {str(e)}")
                messagebox.showerror("Filter Error", str(e))
        
        threading.Thread(target=filter_thread, daemon=True).start()
    
    def _create_export_ui(self, parent):
        """Create the UI for dataset exporting."""
        # Title and description
        ttk.Label(parent, text="Export Dataset", font=("Helvetica", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Label(
            parent,
            text="Export a dataset to a standard format for use with other tools.",
            wraplength=400
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # Export format selection
        format_frame = ttk.LabelFrame(parent, text="Export Format")
        format_frame.pack(fill=tk.X, pady=10)
        
        # Format radio buttons
        format_options = ttk.Frame(format_frame)
        format_options.pack(fill=tk.X, pady=5)
        
        self.export_format = tk.StringVar(value="csv")
        
        formats = [
            ("CSV Inventory", "csv", "Simple CSV listing of files and metadata"),
            ("COCO Format", "coco", "Common Objects in Context format (for object detection)"),
            ("YOLO Format", "yolo", "YOLOv5/YOLOv8 directory structure"),
            ("Pascal VOC", "voc", "Pascal Visual Object Classes format")
        ]
        
        for i, (label, value, desc) in enumerate(formats):
            format_option = ttk.Frame(format_options)
            format_option.pack(fill=tk.X, pady=2)
            
            radio = ttk.Radiobutton(format_option, text=label, value=value, variable=self.export_format)
            radio.pack(side=tk.LEFT, padx=5)
            
            ttk.Label(format_option, text=desc, foreground="#666666").pack(side=tk.LEFT, padx=20)
        
        # Output path
        path_frame = ttk.Frame(parent)
        path_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(path_frame, text="Output Path:").pack(side=tk.LEFT, padx=5)
        
        path_input_frame = ttk.Frame(path_frame)
        path_input_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.export_path = tk.StringVar()
        path_entry = ttk.Entry(path_input_frame, textvariable=self.export_path, width=30)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(path_input_frame, text="Browse...", 
                              command=lambda: self.export_path.set(filedialog.askdirectory()))
        browse_btn.pack(side=tk.RIGHT, padx=5)
        
        # Button to execute export
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=tk.X, pady=20)
        
        export_btn = ttk.Button(action_frame, text="Export Selected Dataset", command=self._export_dataset)
        export_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.export_status = ttk.Label(action_frame, text="")
        self.export_status.pack(side=tk.LEFT, padx=10)
    
    def _export_dataset(self):
        """Export the selected dataset."""
        # Get selected dataset
        if not hasattr(self.explorer, 'selected_dataset') or not self.explorer.selected_dataset:
            messagebox.showerror("Error", "Please select a dataset to export.")
            return
            
        dataset_id = self.explorer.selected_dataset['id']
        
        # Get export format
        format_type = self.export_format.get()
        
        # Get output path
        output_path = self.export_path.get()
        
        # Update status
        self.export_status.config(text="Exporting dataset...")
        
        # Perform export in a separate thread
        def export_thread():
            try:
                result = self.operations.export_dataset(
                    dataset_id,
                    format_type,
                    output_path=output_path
                )
                
                if result:
                    self.export_status.config(text=f"Export completed successfully to {result}")
                else:
                    self.export_status.config(text="Export failed.")
            except Exception as e:
                self.export_status.config(text=f"Error: {str(e)}")
                messagebox.showerror("Export Error", str(e))
        
        threading.Thread(target=export_thread, daemon=True).start()
    
    def _create_analysis_ui(self, parent):
        """Create the UI for dataset analysis."""
        # Title and description
        ttk.Label(parent, text="Dataset Analysis", font=("Helvetica", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Label(
            parent,
            text="Analyze a dataset to get statistics and quality information.",
            wraplength=400
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # Analysis options
        options_frame = ttk.LabelFrame(parent, text="Analysis Options")
        options_frame.pack(fill=tk.X, pady=10)
        
        # Analysis type checkboxes
        self.analyze_files = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="File Statistics", variable=self.analyze_files).pack(anchor=tk.W, padx=5, pady=2)
        
        self.analyze_images = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Image Properties", variable=self.analyze_images).pack(anchor=tk.W, padx=5, pady=2)
        
        self.find_issues = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Find Quality Issues", variable=self.find_issues).pack(anchor=tk.W, padx=5, pady=2)
        
        self.find_duplicates = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Find Duplicate Images", variable=self.find_duplicates).pack(anchor=tk.W, padx=5, pady=2)
        
        # Sample size for performance
        sample_frame = ttk.Frame(options_frame)
        sample_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(sample_frame, text="Sample Size for Image Analysis:").pack(side=tk.LEFT, padx=5)
        self.sample_size = tk.IntVar(value=500)
        sample_spin = ttk.Spinbox(sample_frame, from_=10, to=10000, textvariable=self.sample_size, width=10)
        sample_spin.pack(side=tk.LEFT, padx=5)
        ttk.Label(sample_frame, text="(larger samples take longer)").pack(side=tk.LEFT, padx=5)
        
        # Button to execute analysis
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=tk.X, pady=20)
        
        analyze_btn = ttk.Button(action_frame, text="Analyze Selected Dataset", command=self._analyze_dataset)
        analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.analysis_status = ttk.Label(action_frame, text="")
        self.analysis_status.pack(side=tk.LEFT, padx=10)
        
        # Analysis results display
        results_frame = ttk.LabelFrame(parent, text="Analysis Results")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create a notebook for different result categories
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs for different analysis results
        self.file_stats_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.file_stats_frame, text="Files")
        
        self.image_stats_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.image_stats_frame, text="Images")
        
        self.issues_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.issues_frame, text="Issues")
        
        self.duplicates_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.duplicates_frame, text="Duplicates")
    
    def _analyze_dataset(self):
        """Analyze the selected dataset."""
        # Get selected dataset
        if not hasattr(self.explorer, 'selected_dataset') or not self.explorer.selected_dataset:
            messagebox.showerror("Error", "Please select a dataset to analyze.")
            return
            
        dataset_id = self.explorer.selected_dataset['id']
        
        # Update status
        self.analysis_status.config(text="Analyzing dataset...")
        
        # Clear previous results
        for widget in self.file_stats_frame.winfo_children():
            widget.destroy()
        
        for widget in self.image_stats_frame.winfo_children():
            widget.destroy()
            
        for widget in self.issues_frame.winfo_children():
            widget.destroy()
            
        for widget in self.duplicates_frame.winfo_children():
            widget.destroy()
        
        # Perform analysis in a separate thread
        def analyze_thread():
            try:
                # Run the analysis
                results = self.analyzer.analyze_dataset(dataset_id)
                
                if not results:
                    self.analysis_status.config(text="Analysis failed.")
                    return
                
                # Update the UI with results
                self.analysis_status.config(text="Analysis completed successfully.")
                
                # Update file statistics tab
                if self.analyze_files.get() and "file_stats" in results:
                    self._display_file_stats(results["file_stats"])
                
                # Update image statistics tab
                if self.analyze_images.get() and "image_stats" in results:
                    self._display_image_stats(results["image_stats"])
                
                # Update quality issues tab
                if self.find_issues.get() and "quality_issues" in results:
                    self._display_quality_issues(results["quality_issues"])
                
                # Update duplicates tab
                if self.find_duplicates.get() and "duplicate_analysis" in results:
                    self._display_duplicates(results["duplicate_analysis"])
                
            except Exception as e:
                self.analysis_status.config(text=f"Error: {str(e)}")
                messagebox.showerror("Analysis Error", str(e))
        
        threading.Thread(target=analyze_thread, daemon=True).start()
    
    def _display_file_stats(self, stats):
        """Display file statistics in the UI."""
        # Create a scrollable frame
        canvas = tk.Canvas(self.file_stats_frame)
        scrollbar = ttk.Scrollbar(self.file_stats_frame, orient=tk.VERTICAL, command=canvas.yview)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame inside canvas for content
        content = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=content, anchor=tk.NW)
        
        # Configure scrolling
        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas.create_window(0, 0, window=content, anchor=tk.NW), width=e.width))
        
        # File count statistics
        count_frame = ttk.LabelFrame(content, text="File Counts")
        count_frame.pack(fill=tk.X, pady=10, padx=10)
        
        ttk.Label(count_frame, text=f"Total Files: {stats['total_count']}").pack(anchor=tk.W, padx=10, pady=2)
        ttk.Label(count_frame, text=f"Image Files: {stats['image_count']}").pack(anchor=tk.W, padx=10, pady=2)
        ttk.Label(count_frame, text=f"Video Files: {stats['video_count']}").pack(anchor=tk.W, padx=10, pady=2)
        ttk.Label(count_frame, text=f"Other Files: {stats['other_count']}").pack(anchor=tk.W, padx=10, pady=2)
        
        # Size statistics
        size_frame = ttk.LabelFrame(content, text="Size Statistics")
        size_frame.pack(fill=tk.X, pady=10, padx=10)
        
        # Convert bytes to megabytes for display
        total_mb = stats['total_size_bytes'] / (1024 * 1024)
        image_mb = stats['image_size_bytes'] / (1024 * 1024)
        video_mb = stats['video_size_bytes'] / (1024 * 1024)
        other_mb = stats['other_size_bytes'] / (1024 * 1024)
        
        ttk.Label(size_frame, text=f"Total Size: {total_mb:.2f} MB").pack(anchor=tk.W, padx=10, pady=2)
        ttk.Label(size_frame, text=f"Image Files: {image_mb:.2f} MB").pack(anchor=tk.W, padx=10, pady=2)
        ttk.Label(size_frame, text=f"Video Files: {video_mb:.2f} MB").pack(anchor=tk.W, padx=10, pady=2)
        ttk.Label(size_frame, text=f"Other Files: {other_mb:.2f} MB").pack(anchor=tk.W, padx=10, pady=2)
        
        if stats['total_count'] > 0:
            avg_mb = stats['avg_file_size'] / (1024 * 1024)
            ttk.Label(size_frame, text=f"Average File Size: {avg_mb:.2f} MB").pack(anchor=tk.W, padx=10, pady=2)
        
        # File formats
        format_frame = ttk.LabelFrame(content, text="File Formats")
        format_frame.pack(fill=tk.X, pady=10, padx=10)
        
        # Sort formats by count
        formats_sorted = sorted(
            stats['formats'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        for ext, info in formats_sorted:
            count = info['count']
            size_mb = info['size_bytes'] / (1024 * 1024)
            ttk.Label(format_frame, text=f"{ext}: {count} files, {size_mb:.2f} MB").pack(anchor=tk.W, padx=10, pady=2)
        
        # Largest file
        largest_frame = ttk.LabelFrame(content, text="Largest File")
        largest_frame.pack(fill=tk.X, pady=10, padx=10)
        
        largest_path = stats['largest_file']['path']
        if largest_path:
            largest_name = os.path.basename(largest_path)
            largest_size_mb = stats['largest_file']['size'] / (1024 * 1024)
            ttk.Label(largest_frame, text=f"File: {largest_name}").pack(anchor=tk.W, padx=10, pady=2)
            ttk.Label(largest_frame, text=f"Size: {largest_size_mb:.2f} MB").pack(anchor=tk.W, padx=10, pady=2)
            ttk.Label(largest_frame, text=f"Path: {largest_path}").pack(anchor=tk.W, padx=10, pady=2)
    
    def _display_image_stats(self, stats):
        """Display image statistics in the UI."""
        # Create a scrollable frame
        canvas = tk.Canvas(self.image_stats_frame)
        scrollbar = ttk.Scrollbar(self.image_stats_frame, orient=tk.VERTICAL, command=canvas.yview)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame inside canvas for content
        content = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=content, anchor=tk.NW)
        
        # Configure scrolling
        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas.create_window(0, 0, window=content, anchor=tk.NW), width=e.width))
        
        # Dimension statistics
        if 'dimensions' in stats and stats['dimensions']:
            dim_frame = ttk.LabelFrame(content, text="Image Dimensions")
            dim_frame.pack(fill=tk.X, pady=10, padx=10)
            
            ttk.Label(dim_frame, text=f"Width Range: {stats['min_width']} - {stats['max_width']} pixels").pack(anchor=tk.W, padx=10, pady=2)
            ttk.Label(dim_frame, text=f"Height Range: {stats['min_height']} - {stats['max_height']} pixels").pack(anchor=tk.W, padx=10, pady=2)
            
            if 'avg_width' in stats:
                ttk.Label(dim_frame, text=f"Average Dimensions: {stats['avg_width']:.1f} × {stats['avg_height']:.1f} pixels").pack(anchor=tk.W, padx=10, pady=2)
            
            if 'avg_aspect_ratio' in stats:
                ttk.Label(dim_frame, text=f"Average Aspect Ratio: {stats['avg_aspect_ratio']:.2f} (width/height)").pack(anchor=tk.W, padx=10, pady=2)
            
            # Most common resolution
            if 'most_common_resolution' in stats:
                common_res = stats['most_common_resolution']
                ttk.Label(dim_frame, text=f"Most Common Resolution: {common_res['dimensions']} ({common_res['percentage']:.1f}% of images)").pack(anchor=tk.W, padx=10, pady=2)
        
        # Resolution groups
        if 'resolution_groups' in stats and stats['resolution_groups']:
            res_frame = ttk.LabelFrame(content, text="Resolution Groups")
            res_frame.pack(fill=tk.X, pady=10, padx=10)
            
            # Show top 10 most common resolutions
            count = 0
            for res, info in stats['resolution_groups'].items():
                ttk.Label(res_frame, text=f"{res}: {info['count']} images, aspect ratio {info['aspect_ratio']:.2f}").pack(anchor=tk.W, padx=10, pady=2)
                count += 1
                if count >= 10:
                    break
            
            if len(stats['resolution_groups']) > 10:
                ttk.Label(res_frame, text=f"... and {len(stats['resolution_groups']) - 10} more resolutions").pack(anchor=tk.W, padx=10, pady=2)
        
        # Image formats
        if 'formats' in stats and stats['formats']:
            format_frame = ttk.LabelFrame(content, text="Image Formats")
            format_frame.pack(fill=tk.X, pady=10, padx=10)
            
            for fmt, count in stats['formats'].items():
                ttk.Label(format_frame, text=f"{fmt}: {count} images").pack(anchor=tk.W, padx=10, pady=2)
        
        # Color modes
        if 'color_modes' in stats and stats['color_modes']:
            color_frame = ttk.LabelFrame(content, text="Color Modes")
            color_frame.pack(fill=tk.X, pady=10, padx=10)
            
            for mode, count in stats['color_modes'].items():
                mode_desc = ""
                if mode == "RGB":
                    mode_desc = "(Color)"
                elif mode == "L":
                    mode_desc = "(Grayscale)"
                elif mode == "RGBA":
                    mode_desc = "(Color with transparency)"
                elif mode == "CMYK":
                    mode_desc = "(CMYK color)"
                
                ttk.Label(color_frame, text=f"{mode} {mode_desc}: {count} images").pack(anchor=tk.W, padx=10, pady=2)
    
    def _display_quality_issues(self, issues):
        """Display quality issues in the UI."""
        # Create a scrollable frame
        canvas = tk.Canvas(self.issues_frame)
        scrollbar = ttk.Scrollbar(self.issues_frame, orient=tk.VERTICAL, command=canvas.yview)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame inside canvas for content
        content = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=content, anchor=tk.NW)
        
        # Configure scrolling
        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas.create_window(0, 0, window=content, anchor=tk.NW), width=e.width))
        
        if not issues:
            ttk.Label(content, text="No quality issues found.").pack(anchor=tk.W, padx=20, pady=20)
            return
        
        # Group issues by type
        issue_types = {}
        for issue in issues:
            issue_type = issue['type']
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(issue)
        
        # Display summary
        summary_frame = ttk.Frame(content)
        summary_frame.pack(fill=tk.X, pady=10, padx=10)
        
        ttk.Label(summary_frame, text=f"Found {len(issues)} potential issues:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        
        for issue_type, type_issues in issue_types.items():
            count = len(type_issues)
            
            issue_desc = ""
            if issue_type == "corrupt_image":
                issue_desc = "Corrupt or unreadable images"
            elif issue_type == "invalid_image":
                issue_desc = "Invalid image files"
            elif issue_type == "small_image":
                issue_desc = "Very small images (< 32px)"
            elif issue_type == "dark_image":
                issue_desc = "Very dark images"
            elif issue_type == "bright_image":
                issue_desc = "Very bright or washed-out images"
            else:
                issue_desc = issue_type.replace("_", " ").title()
            
            ttk.Label(summary_frame, text=f"• {issue_desc}: {count} files").pack(anchor=tk.W, padx=20, pady=2)
        
        # Display each issue type in a separate frame
        for issue_type, type_issues in issue_types.items():
            issue_frame = ttk.LabelFrame(content, text=issue_type.replace("_", " ").title())
            issue_frame.pack(fill=tk.X, pady=10, padx=10)
            
            # Show first 10 issues of this type
            count = 0
            for issue in type_issues:
                file_path = issue['path']
                file_name = os.path.basename(file_path)
                
                issue_item = ttk.Frame(issue_frame)
                issue_item.pack(fill=tk.X, pady=2)
                
                ttk.Label(issue_item, text=file_name).pack(side=tk.LEFT, padx=5)
                
                # Show details based on issue type
                details = issue.get('details', {})
                details_text = ""
                
                if issue_type == "small_image" and 'width' in details and 'height' in details:
                    details_text = f"Size: {details['width']}×{details['height']} pixels"
                elif issue_type in ["dark_image", "bright_image"] and 'brightness' in details:
                    details_text = f"Brightness: {details['brightness']:.1f}"
                elif 'error' in details:
                    details_text = f"Error: {details['error']}"
                
                if details_text:
                    ttk.Label(issue_item, text=details_text).pack(side=tk.LEFT, padx=20)
                
                # Add button to view the file
                ttk.Button(
                    issue_item, 
                    text="View", 
                    command=lambda p=file_path: self._open_file(p)
                ).pack(side=tk.RIGHT, padx=5)
                
                count += 1
                if count >= 10:
                    break
            
            if len(type_issues) > 10:
                ttk.Label(issue_frame, text=f"... and {len(type_issues) - 10} more files").pack(anchor=tk.W, padx=10, pady=2)
    
    def _display_duplicates(self, duplicate_data):
        """Display duplicate information in the UI."""
        # Create a scrollable frame
        canvas = tk.Canvas(self.duplicates_frame)
        scrollbar = ttk.Scrollbar(self.duplicates_frame, orient=tk.VERTICAL, command=canvas.yview)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame inside canvas for content
        content = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=content, anchor=tk.NW)
        
        # Configure scrolling
        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas.create_window(0, 0, window=content, anchor=tk.NW), width=e.width))
        
        # Summary information
        summary_frame = ttk.Frame(content)
        summary_frame.pack(fill=tk.X, pady=10, padx=10)
        
        analyzed_count = duplicate_data.get('analyzed_count', 0)
        
        ttk.Label(summary_frame, text=f"Analyzed {analyzed_count} images for duplicates", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        
        exact_count = len(duplicate_data.get('exact_duplicates', []))
        similar_count = len(duplicate_data.get('similar_images', []))
        
        if
        ttk.Label(summary_frame, text=f"Analyzed {analyzed_count} images for duplicates", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        
        exact_count = len(duplicate_data.get('exact_duplicates', []))
        similar_count = len(duplicate_data.get('similar_images', []))
        
        if exact_count == 0 and similar_count == 0:
            ttk.Label(content, text="No duplicate images found.").pack(anchor=tk.W, padx=20, pady=10)
            return
        
        ttk.Label(summary_frame, text=f"Found {exact_count} exact duplicates and {similar_count} similar images").pack(anchor=tk.W, pady=5)
        
        # Display exact duplicates
        if exact_count > 0:
            exact_frame = ttk.LabelFrame(content, text="Exact Duplicates")
            exact_frame.pack(fill=tk.X, pady=10, padx=10)
            
            # Show each duplicate pair
            for i, dup in enumerate(duplicate_data['exact_duplicates']):
            if i >= 20:  # Limit to 20 pairs for performance
                ttk.Label(exact_frame, text=f"... and {exact_count - 20} more duplicate pairs").pack(anchor=tk.W, padx=10, pady=5)
                break
                
            dup_frame = ttk.Frame(exact_frame)
            dup_frame.pack(fill=tk.X, pady=5, padx=5)
            
            # Show original and duplicate file names
            orig_name = os.path.basename(dup['original'])
            dup_name = os.path.basename(dup['duplicate'])
            
            ttk.Label(dup_frame, text=f"Original: {orig_name}").pack(anchor=tk.W)
            ttk.Label(dup_frame, text=f"Duplicate: {dup_name}").pack(anchor=tk.W)
            
            # Add buttons to view files
            btn_frame = ttk.Frame(dup_frame)
            btn_frame.pack(fill=tk.X, pady=5)
            
            ttk.Button(
                btn_frame, 
                text="View Original", 
                command=lambda p=dup['original']: self._open_file(p)
            ).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(
                btn_frame, 
                text="View Duplicate", 
                command=lambda p=dup['duplicate']: self._open_file(p)
            ).pack(side=tk.LEFT, padx=5)
            
            # Add separator between pairs
            if i < exact_count - 1:
                ttk.Separator(exact_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Display similar images
        if similar_count > 0:
            similar_frame = ttk.LabelFrame(content, text="Similar Images")
            similar_frame.pack(fill=tk.X, pady=10, padx=10)
            
            # Sort by similarity (highest first)
            similar_images = sorted(
            duplicate_data['similar_images'],
            key=lambda x: x['similarity'],
            reverse=True
            )
            
            # Show each similar pair
            for i, sim in enumerate(similar_images):
            if i >= 20:  # Limit to 20 pairs for performance
                ttk.Label(similar_frame, text=f"... and {similar_count - 20} more similar pairs").pack(anchor=tk.W, padx=10, pady=5)
                break
                
            sim_frame = ttk.Frame(similar_frame)
            sim_frame.pack(fill=tk.X, pady=5, padx=5)
            
            # Show files and similarity
            img1_name = os.path.basename(sim['image1'])
            img2_name = os.path.basename(sim['image2'])
            similarity = sim['similarity'] * 100  # Convert to percentage
            
            ttk.Label(sim_frame, text=f"Similarity: {similarity:.1f}%").pack(anchor=tk.W)
            ttk.Label(sim_frame, text=f"Image 1: {img1_name}").pack(anchor=tk.W)
            ttk.Label(sim_frame, text=f"Image 2: {img2_name}").pack(anchor=tk.W)
            
            # Add buttons to view files
            btn_frame = ttk.Frame(sim_frame)
            btn_frame.pack(fill=tk.X, pady=5)
            
            ttk.Button(
                btn_frame, 
                text="View Image 1", 
                command=lambda p=sim['image1']: self._open_file(p)
            ).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(
                btn_frame, 
                text="View Image 2", 
                command=lambda p=sim['image2']: self._open_file(p)
            ).pack(side=tk.LEFT, padx=5)
            
            # Add separator between pairs
            if i < similar_count - 1:
                ttk.Separator(sim_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
