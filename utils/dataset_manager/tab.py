"""
Dataset Manager Tab for Dataset Preparation Tool
Main UI tab component for dataset management with CustomTkinter support.
"""

import os
import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import customtkinter as ctk

from utils.dataset_manager.registry import DatasetRegistry
from utils.dataset_manager.explorer import DatasetExplorer
from utils.dataset_manager.operations import DatasetOperations
from utils.dataset_manager.analyzer import DatasetAnalyzer

class DatasetManagerTab:
    """Dataset Manager tab for the main application with CustomTkinter support."""
    
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
        
        # Create the tab frame with CustomTkinter (directly use app instead of app.notebook)
        self.frame = ctk.CTkFrame(app)
        
        # Create the UI components
        self._create_ui()
    
    def _create_ui(self):
        """Create the UI elements for the tab with CustomTkinter."""
        # Create main frame with CustomTkinter
        self.main_frame = ctk.CTkFrame(self.frame)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create header
        header_frame = ctk.CTkFrame(self.main_frame, fg_color=("gray92", "gray25"))
        header_frame.pack(fill=tk.X, padx=5, pady=(0, 10))
        
        header_label = ctk.CTkLabel(
            header_frame, 
            text="Dataset Manager", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        header_label.pack(padx=15, pady=10)
        
        # Use a PanedWindow for resizable sections 
        # Note: CTkinter doesn't have a direct PanedWindow, so we'll use a frame with weight distribution
        self.content_frame = ctk.CTkFrame(self.main_frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure grid weights
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.columnconfigure(1, weight=1)
        self.content_frame.rowconfigure(0, weight=1)
        
        # Create the explorer and operations frames
        self.explorer_frame = ctk.CTkFrame(self.content_frame)
        self.explorer_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        self.operations_frame = ctk.CTkFrame(self.content_frame)
        self.operations_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        # Create explorer with CustomTkinter support
        self.explorer = DatasetExplorer(self.explorer_frame, self.registry)
        
        # Create operations UI
        self._create_operations_ui()
    
    def _create_operations_ui(self):
        """Create the UI for dataset operations with CustomTkinter."""
        # Create a tabview for operations instead of a notebook
        self.ops_tabview = ctk.CTkTabview(self.operations_frame)
        self.ops_tabview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.ops_tabview.add("Split Dataset")
        self.ops_tabview.add("Merge Datasets")
        self.ops_tabview.add("Filter Dataset")
        self.ops_tabview.add("Export Dataset")
        self.ops_tabview.add("Analysis")
        
        # Build UI for each tab
        self._create_split_ui(self.ops_tabview.tab("Split Dataset"))
        self._create_merge_ui(self.ops_tabview.tab("Merge Datasets"))
        self._create_filter_ui(self.ops_tabview.tab("Filter Dataset"))
        self._create_export_ui(self.ops_tabview.tab("Export Dataset"))
        self._create_analysis_ui(self.ops_tabview.tab("Analysis"))
    
    def _create_split_ui(self, parent):
        """Create the UI for dataset splitting with CustomTkinter."""
        # ScrollableFrame for content
        content_frame = ctk.CTkScrollableFrame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Title and description
        title = ctk.CTkLabel(
            content_frame, 
            text="Split Dataset", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.pack(anchor=tk.W, pady=(0, 10))
        
        description = ctk.CTkLabel(
            content_frame,
            text="Split a dataset into training, validation, and test sets with configurable ratios.",
            wraplength=400
        )
        description.pack(anchor=tk.W, pady=(0, 15))
        
        # Form for split settings
        form_frame = ctk.CTkFrame(content_frame)
        form_frame.pack(fill=tk.X, pady=10)
        
        # Split ratios section
        ratio_frame = ctk.CTkFrame(form_frame)
        ratio_frame.pack(fill=tk.X, pady=10, padx=10)
        
        ratio_label = ctk.CTkLabel(
            ratio_frame, 
            text="Split Ratios", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        ratio_label.pack(anchor=tk.W, pady=(0, 10), padx=5)
        
        # Train ratio
        train_frame = ctk.CTkFrame(ratio_frame)
        train_frame.pack(fill=tk.X, pady=5)
        
        ctk.CTkLabel(train_frame, text="Training:").pack(side=tk.LEFT, padx=5)
        self.train_ratio = ctk.DoubleVar(value=0.7)
        train_scale = ctk.CTkSlider(
            train_frame, 
            from_=0.1, 
            to=0.9, 
            variable=self.train_ratio,
            command=self._update_ratio_labels,
            width=200
        )
        train_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.train_label = ctk.CTkLabel(train_frame, text="70%")
        self.train_label.pack(side=tk.LEFT, padx=5)
        
        # Validation ratio
        val_frame = ctk.CTkFrame(ratio_frame)
        val_frame.pack(fill=tk.X, pady=5)
        
        ctk.CTkLabel(val_frame, text="Validation:").pack(side=tk.LEFT, padx=5)
        self.val_ratio = ctk.DoubleVar(value=0.15)
        val_scale = ctk.CTkSlider(
            val_frame, 
            from_=0.0, 
            to=0.5, 
            variable=self.val_ratio,
            command=self._update_ratio_labels,
            width=200
        )
        val_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.val_label = ctk.CTkLabel(val_frame, text="15%")
        self.val_label.pack(side=tk.LEFT, padx=5)
        
        # Test ratio
        test_frame = ctk.CTkFrame(ratio_frame)
        test_frame.pack(fill=tk.X, pady=5)
        
        ctk.CTkLabel(test_frame, text="Test:").pack(side=tk.LEFT, padx=5)
        self.test_ratio = ctk.DoubleVar(value=0.15)
        test_scale = ctk.CTkSlider(
            test_frame, 
            from_=0.0, 
            to=0.5, 
            variable=self.test_ratio,
            command=self._update_ratio_labels,
            width=200
        )
        test_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.test_label = ctk.CTkLabel(test_frame, text="15%")
        self.test_label.pack(side=tk.LEFT, padx=5)
        
        # Options section
        options_frame = ctk.CTkFrame(form_frame)
        options_frame.pack(fill=tk.X, pady=10, padx=10)
        
        options_label = ctk.CTkLabel(
            options_frame, 
            text="Options", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        options_label.pack(anchor=tk.W, pady=(0, 10), padx=5)
        
        # Random seed
        seed_frame = ctk.CTkFrame(options_frame)
        seed_frame.pack(fill=tk.X, pady=5)
        
        ctk.CTkLabel(seed_frame, text="Random Seed:").pack(side=tk.LEFT, padx=5)
        self.random_seed = tk.IntVar(value=42)
        seed_spin = ctk.CTkEntry(seed_frame, textvariable=self.random_seed, width=80)
        seed_spin.pack(side=tk.LEFT, padx=5)
        
        # Stratification option
        self.stratify = tk.BooleanVar(value=False)
        stratify_check = ctk.CTkCheckBox(
            options_frame, 
            text="Stratified Split (maintain class distribution)", 
            variable=self.stratify
        )
        stratify_check.pack(anchor=tk.W, padx=5, pady=5)
        
        # Button to execute split
        action_frame = ctk.CTkFrame(content_frame)
        action_frame.pack(fill=tk.X, pady=20)
        
        split_btn = ctk.CTkButton(
            action_frame, 
            text="Split Selected Dataset", 
            command=self._split_dataset,
            fg_color="#4CAF50",
            hover_color="#388E3C"
        )
        split_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Status label
        self.split_status = ctk.CTkLabel(action_frame, text="")
        self.split_status.pack(side=tk.LEFT, padx=10)
    
    # Placeholder methods that would be implemented similar to the original DatasetManagerTab
    
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
        self.train_label.configure(text=f"{train_norm*100:.1f}%")
        self.val_label.configure(text=f"{val_norm*100:.1f}%")
        self.test_label.configure(text=f"{test_norm*100:.1f}%")
    
    def _split_dataset(self):
        """Split the selected dataset."""
        # Get the selected dataset
        selected_dataset = self.explorer.get_selected_dataset()
        if not selected_dataset:
            messagebox.showinfo("Split Dataset", "Please select a dataset to split.")
            return
        
        # Create a new thread for the split operation
        threading.Thread(target=self._split_thread, args=(selected_dataset,), daemon=True).start()
        
        # Update UI to show processing
        self.split_status.configure(text="Processing...")
    
    def _split_thread(self, dataset):
        """Thread for dataset split to avoid UI blocking."""
        # Implement the actual split operation here
        pass
    
    def _create_merge_ui(self, parent):
        """Create UI for merging datasets."""
        # To be implemented
        pass
    
    def _create_filter_ui(self, parent):
        """Create UI for filtering datasets."""
        # To be implemented
        pass
    
    def _create_export_ui(self, parent):
        """Create UI for exporting datasets."""
        # To be implemented
        pass
    
    def _create_analysis_ui(self, parent):
        """Create UI for dataset analysis."""
        # To be implemented
        pass 