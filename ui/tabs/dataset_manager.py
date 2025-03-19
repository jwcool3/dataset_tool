class DatasetAnalyzer:
    """Analyzes datasets to provide insights and quality assessment."""
    
    def __init__(self, registry):
        """
        Initialize the dataset analyzer.
        
        Args:
            registry: DatasetRegistry instance
        """
        self.registry = registry
    
    def analyze_dataset(self, dataset_id):
        """
        Perform comprehensive analysis of a dataset.
        
        Args:
            dataset_id: Dataset ID to analyze
            
        Returns:
            dict: Analysis results
        """
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset or not os.path.isdir(dataset['path']):
            return None
            
        # Initialize results
        results = {
            "dataset_id": dataset_id,
            "dataset_name": dataset['name'],
            "path": dataset['path'],
            "file_stats": {},
            "image_stats": {},
            "folder_stats": {},
            "quality_issues": [],
            "duplicate_analysis": {}
        }
        
        # Get file statistics
        results["file_stats"] = self._analyze_files(dataset['path'])
        
        # Get image statistics (only if there are images)
        if results["file_stats"]["image_count"] > 0:
            results["image_stats"] = self._analyze_images(dataset['path'])
            
            # Check for image quality issues
            results["quality_issues"] = self._check_quality_issues(dataset['path'])
            
            # Find potential duplicates
            results["duplicate_analysis"] = self._find_duplicates(dataset['path'])
        
        # Analyze folder structure
        results["folder_stats"] = self._analyze_folders(dataset['path'])
        
        # Store analysis results
        timestamp = datetime.datetime.now().isoformat()
        
        # Update dataset with analysis timestamp
        attributes = dataset.get('attributes', {})
        if not attributes:
            attributes = {}
            
        attributes['last_analysis'] = timestamp
        self.registry.update_dataset(dataset_id, attributes=attributes)
        
        # Record the analysis operation
        self.registry.record_processing_step(
            dataset_id,
            "dataset_analysis",
            {"timestamp": timestamp}
        )
        
        return results
    
    def _analyze_files(self, path):
        """Analyze files in the dataset."""
        stats = {
            "total_count": 0,
            "image_count": 0,
            "video_count": 0,
            "other_count": 0,
            "total_size_bytes": 0,
            "image_size_bytes": 0,
            "video_size_bytes": 0,
            "other_size_bytes": 0,
            "formats": {},
            "largest_file": {"path": None, "size": 0},
            "smallest_file": {"path": None, "size": float('inf')},
        }
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
        
        # Scan all files
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                
                # Update total stats
                stats["total_count"] += 1
                stats["total_size_bytes"] += file_size
                
                # Update largest/smallest
                if file_size > stats["largest_file"]["size"]:
                    stats["largest_file"] = {"path": file_path, "size": file_size}
                    
                if file_size < stats["smallest_file"]["size"]:
                    stats["smallest_file"] = {"path": file_path, "size": file_size}
                
                # Categorize by file type
                ext = os.path.splitext(file)[1].lower()
                
                # Update format counts
                if ext not in stats["formats"]:
                    stats["formats"][ext] = {"count": 0, "size_bytes": 0}
                    
                stats["formats"][ext]["count"] += 1
                stats["formats"][ext]["size_bytes"] += file_size
                
                # Categorize as image, video, or other
                if ext in image_extensions:
                    stats["image_count"] += 1
                    stats["image_size_bytes"] += file_size
                elif ext in video_extensions:
                    stats["video_count"] += 1
                    stats["video_size_bytes"] += file_size
                else:
                    stats["other_count"] += 1
                    stats["other_size_bytes"] += file_size
        
        # Calculate averages
        if stats["total_count"] > 0:
            stats["avg_file_size"] = stats["total_size_bytes"] / stats["total_count"]
        else:
            stats["avg_file_size"] = 0
            
        if stats["image_count"] > 0:
            stats["avg_image_size"] = stats["image_size_bytes"] / stats["image_count"]
        else:
            stats["avg_image_size"] = 0
            
        if stats["video_count"] > 0:
            stats["avg_video_size"] = stats["video_size_bytes"] / stats["video_count"]
        else:
            stats["avg_video_size"] = 0
        
        # Handle case where no smallest file was found
        if stats["smallest_file"]["path"] is None:
            stats["smallest_file"] = {"path": None, "size": 0}
            
        return stats
    
    def _analyze_images(self, path, sample_size=500):
        """Analyze image properties in the dataset."""
        stats = {
            "dimensions": [],
            "aspect_ratios": [],
            "formats": {},
            "color_modes": {},
            "min_width": float('inf'),
            "max_width": 0,
            "min_height": float('inf'),
            "max_height": 0,
            "resolution_groups": {}
        }
        
        # Find image files
        image_files = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for root, _, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        
        # Limit sample size for performance
        if sample_size and len(image_files) > sample_size:
            import random
            random.shuffle(image_files)
            image_files = image_files[:sample_size]
        
        # Process images
        for file_path in image_files:
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    # Get dimensions
                    width, height = img.size
                    
                    # Update dimension stats
                    stats["dimensions"].append((width, height))
                    stats["min_width"] = min(stats["min_width"], width)
                    stats["max_width"] = max(stats["max_width"], width)
                    stats["min_height"] = min(stats["min_height"], height)
                    stats["max_height"] = max(stats["max_height"], height)
                    
                    # Calculate aspect ratio
                    aspect_ratio = width / height
                    stats["aspect_ratios"].append(aspect_ratio)
                    
                    # Update format stats
                    format_name = img.format or "Unknown"
                    if format_name not in stats["formats"]:
                        stats["formats"][format_name] = 0
                    stats["formats"][format_name] += 1
                    
                    # Update color mode stats
                    color_mode = img.mode
                    if color_mode not in stats["color_modes"]:
                        stats["color_modes"][color_mode] = 0
                    stats["color_modes"][color_mode] += 1
                    
                    # Group by resolution
                    res_key = f"{width}x{height}"
                    if res_key not in stats["resolution_groups"]:
                        stats["resolution_groups"][res_key] = {
                            "width": width,
                            "height": height,
                            "count": 0,
                            "aspect_ratio": aspect_ratio
                        }
                    stats["resolution_groups"][res_key]["count"] += 1
                    
            except Exception as e:
                print(f"Error analyzing image {file_path}: {str(e)}")
        
        # Calculate averages and distribution
        if stats["dimensions"]:
            avg_width = sum(w for w, _ in stats["dimensions"]) / len(stats["dimensions"])
            avg_height = sum(h for _, h in stats["dimensions"]) / len(stats["dimensions"])
            avg_aspect_ratio = sum(stats["aspect_ratios"]) / len(stats["aspect_ratios"])
            
            stats["avg_width"] = avg_width
            stats["avg_height"] = avg_height
            stats["avg_aspect_ratio"] = avg_aspect_ratio
            
            # Sort resolution groups by frequency
            stats["resolution_groups"] = dict(
                sorted(
                    stats["resolution_groups"].items(),
                    key=lambda x: x[1]["count"],
                    reverse=True
                )
            )
            
            # Find most common resolution
            common_res = list(stats["resolution_groups"].keys())[0]
            stats["most_common_resolution"] = {
                "dimensions": common_res,
                "count": stats["resolution_groups"][common_res]["count"],
                "percentage": stats["resolution_groups"][common_res]["count"] / len(stats["dimensions"]) * 100
            }
        
        # Clean up min/max if no images were processed
        if stats["min_width"] == float('inf'):
            stats["min_width"] = 0
        if stats["min_height"] == float('inf'):
            stats["min_height"] = 0
            
        return stats
    
    def _analyze_folders(self, path):
        """Analyze folder structure in the dataset."""
        stats = {
            "total_folders": 0,
            "max_depth": 0,
            "folders_by_level": {},
            "empty_folders": 0,
            "folder_sizes": {}
        }
        
        # Track root depth
        root_depth = path.count(os.sep)
        
        # Scan folders
        for root, dirs, files in os.walk(path):
            # Calculate depth
            depth = root.count(os.sep) - root_depth
            
            # Update total folders
            stats["total_folders"] += 1
            
            # Update max depth
            stats["max_depth"] = max(stats["max_depth"], depth)
            
            # Update folders by level
            if depth not in stats["folders_by_level"]:
                stats["folders_by_level"][depth] = 0
            stats["folders_by_level"][depth] += 1
            
            # Check if folder is empty
            if not dirs and not files:
                stats["empty_folders"] += 1
            
            # Calculate folder size
            folder_size = 0
            for file in files:
                file_path = os.path.join(root, file)
                folder_size += os.path.getsize(file_path)
                
            # Store folder size (only for main subfolders)
            if depth == 1:
                folder_name = os.path.basename(root)
                stats["folder_sizes"][folder_name] = folder_size
        
        return stats
    
    def _check_quality_issues(self, path, sample_size=500):
        """Check for quality issues in the dataset."""
        issues = []
        
        # Find image files
        image_files = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for root, _, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        
        # Limit sample size for performance
        if sample_size and len(image_files) > sample_size:
            import random
            random.shuffle(image_files)
            image_files = image_files[:sample_size]
        
        # Check for various issues
        for file_path in image_files:
            try:
                from PIL import Image
                img = Image.open(file_path)
                
                # Check image size
                width, height = img.size
                if width < 32 or height < 32:
                    issues.append({
                        "type": "small_image",
                        "path": file_path,
                        "details": {"width": width, "height": height}
                    })
                
                # Check if image can be fully loaded
                try:
                    img.load()
                except Exception as e:
                    issues.append({
                        "type": "corrupt_image",
                        "path": file_path,
                        "details": {"error": str(e)}
                    })
                    continue
                
                # Check if image is very dark or very bright
                try:
                    if img.mode == "RGB":
                        import numpy as np
                        img_array = np.array(img)
                        brightness = np.mean(img_array)
                        
                        if brightness < 30:
                            issues.append({
                                "type": "dark_image",
                                "path": file_path,
                                "details": {"brightness": float(brightness)}
                            })
                        elif brightness > 240:
                            issues.append({
                                "type": "bright_image",
                                "path": file_path,
                                "details": {"brightness": float(brightness)}
                            })
                except:
                    pass
                
                # Close image
                img.close()
                
            except Exception as e:
                # Failed to open image
                issues.append({
                    "type": "invalid_image",
                    "path": file_path,
                    "details": {"error": str(e)}
                })
        
        return issues
    
    def _find_duplicates(self, path, sample_size=1000, threshold=0.9):
        """Find potential duplicate images in the dataset."""
        from PIL import Image
        import numpy as np
        
        results = {
            "exact_duplicates": [],
            "similar_images": [],
            "analyzed_count": 0
        }
        
        # Find image files
        image_files = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for root, _, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        
        # Limit sample size for performance
        if sample_size and len(image_files) > sample_size:
            import random
            random.shuffle(image_files)
            image_files = image_files[:sample_size]
        
        results["analyzed_count"] = len(image_files)
        
        # Find exact duplicates based on file hash
        file_hashes = {}
        
        for file_path in image_files:
            try:
                # Calculate file hash
                import hashlib
                with open(file_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                # Check if hash exists
                if file_hash in file_hashes:
                    # Found an exact duplicate
                    results["exact_duplicates"].append({
                        "original": file_hashes[file_hash],
                        "duplicate": file_path
                    })
                else:
                    file_hashes[file_hash] = file_path
                    
            except Exception as e:
                print(f"Error hashing {file_path}: {str(e)}")
        
        # Find similar images using perceptual hash (if library available)
        try:
            import imagehash
            
            # Calculate perceptual hashes
            image_hashes = []
            
            for file_path in image_files:
                try:
                    img = Image.open(file_path)
                    phash = imagehash.phash(img)
                    image_hashes.append((file_path, phash))
                    img.close()
                except:
                    pass
            
            # Compare hashes
            for i in range(len(image_hashes)):
                for j in range(i+1, len(image_hashes)):
                    path1, hash1 = image_hashes[i]
                    path2, hash2 = image_hashes[j]
                    
                    # Calculate hash difference (0 = identical, higher = more different)
                    difference = hash1 - hash2
                    
                    # Convert to similarity score (1.0 = identical, 0.0 = completely different)
                    # A good threshold is around 0.9 (90% similar)
                    similarity = 1.0 - (difference / 64.0)  # 64 bits in the hash
                    
                    if similarity >= threshold:
                        results["similar_images"].append({
                            "image1": path1,
                            "image2": path2,
                            "similarity": float(similarity)
                        })
        except ImportError:
            # imagehash library not available
            results["similar_images"] = []
        
        return results


class DatasetManagerTab:
    """Dataset Manager tab for the main application."""
    
    def __init__(self, app):
        """
        Initialize the Dataset Manager tab.
        
        Args:
            app: The main application
        """
        self.app = app
        
        # Set up config directory
        config_dir = os.path.join(os.path.expanduser("~"), ".dataset_manager")
        os.makedirs(config_dir, exist_ok=True)
        self.app.config_dir = config_dir
        
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
        self.main_pane.add(self.explorer.parent, weight=1)  # Use explorer.parent for the frame
        
        # Create operations panel on the right
        self.operations_frame = ttk.Frame(self.main_pane)
        self.main_pane.add(self.operations_frame, weight=1)
        
        # Create operations UI
        self._create_operations_ui()
    
    def _create_operations_ui(self):
        """Create the UI for dataset operations."""
        # Tab control for different operation categories
        ops_notebook = ttk.Notebook(self.operations_frame)
        ops_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Split tab
        split_frame = ttk.Frame(ops_notebook, padding=10)
        ops_notebook.add(split_frame, text="Split Dataset")
        self._create_split_ui(split_frame)
        
        # Merge tab
        merge_frame = ttk.Frame(ops_notebook, padding=10)
        ops_notebook.add(merge_frame, text="Merge Datasets")
        self._create_merge_ui(merge_frame)
        
        # Filter tab
        filter_frame = ttk.Frame(ops_notebook, padding=10)
        ops_notebook.add(filter_frame, text="Filter Dataset")
        self._create_filter_ui(filter_frame)
        
        # Export tab
        export_frame = ttk.Frame(ops_notebook, padding=10)
        ops_notebook.add(export_frame, text="Export Dataset")
        self._create_export_ui(export_frame)
        
        # Analysis tab
        analysis_frame = ttk.Frame(ops_notebook, padding=10)
        ops_notebook.add(analysis_frame, text="Analysis")
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
        options_frame."""
Dataset Manager Module for Dataset Preparation Tool
Core components for managing multiple datasets and their versions.
"""

import os
import json
import shutil
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sqlite3
from PIL import Image, ImageTk

class DatasetRegistry:
    """Manages the dataset catalog and metadata storage."""
    
    def __init__(self, app):
        """
        Initialize the dataset registry.
        
        Args:
            app: The main application
        """
        self.app = app
        self.db_path = os.path.join(app.config_dir, "dataset_registry.db")
        self._initialize_database()
        
    def _initialize_database(self):
        """Create or connect to the SQLite database and initialize tables."""
        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create datasets table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            description TEXT,
            created_date TEXT NOT NULL,
            modified_date TEXT NOT NULL,
            category TEXT,
            file_count INTEGER DEFAULT 0,
            image_count INTEGER DEFAULT 0,
            video_count INTEGER DEFAULT 0,
            parent_id INTEGER,
            processing_history TEXT,
            attributes TEXT,
            FOREIGN KEY(parent_id) REFERENCES datasets(id)
        )
        ''')
        
        # Create tags table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
        ''')
        
        # Create dataset_tags mapping table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS dataset_tags (
            dataset_id INTEGER,
            tag_id INTEGER,
            PRIMARY KEY(dataset_id, tag_id),
            FOREIGN KEY(dataset_id) REFERENCES datasets(id),
            FOREIGN KEY(tag_id) REFERENCES tags(id)
        )
        ''')
        
        # Commit changes and close
        conn.commit()
        conn.close()
    
    def add_dataset(self, name, path, description="", category="", parent_id=None, attributes=None):
        """
        Add a new dataset to the registry.
        
        Args:
            name: Dataset name
            path: Path to dataset directory
            description: Optional description
            category: Optional category (e.g., "training", "validation")
            parent_id: ID of parent dataset (if this is a derived dataset)
            attributes: Additional attributes as a dictionary
            
        Returns:
            int: ID of the newly created dataset
        """
        # Normalize the path
        path = os.path.abspath(path)
        
        # Calculate file counts
        file_count, image_count, video_count = self._count_files(path)
        
        # Prepare dates
        now = datetime.datetime.now().isoformat()
        
        # Convert attributes to JSON
        attrs_json = json.dumps(attributes or {})
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert the dataset
        cursor.execute('''
        INSERT INTO datasets 
        (name, path, description, created_date, modified_date, category, 
         file_count, image_count, video_count, parent_id, attributes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, path, description, now, now, category, 
              file_count, image_count, video_count, parent_id, attrs_json))
        
        # Get the ID of the newly inserted dataset
        dataset_id = cursor.lastrowid
        
        # Commit changes and close
        conn.commit()
        conn.close()
        
        return dataset_id
    
    def update_dataset(self, dataset_id, **kwargs):
        """
        Update an existing dataset.
        
        Args:
            dataset_id: ID of the dataset to update
            **kwargs: Fields to update (name, path, description, etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if dataset exists
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return False
        
        # Get current values to update only what's necessary
        updates = {}
        for key, value in kwargs.items():
            if key in dataset and dataset[key] != value:
                updates[key] = value
        
        if not updates:
            return True  # Nothing to update
        
        # Add modified_date
        updates['modified_date'] = datetime.datetime.now().isoformat()
        
        # Special handling for attributes
        if 'attributes' in updates:
            updates['attributes'] = json.dumps(updates['attributes'])
        
        # Generate SQL update statement
        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        values = list(updates.values())
        values.append(dataset_id)
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update the dataset
        try:
            cursor.execute(f'''
            UPDATE datasets
            SET {set_clause}
            WHERE id = ?
            ''', values)
            conn.commit()
            success = True
        except Exception as e:
            print(f"Error updating dataset: {str(e)}")
            conn.rollback()
            success = False
        
        conn.close()
        return success
    
    def get_dataset(self, dataset_id):
        """
        Get a dataset by ID.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            dict: Dataset information or None if not found
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        # Query the dataset
        cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        row = cursor.fetchone()
        
        # Close connection
        conn.close()
        
        if not row:
            return None
            
        # Convert to dict and parse attributes
        dataset = dict(row)
        if dataset.get('attributes'):
            try:
                dataset['attributes'] = json.loads(dataset['attributes'])
            except:
                dataset['attributes'] = {}
                
        return dataset
    
    def get_datasets(self, parent_id=None, category=None, search_term=None):
        """
        Get datasets matching the specified criteria.
        
        Args:
            parent_id: Filter by parent dataset ID
            category: Filter by category
            search_term: Search in name and description
            
        Returns:
            list: List of matching datasets
        """
        # Build query conditions
        conditions = []
        params = []
        
        if parent_id is not None:
            conditions.append("parent_id = ?")
            params.append(parent_id)
        
        if category:
            conditions.append("category = ?")
            params.append(category)
        
        if search_term:
            conditions.append("(name LIKE ? OR description LIKE ?)")
            params.extend([f"%{search_term}%", f"%{search_term}%"])
        
        # Construct query
        query = "SELECT * FROM datasets"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Execute query
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Close connection
        conn.close()
        
        # Convert rows to dicts and parse attributes
        datasets = []
        for row in rows:
            dataset = dict(row)
            if dataset.get('attributes'):
                try:
                    dataset['attributes'] = json.loads(dataset['attributes'])
                except:
                    dataset['attributes'] = {}
            datasets.append(dataset)
            
        return datasets
    
    def delete_dataset(self, dataset_id, delete_files=False):
        """
        Delete a dataset from the registry.
        
        Args:
            dataset_id: ID of the dataset to delete
            delete_files: Whether to also delete the dataset files
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get dataset info first
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return False
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Begin transaction
            conn.execute("BEGIN")
            
            # Delete tags associations
            cursor.execute("DELETE FROM dataset_tags WHERE dataset_id = ?", (dataset_id,))
            
            # Delete the dataset
            cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            
            # Commit transaction
            conn.commit()
            
            # Delete files if requested
            if delete_files and os.path.exists(dataset['path']):
                shutil.rmtree(dataset['path'])
                
            success = True
        except Exception as e:
            print(f"Error deleting dataset: {str(e)}")
            conn.rollback()
            success = False
            
        conn.close()
        return success
    
    def add_tag(self, dataset_id, tag_name):
        """
        Add a tag to a dataset.
        
        Args:
            dataset_id: Dataset ID
            tag_name: Tag name
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not tag_name.strip():
            return False
            
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Begin transaction
            conn.execute("BEGIN")
            
            # Get or create tag
            cursor.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
            row = cursor.fetchone()
            
            if row:
                tag_id = row[0]
            else:
                cursor.execute("INSERT INTO tags (name) VALUES (?)", (tag_name,))
                tag_id = cursor.lastrowid
            
            # Add association (ignore if already exists)
            cursor.execute("""
            INSERT OR IGNORE INTO dataset_tags (dataset_id, tag_id)
            VALUES (?, ?)
            """, (dataset_id, tag_id))
            
            # Commit transaction
            conn.commit()
            success = True
        except Exception as e:
            print(f"Error adding tag: {str(e)}")
            conn.rollback()
            success = False
            
        conn.close()
        return success
    
    def remove_tag(self, dataset_id, tag_name):
        """
        Remove a tag from a dataset.
        
        Args:
            dataset_id: Dataset ID
            tag_name: Tag name
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get tag ID
            cursor.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
            row = cursor.fetchone()
            if not row:
                return False
                
            tag_id = row[0]
            
            # Remove association
            cursor.execute("""
            DELETE FROM dataset_tags
            WHERE dataset_id = ? AND tag_id = ?
            """, (dataset_id, tag_id))
            
            conn.commit()
            success = True
        except Exception as e:
            print(f"Error removing tag: {str(e)}")
            conn.rollback()
            success = False
            
        conn.close()
        return success
    
    def get_tags(self, dataset_id):
        """
        Get all tags for a dataset.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            list: List of tag names
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Query tags
        cursor.execute("""
        SELECT t.name
        FROM tags t
        JOIN dataset_tags dt ON t.id = dt.tag_id
        WHERE dt.dataset_id = ?
        ORDER BY t.name
        """, (dataset_id,))
        
        tags = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return tags
    
    def get_all_tags(self):
        """
        Get all available tags.
        
        Returns:
            list: List of tag names
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Query all tags
        cursor.execute("SELECT name FROM tags ORDER BY name")
        tags = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return tags
    
    def _count_files(self, path):
        """
        Count files in a directory, categorizing by type.
        
        Args:
            path: Directory path
            
        Returns:
            tuple: (total_count, image_count, video_count)
        """
        if not os.path.isdir(path):
            return 0, 0, 0
            
        total_count = 0
        image_count = 0
        video_count = 0
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
        
        for root, _, files in os.walk(path):
            for file in files:
                total_count += 1
                ext = os.path.splitext(file)[1].lower()
                
                if ext in image_extensions:
                    image_count += 1
                elif ext in video_extensions:
                    video_count += 1
        
        return total_count, image_count, video_count
    
    def record_processing_step(self, dataset_id, operation, params=None):
        """
        Record a processing step in the dataset's history.
        
        Args:
            dataset_id: Dataset ID
            operation: Name of the operation
            params: Parameters used in the operation
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get current dataset
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return False
        
        # Create history entry
        history_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "operation": operation,
            "params": params or {}
        }
        
        # Parse existing history
        history = []
        if dataset.get('processing_history'):
            try:
                history = json.loads(dataset['processing_history'])
            except:
                history = []
        
        # Add new entry
        history.append(history_entry)
        
        # Update dataset
        return self.update_dataset(
            dataset_id, 
            processing_history=json.dumps(history)
        )


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


class DatasetOperations:
    """Handles operations that can be performed on datasets."""
    
    def __init__(self, registry):
        """
        Initialize dataset operations.
        
        Args:
            registry: DatasetRegistry instance
        """
        self.registry = registry
    
    def split_dataset(self, dataset_id, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                    random_seed=42, stratify_by=None):
        """
        Split a dataset into training, validation, and test sets.
        
        Args:
            dataset_id: ID of the dataset to split
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_seed: Random seed for reproducibility
            stratify_by: Optional folder name to stratify by (assumes class folders)
            
        Returns:
            tuple: (train_dataset_id, val_dataset_id, test_dataset_id) or None if failed
        """
        # Verify ratios sum to approximately 1
        if not 0.99 <= train_ratio + val_ratio + test_ratio <= 1.01:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Get dataset info
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            return None
            
        source_path = dataset['path']
        if not os.path.isdir(source_path):
            return None
            
        # Create output directories
        dataset_name = dataset['name']
        parent_dir = os.path.dirname(source_path)
        
        train_dir = os.path.join(parent_dir, f"{dataset_name}_train")
        val_dir = os.path.join(parent_dir, f"{dataset_name}_val")
        test_dir = os.path.join(parent_dir, f"{dataset_name}_test")
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Configure random number generator
        import random
        random.seed(random_seed)
        
        # Get list of files or folders (depending on stratification)
        if stratify_by:
            # Stratified split by folder
            return self._split_stratified(
                dataset_id, source_path, train_dir, val_dir, test_dir,
                train_ratio, val_ratio, test_ratio
            )
        else:
            # Simple random split
            return self._split_random(
                dataset_id, source_path, train_dir, val_dir, test_dir,
                train_ratio, val_ratio, test_ratio
            )
    
    def _split_random(self, dataset_id, source_path, train_dir, val_dir, test_dir,
                    train_ratio, val_ratio, test_ratio):
        """Perform a random split of the dataset."""
        # Get all image files
        image_files = []
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    src_path = os.path.join(root, file)
                    # Create relative path from source_path
                    rel_path = os.path.relpath(src_path, source_path)
                    image_files.append(rel_path)
        
        if not image_files:
            return None
            
        # Shuffle files
        import random
        random.shuffle(image_files)
        
        # Split files according to ratios
        n_files = len(image_files)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train+n_val]
        test_files = image_files[n_train+n_val:]
        
        # Copy files to respective directories
        self._copy_files(source_path, train_dir, train_files)
        self._copy_files(source_path, val_dir, val_files)
        self._copy_files(source_path, test_dir, test_files)
        
        # Register new datasets
        dataset = self.registry.get_dataset(dataset_id)
        train_id = self.registry.add_dataset(
            name=f"{dataset['name']}_train",
            path=train_dir,
            description=f"Training split of {dataset['name']}",
            category="Training",
            parent_id=dataset_id,
            attributes={"split_type": "train", "split_ratio": train_ratio}
        )
        
        val_id = self.registry.add_dataset(
            name=f"{dataset['name']}_val",
            path=val_dir,
            description=f"Validation split of {dataset['name']}",
            category="Validation",
            parent_id=dataset_id,
            attributes={"split_type": "val", "split_ratio": val_ratio}
        )
        
        test_id = self.registry.add_dataset(
            name=f"{dataset['name']}_test",
            path=test_dir,
            description=f"Test split of {dataset['name']}",
            category="Testing",
            parent_id=dataset_id,
            attributes={"split_type": "test", "split_ratio": test_ratio}
        )
        
        # Add processing record to parent dataset
        self.registry.record_processing_step(
            dataset_id,
            "dataset_split",
            {
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "stratify": False,
                "train_count": len(train_files),
                "val_count": len(val_files),
                "test_count": len(test_files)
            }
        )
        
        return (train_id, val_id, test_id)
    
    def _split_stratified(self, dataset_id, source_path, train_dir, val_dir, test_dir,
                        train_ratio, val_ratio, test_ratio):
        """Perform a stratified split preserving class distribution."""
        # Get all subfolders (assuming they are class folders)
        import os
        import random
        import shutil
        
        class_folders = []
        for item in os.listdir(source_path):
            item_path = os.path.join(source_path, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                class_folders.append(item)
        
        if not class_folders:
            # No class folders found, fall back to random split
            return self._split_random(
                dataset_id, source_path, train_dir, val_dir, test_dir,
                train_ratio, val_ratio, test_ratio
            )
        
        # Process each class folder and maintain the same split ratio
        for class_folder in class_folders:
            # Create output class folders
            train_class_dir = os.path.join(train_dir, class_folder)
            val_class_dir = os.path.join(val_dir, class_folder)
            test_class_dir = os.path.join(test_dir, class_folder)
            
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)
            
            # Get all images in this class
            class_path = os.path.join(source_path, class_folder)
            class_files = []
            
            for file in os.listdir(class_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    class_files.append(file)
            
            # Shuffle files
            random.shuffle(class_files)
            
            # Split files according to ratios
            n_files = len(class_files)
            n_train = int(n_files * train_ratio)
            n_val = int(n_files * val_ratio)
            
            train_files = class_files[:n_train]
            val_files = class_files[n_train:n_train+n_val]
            test_files = class_files[n_train+n_val:]
            
            # Copy files to respective directories
            for file in train_files:
                src = os.path.join(class_path, file)
                dst = os.path.join(train_class_dir, file)
                shutil.copy2(src, dst)
            
            for file in val_files:
                src = os.path.join(class_path, file)
                dst = os.path.join(val_class_dir, file)
                shutil.copy2(src, dst)
            
            for file in test_files:
                src = os.path.join(class_path, file)
                dst = os.path.join(test_class_dir, file)
                shutil.copy2(src, dst)
        
        # Register new datasets
        dataset = self.registry.get_dataset(dataset_id)
        train_id = self.registry.add_dataset(
            name=f"{dataset['name']}_train",
            path=train_dir,
            description=f"Training split of {dataset['name']} (stratified)",
            category="Training",
            parent_id=dataset_id,
            attributes={"split_type": "train", "split_ratio": train_ratio, "stratified": True}
        )
        
        val_id = self.registry.add_dataset(
            name=f"{dataset['name']}_val",
            path=val_dir,
            description=f"Validation split of {dataset['name']} (stratified)",
            category="Validation",
            parent_id=dataset_id,
            attributes={"split_type": "val", "split_ratio": val_ratio, "stratified": True}
        )
        
        test_id = self.registry.add_dataset(
            name=f"{dataset['name']}_test",
            path=test_dir,
            description=f"Test split of {dataset['name']} (stratified)",
            category="Testing",
            parent_id=dataset_id,
            attributes={"split_type": "test", "split_ratio": test_ratio, "stratified": True}
        )
        
        # Add processing record to parent dataset
        self.registry.record_processing_step(
            dataset_id,
            "dataset_split",
            {
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "stratify": True,
                "class_count": len(class_folders)
            }
        )
        
        return (train_id, val_id, test_id)
    
    def _copy_files(self, source_base, dest_base, file_list):
        """Copy files from source to destination, preserving directory structure."""
        for rel_path in file_list:
            source_path = os.path.join(source_base, rel_path)
            dest_path = os.path.join(dest_base, rel_path)
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Copy the file
            shutil.copy2(source_path, dest_path)
    
    def merge_datasets(self, dataset_ids, merged_name=None, merge_method="copy"):
        """
        Merge multiple datasets into a new dataset.
        
        Args:
            dataset_ids: List of dataset IDs to merge
            merged_name: Name for the merged dataset (default: auto-generated)
            merge_method: Method to use ("copy" or "link")
            
        Returns:
            int: ID of the merged dataset or None if failed
        """
        if not dataset_ids or len(dataset_ids) < 2:
            return None
            
        # Get all datasets
        datasets = []
        for dataset_id in dataset_ids:
            dataset = self.registry.get_dataset(dataset_id)
            if dataset and os.path.isdir(dataset['path']):
                datasets.append(dataset)
        
        if len(datasets) < 2:
            return None
            
        # Create merged name if not provided
        if not merged_name:
            merged_name = f"Merged_{len(datasets)}_datasets"
            
        # Create output directory
        parent_dir = os.path.dirname(datasets[0]['path'])
        merged_dir = os.path.join(parent_dir, merged_name)
        
        if os.path.exists(merged_dir):
            # Add timestamp to make unique
            import time
            timestamp = int(time.time())
            merged_dir = f"{merged_dir}_{timestamp}"
            
        os.makedirs(merged_dir, exist_ok=True)
        
        # Copy or link files from each dataset
        for dataset in datasets:
            source_path = dataset['path']
            
            for root, dirs, files in os.walk(source_path):
                # Create relative path from source
                rel_path = os.path.relpath(root, source_path)
                
                # Create corresponding directory in merged_dir
                if rel_path != '.':
                    dest_dir = os.path.join(merged_dir, rel_path)
                    os.makedirs(dest_dir, exist_ok=True)
                else:
                    dest_dir = merged_dir
                
                # Process each file
                for file in files:
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(dest_dir, file)
                    
                    if merge_method == "copy":
                        shutil.copy2(src_file, dst_file)
                    elif merge_method == "link":
                        # Create a symbolic link instead of copying
                        if os.path.exists(dst_file):
                            os.remove(dst_file)
                        os.symlink(os.path.abspath(src_file), dst_file)
                    else:
                        # Invalid method
                        shutil.rmtree(merged_dir)
                        return None
        
        # Create the merged dataset
        merged_id = self.registry.add_dataset(
            name=merged_name,
            path=merged_dir,
            description=f"Merged dataset from {len(datasets)} source datasets",
            category="Merged",
            attributes={"source_datasets": [d['id'] for d in datasets]}
        )
        
        # Record the merge operation in each source dataset
        for dataset in datasets:
            self.registry.record_processing_step(
                dataset['id'],
                "dataset_merge",
                {"merged_dataset_id": merged_id, "merge_method": merge_method}
            )
        
        return merged_id
    
    def filter_dataset(self, dataset_id, filter_criteria, output_name=None):
        """
        Create a new dataset by filtering an existing one.
        
        Args:
            dataset_id: Source dataset ID
            filter_criteria: Dict of filter criteria (e.g., {"min_width": 800, "extension": ".jpg"})
            output_name: Name for the filtered dataset
            
        Returns:
            int: ID of the filtered dataset or None if failed
        """
        # Get source dataset
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset or not os.path.isdir(dataset['path']):
            return None
            
        # Create output name if not provided
        if not output_name:
            output_name = f"{dataset['name']}_filtered"
            
        # Create output directory
        parent_dir = os.path.dirname(dataset['path'])
        output_dir = os.path.join(parent_dir, output_name)
        
        if os.path.exists(output_dir):
            # Add timestamp to make unique
            import time
            timestamp = int(time.time())
            output_dir = f"{output_dir}_{timestamp}"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Apply filters
        source_path = dataset['path']
        filtered_files = []
        
        for root, _, files in os.walk(source_path):
            for file in files:
                src_file = os.path.join(root, file)
                
                # Apply filters
                if self._apply_filters(src_file, filter_criteria):
                    # File passes all filters
                    rel_path = os.path.relpath(root, source_path)
                    if rel_path != '.':
                        dest_dir = os.path.join(output_dir, rel_path)
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_file = os.path.join(dest_dir, file)
                    else:
                        dest_file = os.path.join(output_dir, file)
                    
                    # Copy the file
                    shutil.copy2(src_file, dest_file)
                    filtered_files.append(dest_file)
        
        if not filtered_files:
            # No files passed the filters
            shutil.rmtree(output_dir)
            return None
        
        # Create the filtered dataset
        filtered_id = self.registry.add_dataset(
            name=output_name,
            path=output_dir,
            description=f"Filtered version of {dataset['name']}",
            category=dataset['category'],
            parent_id=dataset_id,
            attributes={"filter_criteria": filter_criteria}
        )
        
        # Record the filter operation
        self.registry.record_processing_step(
            dataset_id,
            "dataset_filter",
            {
                "filter_criteria": filter_criteria,
                "filtered_dataset_id": filtered_id,
                "filtered_file_count": len(filtered_files)
            }
        )
        
        return filtered_id
    
    def _apply_filters(self, file_path, criteria):
        """Apply filter criteria to a file."""
        # Check file extension
        if 'extension' in criteria:
            ext = os.path.splitext(file_path)[1].lower()
            if isinstance(criteria['extension'], list):
                if ext not in criteria['extension']:
                    return False
            elif ext != criteria['extension'].lower():
                return False
        
        # Check file size
        if 'min_size' in criteria or 'max_size' in criteria:
            file_size = os.path.getsize(file_path)
            
            if 'min_size' in criteria and file_size < criteria['min_size']:
                return False
                
            if 'max_size' in criteria and file_size > criteria['max_size']:
                return False
        
        # Check image dimensions
        if any(k in criteria for k in ['min_width', 'min_height', 'max_width', 'max_height']):
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    width, height = img.size
                    
                    if 'min_width' in criteria and width < criteria['min_width']:
                        return False
                        
                    if 'min_height' in criteria and height < criteria['min_height']:
                        return False
                        
                    if 'max_width' in criteria and width > criteria['max_width']:
                        return False
                        
                    if 'max_height' in criteria and height > criteria['max_height']:
                        return False
            except:
                # Not an image or error opening it
                return False
        
        # Check aspect ratio
        if 'min_aspect_ratio' in criteria or 'max_aspect_ratio' in criteria:
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    width, height = img.size
                    aspect_ratio = width / height
                    
                    if 'min_aspect_ratio' in criteria and aspect_ratio < criteria['min_aspect_ratio']:
                        return False
                        
                    if 'max_aspect_ratio' in criteria and aspect_ratio > criteria['max_aspect_ratio']:
                        return False
            except:
                return False
        
        # All filters passed
        return True
    
    def export_dataset(self, dataset_id, format_type, output_path=None):
        """
        Export a dataset to a specified format.
        
        Args:
            dataset_id: Source dataset ID
            format_type: Format type ("coco", "yolo", "voc", "csv")
            output_path: Output path (default: auto-generated)
            
        Returns:
            str: Path to the exported dataset or None if failed
        """
        # Get source dataset
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset or not os.path.isdir(dataset['path']):
            return None
            
        # Create output path if not provided
        if not output_path:
            parent_dir = os.path.dirname(dataset['path'])
            output_path = os.path.join(parent_dir, f"{dataset['name']}_{format_type}")
            
        os.makedirs(output_path, exist_ok=True)
        
        # Export based on format type
        try:
            if format_type.lower() == "csv":
                self._export_to_csv(dataset, output_path)
            elif format_type.lower() == "coco":
                self._export_to_coco(dataset, output_path)
            elif format_type.lower() == "yolo":
                self._export_to_yolo(dataset, output_path)
            elif format_type.lower() == "voc":
                self._export_to_voc(dataset, output_path)
            else:
                # Unsupported format
                return None
                
            # Record the export operation
            self.registry.record_processing_step(
                dataset_id,
                "dataset_export",
                {"format": format_type, "output_path": output_path}
            )
            
            return output_path
            
        except Exception as e:
            print(f"Export error: {str(e)}")
            return None
    
    def _export_to_csv(self, dataset, output_path):
        """Export dataset to CSV format."""
        import csv
        
        # Create CSV file
        csv_path = os.path.join(output_path, "dataset_inventory.csv")
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['file_path', 'width', 'height', 'format', 'size_bytes', 'folder'])
            
            # Process all files
            source_path = dataset['path']
            for root, _, files in os.walk(source_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, source_path)
                        folder = os.path.relpath(root, source_path)
                        
                        try:
                            # Get file info
                            size_bytes = os.path.getsize(file_path)
                            
                            # Get image dimensions
                            from PIL import Image
                            with Image.open(file_path) as img:
                                width, height = img.size
                                format_name = img.format
                                
                            # Write to CSV
                            writer.writerow([rel_path, width, height, format_name, size_bytes, folder])
                            
                        except Exception as e:
                            print(f"Error processing {file_path}: {str(e)}")
    
    def _export_to_coco(self, dataset, output_path):
        """Export dataset to COCO format (simplified, without annotations)."""
        import json
        import datetime
        
        # Create COCO dataset structure
        coco_data = {
            "info": {
                "description": dataset['description'] or "Exported dataset",
                "url": "",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "contributor": "Dataset Manager",
                "date_created": datetime.datetime.now().isoformat()
            },
            "licenses": [
                {
                    "url": "https://creativecommons.org/licenses/by/4.0/",
                    "id": 1,
                    "name": "Attribution 4.0 International (CC BY 4.0)"
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Process all image files
        source_path = dataset['path']
        image_id = 0
        
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, source_path)
                    
                    try:
                        # Get image dimensions
                        from PIL import Image
                        with Image.open(file_path) as img:
                            width, height = img.size
                        
                        # Add image to COCO format
                        coco_data["images"].append({
                            "id": image_id,
                            "width": width,
                            "height": height,
                            "file_name": rel_path,
                            "license": 1,
                            "date_captured": datetime.datetime.fromtimestamp(
                                os.path.getmtime(file_path)
                            ).isoformat()
                        })
                        
                        # Copy image to output directory
                        target_dir = os.path.join(output_path, os.path.dirname(rel_path))
                        os.makedirs(target_dir, exist_ok=True)
                        shutil.copy2(file_path, os.path.join(output_path, rel_path))
                        
                        image_id += 1
                        
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
        
        # Write COCO JSON file
        with open(os.path.join(output_path, "annotations.json"), 'w') as f:
            json.dump(coco_data, f, indent=2)
    
    def _export_to_yolo(self, dataset, output_path):
        """Export dataset to YOLO format (simplified, without annotations)."""
        # Create directory structure
        images_dir = os.path.join(output_path, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Create empty labels directory
        labels_dir = os.path.join(output_path, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        
        # Create dataset.yaml
        yaml_content = f"""
# YOLO dataset configuration
path: {output_path}
train: images/train
val: images/val
test: images/test

# Classes
nc: 0  # number of classes
names: []  # class names
"""
        
        with open(os.path.join(output_path, "dataset.yaml"), 'w') as f:
            f.write(yaml_content)
        
        # Create train/val/test splits
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(images_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
        
        # Process all image files (put all in train for now)
        source_path = dataset['path']
        
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    file_path = os.path.join(root, file)
                    
                    # Copy to train directory
                    shutil.copy2(file_path, os.path.join(images_dir, "train", file))
    
    def _export_to_voc(self, dataset, output_path):
        """Export dataset to Pascal VOC format (simplified, without annotations)."""
        # Create directory structure
        images_dir = os.path.join(output_path, "JPEGImages")
        os.makedirs(images_dir, exist_ok=True)
        
        annotations_dir = os.path.join(output_path, "Annotations")
        os.makedirs(annotations_dir, exist_ok=True)
        
        imagesets_dir = os.path.join(output_path, "ImageSets", "Main")
        os.makedirs(imagesets_dir, exist_ok=True)
        
        # Process all image files
        source_path = dataset['path']
        image_ids = []
        
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    file_path = os.path.join(root, file)
                    
                    # Generate ID from filename (without extension)
                    image_id = os.path.splitext(file)[0]
                    image_ids.append(image_id)
                    
                    # Copy image to JPEGImages directory
                    target_path = os.path.join(images_dir, f"{image_id}.jpg")
                    
                    # Convert to JPG if needed
                    if file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg'):
                        shutil.copy2(file_path, target_path)
                    else:
                        try:
                            from PIL import Image
                            img = Image.open(file_path)
                            rgb_img = img.convert('RGB')
                            rgb_img.save(target_path)
                        except Exception as e:
                            print(f"Error converting {file_path}: {str(e)}")
                            continue
                    
                    # Create empty XML annotation
                    xml_path = os.path.join(annotations_dir, f"{image_id}.xml")
                    
                    # Get image dimensions
                    try:
                        from PIL import Image
                        with Image.open(file_path) as img:
                            width, height = img.size
                    except:
                        width, height = 0, 0
                    
                    # Create basic XML structure
                    xml_content = f"""
<annotation>
    <folder>JPEGImages</folder>
    <filename>{image_id}.jpg</filename>
    <path>{target_path}</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>{width}</width>
        <height>{height}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
</annotation>
"""
                    with open(xml_path, 'w') as f:
                        f.write(xml_content)
        
        # Create ImageSets files
        import random
        random.shuffle(image_ids)
        
        # Create train/val/test splits
        train_size = int(len(image_ids)



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
                    ttk.Separator(similar_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
    
    def _open_file(self, path):
        """Open a file with the default system application."""
        if not os.path.exists(path):
            messagebox.showerror("Error", f"File not found: {path}")
            return
            
        # Platform-specific file opening
        import platform
        import subprocess
        
        try:
            if platform.system() == 'Windows':
                os.startfile(path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(['open', path])
            else:  # Linux
                subprocess.call(['xdg-open', path])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file: {str(e)}")


def initialize_dataset_manager(app):
    """
    Initialize the Dataset Manager and add it to the application.
    
    Args:
        app: The main application instance
    """
    # Create the Dataset Manager tab
    dataset_manager = DatasetManagerTab(app)
    
    # Add the tab to the notebook
    app.notebook.add(dataset_manager.frame, text="Dataset Manager")
    
    # Store the dataset manager instance on the app for later access
    app.dataset_manager = dataset_manager
    
    # Add to menu
    tools_menu = None
    
    # Find or create Tools menu
    for i in range(app.menu_bar.index('end') + 1):
        if app.menu_bar.type(i) == 'cascade' and app.menu_bar.entrycget(i, 'label') == 'Tools':
            tools_menu = app.menu_bar.nametowidget(app.menu_bar.entrycget(i, 'menu'))
            break
    
    if not tools_menu:
        tools_menu = tk.Menu(app.menu_bar, tearoff=0)
        app.menu_bar.add_cascade(label="Tools", menu=tools_menu)
    
    # Add Dataset Manager commands to Tools menu
    tools_menu.add_separator()
    tools_menu.add_command(label="Dataset Manager", command=lambda: app.notebook.select(app.notebook.index(dataset_manager.frame)))
    tools_menu.add_command(label="Add Current Directory to Datasets", command=lambda: _add_current_directory(app))
    
    # Add help information
    help_menu = None
    for i in range(app.menu_bar.index('end') + 1):
        if app.menu_bar.type(i) == 'cascade' and app.menu_bar.entrycget(i, 'label') == 'Help':
            help_menu = app.menu_bar.nametowidget(app.menu_bar.entrycget(i, 'menu'))
            break
    
    if help_menu:
        help_menu.add_command(label="Dataset Manager Help", command=lambda: _show_dataset_manager_help(app))


def _add_current_directory(app):
    """Add the current input directory to the dataset registry."""
    input_dir = app.input_dir.get()
    
    if not input_dir or not os.path.isdir(input_dir):
        messagebox.showerror("Error", "Please select a valid input directory first.")
        return
    
    # Ask for dataset name
    from tkinter.simpledialog import askstring
    
    name = askstring("Dataset Name", "Enter a name for this dataset:", initialvalue=os.path.basename(input_dir))
    
    if not name:
        return
        
    # Add to registry
    try:
        dataset_id = app.dataset_manager.registry.add_dataset(
            name=name,
            path=input_dir,
            description=f"Added from input directory",
            category="Input"
        )
        
        app.dataset_manager.explorer.refresh_datasets()
        
        # Switch to Dataset Manager tab
        app.notebook.select(app.notebook.index(app.dataset_manager.frame))
        
        # Select the new dataset
        app.dataset_manager.explorer._select_dataset_by_id(dataset_id)
        
        messagebox.showinfo("Success", f"Added '{name}' to datasets.")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to add dataset: {str(e)}")


def _show_dataset_manager_help(app):
    """Show help information for the Dataset Manager."""
    help_dialog = tk.Toplevel(app.root)
    help_dialog.title("Dataset Manager Help")
    help_dialog.geometry("700x500")
    help_dialog.transient(app.root)
    
    # Create scrollable text widget
    frame = ttk.Frame(help_dialog, padding=10)
    frame.pack(fill=tk.BOTH, expand=True)
    
    text_frame = ttk.Frame(frame)
    text_frame.pack(fill=tk.BOTH, expand=True)
    
    text = tk.Text(text_frame, wrap=tk.WORD, padx=10, pady=10)
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text.configure(yscrollcommand=scrollbar.set)
    
    # Help content
    help_text = """Dataset Manager Help

The Dataset Manager helps you organize, track, and manipulate multiple datasets for your AI training needs.

Core Features:

1. Dataset Registry
   • Keep track of all your datasets in one place
   • Store metadata like creation date, purpose, file counts
   • Track relationships between original datasets and processed versions
   • Add tags for easy filtering and categorization

2. Dataset Operations
   • Split datasets into training, validation, and test sets
   • Merge multiple datasets into a single dataset
   • Filter datasets based on criteria like image size or aspect ratio
   • Export datasets to standard formats (CSV, COCO, YOLO, Pascal VOC)

3. Dataset Analysis
   • Get comprehensive statistics about your datasets
   • Analyze image properties like dimensions and color modes
   • Find quality issues like corrupt or extremely dark images
   • Detect duplicate or very similar images

Getting Started:

1. Add Datasets
   • Click "Add Dataset" in the explorer panel
   • Enter a name and select the directory containing your dataset
   • Optionally add a description, category, and tags

2. View Dataset Details
   • Select a dataset in the explorer to view its details
   • See statistics, preview images, and available actions
   • Tags and descriptions help you keep track of dataset purpose

3. Perform Operations
   • Use the operations tabs to perform tasks on selected datasets
   • Results are automatically added to the registry as new datasets
   • Operations maintain parent-child relationships for traceability

Best Practices:

• Use descriptive names and add tags to datasets
• Add datasets immediately after processing to maintain history
• Use the analysis tools before training to catch issues early
• Split datasets strategically for optimal model training
• Export datasets in formats compatible with your training tools

The Dataset Manager integrates with the rest of the Dataset Preparation Tool,
making it easy to process files and then catalog the results for future use.
"""

    text.insert(tk.END, help_text)
    text.config(state=tk.DISABLED)  # Make read-only
    
    # Close button
    close_btn = ttk.Button(frame, text="Close", command=help_dialog.destroy)
    close_btn.pack(pady=10)        # Create train/val/test splits
        train_size = int(len(image_ids) * 0.8)
        val_size = int(len(image_ids) * 0.1)
        
        train_ids = image_ids[:train_size]
        val_ids = image_ids[train_size:train_size+val_size]
        test_ids = image_ids[train_size+val_size:]
        
        # Write split files
        with open(os.path.join(imagesets_dir, "train.txt"), 'w') as f:
            f.write("\n".join(train_ids))
            
        with open(os.path.join(imagesets_dir, "val.txt"), 'w') as f:
            f.write("\n".join(val_ids))
            
        with open(os.path.join(imagesets_dir, "test.txt"), 'w') as f:
            f.write("\n".join(test_ids))
            
        with open(os.path.join(imagesets_dir, "trainval.txt"), 'w') as f:
            f.write("\n".join(train_ids + val_ids))


