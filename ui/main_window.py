"""
Main Window UI Component for Dataset Preparation Tool
Handles the main application window setup and tabs configuration.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

from ui.tabs.input_output_tab import InputOutputTab
from ui.tabs.config_tab import ConfigTab
from ui.tabs.preview_tab import PreviewTab
from ui.tabs.gallery_tab import GalleryTab 
from ui.dialogs import AboutDialog, UsageGuideDialog
from utils.config_manager import ConfigManager
from processors.mask_expander import MaskExpander
from utils.dataset_manager import (
    DatasetRegistry, 
    DatasetExplorer, 
    DatasetOperations, 
    DatasetAnalyzer,
    DatasetManagerTab
)

class MainWindow:
    def __init__(self, root):
        # Set config directory early
        self.config_dir = os.path.join(os.path.expanduser("~"), ".dataset_preparation_tool")
        os.makedirs(self.config_dir, exist_ok=True)

        self.root = root
        
        # Initialize variables FIRST
        self._init_variables()
        
        # Create the menu bar EARLY in the initialization
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)
        
        # Initialize config manager
        from utils.config_manager import ConfigManager
        self.config_manager = ConfigManager(self)
        
        # Create main content and status bar 
        self._create_main_content()
        self._create_status_bar()
        
        # Create the notebook and tabs
        self._create_notebook()
        
        # Create the menu 
        self._create_menu()
        
        # Other initializations
        self._init_image_comparison()
        self._init_outlier_detection()
        
        # Create Dataset Manager Tab
        self.dataset_manager = DatasetManagerTab(self)
        self.notebook.add(self.dataset_manager.frame, text="Dataset Manager")
# Update this method in ui/main_window.py

    def _init_variables(self):
        """Initialize all Tkinter variables used in the application."""
        # Directory paths
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        
        # Frame extraction options
        self.frame_rate = tk.DoubleVar(value=30.0)
        self.video_fps = tk.DoubleVar(value=30.0)
        
        # Mask processing options
        self.fill_ratio = tk.DoubleVar(value=10.0)
        self.use_mask_video = tk.BooleanVar(value=False)
        self.mask_video_path = tk.StringVar()
        
        # Resize options
        self.output_width = tk.IntVar(value=512)
        self.output_height = tk.IntVar(value=512)
        self.use_source_resolution = tk.BooleanVar(value=True)
        self.resize_if_larger = tk.BooleanVar(value=False)
        self.max_width = tk.IntVar(value=1536)
        self.max_height = tk.IntVar(value=1536)
        
        # Square padding options
        self.square_pad_images = tk.BooleanVar(value=False)
        self.padding_color = tk.StringVar(value="black")
        self.use_source_resolution_padding = tk.BooleanVar(value=True)
        self.square_target_size = tk.IntVar(value=512)
        
        # Portrait crop options
        self.portrait_crop_enabled = tk.BooleanVar(value=False)
        self.portrait_crop_position = tk.StringVar(value="center")
        
        # Output options
        self.naming_pattern = tk.StringVar(value="{index:04d}.png")
        
        # Processing flags
        self.extract_frames = tk.BooleanVar(value=False)
        self.crop_mask_regions = tk.BooleanVar(value=False)
        self.resize_images = tk.BooleanVar(value=False)
        self.organize_files = tk.BooleanVar(value=False)
        self.convert_to_video = tk.BooleanVar(value=False)
        self.square_pad_images = tk.BooleanVar(value=False)
        self.reinsert_crops_option = tk.BooleanVar(value=False)  # Make sure this exists
        self.debug_mode = tk.BooleanVar(value=False)

        # Make sure expand_masks variable is defined here
        self.expand_masks = tk.BooleanVar(value=False)

        # Crop reinsertion options
        self.source_images_dir = tk.StringVar()
        self.reinsert_match_method = tk.StringVar(value="name_prefix")
        self.reinsert_padding = tk.IntVar(value=10)  # Default padding percent
        self.use_center_position = tk.BooleanVar(value=True)  # Default to auto-center
        self.reinsert_x = tk.IntVar(value=0)
        self.reinsert_y = tk.IntVar(value=0)
        self.reinsert_width = tk.IntVar(value=0)
        self.reinsert_height = tk.IntVar(value=0)
    
        # Add this new variable
        self.use_enhanced_reinserter = tk.BooleanVar(value=True)

        # New mask alignment options
        self.reinsert_handle_different_masks = tk.BooleanVar(value=False)
        self.reinsert_alignment_method = tk.StringVar(value="centroid")
        self.reinsert_blend_mode = tk.StringVar(value="alpha")
        self.reinsert_blend_extent = tk.IntVar(value=5)
        self.reinsert_preserve_edges = tk.BooleanVar(value=True)
        
        # Add these variables
        self.reinsert_manual_offset_x = tk.IntVar(value=0)
        self.reinsert_manual_offset_y = tk.IntVar(value=0)
        # Add these variables
        self.reinsert_manual_scale_x = tk.DoubleVar(value=1.0)
        self.reinsert_manual_scale_y = tk.DoubleVar(value=1.0)



        # Add bangs extension options
        self.extend_bangs = tk.BooleanVar(value=False)
        self.bangs_extension_amount = tk.IntVar(value=30)
        self.bangs_width_ratio = tk.DoubleVar(value=0.3)

        # Mask expansion options
        self.mask_expand_iterations = tk.IntVar(value=5)
        self.mask_expand_kernel_size = tk.IntVar(value=3)
        self.mask_expand_preserve_structure = tk.BooleanVar(value=True)
        # Add new variable for mask-only reinsertion
        self.reinsert_mask_only = tk.BooleanVar(value=False)
        # Preview image storage
        self.preview_image = None
        self.preview_mask = None
    
    def _create_menu(self):
        # Clear any existing menus
        for i in range(self.menu_bar.index('end') + 1):
            self.menu_bar.delete(0)
        
        # File menu
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Configuration", command=self.config_manager.save_config)
        file_menu.add_command(label="Load Configuration", command=self.config_manager.load_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Tools", menu=tools_menu)
        
        # Add Dataset Manager to Tools menu
        tools_menu.add_command(
            label="Dataset Manager", 
            command=lambda: self.notebook.select(self.notebook.index(self.dataset_manager.frame))
        )
        
        # Outlier detection option
        tools_menu.add_command(
            label="Find Outlier Groups...", 
            command=self._run_outlier_detection
        )
        
        # Help menu
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="Usage Guide", command=self._show_usage_guide)

    

    def _add_integration_hooks(self):
        # Add button to Input/Output tab
        add_dataset_btn = ttk.Button(
            self.input_output_tab.frame,  # Use .frame to access the actual frame 
            text="Add to Datasets", 
            command=self._add_current_directory_to_datasets
        )
        add_dataset_btn.pack(side=tk.LEFT, padx=5)

    def _add_current_directory_to_datasets(self):
        input_dir = self.input_dir.get()
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("Error", "Please select a valid input directory first.")
            return
        
        # Add to dataset registry
        dataset_id = self.dataset_manager.registry.add_dataset(
            name=os.path.basename(input_dir),
            path=input_dir,
            description="Added from input directory",
            category="Input"
        )
        
        # Refresh dataset explorer
        self.dataset_manager.explorer.refresh_datasets()
        
        # Switch to Dataset Manager tab
        self.notebook.select(self.notebook.index(self.dataset_manager.frame))


    def _run_outlier_detection(self):
        """Run the outlier detection feature."""
        from utils.outlier_detection import OutlierGroupDetector, run_outlier_scan
        
        # Create detector instance
        detector = OutlierGroupDetector(self.gallery_tab)
        
        # Run the outlier scan
        run_outlier_scan(self, detector)



    def _create_main_content(self):
        """Create the main content area of the window."""
        self.main_content = ttk.Frame(self.root)
        self.main_content.pack(fill=tk.BOTH, expand=True)
    
    def _create_status_bar(self):
        """Create the status bar at the bottom of the window."""
        self.status_frame = ttk.Frame(self.root, padding="10")
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.progress_bar = ttk.Progressbar(self.status_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(self.status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5)
    
    def _create_notebook(self):
        """Create the notebook with tabs for different functionality."""
        self.notebook = ttk.Notebook(self.main_content)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        # Create tabs
        self.input_output_tab = InputOutputTab(self)
        self.config_tab = ConfigTab(self)
        self.preview_tab = PreviewTab(self)
        self.gallery_tab = GalleryTab(self)
        
        # Add tabs to notebook
        self.notebook.add(self.input_output_tab.frame, text="Input/Output")
        self.notebook.add(self.config_tab.frame, text="Configuration")
        self.notebook.add(self.preview_tab.frame, text="Preview")
        self.notebook.add(self.gallery_tab.frame, text="Gallery")
        
        # Initialize image comparison integration
        self._init_image_comparison()

    def _init_image_comparison(self):
        # Import here to avoid circular imports
        from utils.gallery_integration import GalleryComparisonIntegrator
        
        # Create instance of the gallery comparison integrator
        self.gallery_integrator = GalleryComparisonIntegrator(self.gallery_tab)
        
        # Add Image Comparison to Help menu
        help_menu = None
        for i in range(self.menu_bar.index('end') + 1):
            if self.menu_bar.type(i) == 'cascade' and self.menu_bar.entrycget(i, 'label') == 'Help':
                help_menu = self.menu_bar.nametowidget(self.menu_bar.entrycget(i, 'menu'))
                break
        
        if help_menu:
            help_menu.add_separator()
            help_menu.add_command(label="Image Comparison Help", command=self._show_comparison_help)

    def _init_outlier_detection(self):
        """Initialize outlier detection functionality."""
        # Import here to avoid circular imports
        from utils.outlier_detection import integrate_outlier_detection
        
        # Integrate outlier detection features
        integrate_outlier_detection(self)


    def _show_comparison_help(self):
        """Show help information about image comparison."""
        help_dialog = tk.Toplevel(self.root)
        help_dialog.title("Image Comparison Help")
        help_dialog.geometry("600x500")
        help_dialog.transient(self.root)
        
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
        help_text = """Image Comparison Tool

    The Image Comparison tool helps you compare multiple versions of the same image to:
    1. Identify outliers or anomalies in image sets
    2. Visualize differences between images
    3. Generate similarity metrics for quantitative analysis

    Using Image Comparison:

    1. Navigate to the Gallery tab to view your image sets
    2. Click "Compare Images" to analyze the current image group
    3. In the comparison dialog:
    - Select a comparison method (different methods are sensitive to different types of changes)
    - Run the comparison to see results across three tabs:
        a) Overview: Shows similarity matrix and outlier scores
        b) Outliers: Lists images ranked by difference from the group
        c) Detailed Comparison: Direct side-by-side comparison with difference visualization

    Comparison Methods:

    1. histogram_correlation: Compares color distribution similarity
    - Good for detecting overall color/tone differences
    - Range: -1.0 to 1.0 (higher = more similar)

    2. histogram_chi: Chi-square test for color histograms
    - Sensitive to smaller color distribution differences
    - Good for detecting subtle color changes
    - Range: 0.0 to unbounded (lower = more similar)

    3. histogram_intersection: Measures overlap between color histograms
    - Useful for finding images with similar dominant colors
    - Range: 0.0 to 1.0 (higher = more similar)

    4. histogram_bhattacharyya: Statistical distance between distributions
    - Good balance between sensitivity and noise tolerance
    - Range: 0.0 to 1.0 (lower = more similar)

    5. ssim: Structural Similarity Index
    - Considers image structure, not just colors
    - Better matches human perception of similarity
    - Sensitive to position changes, blurring, compression
    - Range: 0.0 to 1.0 (higher = more similar)

    6. mse: Mean Squared Error
    - Direct pixel-by-pixel comparison
    - Very sensitive to positioning and small changes
    - Range: 0.0 to unbounded (lower = more similar)

    Batch Analysis:

    1. From the Tools menu, select "Image Comparison > Batch Compare All Groups"
    2. Choose analysis mode:
    - Find Outliers: Identify potential anomalies in each image group
    - Generate Similarity Report: Measure overall similarity within groups
    3. Results are saved to a CSV file for further analysis

    Tips:

    - Different comparison methods highlight different types of differences
    - The SSIM method often provides the most intuitive results for most scenarios
    - For detecting color/tone issues, histogram methods work better
    - Use the detailed comparison view to see exactly where images differ
    - Export results for documentation or further analysis

    The Image Comparison tool is valuable for:
    - Quality control during dataset preparation
    - Detecting processing anomalies or corrupted images
    - Verifying consistency across image variations
    - Understanding the effects of different processing steps
    """

        text.insert(tk.END, help_text)
        text.config(state=tk.DISABLED)  # Make read-only
        
        # Add close button
        close_btn = ttk.Button(help_frame, text="Close", command=help_dialog.destroy)
        close_btn.pack(pady=10)

    def _show_about(self):
        """Show the about dialog."""
        AboutDialog(self.root)
    
    def _show_usage_guide(self):
        """Show the usage guide dialog."""
        UsageGuideDialog(self.root)
    
# Update to the start_processing method in ui/main_window.py

# Update this method in MainWindow class to include the expand_masks option in the validation check

    def start_processing(self):
        """Start the processing pipeline in a separate thread."""
        
# At the start of your processing function
        print("Current Configuration:")
        print(f"- Alignment Method: {self.reinsert_alignment_method.get()}")
        print(f"- Blend Mode: {self.reinsert_blend_mode.get()}")
        print(f"- Blend Extent: {self.reinsert_blend_extent.get()}")
        print(f"- Preserve Edges: {self.reinsert_preserve_edges.get()}")
        print(f"- Using Enhanced Reinserter: {self.use_enhanced_reinserter.get()}")
        print(f"- Handle Different Masks: {self.reinsert_handle_different_masks.get()}")

        if not self.input_dir.get() or not os.path.isdir(self.input_dir.get()):
            messagebox.showerror("Error", "Please select a valid input directory.")
            return
        
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please select an output directory.")
            return
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir.get()):
            try:
                os.makedirs(self.output_dir.get())
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create output directory: {str(e)}")
                return
        
        # Check if any processing steps are selected - INCLUDE expand_masks
        if not any([self.extract_frames.get(), self.crop_mask_regions.get(), 
                self.expand_masks.get(),  # Make sure this line is included
                self.square_pad_images.get(), self.resize_images.get(), 
                self.organize_files.get(), self.convert_to_video.get(),
                self.reinsert_crops_option.get()]):
            messagebox.showerror("Error", "Please select at least one processing step.")
            return
        
        # Add debugging print to see which steps are selected
        print("Selected processing steps:")
        print(f"- Extract frames: {self.extract_frames.get()}")
        print(f"- Crop mask regions: {self.crop_mask_regions.get()}")
        print(f"- Expand mask regions: {self.expand_masks.get()}")  # Debug print
        print(f"- Square pad images: {self.square_pad_images.get()}")
        print(f"- Resize images: {self.resize_images.get()}")
        print(f"- Organize files: {self.organize_files.get()}")
        print(f"- Convert to video: {self.convert_to_video.get()}")
        print(f"- Reinsert crops: {self.reinsert_crops_option.get()}")
        
        # Additional check for reinsert_crops_option
        if self.reinsert_crops_option.get():
            if not self.source_images_dir.get() or not os.path.isdir(self.source_images_dir.get()):
                messagebox.showerror("Error", "For image reinsertion, please select a valid source images directory in the Config tab.")
                return
        
        # Get confirmation before starting processing
        if not self._confirm_processing():
            return
        
        # Update UI state
        self.processing = True
        self.input_output_tab.process_button.config(state=tk.DISABLED)
        self.input_output_tab.cancel_button.config(state=tk.NORMAL)
        self.progress_bar['value'] = 0
        self.status_label.config(text="Initializing processing...")
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(target=self._process_data_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Check progress periodically
        self.root.after(100, self._check_progress)
# Update to the _confirm_processing method in ui/main_window.py

    def _confirm_processing(self):
        """Show a confirmation dialog with processing settings."""
        steps_selected = []
        if self.extract_frames.get():
            steps_selected.append(f"- Extract frames (at {self.frame_rate.get()} fps)")
        if self.crop_mask_regions.get():
            steps_selected.append(f"- Crop mask regions (padding: {self.fill_ratio.get()}%)")
        if self.expand_masks.get():
            steps_selected.append(f"- Expand mask regions (iterations: {self.mask_expand_iterations.get()}, kernel size: {self.mask_expand_kernel_size.get()})")
        if self.square_pad_images.get():
            steps_selected.append(f"- Add padding to make images square (color: {self.padding_color.get()})")
        if self.resize_images.get():
            if self.use_source_resolution.get():
                steps_selected.append(f"- Resize images (use source resolution)")
            else:
                steps_selected.append(f"- Resize images ({self.output_width.get()}x{self.output_height.get()})")
        if self.organize_files.get():
            steps_selected.append(f"- Organize files (pattern: {self.naming_pattern.get()})")
        if self.convert_to_video.get():
            steps_selected.append(f"- Convert to video (fps: {self.video_fps.get()})")
        if self.reinsert_crops_option.get():
            steps_selected.append(f"- Reinsert cropped images (source dir: {os.path.basename(self.source_images_dir.get())})")
        
        # Check if only mask expansion is selected
        if len(steps_selected) == 1 and self.expand_masks.get():
            confirmation_message = f"You are running Mask Expansion as a standalone process.\n\n"
            confirmation_message += steps_selected[0] + "\n\n"
            confirmation_message += f"Input: {self.input_dir.get()}\nOutput: {os.path.join(self.output_dir.get(), 'expanded_masks')}"
        else:
            confirmation_message = f"Ready to process with the following steps:\n\n" + "\n".join(steps_selected)
            confirmation_message += f"\n\nInput: {self.input_dir.get()}\nOutput: {self.output_dir.get()}"
        
        return messagebox.askyesno("Confirm Processing", confirmation_message)
    
    def _process_data_thread(self):
        """Processing thread implementation."""

            # Add these debug prints to check paths
        print(f"Input directory: {self.input_dir.get()}")
        print(f"Source images directory: {self.source_images_dir.get()}")
        print(f"Output directory: {self.output_dir.get()}")
        # Initialize current_input
        current_input = self.input_dir.get()
        
        try:
            # Import processors
            from processors.frame_extractor import FrameExtractor
            from processors.mask_processor import MaskProcessor
            from processors.image_resizer import ImageResizer
            from processors.file_organizer import FileOrganizer
            from processors.video_converter import VideoConverter
            from processors.square_padder import SquarePadder
            from processors.crop_reinserter import CropReinserter
            from processors.mask_expander import MaskExpander
            from processors.enhanced_crop_reinserter import EnhancedCropReinserter


            # Initialize processor instances
            frame_extractor = FrameExtractor(self)
            mask_processor = MaskProcessor(self)
            image_resizer = ImageResizer(self)
            file_organizer = FileOrganizer(self)
            video_converter = VideoConverter(self)
            square_padder = SquarePadder(self)
            crop_reinserter = CropReinserter(self)
            mask_expander = MaskExpander(self)
            enhanced_crop_reinserter = EnhancedCropReinserter(self)


            # Define pipeline steps in order
            pipeline_steps = []
            if self.extract_frames.get():
                pipeline_steps.append(("extract_frames", frame_extractor.extract_frames))
            if self.crop_mask_regions.get():
                pipeline_steps.append(("crop_mask_regions", mask_processor.process_masks))
            if self.expand_masks.get():
                pipeline_steps.append(("expand_masks", mask_expander.expand_masks))
            if self.square_pad_images.get():
                pipeline_steps.append(("square_pad_images", square_padder.add_square_padding))
            if self.resize_images.get():
                pipeline_steps.append(("resize_images", image_resizer.resize_images))
            if self.organize_files.get():
                pipeline_steps.append(("organize_files", file_organizer.organize_files))
            if self.convert_to_video.get():
                pipeline_steps.append(("convert_to_video", video_converter.convert_to_video))
            if self.reinsert_crops_option.get():
                if self.use_enhanced_reinserter.get():
                    print("Using enhanced reinserter")
                    # Make sure the directories are being passed correctly
                    print(f"Enhanced reinserter input: {current_input}")
                    print(f"Enhanced reinserter source: {self.source_images_dir.get()}")
                    print(f"Enhanced reinserter output: {self.output_dir.get()}")
                    success = enhanced_crop_reinserter.reinsert_crops(current_input, self.output_dir.get())
                else:
                    print("Using original reinserter")
                    success = crop_reinserter.reinsert_crops(current_input, self.output_dir.get())




            # Print the pipeline for debugging
            print("Processing pipeline steps:", [step[0] for step in pipeline_steps])
            
            # Dictionary to track all directories created during processing
            output_directories = {
                "original": self.input_dir.get(),
                "frames": os.path.join(self.output_dir.get(), "frames"),
                "cropped": os.path.join(self.output_dir.get(), "cropped"),
                "expanded_masks": os.path.join(self.output_dir.get(), "expanded_masks"),
                "square_padded": os.path.join(self.output_dir.get(), "square_padded"),
                "resized": os.path.join(self.output_dir.get(), "resized"),
                "organized": os.path.join(self.output_dir.get(), "organized"),
                "videos": os.path.join(self.output_dir.get(), "videos"),
                "reinserted": os.path.join(self.output_dir.get(), "reinserted")
            }
            
            # Keep track of which directory to use as input for each step
            current_input = self.input_dir.get()
            
            # Processing progress tracking
            current_progress = 0
            progress_per_step = 100 / len(pipeline_steps) if pipeline_steps else 0
            
            # Execute each step in the pipeline
            for step_index, (step_name, step_func) in enumerate(pipeline_steps):
                if not self.processing:  # Check if processing was cancelled
                    break
                
                self.status_label.config(text=f"Starting step {step_index+1}/{len(pipeline_steps)}: {step_name}")
                print(f"Processing: {step_name} using input directory: {current_input}")
                self.root.update_idletasks()
                
                try:
                    # Special handling for expand_masks if it's the only step selected
                    if step_name == "expand_masks" and len(pipeline_steps) == 1:
                        self.status_label.config(text="Processing mask expansion as a standalone step...")
                        success = step_func(current_input, self.output_dir.get())
                    else:
                        # Execute the current step normally
                        success = step_func(current_input, self.output_dir.get())
                    
                    # If successful, update the input directory for the next step
                    if success:
                        # Find the appropriate output directory for this step
                        if step_name == "extract_frames" and os.path.exists(output_directories["frames"]):
                            current_input = output_directories["frames"]
                        elif step_name == "crop_mask_regions" and os.path.exists(output_directories["cropped"]):
                            current_input = output_directories["cropped"]
                        elif step_name == "expand_masks" and os.path.exists(output_directories["expanded_masks"]):
                            current_input = output_directories["expanded_masks"]
                        elif step_name == "square_pad_images" and os.path.exists(output_directories["square_padded"]):
                            current_input = output_directories["square_padded"]
                        elif step_name == "resize_images" and os.path.exists(output_directories["resized"]):
                            current_input = output_directories["resized"]
                        elif step_name == "organize_files" and os.path.exists(output_directories["organized"]):
                            current_input = output_directories["organized"]
                                # Inside the pipeline steps loop
                        elif step_name == "reinsert_crops":
                            # Validate that source directories are set correctly
                            self.status_label.config(text="Validating directories for crop reinsertion...")
                            
                            if not self.source_images_dir.get() or not os.path.isdir(self.source_images_dir.get()):
                                error_msg = "Error: Source images directory not set or invalid for crop reinsertion"
                                self.status_label.config(text=error_msg)
                                messagebox.showerror("Directory Error", error_msg)
                                continue
                            
                            # Verify that the input directory has cropped images (not masks)
                            mask_dir_found = False
                            for root, dirs, _ in os.walk(current_input):
                                if "masks" in dirs:
                                    mask_dir_found = True
                                    break
                            
                            if not mask_dir_found:
                                self.status_label.config(text="Warning: No 'masks' subdirectory found in input. "
                                                        "Ensure input contains cropped images, not masks.")
                                                        
                            # Now execute crop reinsertion
                            success = step_func(current_input, self.output_dir.get())
                        
                        self.status_label.config(text=f"Completed step: {step_name}. Using {current_input} for next step.")
                    else:
                        self.status_label.config(text=f"Warning: Step {step_name} did not produce expected output. Continuing with same input.")
                    
                    # Automatically add datasets if auto-add is enabled
                    if hasattr(self, 'dataset_manager') and getattr(self, 'auto_add_datasets', True):
                        output_dirs = {
                            "frames": os.path.join(self.output_dir.get(), "frames"),
                            "cropped": os.path.join(self.output_dir.get(), "cropped"),
                            "expanded_masks": os.path.join(self.output_dir.get(), "expanded_masks"),
                            "square_padded": os.path.join(self.output_dir.get(), "square_padded"),
                            "resized": os.path.join(self.output_dir.get(), "resized"),
                            "organized": os.path.join(self.output_dir.get(), "organized")
                        }
                        
                        for step_name, dir_path in output_dirs.items():
                            if os.path.isdir(dir_path) and os.listdir(dir_path):
                                try:
                                    self.dataset_manager.registry.add_dataset(
                                        name=f"{os.path.basename(self.input_dir.get())}_{step_name}",
                                        path=dir_path,
                                        description=f"Processed output from {step_name} step",
                                        category="Processed"
                                    )
                                except Exception as e:
                                    print(f"Failed to add {step_name} dataset: {str(e)}")
                        
                        # Refresh the dataset explorer
                        self.dataset_manager.explorer.refresh_datasets()


                except Exception as e:
                    error_msg = f"Error during {step_name} step: {str(e)}"
                    self.status_label.config(text=error_msg)
                    import traceback
                    print(error_msg)
                    print(traceback.format_exc())
                
                # Update progress
                current_progress += progress_per_step
                self.progress_bar['value'] = min(current_progress, 100)
                self.root.update_idletasks()
            
            # Processing completed
            if self.processing:
                self.status_label.config(text="Processing completed successfully.")
                from tkinter import messagebox
                messagebox.showinfo("Success", "Processing completed successfully.")
        
        except Exception as e:
            # Handle any unexpected errors
            error_msg = f"Unexpected error: {str(e)}"
            self.status_label.config(text=error_msg)
            import traceback
            print(error_msg)
            print(traceback.format_exc())
        
        finally:
            # Reset UI state
            self.processing = False
            self.input_output_tab.process_button.config(state=tk.NORMAL)
            self.input_output_tab.cancel_button.config(state=tk.DISABLED)
    
    def _check_progress(self):
        """Check the progress of the processing thread."""
        if self.processing:
            # Schedule next check
            self.root.after(100, self._check_progress)
    
    def cancel_processing(self):
        """Cancel the processing."""
        if self.processing:
            # Set a flag to stop processing
            self.processing = False
            self.status_label.config(text="Processing cancelled.")
            messagebox.showinfo("Cancelled", "Processing has been cancelled.")
            
            # Reset UI state
            self.input_output_tab.process_button.config(state=tk.NORMAL)
            self.input_output_tab.cancel_button.config(state=tk.DISABLED)


    def _add_gallery_to_datasets(self):
        # Check if gallery has images
        if not hasattr(self, 'gallery_tab') or not self.gallery_tab.images_data:
            messagebox.showinfo("No Gallery Data", "No images are currently in the gallery.")
            return
        
        # Collect unique image directories
        image_paths = set()
        for image_group in self.gallery_tab.images_data:
            for version in image_group['versions']:
                image_dir = os.path.dirname(version['path'])
                image_paths.add(image_dir)
        
        # If multiple directories, show selection dialog
        if len(image_paths) > 1:
            selected_dir = self._show_directory_selection_dialog(image_paths)
        else:
            selected_dir = list(image_paths)[0]
        
        if selected_dir:
            # Add to dataset registry
            dataset_id = self.dataset_manager.registry.add_dataset(
                name=f"Gallery_{os.path.basename(selected_dir)}",
                path=selected_dir,
                description="Dataset created from gallery view",
                category="Gallery"
            )
            
            # Refresh dataset explorer
            self.dataset_manager.explorer.refresh_datasets()