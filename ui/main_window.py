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

class MainWindow:
    """Main window class that sets up the UI and connects functionality."""
    
    def __init__(self, root):
        """
        Initialize the main window UI.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.processing = False  # Flag to track processing status
        
        # Initialize application variables
        self._init_variables()
        
        # Initialize the config manager first
        from utils.config_manager import ConfigManager
        self.config_manager = ConfigManager(self)
        
        # Create the UI components
        self._create_main_content()
        self._create_status_bar()
        self._create_menu()
        

        self.reinsert_crops_option = tk.BooleanVar(value=False)
        self.original_images_dir = tk.StringVar()
        self.selected_original_image = tk.StringVar()
        self.crop_x_position = tk.IntVar(value=0)
        self.crop_y_position = tk.IntVar(value=0)
        self.crop_width = tk.IntVar(value=0)
        self.crop_height = tk.IntVar(value=0)
        self.reinsert_padding = tk.IntVar(value=0)

        # Create the notebook and tabs
        self._create_notebook()

        # Initialize image comparison integration
        self._init_image_comparison()

        # Initialize outlier detection - ADD THIS LINE
        self._init_outlier_detection()


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


        # Crop reinsertion options
        self.reinsert_crops_option = tk.BooleanVar(value=False)
        self.source_images_dir = tk.StringVar()
        self.reinsert_match_method = tk.StringVar(value="name_prefix")
        self.reinsert_padding = tk.IntVar(value=10)  # Default padding percent
        self.use_center_position = tk.BooleanVar(value=True)  # Default to auto-center
        self.reinsert_x = tk.IntVar(value=0)
        self.reinsert_y = tk.IntVar(value=0)
        self.reinsert_width = tk.IntVar(value=0)
        self.reinsert_height = tk.IntVar(value=0)

        # Preview image storage
        self.preview_image = None
        self.preview_mask = None
    
    def _create_menu(self):
        """Create the application menu bar."""
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)
        
        # File menu
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Configuration", command=self.config_manager.save_config)
        file_menu.add_command(label="Load Configuration", command=self.config_manager.load_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu (add this)
        tools_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Tools", menu=tools_menu)
        
        # Add outlier detection option (add this)
        tools_menu.add_command(
            label="Find Outlier Groups...", 
            command=self._run_outlier_detection
    )

        # Help menu
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="Usage Guide", command=self._show_usage_guide)
    
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
        """Initialize image comparison functionality."""
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
    
    def start_processing(self):
        """Start the processing pipeline in a separate thread."""
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
        
        # Check if any processing steps are selected
        if not any([self.extract_frames.get(), self.crop_mask_regions.get(), 
                self.square_pad_images.get(), self.resize_images.get(), 
                self.organize_files.get(), self.convert_to_video.get()]):
            messagebox.showerror("Error", "Please select at least one processing step.")
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
    
    def _confirm_processing(self):
        """Show a confirmation dialog with processing settings."""
        steps_selected = []
        if self.extract_frames.get():
            steps_selected.append(f"- Extract frames (at {self.frame_rate.get()} fps)")
        if self.crop_mask_regions.get():
            steps_selected.append(f"- Crop mask regions (padding: {self.fill_ratio.get()}%)")
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
        
        confirmation_message = f"Ready to process with the following steps:\n\n" + "\n".join(steps_selected)
        confirmation_message += f"\n\nInput: {self.input_dir.get()}\nOutput: {self.output_dir.get()}"
        
        return messagebox.askyesno("Confirm Processing", confirmation_message)
    
    def _process_data_thread(self):
        """Processing thread implementation."""
        try:
            # Import processors
            from processors.frame_extractor import FrameExtractor
            from processors.mask_processor import MaskProcessor
            from processors.image_resizer import ImageResizer
            from processors.file_organizer import FileOrganizer
            from processors.video_converter import VideoConverter
            from processors.square_padder import SquarePadder
            from processors.crop_reinserter import CropReinserter  # Add new import
            
            # Initialize processor instances
            frame_extractor = FrameExtractor(self)
            mask_processor = MaskProcessor(self)
            image_resizer = ImageResizer(self)
            file_organizer = FileOrganizer(self)
            video_converter = VideoConverter(self)
            square_padder = SquarePadder(self)
            crop_reinserter = CropReinserter(self)  # Add new instance
            
            # Define pipeline steps in order
            pipeline_steps = []
            if self.extract_frames.get():
                pipeline_steps.append(("extract_frames", frame_extractor.extract_frames))
            if self.crop_mask_regions.get():
                pipeline_steps.append(("crop_mask_regions", mask_processor.process_masks))
            if self.square_pad_images.get():
                pipeline_steps.append(("square_pad_images", square_padder.add_square_padding))
            if self.resize_images.get():
                pipeline_steps.append(("resize_images", image_resizer.resize_images))
            if self.organize_files.get():
                pipeline_steps.append(("organize_files", file_organizer.organize_files))
            if self.convert_to_video.get():
                pipeline_steps.append(("convert_to_video", video_converter.convert_to_video))
            if self.reinsert_crops_option.get():
                pipeline_steps.append(("reinsert_crops", crop_reinserter.reinsert_crops))  # Add new step
            # Calculate progress per step
            if pipeline_steps:
                progress_per_step = 100 / len(pipeline_steps)
            else:
                progress_per_step = 0
                
            # Dictionary to track all directories created during processing
            output_directories = {
                "original": self.input_dir.get(),
                "frames": os.path.join(self.output_dir.get(), "frames"),
                "cropped": os.path.join(self.output_dir.get(), "cropped"),
                "square_padded": os.path.join(self.output_dir.get(), "square_padded"),
                "resized": os.path.join(self.output_dir.get(), "resized"),
                "organized": os.path.join(self.output_dir.get(), "organized"),
                "videos": os.path.join(self.output_dir.get(), "videos"),
                "reinserted": os.path.join(self.output_dir.get(), "reinserted")  # Add new directory
            }
            
            # Keep track of which directory to use as input for each step
            current_input = self.input_dir.get()
            
            # Processing progress tracking
            current_progress = 0
            
            # Execute each step in the pipeline
            for step_index, (step_name, step_func) in enumerate(pipeline_steps):
                if not self.processing:  # Check if processing was cancelled
                    break
                
                self.status_label.config(text=f"Starting step {step_index+1}/{len(pipeline_steps)}: {step_name}")
                self.root.update_idletasks()
                
                try:
                    # Execute the current step
                    success = step_func(current_input, self.output_dir.get())
                    
                    # If successful, update the input directory for the next step
                    if success:
                        # Find the appropriate output directory for this step
                        if step_name == "extract_frames" and os.path.exists(output_directories["frames"]):
                            current_input = output_directories["frames"]
                        elif step_name == "crop_mask_regions" and os.path.exists(output_directories["cropped"]):
                            current_input = output_directories["cropped"]
                        elif step_name == "square_pad_images" and os.path.exists(output_directories["square_padded"]):
                            current_input = output_directories["square_padded"]
                        elif step_name == "resize_images" and os.path.exists(output_directories["resized"]):
                            current_input = output_directories["resized"]
                        elif step_name == "organize_files" and os.path.exists(output_directories["organized"]):
                            current_input = output_directories["organized"]
                        
                        self.status_label.config(text=f"Completed step: {step_name}. Using output directory for next step.")
                    else:
                        # If the step failed, continue with the same input directory
                        self.status_label.config(text=f"Warning: Step {step_name} did not produce expected output. Continuing with same input.")
                
                except Exception as e:
                    error_msg = f"Error during {step_name} step: {str(e)}"
                    self.status_label.config(text=error_msg)
                    from tkinter import messagebox
                    messagebox.showerror("Processing Error", error_msg)
                    import traceback
                    print(error_msg)
                    print(traceback.format_exc())
                
                # Update progress
                current_progress += progress_per_step
                self.progress_bar['value'] = min(current_progress, 100)
                self.root.update_idletasks()
            
            # Processing completed successfully
            if self.processing:  # Only show success if not cancelled
                self.status_label.config(text="Processing completed successfully.")
                from tkinter import messagebox
                messagebox.showinfo("Success", "Dataset preparation completed successfully.")
        
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            from tkinter import messagebox
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            # Log the full error with traceback
            import traceback
            print(f"Error during processing: {str(e)}")
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