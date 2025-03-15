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
        
        # Create the notebook and tabs
        self._create_notebook()
    
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
        self.debug_mode = tk.BooleanVar(value=False)
        
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
        
        # Help menu
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="Usage Guide", command=self._show_usage_guide)
    
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
        
        # Add tabs to notebook
        self.notebook.add(self.input_output_tab.frame, text="Input/Output")
        self.notebook.add(self.config_tab.frame, text="Configuration")
        self.notebook.add(self.preview_tab.frame, text="Preview")
    
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
            
            # Initialize processor instances
            frame_extractor = FrameExtractor(self)
            mask_processor = MaskProcessor(self)
            image_resizer = ImageResizer(self)
            file_organizer = FileOrganizer(self)
            video_converter = VideoConverter(self)
            square_padder = SquarePadder(self)
            
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
                "videos": os.path.join(self.output_dir.get(), "videos")
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