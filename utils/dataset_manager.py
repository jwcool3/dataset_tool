"""
Dataset Manager Integration for the Dataset Preparation Tool
Adds the Dataset Manager functionality to the main application.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox

# Import from the dataset manager module
from dataset_manager import (
    DatasetRegistry, 
    DatasetExplorer, 
    DatasetOperations, 
    DatasetAnalyzer,
    DatasetManagerTab,
    initialize_dataset_manager
)

def add_dataset_manager_to_app(app):
    """
    Add the Dataset Manager to the application.
    
    Args:
        app: The main application instance
    """
    # Ensure the config directory exists
    config_dir = os.path.join(os.path.expanduser("~"), ".dataset_manager")
    os.makedirs(config_dir, exist_ok=True)
    app.config_dir = config_dir
    
    # Initialize the Dataset Manager
    initialize_dataset_manager(app)
    
    # Link the Dataset Manager with existing functionality
    _add_integration_hooks(app)

def _add_integration_hooks(app):
    """
    Add hooks to integrate Dataset Manager with existing functionality.
    
    Args:
        app: The main application instance
    """
    # Add buttons to Input/Output tab to interact with Dataset Manager
    _add_input_output_integration(app)
    
    # Add option to save processing results as a dataset
    _add_processing_integration(app)
    
    # Add gallery integration
    _add_gallery_integration(app)

def _add_input_output_integration(app):
    """Add Dataset Manager integration to the Input/Output tab."""
    # Get the Input/Output tab frame
    input_output_tab = app.input_output_tab
    
    # Add a frame for dataset interactions
    dataset_frame = ttk.LabelFrame(input_output_tab.frame, text="Dataset Manager Integration")
    
    # Find the right spot to insert the frame
    # Try to place after the directory selection section
    for i, child in enumerate(input_output_tab.frame.winfo_children()):
        if isinstance(child, ttk.LabelFrame) and child.cget("text") == "Directory Selection":
            dataset_frame.pack(fill=tk.X, pady=5, after=child)
            break
    else:
        # If not found, just add to the end
        dataset_frame.pack(fill=tk.X, pady=5)
    
    # Create buttons for common actions
    ttk.Button(
        dataset_frame, 
        text="Add Current Directory as Dataset", 
        command=lambda: _add_current_directory(app)
    ).pack(side=tk.LEFT, padx=5, pady=5)
    
    ttk.Button(
        dataset_frame, 
        text="Browse Datasets", 
        command=lambda: app.notebook.select(app.notebook.index(app.dataset_manager.frame))
    ).pack(side=tk.LEFT, padx=5, pady=5)
    
    # Checkbox to automatically add processing results as datasets
    app.auto_add_datasets = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        dataset_frame,
        text="Automatically add processing results to dataset registry",
        variable=app.auto_add_datasets
    ).pack(side=tk.LEFT, padx=20, pady=5)

def _add_processing_integration(app):
    """Add integration hooks to the processing pipeline."""
    # Store original _process_data_thread method
    original_process = app._process_data_thread
    
    # Replace with wrapped version that adds datasets
    def wrapped_process_data_thread():
        # Call the original method
        original_process()
        
        # Check if auto-add is enabled
        if hasattr(app, 'auto_add_datasets') and app.auto_add_datasets.get():
            # Get output directories from the processing
            output_dirs = {}
            for step_name in [
                "frames", "cropped", "expanded_masks", "square_padded", 
                "resized", "organized", "videos", "reinserted"
            ]:
                dir_path = os.path.join(app.output_dir.get(), step_name)
                if os.path.isdir(dir_path) and os.listdir(dir_path):  # Only if exists and not empty
                    output_dirs[step_name] = dir_path
            
            # Add each output directory as a dataset
            if hasattr(app, 'dataset_manager'):
                for step_name, dir_path in output_dirs.items():
                    try:
                        # Generate a descriptive name
                        name = f"{os.path.basename(app.input_dir.get())}_{step_name}"
                        
                        # Add to registry
                        app.dataset_manager.registry.add_dataset(
                            name=name,
                            path=dir_path,
                            description=f"Processed output from {step_name} step",
                            category="Processed"
                        )
                        
                    except Exception as e:
                        print(f"Failed to add {step_name} dataset: {str(e)}")
                
                # Refresh the dataset explorer
                app.dataset_manager.explorer.refresh_datasets()
    
    # Replace the method
    app._process_data_thread = wrapped_process_data_thread

def _add_gallery_integration(app):
    """Add integration with the Gallery tab."""
    # Make sure the Gallery tab exists
    if not hasattr(app, 'gallery_tab'):
        return
    
    # Get the Gallery tab
    gallery_tab = app.gallery_tab
    
    # Add a button to add the current gallery view as a dataset
    if hasattr(gallery_tab, 'toolbar'):
        # Add button to toolbar if it exists
        add_dataset_btn = ttk.Button(
            gallery_tab.toolbar, 
            text="Add to Datasets", 
            command=lambda: _add_gallery_to_datasets(app)
        )
        add_dataset_btn.pack(side=tk.LEFT, padx=5)
    
def _add_gallery_to_datasets(app):
    """Add the current gallery view as a dataset."""
    # Make sure the gallery tab exists and has images
    if not hasattr(app, 'gallery_tab') or not hasattr(app.gallery_tab, 'images_data'):
        messagebox.showinfo("No Gallery Data", "No images are currently loaded in the gallery.")
        return
    
    if not app.gallery_tab.images_data:
        messagebox.showinfo("No Gallery Data", "No images are currently loaded in the gallery.")
        return
    
    # Get image paths from current gallery view
    image_paths = set()
    for image_group in app.gallery_tab.images_data:
        for version in image_group['versions']:
            image_dir = os.path.dirname(version['path'])
            image_paths.add(image_dir)
    
    if not image_paths:
        messagebox.showinfo("No Images", "No valid image directories found in the gallery.")
        return
    
    # If multiple directories found, let the user choose one
    selected_dir = None
    if len(image_paths) == 1:
        selected_dir = list(image_paths)[0]
    else:
        # Create a dialog to select which directory to add
        dir_dialog = tk.Toplevel(app.root)
        dir_dialog.title("Select Directory to Add")
        dir_dialog.geometry("500x300")
        dir_dialog.transient(app.root)
        dir_dialog.grab_set()
        
        ttk.Label(dir_dialog, text="Select which directory to add as a dataset:", 
                 font=("Helvetica", 10, "bold")).pack(pady=10)
        
        # Create a listbox for directory selection
        dir_frame = ttk.Frame(dir_dialog)
        dir_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = ttk.Scrollbar(dir_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        dir_listbox = tk.Listbox(dir_frame)
        dir_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=dir_listbox.yview)
        dir_listbox.config(yscrollcommand=scrollbar.set)
        
        # Add directories to the listbox
        for directory in sorted(image_paths):
            dir_listbox.insert(tk.END, directory)
        
        # Button frame
        btn_frame = ttk.Frame(dir_dialog)
        btn_frame.pack(fill=tk.X, pady=10)
        
        # Function to get selection
        def select_directory():
            selection = dir_listbox.curselection()
            nonlocal selected_dir
            
            if selection:
                selected_dir = dir_listbox.get(selection[0])
                dir_dialog.destroy()
            else:
                messagebox.showwarning("No Selection", "Please select a directory first.")
        
        ttk.Button(btn_frame, text="Add Selected Directory", command=select_directory).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Cancel", command=dir_dialog.destroy).pack(side=tk.RIGHT, padx=10)
        
        # Wait for dialog to close
        app.root.wait_window(dir_dialog)
        
        if not selected_dir:
            return  # User cancelled
    
    # Get a name for the dataset
    from tkinter.simpledialog import askstring
    
    name = askstring("Dataset Name", "Enter a name for this dataset:", 
                     initialvalue=os.path.basename(selected_dir))
    
    if not name:
        return  # User cancelled
    
    # Add the dataset to the registry
    try:
        dataset_id = app.dataset_manager.registry.add_dataset(
            name=name,
            path=selected_dir,
            description=f"Added from gallery view",
            category="Gallery"
        )
        
        app.dataset_manager.explorer.refresh_datasets()
        
        # Switch to Dataset Manager tab
        app.notebook.select(app.notebook.index(app.dataset_manager.frame))
        
        # Select the new dataset
        app.dataset_manager.explorer._select_dataset_by_id(dataset_id)
        
        messagebox.showinfo("Success", f"Added '{name}' to datasets.")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to add dataset: {str(e)}")

def _add_current_directory(app):
    """Add the current input directory to the dataset registry."""
    input_dir = app.input_dir.get()
    
    if not input_dir or not os.path.isdir(input_dir):
        messagebox.showerror("Error", "Please select a valid input directory first.")
        return
    
    # Ask for dataset name
    from tkinter.simpledialog import askstring
    
    name = askstring("Dataset Name", "Enter a name for this dataset:", 
                    initialvalue=os.path.basename(input_dir))
    
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

# Add the main function to create a standalone Dataset Manager
def create_standalone_dataset_manager():
    """Create a standalone Dataset Manager application."""
    import tkinter as tk
    from tkinter import ttk
    
    root = tk.Tk()
    root.title("Dataset Manager")
    root.geometry("1200x800")
    
    # Create a minimal app object to satisfy dependencies
    class MinimalApp:
        def __init__(self, root):
            self.root = root
            self.menu_bar = tk.Menu(root)
            self.notebook = ttk.Notebook(root)
            self.notebook.pack(fill=tk.BOTH, expand=True)
            
            # Set up config directory
            self.config_dir = os.path.join(os.path.expanduser("~"), ".dataset_manager")
            os.makedirs(self.config_dir, exist_ok=True)
    
    app = MinimalApp(root)
    root.config(menu=app.menu_bar)
    
    # Initialize Dataset Manager
    dataset_manager = DatasetManagerTab(app)
    
    # Add the tab to the notebook
    app.notebook.add(dataset_manager.frame, text="Dataset Manager")
    
    # Store the dataset manager instance on the app
    app.dataset_manager = dataset_manager
    
    # Add File menu
    file_menu = tk.Menu(app.menu_bar, tearoff=0)
    app.menu_bar.add_cascade(label="File", menu=file_menu)
    
    file_menu.add_command(label="Add Dataset", command=lambda: dataset_manager.explorer._add_dataset())
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    
    # Add Help menu
    help_menu = tk.Menu(app.menu_bar, tearoff=0)
    app.menu_bar.add_cascade(label="Help", menu=help_menu)
    
    help_menu.add_command(label="Dataset Manager Help", 
                         command=lambda: _show_dataset_manager_help(app))
    help_menu.add_command(label="About", command=lambda: _show_about_dialog(app))
    
    return root, app

def _show_about_dialog(app):
    """Show about dialog for standalone application."""
    about_dialog = tk.Toplevel(app.root)
    about_dialog.title("About Dataset Manager")
    about_dialog.geometry("400x250")
    about_dialog.transient(app.root)
    about_dialog.grab_set()
    
    frame = ttk.Frame(about_dialog, padding=20)
    frame.pack(fill=tk.BOTH, expand=True)
    
    ttk.Label(frame, text="Dataset Manager", font=("Helvetica", 16, "bold")).pack(pady=(0, 10))
    ttk.Label(frame, text="Version 1.0").pack()
    ttk.Label(frame, text="A tool for managing and operating on datasets").pack(pady=5)
    ttk.Label(frame, text="© 2025").pack(pady=10)
    
    ttk.Button(frame, text="OK", command=about_dialog.destroy).pack(pady=10)

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
    close_btn.pack(pady=10)

# If this script is run directly, launch the standalone version
if __name__ == "__main__":
    root, _ = create_standalone_dataset_manager()
    root.mainloop()