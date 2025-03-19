"""
Test Script for Dataset Manager

This script creates a simple test environment to ensure the Dataset Manager
works properly both standalone and integrated with the main application.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

def run_standalone_test():
    """Run the standalone version of the Dataset Manager."""
    try:
        # Import the standalone creator function
        from dataset_manager_integration import create_standalone_dataset_manager
        
        # Create and run the standalone application
        root, app = create_standalone_dataset_manager()
        
        print("Standalone Dataset Manager initialized successfully")
        print("Running application...")
        
        root.mainloop()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure the dataset_manager_integration.py file is in the Python path")
        sys.exit(1)
    except Exception as e:
        print(f"Error running standalone test: {e}")
        sys.exit(1)

def run_integration_test():
    """
    Run a test of the Dataset Manager integrated with a mock main application.
    """
    try:
        # Create a mock main application
        class MockMainApp:
            def __init__(self):
                self.root = tk.Tk()
                self.root.title("Dataset Preparation Tool (Mock)")
                self.root.geometry("1200x800")
                
                # Create menu bar
                self.menu_bar = tk.Menu(self.root)
                self.root.config(menu=self.menu_bar)
                
                # Create file menu
                file_menu = tk.Menu(self.menu_bar, tearoff=0)
                self.menu_bar.add_cascade(label="File", menu=file_menu)
                file_menu.add_command(label="Exit", command=self.root.quit)
                
                # Create help menu
                help_menu = tk.Menu(self.menu_bar, tearoff=0)
                self.menu_bar.add_cascade(label="Help", menu=help_menu)
                
                # Create notebook
                self.notebook = ttk.Notebook(self.root)
                self.notebook.pack(fill=tk.BOTH, expand=True)
                
                # Add tabs
                self.input_output_tab = self._create_mock_input_output_tab()
                self.config_tab = self._create_mock_tab("Configuration")
                self.preview_tab = self._create_mock_tab("Preview")
                self.gallery_tab = self._create_mock_gallery_tab()
                
                # Add tabs to notebook
                self.notebook.add(self.input_output_tab.frame, text="Input/Output")
                self.notebook.add(self.config_tab, text="Configuration")
                self.notebook.add(self.preview_tab, text="Preview")
                self.notebook.add(self.gallery_tab.frame, text="Gallery")
                
                # Create status bar
                self.status_frame = ttk.Frame(self.root, padding="10")
                self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
                
                self.progress_bar = ttk.Progressbar(self.status_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
                self.progress_bar.pack(fill=tk.X, pady=5)
                
                self.status_label = ttk.Label(self.status_frame, text="Ready")
                self.status_label.pack(side=tk.LEFT, padx=5)
                
                # Mock variables
                self.input_dir = tk.StringVar(value=os.path.expanduser("~/Documents"))
                self.output_dir = tk.StringVar(value=os.path.expanduser("~/Documents/output"))
                
                # Create config directory
                self.config_dir = os.path.join(os.path.expanduser("~"), ".dataset_manager_test")
                os.makedirs(self.config_dir, exist_ok=True)
            
            def _create_mock_tab(self, name):
                """Create a mock tab with a label."""
                frame = ttk.Frame(self.notebook, padding="10")
                ttk.Label(frame, text=f"Mock {name} Tab").pack(pady=20)
                return frame
            
            def _create_mock_input_output_tab(self):
                """Create a mock input/output tab with directory selection."""
                class MockInputOutputTab:
                    def __init__(self, parent):
                        self.parent = parent
                        self.frame = ttk.Frame(parent.notebook, padding="10")
                        
                        # Directory selection
                        dir_frame = ttk.LabelFrame(self.frame, text="Directory Selection", padding="10")
                        dir_frame.pack(fill=tk.X, pady=5)
                        
                        # Input directory
                        input_dir_frame = ttk.Frame(dir_frame)
                        input_dir_frame.pack(fill=tk.X, pady=5)
                        
                        ttk.Label(input_dir_frame, text="Input Directory:").pack(side=tk.LEFT, padx=5)
                        ttk.Entry(input_dir_frame, textvariable=parent.input_dir, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
                        ttk.Button(input_dir_frame, text="Browse...").pack(side=tk.LEFT, padx=5)
                        
                        # Output directory
                        output_dir_frame = ttk.Frame(dir_frame)
                        output_dir_frame.pack(fill=tk.X, pady=5)
                        
                        ttk.Label(output_dir_frame, text="Output Directory:").pack(side=tk.LEFT, padx=5)
                        ttk.Entry(output_dir_frame, textvariable=parent.output_dir, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
                        ttk.Button(output_dir_frame, text="Browse...").pack(side=tk.LEFT, padx=5)
                        
                        # Processing options
                        options_frame = ttk.LabelFrame(self.frame, text="Processing Options", padding="10")
                        options_frame.pack(fill=tk.X, pady=5)
                        
                        ttk.Checkbutton(options_frame, text="Extract frames from videos").pack(anchor=tk.W, padx=5, pady=2)
                        ttk.Checkbutton(options_frame, text="Crop mask regions").pack(anchor=tk.W, padx=5, pady=2)
                        ttk.Checkbutton(options_frame, text="Resize images").pack(anchor=tk.W, padx=5, pady=2)
                        
                        # Buttons
                        button_frame = ttk.Frame(self.frame)
                        button_frame.pack(fill=tk.X, pady=10)
                        
                        ttk.Button(button_frame, text="Preview").pack(side=tk.LEFT, padx=5)
                        ttk.Button(button_frame, text="Start Processing").pack(side=tk.LEFT, padx=5)
                
                return MockInputOutputTab(self)
            
            def _create_mock_gallery_tab(self):
                """Create a mock gallery tab with toolbar."""
                class MockGalleryTab:
                    def __init__(self, parent):
                        self.parent = parent
                        self.frame = ttk.Frame(parent.notebook, padding="10")
                        
                        # Create toolbar
                        self.toolbar = ttk.Frame(self.frame)
                        self.toolbar.pack(fill=tk.X, pady=5)
                        
                        self.refresh_button = ttk.Button(self.toolbar, text="Refresh Gallery")
                        self.refresh_button.pack(side=tk.LEFT, padx=5)
                        
                        # Add mock data
                        self.images_data = [
                            {
                                'filename': 'image1.jpg',
                                'versions': [
                                    {
                                        'path': os.path.join(parent.input_dir.get(), 'image1.jpg'),
                                        'is_mask': False
                                    }
                                ]
                            },
                            {
                                'filename': 'image2.jpg',
                                'versions': [
                                    {
                                        'path': os.path.join(parent.input_dir.get(), 'image2.jpg'),
                                        'is_mask': False
                                    }
                                ]
                            }
                        ]
                
                return MockGalleryTab(self)
            
            def _process_data_thread(self):
                """Mock processing method."""
                print("Mock processing started")
                messagebox.showinfo("Processing", "Mock processing completed")
                self.status_label.config(text="Processing completed")
        
        # Create the mock app
        app = MockMainApp()
        
        # Import and add the Dataset Manager
        from dataset_manager_integration import add_dataset_manager_to_app
        add_dataset_manager_to_app(app)
        
        print("Integrated Dataset Manager initialized successfully")
        print("Running application...")
        
        app.root.mainloop()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure the dataset_manager_integration.py file is in the Python path")
        sys.exit(1)
    except Exception as e:
        print(f"Error running integration test: {e}")
        sys.exit(1)

def main():
    """Main function to run tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Dataset Manager")
    parser.add_argument("--mode", choices=["standalone", "integrated"], default="standalone",
                      help="Test mode: standalone or integrated with mock app")
    
    args = parser.parse_args()
    
    if args.mode == "standalone":
        print("Running standalone Dataset Manager test...")
        run_standalone_test()
    else:
        print("Running integrated Dataset Manager test...")
        run_integration_test()

if __name__ == "__main__":
    main()