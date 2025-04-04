"""
Dataset Preparation Tool - Main Application
Initializes the UI and connects it with processing functionality.
"""

import tkinter as tk
import sys
import os
import customtkinter as ctk

# Add the current directory to the path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import MainWindow

# Configure CustomTkinter appearance
ctk.set_appearance_mode("system")  # Options: "system" (default), "light", "dark"
ctk.set_default_color_theme("blue")  # Options: "blue" (default), "green", "dark-blue"

class DatasetPreparationApp:
    """Main application class that initializes and runs the application."""
    
    def __init__(self):
        """Initialize the application."""
        # Create the root window with CustomTkinter
        self.root = ctk.CTk()
        self.root.title("Dataset Preparation Tool")
        
        # Set a minimum window size
        self.root.minsize(1000, 750)
        
        # Add window icon if available
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass  # No icon available, continue without it
        
        # Center the window on screen
        window_width = 1000
        window_height = 750
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Initialize the main window and UI
        self.main_window = MainWindow(self.root)
        
        # Add appearance mode toggle
        self._add_appearance_toggle()
    
    def _add_appearance_toggle(self):
        """Add a toggle for switching between light and dark mode."""
        # Create a frame for the toggle at the bottom-right
        toggle_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        toggle_frame.place(relx=0.98, rely=0.98, anchor="se")
        
        # Create the toggle button
        mode_var = ctk.StringVar(value=ctk.get_appearance_mode())
        
        def toggle_appearance_mode():
            """Toggle between light and dark mode."""
            current_mode = ctk.get_appearance_mode()
            new_mode = "Light" if current_mode == "Dark" else "Dark"
            ctk.set_appearance_mode(new_mode)
            mode_var.set(new_mode)
        
        # Add light/dark mode toggle
        toggle_btn = ctk.CTkButton(
            toggle_frame,
            text=f"{'🌙' if mode_var.get() == 'Light' else '☀️'} Mode",
            width=100,
            height=28,
            command=toggle_appearance_mode,
            fg_color="transparent",
            border_width=1,
            text_color=("gray50", "gray90")
        )
        toggle_btn.pack(padx=10, pady=10)
    
    def run(self):
        """Run the application main loop."""
        self.root.mainloop()