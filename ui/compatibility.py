import tkinter as tk
from tkinter import ttk
import importlib

# Try to import customtkinter
try:
    import customtkinter as ctk
    HAVE_CTK = True
except ImportError:
    # If customtkinter isn't available, make a simple compatibility layer
    HAVE_CTK = False
    class DummyCTk:
        def __getattr__(self, name):
            # Return appropriate ttk/tk widgets for each customtkinter widget
            if name == 'CTkFrame':
                return ttk.Frame
            elif name == 'CTkLabel':
                return ttk.Label
            elif name == 'CTkButton':
                return ttk.Button
            elif name == 'CTkEntry':
                return ttk.Entry
            elif name == 'CTkCheckbutton':
                return ttk.Checkbutton
            elif name == 'CTkRadiobutton':
                return ttk.Radiobutton
            elif name == 'CTkCombobox':
                return ttk.Combobox
            elif name == 'CTkScrollableFrame':
                # Return a frame with a scrollbar
                return ttk.Frame
            elif name == 'CTkSpinbox':
                return ttk.Spinbox if hasattr(ttk, 'Spinbox') else tk.Spinbox
            elif name == 'CTkFont':
                return tk.font.Font
            elif name == 'CTkScale':
                return ttk.Scale
            elif name == 'CTkProgressBar':
                return ttk.Progressbar
            elif name == 'CTkTabview':
                return ttk.Notebook
            elif name == 'CTkOptionMenu':
                return ttk.OptionMenu
            elif name == 'CTkCanvas':
                return tk.Canvas
            else:
                # Default fallback
                return ttk.Widget
    
    ctk = DummyCTk()

# Function to get the appropriate widget class with fallbacks
def get_widget(widget_name):
    if HAVE_CTK:
        try:
            return getattr(ctk, widget_name)
        except AttributeError:
            # If widget doesn't exist in customtkinter, fall back to ttk/tk
            pass
    
    # Fallbacks for common widgets
    if widget_name == 'CTkFrame':
        return ttk.Frame
    elif widget_name == 'CTkLabel':
        return ttk.Label
    elif widget_name == 'CTkButton':
        return ttk.Button
    elif widget_name == 'CTkEntry':
        return ttk.Entry
    elif widget_name == 'CTkCheckbutton':
        return ttk.Checkbutton
    elif widget_name == 'CTkRadiobutton':
        return ttk.Radiobutton
    elif widget_name == 'CTkCombobox':
        return ttk.Combobox
    elif widget_name == 'CTkScrollableFrame':
        # Return a frame with a scrollbar
        return ttk.Frame
    elif widget_name == 'CTkSpinbox':
        return ttk.Spinbox if hasattr(ttk, 'Spinbox') else tk.Spinbox
    elif widget_name == 'CTkScale':
        return ttk.Scale
    else:
        # Default fallback
        return ttk.Widget