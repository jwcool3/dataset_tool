"""
Enhanced compatibility layer for CustomTkinter
Provides fallback mechanisms for missing CustomTkinter widgets or versions.
"""

import tkinter as tk
from tkinter import ttk
import importlib
from functools import partial
import inspect

# Try to import customtkinter
try:
    import customtkinter as ctk_original
    HAVE_CTK = True
    
    # Check for specific classes to determine version compatibility
    HAVE_CTK_SPINBOX = hasattr(ctk_original, 'CTkSpinbox')
    HAVE_CTK_SCROLLABLE = hasattr(ctk_original, 'CTkScrollableFrame')
    CTK_VERSION = getattr(ctk_original, '__version__', '0.0.0')
except ImportError:
    # If customtkinter isn't available at all
    HAVE_CTK = False
    HAVE_CTK_SPINBOX = False
    HAVE_CTK_SCROLLABLE = False
    CTK_VERSION = '0.0.0'

# Parameter mappings between CustomTkinter and standard widgets
PARAM_MAPPINGS = {
    'fg_color': 'background',
    'text_color': 'foreground',
    'border_color': 'highlightbackground',
    'border_width': 'highlightthickness',
    'corner_radius': None,  # No equivalent
    'hover_color': None,  # No equivalent
    'font': 'font',
    'textvariable': 'textvariable',
    'variable': 'variable',
    'from_': 'from_',
    'to': 'to',
    'increment': 'increment',
    'padding': 'padding',
    'command': 'command',
    'state': 'state',
    'width': 'width',
    'height': 'height'
}

def transform_params(widget_class, params):
    """Transform CustomTkinter-style parameters to standard ttk/tk parameters."""
    transformed = {}
    
    # Get the signature of the widget's constructor to check valid parameters
    try:
        sig = inspect.signature(widget_class.__init__)
        valid_params = list(sig.parameters.keys())
    except (ValueError, TypeError):
        valid_params = []  # If we can't get the signature, assume all params are valid
    
    for key, value in params.items():
        if key in valid_params or key == 'class_' or '__' not in key:
            transformed[key] = value
        elif key in PARAM_MAPPINGS and PARAM_MAPPINGS[key] is not None:
            # Map to equivalent ttk/tk parameter
            new_key = PARAM_MAPPINGS[key]
            if new_key in valid_params or new_key == 'class_' or '__' not in new_key:
                transformed[new_key] = value
    
    return transformed

class CustomTkinterCompat:
    """
    Compatibility class that provides fallback widgets when CustomTkinter widgets
    are unavailable or incompatible.
    """
    def __init__(self):
        # If we have actual CustomTkinter, use it as the basis
        if HAVE_CTK:
            self._ctk = ctk_original
        else:
            self._ctk = None
        
        # Initialize cache for widget factory functions
        self._widget_factories = {}
    
    def __getattr__(self, name):
        """
        Dynamically handle attribute access to provide compatibility.
        
        Will return the original CustomTkinter class if available, or a compatible
        replacement if not.
        """
        # First try to get from the original CustomTkinter
        if self._ctk is not None:
            try:
                return getattr(self._ctk, name)
            except AttributeError:
                pass
        
        # Create a widget factory function on demand if it's a widget class
        if name.startswith('CTk') and name not in self._widget_factories:
            self._widget_factories[name] = self._create_widget_factory(name)
        
        if name in self._widget_factories:
            return self._widget_factories[name]
            
        # Handle commonly used functions/attributes
        if name == 'set_appearance_mode':
            return lambda mode: None
        elif name == 'set_default_color_theme':
            return lambda theme: None
        elif name == 'CTkFont':
            return tk.font.Font
        
        # For unknown attributes, raise an error
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def _create_widget_factory(self, widget_type):
        """Create a factory function for the specified widget type."""
        def factory(parent=None, **kwargs):
            return self._create_widget(parent, widget_type, **kwargs)
        
        return factory
    
    def _create_widget(self, parent, widget_type, **kwargs):
        """
        Create a widget with appropriate fallbacks.
        
        Args:
            parent: Parent widget
            widget_type: String name of the CustomTkinter widget type
            **kwargs: Widget parameters
            
        Returns:
            Created widget instance
        """
        # For CustomTkinter without certain classes, or plain ttk
        # Map widget types to suitable replacements
        if widget_type == 'CTkFrame':
            widget_class = ttk.Frame
        elif widget_type == 'CTkLabel':
            widget_class = ttk.Label
        elif widget_type == 'CTkButton':
            widget_class = ttk.Button
        elif widget_type == 'CTkEntry':
            widget_class = ttk.Entry
        elif widget_type == 'CTkCheckbox' or widget_type == 'CTkCheckbutton':
            widget_class = ttk.Checkbutton
        elif widget_type == 'CTkRadiobutton':
            widget_class = ttk.Radiobutton
        elif widget_type == 'CTkCombobox':
            widget_class = ttk.Combobox
        elif widget_type == 'CTkSpinbox' or widget_type == 'CTkSpinBox':
            widget_class = ttk.Spinbox if hasattr(ttk, 'Spinbox') else tk.Spinbox
        elif widget_type == 'CTkScale':
            widget_class = ttk.Scale
        elif widget_type == 'CTkProgressBar':
            widget_class = ttk.Progressbar
        elif widget_type == 'CTkScrollableFrame':
            # Create a frame with a scrollbar
            frame = ttk.Frame(parent)
            canvas = tk.Canvas(frame)
            scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Handle mousewheel scrolling
            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            
            # Attach the canvas and scrollbar as attributes
            scrollable_frame.canvas = canvas
            scrollable_frame.scrollbar = scrollbar
            return scrollable_frame
            
        elif widget_type == 'CTkCanvas':
            widget_class = tk.Canvas
        elif widget_type == 'CTkTabview':
            widget_class = ttk.Notebook
        elif widget_type == 'CTkOptionMenu':
            widget_class = ttk.OptionMenu
            # Special handling for OptionMenu which has a different constructor
            if 'values' in kwargs and parent is not None:
                variable = kwargs.get('variable', tk.StringVar())
                values = kwargs.pop('values', [])
                if not values:
                    values = [""]  # OptionMenu needs at least one value
                transformed_params = transform_params(widget_class, kwargs)
                return widget_class(parent, variable, values[0], *values, **transformed_params)
        elif widget_type == 'CTkSwitch':
            widget_class = ttk.Checkbutton  # Best approximation
        elif widget_type == 'CTkTextbox':
            widget_class = tk.Text
        elif widget_type == 'CTkSegmentedButton':
            # Create a frame with multiple buttons
            frame = ttk.Frame(parent)
            if 'values' in kwargs and 'command' in kwargs:
                values = kwargs.pop('values', [])
                command = kwargs.pop('command', None)
                variable = kwargs.pop('variable', tk.StringVar())
                
                # Create a button for each value
                for value in values:
                    btn = ttk.Button(
                        frame, 
                        text=value,
                        command=lambda v=value: (variable.set(v), command() if command else None)
                    )
                    btn.pack(side=tk.LEFT)
                
            return frame
        elif widget_type == 'CTkLabelFrame':
            widget_class = ttk.LabelFrame
        else:
            # Default fallback
            widget_class = ttk.Frame
        
        # Transform parameters for ttk/tk widgets
        transformed_params = transform_params(widget_class, kwargs)
        
        return widget_class(parent, **transformed_params)

# Create the compatibility layer
ctk = CustomTkinterCompat()

# Function to get the appropriate widget class with fallbacks
def get_widget(widget_name):
    """
    Get the appropriate widget class with fallbacks.
    
    Args:
        widget_name: String name of the widget class
        
    Returns:
        Widget class
    """
    # First try to get from actual CustomTkinter if available
    if HAVE_CTK:
        try:
            return getattr(ctk_original, widget_name)
        except AttributeError:
            pass
    
    # Standard mappings for common widgets
    mappings = {
        'CTkFrame': ttk.Frame,
        'CTkLabel': ttk.Label,
        'CTkButton': ttk.Button,
        'CTkEntry': ttk.Entry,
        'CTkCheckbox': ttk.Checkbutton,
        'CTkCheckbutton': ttk.Checkbutton,
        'CTkRadiobutton': ttk.Radiobutton,
        'CTkCombobox': ttk.Combobox,
        'CTkScrollableFrame': ttk.Frame,
        'CTkSpinbox': ttk.Spinbox if hasattr(ttk, 'Spinbox') else tk.Spinbox,
        'CTkProgressBar': ttk.Progressbar,
        'CTkCanvas': tk.Canvas,
        'CTkTabview': ttk.Notebook,
        'CTkOptionMenu': ttk.OptionMenu,
        'CTkSwitch': ttk.Checkbutton,
        'CTkTextbox': tk.Text,
        'CTkLabelFrame': ttk.LabelFrame,
        'CTkScale': ttk.Scale
    }
    
    if widget_name in mappings:
        return mappings[widget_name]
    
    # Default fallback
    return ttk.Widget