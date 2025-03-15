"""
Configuration Manager for Dataset Preparation Tool
Handles saving and loading of application configuration.
"""

import json
import os
from tkinter import filedialog, messagebox

class ConfigManager:
    """Manages saving and loading configuration settings."""
    
    def __init__(self, app):
        """
        Initialize the configuration manager.
        
        Args:
            app: The main application with shared variables
        """
        self.app = app
    
    def save_config(self):
        """Save the current configuration to a file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Configuration"
        )
        
        if not file_path:
            return
        
        try:
            # Collect configuration
            config = {
                "input_dir": self.app.input_dir.get(),
                "output_dir": self.app.output_dir.get(),
                "frame_rate": self.app.frame_rate.get(),
                "fill_ratio": self.app.fill_ratio.get(),
                "output_width": self.app.output_width.get(),
                "output_height": self.app.output_height.get(),
                "naming_pattern": self.app.naming_pattern.get(),
                "video_fps": self.app.video_fps.get(),
                "use_source_resolution": self.app.use_source_resolution.get(),
                "extract_frames": self.app.extract_frames.get(),
                "crop_mask_regions": self.app.crop_mask_regions.get(),
                "resize_images": self.app.resize_images.get(),
                "organize_files": self.app.organize_files.get(),
                "convert_to_video": self.app.convert_to_video.get(),
                "debug_mode": self.app.debug_mode.get(),
                "use_mask_video": self.app.use_mask_video.get(),
                "mask_video_path": self.app.mask_video_path.get(),
                "square_pad_images": self.app.square_pad_images.get(),
                "padding_color": self.app.padding_color.get(),
                "use_source_resolution_padding": self.app.use_source_resolution_padding.get(),
                "square_target_size": self.app.square_target_size.get(),
                "resize_if_larger": self.app.resize_if_larger.get(),
                "max_width": self.app.max_width.get(),
                "max_height": self.app.max_height.get(),
                "portrait_crop_enabled": self.app.portrait_crop_enabled.get(),
                "portrait_crop_position": self.app.portrait_crop_position.get(),
            }
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=4)
                
            self.app.status_label.config(text=f"Configuration saved to {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
    
    def load_config(self):
        """Load configuration from a file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Configuration"
        )
        
        if not file_path:
            return
        
        try:
            # Load from file
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            # Apply configuration
            if "input_dir" in config:
                self.app.input_dir.set(config["input_dir"])
            if "output_dir" in config:
                self.app.output_dir.set(config["output_dir"])
            if "frame_rate" in config:
                self.app.frame_rate.set(config["frame_rate"])
            if "fill_ratio" in config:
                self.app.fill_ratio.set(config["fill_ratio"])
            if "output_width" in config:
                self.app.output_width.set(config["output_width"])
            if "output_height" in config:
                self.app.output_height.set(config["output_height"])
            if "naming_pattern" in config:
                self.app.naming_pattern.set(config["naming_pattern"])
            if "video_fps" in config:
                self.app.video_fps.set(config["video_fps"])
            if "use_source_resolution" in config:
                self.app.use_source_resolution.set(config["use_source_resolution"])
            if "extract_frames" in config:
                self.app.extract_frames.set(config["extract_frames"])
            if "crop_mask_regions" in config:
                self.app.crop_mask_regions.set(config["crop_mask_regions"])
            if "resize_images" in config:
                self.app.resize_images.set(config["resize_images"])
            if "organize_files" in config:
                self.app.organize_files.set(config["organize_files"])
            if "convert_to_video" in config:
                self.app.convert_to_video.set(config["convert_to_video"])
            if "debug_mode" in config:
                self.app.debug_mode.set(config["debug_mode"])
            if "use_mask_video" in config:
                self.app.use_mask_video.set(config["use_mask_video"])
            if "mask_video_path" in config:
                self.app.mask_video_path.set(config["mask_video_path"])
            if "square_pad_images" in config:
                self.app.square_pad_images.set(config["square_pad_images"])
            if "padding_color" in config:
                self.app.padding_color.set(config["padding_color"])
            if "use_source_resolution_padding" in config:
                self.app.use_source_resolution_padding.set(config["use_source_resolution_padding"])
            if "square_target_size" in config:
                self.app.square_target_size.set(config["square_target_size"])
            if "resize_if_larger" in config:
                self.app.resize_if_larger.set(config["resize_if_larger"])
            if "max_width" in config:
                self.app.max_width.set(config["max_width"])
            if "max_height" in config:
                self.app.max_height.set(config["max_height"])
            if "portrait_crop_enabled" in config:
                self.app.portrait_crop_enabled.set(config["portrait_crop_enabled"])
            if "portrait_crop_position" in config:
                self.app.portrait_crop_position.set(config["portrait_crop_position"])
            
            # Update UI controls
            self.app.config_tab._toggle_resolution_controls()
            self.app.config_tab._toggle_mask_video_controls()
            self.app.config_tab._toggle_conditional_resize_controls()
            self.app.config_tab._toggle_square_padding_controls()
            self.app.config_tab._toggle_portrait_crop_controls()
            
            # Try to load preview if input directory exists
            if os.path.isdir(self.app.input_dir.get()):
                self.app.input_output_tab._load_preview_images()
            
            self.app.status_label.config(text=f"Configuration loaded from {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
