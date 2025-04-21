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
                # Basic settings
                "input_dir": self.app.input_dir.get(),
                "output_dir": self.app.output_dir.get(),
                "frame_rate": self.app.frame_rate.get(),
                "fill_ratio": self.app.fill_ratio.get(),
                "output_width": self.app.output_width.get(),
                "output_height": self.app.output_height.get(),
                "naming_pattern": self.app.naming_pattern.get(),
                "video_fps": self.app.video_fps.get(),
                "use_source_resolution": self.app.use_source_resolution.get(),
                
                # Processing flags
                "extract_frames": self.app.extract_frames.get(),
                "crop_mask_regions": self.app.crop_mask_regions.get(),
                "resize_images": self.app.resize_images.get(),
                "organize_files": self.app.organize_files.get(),
                "convert_to_video": self.app.convert_to_video.get(),
                "debug_mode": self.app.debug_mode.get(),
                "expand_masks": self.app.expand_masks.get(),
                "square_pad_images": self.app.square_pad_images.get(),
                "reinsert_crops_option": self.app.reinsert_crops_option.get(),
                "export_cropped_only": self.app.export_cropped_only.get(),
                
                # Frame extraction and mask settings
                "use_mask_video": self.app.use_mask_video.get(),
                "mask_video_path": self.app.mask_video_path.get(),
                
                # Square padding options
                "padding_color": self.app.padding_color.get(),
                "use_source_resolution_padding": self.app.use_source_resolution_padding.get(),
                "square_target_size": self.app.square_target_size.get(),
                
                # Resize options
                "resize_if_larger": self.app.resize_if_larger.get(),
                "max_width": self.app.max_width.get(),
                "max_height": self.app.max_height.get(),
                
                # Portrait crop options
                "portrait_crop_enabled": self.app.portrait_crop_enabled.get(),
                "portrait_crop_position": self.app.portrait_crop_position.get(),
                
                # Mask expansion options
                "mask_expand_iterations": self.app.mask_expand_iterations.get(),
                "mask_expand_kernel_size": self.app.mask_expand_kernel_size.get(),
                "mask_expand_preserve_structure": self.app.mask_expand_preserve_structure.get(),
                
                # Crop reinsertion options
                "source_images_dir": self.app.source_images_dir.get(),
                "reinsert_match_method": self.app.reinsert_match_method.get(),
                "reinsert_padding": self.app.reinsert_padding.get(),
                "use_center_position": self.app.use_center_position.get(),
                "reinsert_x": self.app.reinsert_x.get(),
                "reinsert_y": self.app.reinsert_y.get(),
                "reinsert_width": self.app.reinsert_width.get(),
                "reinsert_height": self.app.reinsert_height.get(),
                "use_enhanced_reinserter": self.app.use_enhanced_reinserter.get(),
                "reinsert_mask_only": self.app.reinsert_mask_only.get(),
                
                # Mask alignment options
                "reinsert_handle_different_masks": self.app.reinsert_handle_different_masks.get(),
                "reinsert_alignment_method": self.app.reinsert_alignment_method.get(),
                "reinsert_blend_mode": self.app.reinsert_blend_mode.get(),
                "reinsert_blend_extent": self.app.reinsert_blend_extent.get(),
                "reinsert_preserve_edges": self.app.reinsert_preserve_edges.get(),
                "reinsert_manual_offset_x": self.app.reinsert_manual_offset_x.get(),
                "reinsert_manual_offset_y": self.app.reinsert_manual_offset_y.get(),
                "reinsert_manual_scale_x": self.app.reinsert_manual_scale_x.get(),
                "reinsert_manual_scale_y": self.app.reinsert_manual_scale_y.get(),
                "reinsert_manual_rotation": self.app.reinsert_manual_rotation.get(),
                "use_translation_only": self.app.use_translation_only.get(),
                
                # Hair preservation options
                "preserve_hair_parting": self.app.preserve_hair_parting.get(),
                "extend_bangs": self.app.extend_bangs.get(),
                "bangs_extension_amount": self.app.bangs_extension_amount.get(),
                "bangs_width_ratio": self.app.bangs_width_ratio.get(),
                "bangs_min_opacity": self.app.bangs_min_opacity.get(),
                "use_bangs_only": self.app.use_bangs_only.get(),
                "protect_face_from_bangs": self.app.protect_face_from_bangs.get(),
                "face_protection_strength": self.app.face_protection_strength.get(),
                
                # Gap filling and artifact removal
                "fill_mask_gaps": self.app.fill_mask_gaps.get(),
                "use_color_specific_fill": self.app.use_color_specific_fill.get(),
                "remove_artifacts": self.app.remove_artifacts.get(),
                "use_textured_gap_filling": self.app.use_textured_gap_filling.get(),
                
                # Export options
                "export_with_alpha": self.app.export_with_alpha.get(),
                "export_cropped_video": self.app.export_cropped_video.get(),
                "cropped_video_fps": self.app.cropped_video_fps.get(),
                "cropped_video_format": self.app.cropped_video_format.get()
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
            
            # Apply configuration - basic settings
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
            
            # Processing flags
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
            if "expand_masks" in config:
                self.app.expand_masks.set(config["expand_masks"])
            if "square_pad_images" in config:
                self.app.square_pad_images.set(config["square_pad_images"])
            if "reinsert_crops_option" in config:
                self.app.reinsert_crops_option.set(config["reinsert_crops_option"])
            if "export_cropped_only" in config:
                self.app.export_cropped_only.set(config["export_cropped_only"])
                
            # Frame extraction and mask settings
            if "use_mask_video" in config:
                self.app.use_mask_video.set(config["use_mask_video"])
            if "mask_video_path" in config:
                self.app.mask_video_path.set(config["mask_video_path"])
            
            # Square padding options
            if "padding_color" in config:
                self.app.padding_color.set(config["padding_color"])
            if "use_source_resolution_padding" in config:
                self.app.use_source_resolution_padding.set(config["use_source_resolution_padding"])
            if "square_target_size" in config:
                self.app.square_target_size.set(config["square_target_size"])
            
            # Resize options
            if "resize_if_larger" in config:
                self.app.resize_if_larger.set(config["resize_if_larger"])
            if "max_width" in config:
                self.app.max_width.set(config["max_width"])
            if "max_height" in config:
                self.app.max_height.set(config["max_height"])
            
            # Portrait crop options
            if "portrait_crop_enabled" in config:
                self.app.portrait_crop_enabled.set(config["portrait_crop_enabled"])
            if "portrait_crop_position" in config:
                self.app.portrait_crop_position.set(config["portrait_crop_position"])
                
            # Mask expansion options
            if "mask_expand_iterations" in config:
                self.app.mask_expand_iterations.set(config["mask_expand_iterations"])
            if "mask_expand_kernel_size" in config:
                self.app.mask_expand_kernel_size.set(config["mask_expand_kernel_size"])
            if "mask_expand_preserve_structure" in config:
                self.app.mask_expand_preserve_structure.set(config["mask_expand_preserve_structure"])
                
            # Crop reinsertion options
            if "source_images_dir" in config:
                self.app.source_images_dir.set(config["source_images_dir"])
            if "reinsert_match_method" in config:
                self.app.reinsert_match_method.set(config["reinsert_match_method"])
            if "reinsert_padding" in config:
                self.app.reinsert_padding.set(config["reinsert_padding"])
            if "use_center_position" in config:
                self.app.use_center_position.set(config["use_center_position"])
            if "reinsert_x" in config:
                self.app.reinsert_x.set(config["reinsert_x"])
            if "reinsert_y" in config:
                self.app.reinsert_y.set(config["reinsert_y"])
            if "reinsert_width" in config:
                self.app.reinsert_width.set(config["reinsert_width"])
            if "reinsert_height" in config:
                self.app.reinsert_height.set(config["reinsert_height"])
            if "use_enhanced_reinserter" in config:
                self.app.use_enhanced_reinserter.set(config["use_enhanced_reinserter"])
            if "reinsert_mask_only" in config:
                self.app.reinsert_mask_only.set(config["reinsert_mask_only"])
                
            # Mask alignment options
            if "reinsert_handle_different_masks" in config:
                self.app.reinsert_handle_different_masks.set(config["reinsert_handle_different_masks"])
            if "reinsert_alignment_method" in config:
                self.app.reinsert_alignment_method.set(config["reinsert_alignment_method"])
            if "reinsert_blend_mode" in config:
                self.app.reinsert_blend_mode.set(config["reinsert_blend_mode"])
            if "reinsert_blend_extent" in config:
                self.app.reinsert_blend_extent.set(config["reinsert_blend_extent"])
            if "reinsert_preserve_edges" in config:
                self.app.reinsert_preserve_edges.set(config["reinsert_preserve_edges"])
            if "reinsert_manual_offset_x" in config:
                self.app.reinsert_manual_offset_x.set(config["reinsert_manual_offset_x"])
            if "reinsert_manual_offset_y" in config:
                self.app.reinsert_manual_offset_y.set(config["reinsert_manual_offset_y"])
            if "reinsert_manual_scale_x" in config:
                self.app.reinsert_manual_scale_x.set(config["reinsert_manual_scale_x"])
            if "reinsert_manual_scale_y" in config:
                self.app.reinsert_manual_scale_y.set(config["reinsert_manual_scale_y"])
            if "reinsert_manual_rotation" in config:
                self.app.reinsert_manual_rotation.set(config["reinsert_manual_rotation"])
            if "use_translation_only" in config:
                self.app.use_translation_only.set(config["use_translation_only"])
                
            # Hair preservation options
            if "preserve_hair_parting" in config:
                self.app.preserve_hair_parting.set(config["preserve_hair_parting"])
            if "extend_bangs" in config:
                self.app.extend_bangs.set(config["extend_bangs"])
            if "bangs_extension_amount" in config:
                self.app.bangs_extension_amount.set(config["bangs_extension_amount"])
            if "bangs_width_ratio" in config:
                self.app.bangs_width_ratio.set(config["bangs_width_ratio"])
            if "bangs_min_opacity" in config:
                self.app.bangs_min_opacity.set(config["bangs_min_opacity"])
            if "use_bangs_only" in config:
                self.app.use_bangs_only.set(config["use_bangs_only"])
            if "protect_face_from_bangs" in config:
                self.app.protect_face_from_bangs.set(config["protect_face_from_bangs"])
            if "face_protection_strength" in config:
                self.app.face_protection_strength.set(config["face_protection_strength"])
                
            # Gap filling and artifact removal
            if "fill_mask_gaps" in config:
                self.app.fill_mask_gaps.set(config["fill_mask_gaps"])
            if "use_color_specific_fill" in config:
                self.app.use_color_specific_fill.set(config["use_color_specific_fill"])
            if "remove_artifacts" in config:
                self.app.remove_artifacts.set(config["remove_artifacts"])
            if "use_textured_gap_filling" in config:
                self.app.use_textured_gap_filling.set(config["use_textured_gap_filling"])
                
            # Export options
            if "export_with_alpha" in config:
                self.app.export_with_alpha.set(config["export_with_alpha"])
            if "export_cropped_video" in config:
                self.app.export_cropped_video.set(config["export_cropped_video"])
            if "cropped_video_fps" in config:
                self.app.cropped_video_fps.set(config["cropped_video_fps"])
            if "cropped_video_format" in config:
                self.app.cropped_video_format.set(config["cropped_video_format"])
            
            # Update UI controls
            self.app.config_tab._toggle_resolution_controls()
            self.app.config_tab._toggle_mask_video_controls()
            self.app.config_tab._toggle_conditional_resize_controls()
            self.app.config_tab._toggle_square_padding_controls()
            self.app.config_tab._toggle_portrait_crop_controls()
            
            # Additional UI toggles for newer features
            if hasattr(self.app.config_tab, '_toggle_mask_alignment_controls'):
                self.app.config_tab._toggle_mask_alignment_controls()
            if hasattr(self.app.config_tab, '_toggle_video_export_controls'):
                self.app.config_tab._toggle_video_export_controls()
            
            # Try to load preview if input directory exists
            if os.path.isdir(self.app.input_dir.get()):
                self.app.input_output_tab._load_preview_images()
            
            self.app.status_label.config(text=f"Configuration loaded from {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
