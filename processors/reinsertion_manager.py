"""
Unified Reinsertion Manager for Dataset Preparation Tool
Handles all reinsertion types with a simplified interface.
"""

import os
import cv2
import numpy as np
import re
import json

class ReinsertionManager:
    """Manages all reinsertion types with a simplified interface."""
    
    def __init__(self, app):
        """
        Initialize the reinsertion manager.
        
        Args:
            app: The main application with shared variables and UI controls
        """
        self.app = app
    
    def reinsert_crops(self, input_dir, output_dir):
        """
        Main entry point for all reinsertion types.
        
        Args:
            input_dir: Input directory containing processed images
            output_dir: Output directory for reinserted images
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create output directory for reinserted images
        reinsert_output_dir = os.path.join(output_dir, "reinserted")
        os.makedirs(reinsert_output_dir, exist_ok=True)
        
        # Create debug directory if debug mode is enabled
        debug_dir = None
        if self.app.debug_mode.get():
            debug_dir = os.path.join(output_dir, "reinsert_debug")
            os.makedirs(debug_dir, exist_ok=True)
            print(f"Debug mode enabled, saving visualizations to: {debug_dir}")
        
        # Determine which reinsertion method to use
        if self.app.use_smart_hair_reinserter.get():
            print("Using Smart Hair Reinserter")
            return self._reinsert_hair(input_dir, reinsert_output_dir, debug_dir)
        elif self.app.use_enhanced_reinserter.get():
            print("Using Enhanced Reinserter")
            return self._reinsert_enhanced(input_dir, reinsert_output_dir, debug_dir)
        else:
            print("Using Standard Reinserter")
            return self._reinsert_standard(input_dir, reinsert_output_dir, debug_dir)
    
    def _reinsert_standard(self, input_dir, output_dir, debug_dir=None):
        """Standard reinsertion (original functionality)."""
        # Log settings
        print(f"Standard Reinsertion: Input Dir: {input_dir}")
        print(f"Standard Reinsertion: Source Dir: {self.app.source_images_dir.get()}")
        print(f"Standard Reinsertion: Output Dir: {output_dir}")
        print(f"Standard Reinsertion: Mask-only mode: {self.app.reinsert_mask_only.get()}")
        
        # Find all processed images and their corresponding masks
        processed_images = self._find_processed_images(input_dir)
        
        if not processed_images:
            self.app.status_label.config(text="No processed images with masks found in input directory.")
            return False
        
        # Get source images
        source_images = self._load_source_images()
        
        if not source_images:
            self.app.status_label.config(text="Source images directory not set or invalid.")
            return False
        
        # Process each image
        total_images = len(processed_images)
        processed_count = 0
        failed_count = 0
        
        for idx, img_data in enumerate(processed_images):
            if not self.app.processing:  # Check if processing was cancelled
                break
            
            processed_path = img_data['processed_path']
            mask_path = img_data['mask_path']
            filename = img_data['filename']
            
            try:
                # Match processed image to source image
                source_filename = self._match_source_image(filename, source_images)
                
                if not source_filename:
                    self.app.status_label.config(text=f"Source image not found for {filename}")
                    failed_count += 1
                    continue
                
                source_path = source_images[source_filename]
                
                # Load images
                source_img = cv2.imread(source_path)
                processed_img = cv2.imread(processed_path)
                
                if source_img is None or processed_img is None:
                    self.app.status_label.config(text=f"Error loading images for {filename}")
                    failed_count += 1
                    continue
                
                # Load mask
                mask = None
                if mask_path and os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        print(f"Warning: Could not load mask from {mask_path}")
                
                # Get dimensions
                source_h, source_w = source_img.shape[:2]
                processed_h, processed_w = processed_img.shape[:2]
                
                # Resize processed image and mask to match source if dimensions differ
                if source_w != processed_w or source_h != processed_h:
                    print(f"Resizing processed image from {processed_w}x{processed_h} to {source_w}x{source_h}")
                    processed_img = cv2.resize(processed_img, (source_w, source_h), 
                                            interpolation=cv2.INTER_LANCZOS4)
                    if mask is not None:
                        mask = cv2.resize(mask, (source_w, source_h), 
                                       interpolation=cv2.INTER_NEAREST)
                
                # CRITICAL FIX: Always create a result_img based on source_img, not processed_img
                result_img = source_img.copy()
                
                # Apply insertion based on mask option
                if self.app.reinsert_mask_only.get() and mask is not None:
                    # Mask-only insertion (only replace masked pixels)
                    mask_float = mask.astype(float) / 255.0
                    mask_float_3d = np.stack([mask_float] * 3, axis=2)
                    result_img = source_img * (1 - mask_float_3d) + processed_img * mask_float_3d
                else:
                    # Check for crop info to position the processed image
                    crop_info_path = os.path.splitext(processed_path)[0] + "_crop_info.json"
                    if os.path.exists(crop_info_path):
                        try:
                            with open(crop_info_path, 'r') as f:
                                crop_info = json.load(f)
                            
                            # Get position from crop info
                            if all(k in crop_info for k in ["crop_x", "crop_y", "crop_width", "crop_height"]):
                                x_pos = crop_info["crop_x"]
                                y_pos = crop_info["crop_y"]
                                width = crop_info["crop_width"]
                                height = crop_info["crop_height"]
                                
                                # Create a region of interest and place the processed image there
                                roi = result_img[y_pos:y_pos+height, x_pos:x_pos+width]
                                processed_roi = processed_img[y_pos:y_pos+height, x_pos:x_pos+width]
                                
                                # Only copy if dimensions match
                                if roi.shape == processed_roi.shape:
                                    result_img[y_pos:y_pos+height, x_pos:x_pos+width] = processed_roi
                                else:
                                    print(f"Warning: ROI dimensions mismatch: {roi.shape} vs {processed_roi.shape}")
                                    # Fall back to full image replacement with blending
                                    alpha = 0.7  # Blend factor
                                    result_img = cv2.addWeighted(source_img, 1-alpha, processed_img, alpha, 0)
                        except Exception as e:
                            print(f"Error using crop info: {str(e)}")
                            # Fall back to full image replacement
                            alpha = 0.7  # Blend factor
                            result_img = cv2.addWeighted(source_img, 1-alpha, processed_img, alpha, 0)
                    else:
                        # If no crop info, use center positioning
                        center_x = (source_w - processed_w) // 2 if processed_w < source_w else 0
                        center_y = (source_h - processed_h) // 2 if processed_h < source_h else 0
                        
                        # If dimensions match, we can do a direct replacement of the center region
                        if processed_w < source_w and processed_h < source_h:
                            result_img[center_y:center_y+processed_h, center_x:center_x+processed_w] = processed_img
                        else:
                            # If processed image is same size or larger, use alpha blending
                            alpha = 0.7  # Blend factor
                            result_img = cv2.addWeighted(source_img, 1-alpha, processed_img, alpha, 0)
                
                # Save the result
                output_path = os.path.join(output_dir, f"reinserted_{filename}")
                cv2.imwrite(output_path, result_img)
                
                # Save debug comparison if enabled
                if debug_dir:
                    # Create side-by-side comparison of source, processed, and result
                    # Resize all images to the same height for proper comparison
                    comp_height = 300
                    
                    # Calculate scaling factors
                    src_scale = comp_height / source_h
                    src_width = int(source_w * src_scale)
                    
                    proc_scale = comp_height / processed_h
                    proc_width = int(processed_w * proc_scale)
                    
                    res_scale = comp_height / result_img.shape[0]
                    res_width = int(result_img.shape[1] * res_scale)
                    
                    # Resize images
                    src_resized = cv2.resize(source_img, (src_width, comp_height))
                    proc_resized = cv2.resize(processed_img, (proc_width, comp_height))
                    res_resized = cv2.resize(result_img, (res_width, comp_height))
                    
                    # Create comparison image
                    comparison = np.hstack((src_resized, proc_resized, res_resized))
                    cv2.imwrite(os.path.join(debug_dir, f"comparison_{filename}"), comparison)
                    
                    # Add labels
                    cv2.putText(comparison, "Source", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(comparison, "Processed", (src_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(comparison, "Result", (src_width + proc_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imwrite(os.path.join(debug_dir, f"labeled_comparison_{filename}"), comparison)
                
                processed_count += 1
                
            except Exception as e:
                self.app.status_label.config(text=f"Error processing {filename}: {str(e)}")
                print(f"Error in _reinsert_standard: {str(e)}")
                import traceback
                traceback.print_exc()
                failed_count += 1
                continue
            
            # Update progress
            self._update_progress(idx, total_images)
        
        # Final status update
        self._update_final_status(processed_count, failed_count)
        return processed_count > 0
    
    def _reinsert_enhanced(self, input_dir, output_dir, debug_dir=None):
        """Enhanced reinsertion with resolution handling."""
        # Log settings
        print(f"Enhanced Reinsertion: Input Dir: {input_dir}")
        print(f"Enhanced Reinsertion: Source Dir: {self.app.source_images_dir.get()}")
        print(f"Enhanced Reinsertion: Output Dir: {output_dir}")
        
        # Find all processed images and their corresponding masks
        processed_images = self._find_processed_images(input_dir)
        
        if not processed_images:
            self.app.status_label.config(text="No processed images with masks found in input directory.")
            return False
        
        # Get source images
        source_images = self._load_source_images()
        
        if not source_images:
            self.app.status_label.config(text="Source images directory not set or invalid.")
            return False
        
        # Process each image
        total_images = len(processed_images)
        processed_count = 0
        failed_count = 0
        
        for idx, img_data in enumerate(processed_images):
            if not self.app.processing:  # Check if processing was cancelled
                break
            
            processed_path = img_data['processed_path']
            mask_path = img_data['mask_path']
            filename = img_data['filename']
            
            try:
                # Match processed image to source image
                source_filename = self._match_source_image(filename, source_images)
                
                if not source_filename:
                    self.app.status_label.config(text=f"Source image not found for {filename}")
                    failed_count += 1
                    continue
                
                source_path = source_images[source_filename]
                
                # Load images
                source_img = cv2.imread(source_path)
                processed_img = cv2.imread(processed_path)
                
                if source_img is None or processed_img is None:
                    self.app.status_label.config(text=f"Error loading images for {filename}")
                    failed_count += 1
                    continue
                
                # Load mask
                mask = None
                if mask_path and os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                else:
                    self.app.status_label.config(text=f"Mask not found for {filename}")
                    failed_count += 1
                    continue
                
                # Get dimensions
                source_h, source_w = source_img.shape[:2]
                processed_h, processed_w = processed_img.shape[:2]
                
                # Resize processed image and mask to match source if dimensions differ
                if source_w != processed_w or source_h != processed_h:
                    processed_img = cv2.resize(processed_img, (source_w, source_h), 
                                            interpolation=cv2.INTER_LANCZOS4)
                    mask = cv2.resize(mask, (source_w, source_h), 
                                    interpolation=cv2.INTER_NEAREST)
                
                # Check if we need to handle different masks
                source_mask = self._find_source_mask(source_path)
                
                # Apply blending based on settings
                if self.app.reinsert_handle_different_masks.get() and source_mask is not None:
                    # Enhanced blending with mask alignment
                    result_img = self._blend_with_mask_alignment(
                        source_img, 
                        processed_img, 
                        mask, 
                        source_mask,
                        self.app.reinsert_blend_mode.get(),
                        self.app.reinsert_blend_extent.get(),
                        self.app.vertical_alignment_bias.get(),
                        self.app.soft_edge_width.get(),
                        debug_dir
                    )
                else:
                    # Standard alpha blending with feathering
                    result_img = self._apply_feathered_alpha_blend(
                        source_img,
                        processed_img,
                        mask,
                        self.app.soft_edge_width.get()
                    )
                
                # Save the result
                output_path = os.path.join(output_dir, f"reinserted_{filename}")
                cv2.imwrite(output_path, result_img)
                
                # Save debug comparison if enabled
                if debug_dir:
                    comparison = np.hstack((source_img, result_img))
                    cv2.imwrite(os.path.join(debug_dir, f"comparison_{filename}"), comparison)
                
                processed_count += 1
                
            except Exception as e:
                self.app.status_label.config(text=f"Error processing {filename}: {str(e)}")
                print(f"Error in _reinsert_enhanced: {str(e)}")
                import traceback
                traceback.print_exc()
                failed_count += 1
                continue
            
            # Update progress
            self._update_progress(idx, total_images)
        
        # Final status update
        self._update_final_status(processed_count, failed_count)
        return processed_count > 0
    
    def _reinsert_hair(self, input_dir, output_dir, debug_dir=None):
        """Smart hair reinsertion."""
        # Log settings
        print(f"Smart Hair Reinsertion: Input Dir: {input_dir}")
        print(f"Smart Hair Reinsertion: Source Dir: {self.app.source_images_dir.get()}")
        print(f"Smart Hair Reinsertion: Output Dir: {output_dir}")
        print(f"Smart Hair Reinsertion: Vertical Bias: {self.app.vertical_alignment_bias.get()}")
        print(f"Smart Hair Reinsertion: Soft Edge Width: {self.app.soft_edge_width.get()}")
        
        # Find all processed images and their corresponding masks
        processed_images = self._find_processed_images(input_dir)
        
        if not processed_images:
            self.app.status_label.config(text="No processed images with masks found in input directory.")
            return False
        
        # Get source images
        source_images = self._load_source_images()
        
        if not source_images:
            self.app.status_label.config(text="Source images directory not set or invalid.")
            return False
        
        # Process each image
        total_images = len(processed_images)
        processed_count = 0
        failed_count = 0
        
        for idx, img_data in enumerate(processed_images):
            if not self.app.processing:  # Check if processing was cancelled
                break
            
            processed_path = img_data['processed_path']
            mask_path = img_data['mask_path']
            filename = img_data['filename']
            
            try:
                # Match processed image to source image
                source_filename = self._match_source_image(filename, source_images)
                
                if not source_filename:
                    self.app.status_label.config(text=f"Source image not found for {filename}")
                    failed_count += 1
                    continue
                
                source_path = source_images[source_filename]
                
                # Find source mask
                source_mask_path = self._find_source_mask_path(source_path)
                
                # Load images
                source_img = cv2.imread(source_path)
                processed_img = cv2.imread(processed_path)
                
                if source_img is None or processed_img is None:
                    self.app.status_label.config(text=f"Error loading images for {filename}")
                    failed_count += 1
                    continue
                
                # Load mask
                mask = None
                if mask_path and os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                else:
                    self.app.status_label.config(text=f"Mask not found for {filename}")
                    failed_count += 1
                    continue
                
                # Load source mask if available
                source_mask = None
                if source_mask_path and os.path.exists(source_mask_path):
                    source_mask = cv2.imread(source_mask_path, cv2.IMREAD_GRAYSCALE)
                
                # Get dimensions
                source_h, source_w = source_img.shape[:2]
                processed_h, processed_w = processed_img.shape[:2]
                
                # Resize processed image and mask to match source if dimensions differ
                if source_w != processed_w or source_h != processed_h:
                    processed_img = cv2.resize(processed_img, (source_w, source_h), 
                                            interpolation=cv2.INTER_LANCZOS4)
                    mask = cv2.resize(mask, (source_w, source_h), 
                                    interpolation=cv2.INTER_NEAREST)
                
                # Save original images for debug purposes
                if debug_dir:
                    cv2.imwrite(os.path.join(debug_dir, f"source_original_{filename}"), source_img)
                    cv2.imwrite(os.path.join(debug_dir, f"processed_original_{filename}"), processed_img)
                    cv2.imwrite(os.path.join(debug_dir, f"mask_original_{filename}"), mask)
                    if source_mask is not None:
                        cv2.imwrite(os.path.join(debug_dir, f"source_mask_original_{filename}"), source_mask)
                
                # Apply color correction if enabled
                if self.app.hair_color_correction.get():
                    processed_img = self._match_hair_color(
                        source_img, 
                        processed_img, 
                        source_mask, 
                        mask, 
                        strength=0.5
                    )
                
                # Apply smart hair alignment
                result_img = self._apply_smart_hair_alignment(
                    source_img,
                    processed_img,
                    mask,
                    source_mask,
                    self.app.vertical_alignment_bias.get(),
                    self.app.soft_edge_width.get(),
                    self.app.hair_top_alignment.get(),
                    debug_dir
                )
                
                # Save the result
                output_path = os.path.join(output_dir, f"reinserted_{filename}")
                cv2.imwrite(output_path, result_img)
                
                # Create comparison image for debugging
                if debug_dir:
                    comparison = np.hstack((source_img, result_img))
                    cv2.imwrite(os.path.join(debug_dir, f"comparison_{filename}"), comparison)
                
                processed_count += 1
                
            except Exception as e:
                self.app.status_label.config(text=f"Error processing {filename}: {str(e)}")
                print(f"Error in _reinsert_hair: {str(e)}")
                import traceback
                traceback.print_exc()
                failed_count += 1
                continue
            
            # Update progress
            self._update_progress(idx, total_images)
        
        # Final status update
        self._update_final_status(processed_count, failed_count)
        return processed_count > 0
    
    # ===== HELPER METHODS =====
    
    def _find_processed_images(self, input_dir):
        """Find all processed images and their corresponding masks."""
        processed_images = []
        
        for root, dirs, files in os.walk(input_dir):
            # Skip if this is a 'masks' directory
            if os.path.basename(root).lower() == "masks":
                continue
                
            # Find image files
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    processed_path = os.path.join(root, file)
                    
                    # Look for corresponding mask
                    mask_path = None
                    masks_dir = os.path.join(root, "masks")
                    if os.path.isdir(masks_dir):
                        # Try exact filename match first
                        potential_mask = os.path.join(masks_dir, file)
                        if os.path.exists(potential_mask):
                            mask_path = potential_mask
                        else:
                            # Try different extensions
                            base_name = os.path.splitext(file)[0]
                            for ext in ['.png', '.jpg', '.jpeg']:
                                potential_mask = os.path.join(masks_dir, base_name + ext)
                                if os.path.exists(potential_mask):
                                    mask_path = potential_mask
                                    break
                    
                    # Add to processing list
                    processed_images.append({
                        'processed_path': processed_path,
                        'mask_path': mask_path,
                        'filename': file
                    })
        
        return processed_images
    
    def _load_source_images(self):
        """Load all source images."""
        source_dir = self.app.source_images_dir.get()
        if not source_dir or not os.path.isdir(source_dir):
            return {}
        
        source_images = {}
        for root, dirs, files in os.walk(source_dir):
            # Skip if this is a 'masks' directory
            if os.path.basename(root).lower() == "masks":
                continue
                
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    source_images[file] = os.path.join(root, file)
        
        return source_images
    
    def _match_source_image(self, processed_filename, source_images):
        """Match processed image filename to source image."""
        # Try exact match first
        if processed_filename in source_images:
            return processed_filename
        
        # Try matching base name without extension
        base_name = os.path.splitext(processed_filename)[0]
        for source_file in source_images:
            source_base = os.path.splitext(source_file)[0]
            if source_base == base_name:
                return source_file
        
        # Try looking for crop info JSON for original image reference
        crop_info_path = os.path.join(
            os.path.dirname(self.app.input_dir.get()),
            f"{base_name}_crop_info.json"
        )
        
        if os.path.exists(crop_info_path):
            try:
                with open(crop_info_path, 'r') as f:
                    crop_info = json.load(f)
                    if 'original_image' in crop_info:
                        original_filename = crop_info['original_image']
                        if original_filename in source_images:
                            return original_filename
            except Exception as e:
                print(f"Error reading crop info: {str(e)}")
        
        # If there's only one source image, use it
        if len(source_images) == 1:
            return list(source_images.keys())[0]
        
        return None
    
    def _find_source_mask(self, source_path):
        """Find mask for a source image."""
        source_dir = os.path.dirname(source_path)
        source_name = os.path.basename(source_path)
        potential_mask_dir = os.path.join(source_dir, "masks")
        
        if os.path.isdir(potential_mask_dir):
            # Check for mask with same name
            potential_mask = os.path.join(potential_mask_dir, source_name)
            if os.path.exists(potential_mask):
                return cv2.imread(potential_mask, cv2.IMREAD_GRAYSCALE)
            
            # Try different extensions
            base_name = os.path.splitext(source_name)[0]
            for ext in ['.png', '.jpg', '.jpeg']:
                alt_mask_path = os.path.join(potential_mask_dir, base_name + ext)
                if os.path.exists(alt_mask_path):
                    return cv2.imread(alt_mask_path, cv2.IMREAD_GRAYSCALE)
        
        return None
    
    def _find_source_mask_path(self, source_path):
        """Find the path to a source mask."""
        source_dir = os.path.dirname(source_path)
        source_name = os.path.basename(source_path)
        potential_mask_dir = os.path.join(source_dir, "masks")
        
        if os.path.isdir(potential_mask_dir):
            # Check for mask with same name
            potential_mask = os.path.join(potential_mask_dir, source_name)
            if os.path.exists(potential_mask):
                return potential_mask
            
            # Try different extensions
            base_name = os.path.splitext(source_name)[0]
            for ext in ['.png', '.jpg', '.jpeg']:
                alt_mask_path = os.path.join(potential_mask_dir, base_name + ext)
                if os.path.exists(alt_mask_path):
                    return alt_mask_path
        
        return None
    
    def _update_progress(self, idx, total):
        """Update progress bar and status label."""
        progress = (idx + 1) / total * 100
        self.app.progress_bar['value'] = min(progress, 100)
        self.app.status_label.config(text=f"Processed {idx+1}/{total} images")
        self.app.root.update_idletasks()
    
    def _update_final_status(self, processed_count, failed_count):
        """Update final status message."""
        if failed_count > 0:
            self.app.status_label.config(
                text=f"Reinsertion completed. Processed {processed_count} images. Failed: {failed_count}."
            )
        else:
            self.app.status_label.config(
                text=f"Reinsertion completed. Successfully processed {processed_count} images."
            )
        
        self.app.progress_bar['value'] = 100
    
    def _blend_with_mask_alignment(self, source_img, processed_img, processed_mask, source_mask, 
                                 blend_mode, blend_extent, vertical_bias, soft_edge_width, debug_dir=None):
        """Blend images with mask alignment."""
        # Ensure we have binary masks
        _, proc_mask_binary = cv2.threshold(processed_mask, 127, 255, cv2.THRESH_BINARY)
        _, source_mask_binary = cv2.threshold(source_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Calculate vertical alignment with bias
        proc_points = np.argwhere(proc_mask_binary > 0)
        source_points = np.argwhere(source_mask_binary > 0)
        
        if len(proc_points) == 0 or len(source_points) == 0:
            # Fall back to simple alpha blend if either mask is empty
            return self._apply_feathered_alpha_blend(
                source_img, 
                processed_img, 
                processed_mask,
                soft_edge_width
            )
        
        # Get top of masks
        proc_top = proc_points[:, 0].min()
        source_top = source_points[:, 0].min()
        
        # Calculate vertical and horizontal alignment
        dy = (source_top - proc_top) - vertical_bias
        
        # Get centroids for horizontal alignment
        source_moments = cv2.moments(source_mask_binary)
        proc_moments = cv2.moments(proc_mask_binary)
        
        if source_moments["m00"] > 0 and proc_moments["m00"] > 0:
            source_cx = int(source_moments["m10"] / source_moments["m00"])
            proc_cx = int(proc_moments["m10"] / proc_moments["m00"])
            dx = source_cx - proc_cx
        else:
            dx = 0
        
        # Apply alignment transformation
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned_img = cv2.warpAffine(processed_img, M, (processed_img.shape[1], processed_img.shape[0]))
        aligned_mask = cv2.warpAffine(processed_mask, M, (processed_mask.shape[1], processed_mask.shape[0]))
        
        # Save alignment debug images
        if debug_dir:
            # Create a colored visualization of mask alignment
            mask_viz = np.zeros((source_img.shape[0], source_img.shape[1], 3), dtype=np.uint8)
            mask_viz[source_mask_binary > 0] = [0, 255, 0]  # Source mask in green
            mask_viz[aligned_mask > 0] = [0, 0, 255]  # Aligned mask in red
            # Overlap will appear as cyan
            cv2.imwrite(os.path.join(debug_dir, "mask_alignment_viz.png"), mask_viz)
        
        # Apply blending based on mode
        if blend_mode == "alpha":
            result_img = self._apply_alpha_blend(source_img, aligned_img, aligned_mask, blend_extent)
        elif blend_mode == "poisson":
            result_img = self._apply_poisson_blend(source_img, aligned_img, aligned_mask)
        else:  # feathered
            result_img = self._apply_feathered_alpha_blend(source_img, aligned_img, aligned_mask, soft_edge_width)
        
        return result_img
    
    def _apply_alpha_blend(self, source_img, processed_img, mask, blend_extent=0):
        """Apply simple alpha blending."""
        mask_float = mask.astype(float) / 255.0
        
        # Apply feathering if blend_extent > 0
        if blend_extent > 0:
            kernel = np.ones((blend_extent, blend_extent), np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=1)
            border = dilated & ~mask
            
            dist = cv2.distanceTransform(~border, cv2.DIST_L2, 3)
            dist[dist > blend_extent] = blend_extent
            
            feather = dist / blend_extent
            
            mask_float[border > 0] = 1.0 - feather[border > 0]
        
        # Create 3-channel mask
        mask_float_3d = np.stack([mask_float] * 3, axis=2)
        
        # Apply blending
        result_img = source_img * (1 - mask_float_3d) + processed_img * mask_float_3d
        
        return np.clip(result_img, 0, 255).astype(np.uint8)
    
    def _apply_poisson_blend(self, source_img, processed_img, mask):
        """Apply Poisson blending."""
        try:
            # Ensure mask is uint8
            mask_uint8 = mask.astype(np.uint8)
            
            # Find center of mask
            moments = cv2.moments(mask_uint8)
            if moments["m00"] > 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
                center = (center_x, center_y)
                
                # Apply seamless cloning
                result_img = cv2.seamlessClone(processed_img, source_img, mask_uint8, center, cv2.NORMAL_CLONE)
                return result_img
            else:
                # Fallback to alpha blending if mask is empty
                mask_float = mask.astype(float) / 255.0
                mask_float_3d = np.stack([mask_float] * 3, axis=2)
                result_img = source_img * (1 - mask_float_3d) + processed_img * mask_float_3d
                return np.clip(result_img, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"Poisson blending failed: {str(e)}, falling back to alpha blending")
            # Fallback to alpha blending
            mask_float = mask.astype(float) / 255.0
            mask_float_3d = np.stack([mask_float] * 3, axis=2)
            result_img = source_img * (1 - mask_float_3d) + processed_img * mask_float_3d
            return np.clip(result_img, 0, 255).astype(np.uint8)
    
    def _apply_feathered_alpha_blend(self, source_img, processed_img, mask, feather_pixels=15):
        """Apply alpha blending with feathered edges for smoother transitions."""
        # Ensure mask is binary
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Create a soft-edged mask for natural transitions
        kernel_size = max(1, feather_pixels // 2)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Dilate and erode mask to get transition regions
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=2)
        eroded_mask = cv2.erode(binary_mask, kernel, iterations=2)
        
        # Create a distance transform for the transition region
        inner_region = eroded_mask
        outer_region = dilated_mask & ~eroded_mask
        
        # Create a soft, feathered alpha mask
        alpha_mask = np.zeros_like(binary_mask, dtype=np.float32)
        alpha_mask[inner_region > 0] = 1.0  # Fully processed image
        
        # Calculate feathering for transition region
        if np.any(outer_region):
            # Distance transform for transition region
            dist_transform = cv2.distanceTransform(outer_region, cv2.DIST_L2, 5)
            max_dist = np.max(dist_transform)
            if max_dist > 0:
                # Normalize distances and invert (further = lower alpha)
                normalized_dist = dist_transform / max_dist
                # Alpha decreases as we move away from inner region
                alpha_mask[outer_region > 0] = 1.0 - normalized_dist[outer_region > 0]
        
        # Create 3-channel alpha mask for blending
        alpha_mask_3d = np.stack([alpha_mask] * 3, axis=2)
        
        # Apply the blend
        result_img = source_img * (1.0 - alpha_mask_3d) + processed_img * alpha_mask_3d
        
        return np.clip(result_img, 0, 255).astype(np.uint8)
    
    def _apply_smart_hair_alignment(self, source_img, processed_img, processed_mask, source_mask=None, 
                                 vertical_bias=10, feather_pixels=15, focus_on_top=True, debug_dir=None):
        """Apply smart hair alignment between source and processed images."""
        # Ensure we have binary masks
        _, proc_mask_binary = cv2.threshold(processed_mask, 127, 255, cv2.THRESH_BINARY)
        
        # If source mask is available, use it for alignment
        if source_mask is not None:
            _, source_mask_binary = cv2.threshold(source_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Find top points of masks (hair typically starts from top)
            proc_points = np.argwhere(proc_mask_binary > 0)
            source_points = np.argwhere(source_mask_binary > 0)
            
            if len(proc_points) == 0 or len(source_points) == 0:
                print("Empty mask detected, using standard alpha blend")
                return self._apply_feathered_alpha_blend(
                    source_img, 
                    processed_img, 
                    processed_mask,
                    feather_pixels
                )
            
            # Get top rows of hair in each mask
            proc_top = proc_points[:, 0].min()
            source_top = source_points[:, 0].min()
            
            # Calculate vertical alignment with bias
            dy = (source_top - proc_top) - vertical_bias
            
            # Use center of mass in top third of hair for better horizontal alignment
            # Get top third of hair mask
            proc_top_third_height = proc_mask_binary.shape[0] // 3
            source_top_third_height = source_mask_binary.shape[0] // 3
            
            proc_top_third = proc_mask_binary[:proc_top_third_height, :]
            source_top_third = source_mask_binary[:source_top_third_height, :]
            
            # Calculate centers of mass
            proc_top_moments = cv2.moments(proc_top_third)
            source_top_moments = cv2.moments(source_top_third)
            
            # Handle potential division by zero
            if proc_top_moments["m00"] > 0 and source_top_moments["m00"] > 0:
                proc_top_cx = int(proc_top_moments["m10"] / proc_top_moments["m00"])
                source_top_cx = int(source_top_moments["m10"] / source_top_moments["m00"])
                dx = source_top_cx - proc_top_cx
            else:
                # Use full mask if top third is empty
                proc_moments = cv2.moments(proc_mask_binary)
                source_moments = cv2.moments(source_mask_binary)
                
                if proc_moments["m00"] > 0 and source_moments["m00"] > 0:
                    proc_cx = int(proc_moments["m10"] / proc_moments["m00"])
                    source_cx = int(source_moments["m10"] / source_moments["m00"])
                    dx = source_cx - proc_cx
                else:
                    dx = 0
        else:
            # Without source mask, apply simpler alignment using only processed mask
            # Focus on aligning the top for hair
            proc_points = np.argwhere(proc_mask_binary > 0)
            if len(proc_points) == 0:
                return self._apply_feathered_alpha_blend(
                    source_img, 
                    processed_img, 
                    processed_mask,
                    feather_pixels
                )
            
            # Get center of mass for processed mask
            proc_moments = cv2.moments(proc_mask_binary)
            if proc_moments["m00"] > 0:
                proc_cx = int(proc_moments["m10"] / proc_moments["m00"])
                proc_cy = int(proc_moments["m01"] / proc_moments["m00"])
            else:
                proc_cx = proc_mask_binary.shape[1] // 2
                proc_cy = proc_mask_binary.shape[0] // 2
            
            # Center the mask horizontally, apply vertical bias
            dx = (source_img.shape[1] // 2) - proc_cx
            dy = -vertical_bias  # Apply only the vertical bias since we don't have source mask position
        
        # Apply alignment transformation
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        
        # Warp processed image and mask
        aligned_img = cv2.warpAffine(processed_img, M, (processed_img.shape[1], processed_img.shape[0]))
        aligned_mask = cv2.warpAffine(processed_mask, M, (processed_mask.shape[1], processed_mask.shape[0]))
        
        # Save alignment debug images
        if debug_dir:
            # Create a colored visualization of mask alignment
            if source_mask is not None:
                mask_viz = np.zeros((source_img.shape[0], source_img.shape[1], 3), dtype=np.uint8)
                mask_viz[source_mask_binary > 0] = [0, 255, 0]  # Source mask in green
                mask_viz[cv2.warpAffine(proc_mask_binary, M, (proc_mask_binary.shape[1], proc_mask_binary.shape[0])) > 0] = [0, 0, 255]  # Aligned mask in red
                cv2.imwrite(os.path.join(debug_dir, "mask_alignment_viz.png"), mask_viz)
        
        # Apply feathered alpha blending for smooth transitions
        result_img = self._apply_feathered_alpha_blend(source_img, aligned_img, aligned_mask, feather_pixels)
        
        return result_img
    
    def _match_hair_color(self, source_img, processed_img, source_mask, processed_mask, strength=0.5):
        """Match the color characteristics of processed hair to better blend with source hair."""
        # If either mask is None, return the processed image unchanged
        if source_mask is None or processed_mask is None:
            return processed_img
        
        # Ensure we have binary masks
        _, source_mask_bin = cv2.threshold(source_mask, 127, 255, cv2.THRESH_BINARY)
        _, processed_mask_bin = cv2.threshold(processed_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Check if masks have content
        if np.sum(source_mask_bin) == 0 or np.sum(processed_mask_bin) == 0:
            return processed_img
        
        # Extract only hair regions
        source_hair = cv2.bitwise_and(source_img, source_img, mask=source_mask_bin)
        processed_hair = cv2.bitwise_and(processed_img, processed_img, mask=processed_mask_bin)
        
        # Convert to LAB color space for better color matching
        source_lab = cv2.cvtColor(source_hair, cv2.COLOR_BGR2LAB)
        processed_lab = cv2.cvtColor(processed_hair, cv2.COLOR_BGR2LAB)
        
        # Calculate color statistics for each channel
        source_stats = []
        processed_stats = []
        
        for i in range(3):  # L, A, B channels
            # Get non-zero pixels (only hair pixels)
            source_channel = source_lab[:,:,i][source_mask_bin > 0]
            processed_channel = processed_lab[:,:,i][processed_mask_bin > 0]
            
            if len(source_channel) == 0 or len(processed_channel) == 0:
                continue
            
            # Calculate statistics
            source_mean = np.mean(source_channel)
            source_std = np.std(source_channel)
            processed_mean = np.mean(processed_channel)
            processed_std = np.std(processed_channel)
            
            source_stats.append((source_mean, source_std))
            processed_stats.append((processed_mean, processed_std))
        
        # Create a corrected version of the processed image
        corrected_img = processed_img.copy()
        corrected_lab = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2LAB)
        
        # Apply color correction (scale and shift) for each channel
        for i in range(3):
            if i >= len(source_stats) or i >= len(processed_stats):
                continue
                
            source_mean, source_std = source_stats[i]
            processed_mean, processed_std = processed_stats[i]
            
            # Skip if std is zero (avoid division by zero)
            if processed_std == 0:
                continue
            
            # Calculate scaling factor and shift
            scale = source_std / processed_std
            shift = source_mean - processed_mean * scale
            
            # Apply transformation to the hair region only
            channel = corrected_lab[:,:,i]
            mask_float = processed_mask_bin.astype(float) / 255.0
            
            # Calculate corrected channel values
            correction = (channel * scale + shift - channel) * mask_float
            
            # Apply correction with strength factor
            channel += correction * strength
        
        # Convert back to BGR
        corrected_img = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        
        return np.clip(corrected_img, 0, 255).astype(np.uint8)