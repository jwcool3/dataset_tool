"""
Enhanced Crop Reinserter for Dataset Preparation Tool
Specialized for handling resolution differences between source and processed images.
"""

import os
import cv2
import numpy as np
import re
import json
from skimage.transform import resize as skimage_resize
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_ubyte, img_as_float
from processors.mask_alignment_handler import MaskAlignmentHandler
import tkinter as tk

class EnhancedCropReinserter:
    """Reinserts processed regions back into original images, handling resolution differences."""
    
    def __init__(self, app):
        """
        Initialize enhanced crop reinserter.
        
        Args:
            app: The main application with shared variables and UI controls
        """
        self.app = app
    
    def reinsert_crops(self, input_dir, output_dir):
        """
        Reinsert processed regions back into original images with enhanced resolution handling.
        
        Args:
            input_dir: Input directory containing processed images
            output_dir: Output directory for reinserted images
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create output directory for reinserted images
        reinsert_output_dir = os.path.join(output_dir, "reinserted")
        os.makedirs(reinsert_output_dir, exist_ok=True)
        
        # Output debug information
        print(f"Enhanced Reinsertion: Input (Processed) Dir: {input_dir}")
        print(f"Enhanced Reinsertion: Source (Original) Dir: {self.app.source_images_dir.get()}")
        print(f"Enhanced Reinsertion: Output Dir: {reinsert_output_dir}")
        print(f"Enhanced Reinsertion: Mask-only mode: {self.app.reinsert_mask_only.get()}")
        
        # Create a debug directory for visualization if in debug mode
        debug_dir = None
        if self.app.debug_mode.get():
            debug_dir = os.path.join(output_dir, "reinsert_debug")
            os.makedirs(debug_dir, exist_ok=True)
            print(f"Debug mode enabled, saving visualizations to: {debug_dir}")
        
        # Find all processed images and corresponding masks
        processed_images = []
        
        for root, dirs, files in os.walk(input_dir):
            # Skip if this is a 'masks' directory - we'll handle masks separately
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
        
        # Get source directory (original images)
        source_dir = self.app.source_images_dir.get()
        if not source_dir or not os.path.isdir(source_dir):
            self.app.status_label.config(text="Source images directory not set or invalid.")
            return False
        
        # Load all source images
        source_images = {}
        for root, dirs, files in os.walk(source_dir):
            # Skip if this is a 'masks' directory
            if os.path.basename(root).lower() == "masks":
                continue
                
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    source_images[file] = os.path.join(root, file)
        
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
                matched_source = self._match_source_image(filename, source_images)
                
                if not matched_source:
                    self.app.status_label.config(text=f"Source image not found for {filename}")
                    failed_count += 1
                    continue
                
                source_path = source_images[matched_source]
                
                # Perform the resolution-aware reinsertion
                success = self._reinsert_with_resolution_handling(
                    source_path,
                    processed_path,
                    mask_path,
                    os.path.join(reinsert_output_dir, f"reinserted_{filename}"),
                    debug_dir
                )
                
                if success:
                    processed_count += 1
                else:
                    failed_count += 1
                
            except Exception as e:
                self.app.status_label.config(text=f"Error processing {filename}: {str(e)}")
                print(f"Error in enhanced_reinsert_crops: {str(e)}")
                import traceback
                traceback.print_exc()
                failed_count += 1
                continue
            
            # Update progress
            progress = (idx + 1) / total_images * 100
            self.app.progress_bar['value'] = min(progress, 100)
            self.app.status_label.config(text=f"Processed {idx+1}/{total_images} images")
            self.app.root.update_idletasks()
        
        # Final status update
        if failed_count > 0:
            self.app.status_label.config(text=f"Enhanced reinsertion completed. Processed {processed_count} images. Failed: {failed_count}.")
        else:
            self.app.status_label.config(text=f"Enhanced reinsertion completed. Successfully processed {processed_count} images.")
        
        self.app.progress_bar['value'] = 100
        return processed_count > 0
    
    def _match_source_image(self, processed_filename, source_images):
        """Match processed image filename to source image."""
        # First, just try a direct approach - look for the exact same filename in the source dir
        source_dir = self.app.source_images_dir.get()
        source_path = os.path.join(source_dir, processed_filename)
        
        # Check if this exact filename exists in the source directory
        if os.path.exists(source_path):
            print(f"Found direct match for {processed_filename} in source directory")
            return processed_filename
        
        # If not, try to find a file with the same base name regardless of extension
        base_name = os.path.splitext(processed_filename)[0]
        for file in os.listdir(source_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                source_base = os.path.splitext(file)[0]
                if source_base == base_name:
                    print(f"Found match by base name: {file}")
                    return file
        
        # As a last resort, if there's only one file in the source directory, use that
        image_files = [f for f in os.listdir(source_dir) 
                    if os.path.isfile(os.path.join(source_dir, f)) and 
                    f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(image_files) == 1:
            print(f"Only one source image found, using {image_files[0]}")
            return image_files[0]
        
        print(f"WARNING: No matching source image found for {processed_filename}")
        return None
        
    def _reinsert_with_resolution_handling(self, source_path, processed_path, mask_path, output_path, debug_dir=None):
        """
        Reinsert a processed image region into the source image with enhanced mask alignment.
        
        Args:
            source_path: Path to source (original) image
            processed_path: Path to processed image
            mask_path: Path to mask (if available)
            output_path: Path to save the result
            debug_dir: Directory to save debug visualizations (if enabled)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Load images
        source_img = cv2.imread(source_path)
        processed_img = cv2.imread(processed_path)
        
        if source_img is None or processed_img is None:
            print(f"Failed to load source or processed image: {source_path} / {processed_path}")
            return False
        
        # Get dimensions
        source_h, source_w = source_img.shape[:2]
        processed_h, processed_w = processed_img.shape[:2]
        
        # Load and prepare mask
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to load mask: {mask_path}")
                return False
        else:
            print(f"No mask found at {mask_path}")
            return False
        
        # Print diagnostic information
        print(f"Source dimensions: {source_w}x{source_h}")
        print(f"Processed dimensions: {processed_w}x{processed_h}")
        print(f"Mask dimensions: {mask.shape[1]}x{mask.shape[0]}")
        
        # Check if dimensions match
        resolution_diff = (abs(source_w - processed_w) > 2 or 
                        abs(source_h - processed_h) > 2)
        
        # If resolution differs, resize processed image and mask to match source
        if resolution_diff:
            print(f"Resizing processed image from {processed_w}x{processed_h} to {source_w}x{source_h}")
            processed_img_resized = cv2.resize(processed_img, (source_w, source_h), 
                                            interpolation=cv2.INTER_LANCZOS4)
            mask_resized = cv2.resize(mask, (source_w, source_h), 
                                    interpolation=cv2.INTER_NEAREST)
        else:
            processed_img_resized = processed_img
            mask_resized = mask
        
        # Save original images for debugging
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "source_original.png"), source_img)
            cv2.imwrite(os.path.join(debug_dir, "processed_original.png"), processed_img)
            cv2.imwrite(os.path.join(debug_dir, "mask_original.png"), mask)
            
            if resolution_diff:
                cv2.imwrite(os.path.join(debug_dir, "processed_resized.png"), processed_img_resized)
                cv2.imwrite(os.path.join(debug_dir, "mask_resized.png"), mask_resized)
        
        # Check if the source has a different hair mask (try to find it in the same directory)
        source_mask_path = None
        source_dir = os.path.dirname(source_path)
        source_name = os.path.basename(source_path)
        potential_mask_dir = os.path.join(source_dir, "masks")
        
        if os.path.isdir(potential_mask_dir):
            # Check for mask with same name
            potential_mask = os.path.join(potential_mask_dir, source_name)
            if os.path.exists(potential_mask):
                source_mask_path = potential_mask
            else:
                # Try different extensions
                base_name = os.path.splitext(source_name)[0]
                for ext in ['.png', '.jpg', '.jpeg']:
                    alt_mask_path = os.path.join(potential_mask_dir, base_name + ext)
                    if os.path.exists(alt_mask_path):
                        source_mask_path = alt_mask_path
                        break
        
        # Load source mask if found
        source_mask = None
        if source_mask_path and os.path.exists(source_mask_path):
            print(f"Found source mask: {source_mask_path}")
            source_mask = cv2.imread(source_mask_path, cv2.IMREAD_GRAYSCALE)
            if source_mask is not None and source_mask.shape[:2] != (source_h, source_w):
                source_mask = cv2.resize(source_mask, (source_w, source_h), 
                                        interpolation=cv2.INTER_NEAREST)
            if debug_dir and source_mask is not None:
                cv2.imwrite(os.path.join(debug_dir, "source_mask.png"), source_mask)
        
        # Use the MaskAlignmentHandler for alignment and blending if both masks are available
        # In _reinsert_with_resolution_handling method, replace the MaskAlignmentHandler block with:

        if source_mask is not None and self.app.reinsert_handle_different_masks.get():
            # Pre-process both masks for better comparison
            _, source_mask_bin = cv2.threshold(source_mask, 127, 255, cv2.THRESH_BINARY)
            _, processed_mask_bin = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
            
            # Save the pre-processed masks for debugging
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "source_mask_binary.png"), source_mask_bin)
                cv2.imwrite(os.path.join(debug_dir, "processed_mask_binary.png"), processed_mask_bin)
            
            # Find the top point of each mask (for hair, this is usually what we want to align)
            source_points = np.argwhere(source_mask_bin > 0)
            processed_points = np.argwhere(processed_mask_bin > 0)
            
            if len(source_points) > 0 and len(processed_points) > 0:
                # Find the top-most point (minimum y-coordinate) in each mask
                source_top_y = source_points[:, 0].min()
                source_top_indices = np.where(source_points[:, 0] == source_top_y)[0]
                source_top_x = np.median(source_points[source_top_indices, 1])
                
                processed_top_y = processed_points[:, 0].min()
                processed_top_indices = np.where(processed_points[:, 0] == processed_top_y)[0]
                processed_top_x = np.median(processed_points[processed_top_indices, 1])
                
                # Calculate shift needed to align top points
                dx = int(source_top_x - processed_top_x)
                dy = int(source_top_y - processed_top_y)
                
                print(f"Aligning masks: dx={dx}, dy={dy}")
                
                # Create a transformation matrix for the shift
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                
                # Apply the transformation to the processed mask
                aligned_mask = cv2.warpAffine(processed_mask_bin, M, 
                                            (processed_mask_bin.shape[1], processed_mask_bin.shape[0]))
                
                # Apply the same transformation to the processed image
                aligned_img = cv2.warpAffine(processed_img_resized, M,
                                        (processed_img_resized.shape[1], processed_img_resized.shape[0]))
                
                # Save aligned results for debugging
                if debug_dir:
                    cv2.imwrite(os.path.join(debug_dir, "aligned_mask.png"), aligned_mask)
                    cv2.imwrite(os.path.join(debug_dir, "aligned_img.png"), aligned_img)
                    
                    # Create a visualization of the alignment
                    mask_viz = np.zeros((source_mask_bin.shape[0], source_mask_bin.shape[1], 3), dtype=np.uint8)
                    mask_viz[source_mask_bin > 0] = [0, 0, 255]  # Red for source mask
                    mask_viz[aligned_mask > 0] = [0, 255, 0]  # Green for aligned mask
                    cv2.imwrite(os.path.join(debug_dir, "mask_alignment_viz.png"), mask_viz)
                
                # Create a feathered mask with a large blend extent for hair
                blend_extent = 20  # Use a large blend extent for hair
                
                # Create a distance transform for smooth blending
                # First dilate the aligned mask to create a border region
                kernel = np.ones((blend_extent, blend_extent), np.uint8)
                dilated = cv2.dilate(aligned_mask, kernel, iterations=1)
                border = dilated & ~aligned_mask  # Border pixels
                
                # Create distance map from border
                dist = cv2.distanceTransform(~border, cv2.DIST_L2, 3)
                dist[dist > blend_extent] = blend_extent
                
                # Normalize distances to create feathered mask
                feather = dist / blend_extent
                
                # Create the final alpha mask for blending
                mask_float = aligned_mask.astype(float) / 255.0
                mask_float[border > 0] = 1.0 - feather[border > 0]
                
                # Create 3-channel mask for RGB
                mask_float_3d = np.stack([mask_float] * 3, axis=2)
                
                # Perform the blending
                result_img = source_img * (1 - mask_float_3d) + aligned_img * mask_float_3d
                result_img = np.clip(result_img, 0, 255).astype(np.uint8)
                
                # Save the result
                cv2.imwrite(output_path, result_img)
                
                # Create comparison image
                if debug_dir:
                    comparison = np.hstack((source_img, aligned_img, result_img))
                    cv2.imwrite(os.path.join(debug_dir, f"comparison_{os.path.basename(output_path)}"), comparison)
                
                return True
            else:
                print("Cannot find points in the masks for alignment")