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
        Simplified version focused on alpha blending for hair replacement.
        """
        # Load images
        source_img = cv2.imread(source_path)
        processed_img = cv2.imread(processed_path)
        
        if source_img is None or processed_img is None:
            print(f"Failed to load source or processed image: {source_path} / {processed_path}")
            return False
        
        # Load mask
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to load mask: {mask_path}")
                return False
        else:
            print(f"No mask found at {mask_path}")
            return False
        
        # Print dimensions
        print(f"Source dimensions: {source_img.shape[:2]}")
        print(f"Processed dimensions: {processed_img.shape[:2]}")
        print(f"Mask dimensions: {mask.shape[:2]}")
        
        # Resize processed image and mask to match source if needed
        if source_img.shape[:2] != processed_img.shape[:2]:
            processed_img = cv2.resize(processed_img, (source_img.shape[1], source_img.shape[0]), 
                                    interpolation=cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, (source_img.shape[1], source_img.shape[0]), 
                            interpolation=cv2.INTER_LINEAR)
        
        # Get blend extent from UI
        blend_extent = self.app.reinsert_blend_extent.get()
        print(f"Using blend extent: {blend_extent}")
        
        # Simple alpha blending with optional feathering
        if self.app.reinsert_handle_different_masks.get():
            # Create alpha mask
            mask_float = mask.astype(float) / 255.0
            
            # Apply feathering if extent > 0
            if blend_extent > 0:
                # Create a binary mask for edge detection
                _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                
                # Dilate to create border region
                kernel = np.ones((blend_extent, blend_extent), np.uint8)
                dilated = cv2.dilate(binary_mask, kernel, iterations=1)
                border = dilated & ~binary_mask  # Border pixels
                
                # Create distance map from border
                dist = cv2.distanceTransform(~border, cv2.DIST_L2, 3)
                dist[dist > blend_extent] = blend_extent
                
                # Normalize distances
                feather = dist / blend_extent
                
                # Apply feathering at borders
                mask_float[border > 0] = 1.0 - feather[border > 0]
            
            # Create 3-channel mask for RGB blending
            mask_float_3d = np.stack([mask_float] * 3, axis=2)
            
            # Apply blending
            result_img = source_img * (1 - mask_float_3d) + processed_img * mask_float_3d
            
            # Convert to uint8
            result_img = np.clip(result_img, 0, 255).astype(np.uint8)
        else:
            # Default behavior without special handling
            result_img = processed_img
        
        # Save output
        cv2.imwrite(output_path, result_img)
        
        # Create debug visualizations
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "source_img.png"), source_img)
            cv2.imwrite(os.path.join(debug_dir, "processed_img.png"), processed_img)
            cv2.imwrite(os.path.join(debug_dir, "mask.png"), mask)
            
            # Create alpha mask visualization
            if self.app.reinsert_handle_different_masks.get():
                alpha_viz = (mask_float * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(debug_dir, "alpha_mask.png"), alpha_viz)
            
            # Create comparison
            comparison = np.hstack((source_img, processed_img, result_img))
            cv2.imwrite(os.path.join(debug_dir, f"comparison_{os.path.basename(output_path)}"), comparison)
        
        return True