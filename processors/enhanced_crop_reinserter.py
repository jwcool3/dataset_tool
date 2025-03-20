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
        if source_mask is not None and self.app.reinsert_handle_different_masks.get():
            from mask_alignment_handler import MaskAlignmentHandler
            
            # Get alignment method from config (or use default)
            alignment_method = getattr(self.app, 'reinsert_alignment_method', tk.StringVar(value="centroid")).get()
            blend_mode = getattr(self.app, 'reinsert_blend_mode', tk.StringVar(value="alpha")).get()
            blend_extent = getattr(self.app, 'reinsert_blend_extent', tk.IntVar(value=5)).get()
            preserve_edges = getattr(self.app, 'reinsert_preserve_edges', tk.BooleanVar(value=True)).get()
            
            # Create mask alignment handler
            aligner = MaskAlignmentHandler(debug_dir=debug_dir)
            
            # Align and blend images
            print(f"Using mask alignment with method: {alignment_method}, blend mode: {blend_mode}")
            result_img, aligned_mask = aligner.align_masks(
                source_img, 
                processed_img_resized, 
                source_mask=source_mask, 
                processed_mask=mask_resized,
                alignment_method=alignment_method,
                blend_mode=blend_mode, 
                blend_extent=blend_extent,
                preserve_original_edges=preserve_edges
            )
            
            # Save the result
            cv2.imwrite(output_path, result_img)
            
            # If debug mode, create a comparison image
            if debug_dir:
                comparison = np.hstack((source_img, processed_img_resized, result_img))
                cv2.imwrite(os.path.join(debug_dir, f"comparison_{os.path.basename(output_path)}"), comparison)
                
            return True
        
        # If not using the mask alignment handler, use standard blending
        # Standard blending with mask-only option
        if self.app.reinsert_mask_only.get():
            # Ensure mask is binary
            _, binary_mask = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
            
            # Check if mask has any white pixels
            if np.max(binary_mask) == 0:
                print("ERROR: Mask is completely black, nothing will be copied!")
                return False
            
            # Print mask statistics for debugging
            white_pixel_count = np.count_nonzero(binary_mask)
            total_pixels = binary_mask.size
            white_percentage = (white_pixel_count / total_pixels) * 100
            print(f"Mask statistics: {white_pixel_count} white pixels ({white_percentage:.2f}% of total)")
            
            # Create normalized floating point mask (0.0 to 1.0)
            mask_float = binary_mask.astype(float) / 255.0
            
            # Expand mask to 3 channels for RGB blending
            mask_float_3d = np.stack([mask_float] * 3, axis=2)
            
            # Create visualization of what's being transferred (just the masked part)
            if debug_dir:
                transferred = processed_img_resized.copy()
                transferred[binary_mask < 127] = 0  # Zero out non-mask regions
                cv2.imwrite(os.path.join(debug_dir, "transferred_content.png"), transferred)
            
            # Apply feathering to the mask edges for smoother transition
            blend_extent = getattr(self.app, 'reinsert_blend_extent', tk.IntVar(value=5)).get()
            if blend_extent > 0:
                # Create a dilated mask for feathering
                kernel = np.ones((blend_extent, blend_extent), np.uint8)
                dilated = cv2.dilate(binary_mask, kernel, iterations=1)
                border = dilated & ~binary_mask  # Border pixels
                
                # Create distance map from border
                dist = cv2.distanceTransform(~border, cv2.DIST_L2, 3)
                dist[dist > blend_extent] = blend_extent
                
                # Normalize distances to create feathered mask
                feather = dist / blend_extent
                
                # Apply feathering to mask edges
                mask_float[border > 0] = 1.0 - feather[border > 0]
                
                # Recreate 3D mask after feathering
                mask_float_3d = np.stack([mask_float] * 3, axis=2)
            
            # Perform alpha blending: source * (1-alpha) + processed * alpha
            result_img = source_img.copy()
            
            # Apply blending only at mask locations for efficiency
            mask_indices = binary_mask > 0
            if np.any(mask_indices):
                # Create a 3-channel mask for indexing
                mask_indices_3d = np.stack([mask_indices] * 3, axis=2)
                
                # Apply blending only at mask locations
                result_img[mask_indices_3d] = (source_img[mask_indices_3d] * (1 - mask_float_3d[mask_indices_3d]) + 
                                            processed_img_resized[mask_indices_3d] * mask_float_3d[mask_indices_3d])
            else:
                print("WARNING: No pixels to blend after resizing mask!")
        else:
            # Regular insertion (full crop replacement)
            result_img = processed_img_resized
        
        # Convert result back to uint8 (just to be safe)
        result_img = np.clip(result_img, 0, 255).astype(np.uint8)
        
        # Save the result
        cv2.imwrite(output_path, result_img)
        
        # Create final side-by-side comparison
        if debug_dir:
            comparison = np.hstack((source_img, processed_img_resized, result_img))
            cv2.imwrite(os.path.join(debug_dir, f"comparison_{os.path.basename(output_path)}"), comparison)
        
        return True