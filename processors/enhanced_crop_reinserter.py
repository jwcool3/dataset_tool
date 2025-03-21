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
        

        # Apply bangs extension if enabled (add after loading the mask but before any other mask processing)
        if self.app.extend_bangs.get() and mask is not None:
            print("Extending mask in bangs/forehead area")
            extension_amount = self.app.bangs_extension_amount.get()
            width_ratio = self.app.bangs_width_ratio.get()
            
            mask = self._extend_bangs_area(
                mask, 
                extend_pixels=extension_amount,
                forehead_ratio=width_ratio,
                min_opacity=self.app.bangs_min_opacity.get()
            )
            
            # Save debug image
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "extended_bangs_mask.png"), mask)

        # Resize processed image and mask to match source if dimensions differ
        if source_w != processed_w or source_h != processed_h:
            print(f"Resizing processed image from {processed_w}x{processed_h} to {source_w}x{source_h}")
            processed_img_resized = cv2.resize(processed_img, (source_w, source_h), 
                                            interpolation=cv2.INTER_LANCZOS4)
            mask_resized = cv2.resize(mask, (source_w, source_h), 
                                    interpolation=cv2.INTER_NEAREST)
        else:
            processed_img_resized = processed_img
            mask_resized = mask
        
        # Debug: Save original and resized images
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "source_original.png"), source_img)
            cv2.imwrite(os.path.join(debug_dir, "processed_original.png"), processed_img)
            cv2.imwrite(os.path.join(debug_dir, "mask_original.png"), mask)
            
            if source_w != processed_w or source_h != processed_h:
                cv2.imwrite(os.path.join(debug_dir, "processed_resized.png"), processed_img_resized)
                cv2.imwrite(os.path.join(debug_dir, "mask_resized.png"), mask_resized)
        
        # Check if the source has a different hair mask
        source_mask = self._find_source_mask(source_path)
        
        # If we're not handling different masks, simply use standard blending
        if not self.app.reinsert_handle_different_masks.get() or source_mask is None:
            # Basic alpha blending
            mask_float = mask_resized.astype(float) / 255.0
            mask_float_3d = np.stack([mask_float] * 3, axis=2)
            result_img = source_img * (1 - mask_float_3d) + processed_img_resized * mask_float_3d
            result_img = np.clip(result_img, 0, 255).astype(np.uint8)
            
            cv2.imwrite(output_path, result_img)
            return True
        
        # Get manual offset values
        manual_offset_x = self.app.reinsert_manual_offset_x.get()
        manual_offset_y = self.app.reinsert_manual_offset_y.get()

        # Apply manual offset if any is set
        if manual_offset_x != 0 or manual_offset_y != 0:
            print(f"Applying manual offset: X={manual_offset_x}, Y={manual_offset_y}")
            
            # Create transformation matrix for the offset
            M = np.float32([[1, 0, manual_offset_x], [0, 1, manual_offset_y]])
            
            # Apply to both processed image and mask AFTER resizing
            processed_img_resized = cv2.warpAffine(processed_img_resized, M, 
                                                (processed_img_resized.shape[1], processed_img_resized.shape[0]))
            mask_resized = cv2.warpAffine(mask_resized, M, 
                                        (mask_resized.shape[1], mask_resized.shape[0]))
            
            # Save offset versions for debugging
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "offset_processed.png"), processed_img_resized)
                cv2.imwrite(os.path.join(debug_dir, "offset_mask.png"), mask_resized)


        # Get manual scaling values
        scale_x = self.app.reinsert_manual_scale_x.get()
        scale_y = self.app.reinsert_manual_scale_y.get()

        # Apply scaling if not at default value (1.0)
        if scale_x != 1.0 or scale_y != 1.0:
            print(f"Applying manual scaling: X={scale_x}, Y={scale_y}")
            
            # Get dimensions
            h, w = processed_img_resized.shape[:2]
            
            # Calculate new dimensions
            new_w = int(w * scale_x)
            new_h = int(h * scale_y)
            
            # Calculate center point for scaling around the center
            center_x = w // 2
            center_y = h // 2
            
            # Create transformation matrix
            # First translate to origin, then scale, then translate back
            M = np.float32([
                [scale_x, 0, center_x * (1 - scale_x)],
                [0, scale_y, center_y * (1 - scale_y)]
            ])
            
            # Apply to processed image and mask
            processed_img_resized = cv2.warpAffine(
                processed_img_resized, M, (w, h),
                flags=cv2.INTER_LANCZOS4
            )
            mask_resized = cv2.warpAffine(
                mask_resized, M, (w, h),
                flags=cv2.INTER_NEAREST
            )
            
            # Save scaled versions for debugging
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "scaled_processed.png"), processed_img_resized)
                cv2.imwrite(os.path.join(debug_dir, "scaled_mask.png"), mask_resized)


        # If handling different masks, get configuration settings
        alignment_method = self.app.reinsert_alignment_method.get()
        blend_mode = self.app.reinsert_blend_mode.get()
        blend_extent = self.app.reinsert_blend_extent.get()
        preserve_edges = self.app.reinsert_preserve_edges.get()
        
        print(f"Using config settings: alignment={alignment_method}, blend={blend_mode}, extent={blend_extent}")
        
        # Align mask and image if not using "none" alignment
        aligned_mask = mask_resized.copy()
        aligned_img = processed_img_resized.copy()
        
        if alignment_method != "none":
            # Perform the alignment based on the selected method
            aligned_mask, aligned_img = self._align_masks(
                source_mask, mask_resized, 
                source_img, processed_img_resized, 
                alignment_method, 
                debug_dir
            )
        
        # Blending stage
        if blend_mode == "alpha":
            result_img = self._alpha_blend(
                source_img, aligned_img, 
                aligned_mask, 
                blend_extent
            )
        elif blend_mode == "poisson":
            try:
                # Convert mask to correct format
                mask_uint8 = aligned_mask.astype(np.uint8)
                
                # Find center of mask
                moments = cv2.moments(mask_uint8)
                if moments["m00"] > 0:
                    center_x = int(moments["m10"] / moments["m00"])
                    center_y = int(moments["m01"] / moments["m00"])
                    
                    # Make sure center point is within safe boundaries
                    # (at least 1/4 of the image dimensions from any edge)
                    h, w = source_img.shape[:2]
                    min_distance = min(w, h) // 4
                    
                    center_x = max(min_distance, min(w - min_distance, center_x))
                    center_y = max(min_distance, min(h - min_distance, center_y))
                    
                    center = (center_x, center_y)
                    
                    # Ensure the mask has non-zero values (required for seamlessClone)
                    if np.any(mask_uint8 > 0):
                        # Apply seamless cloning
                        result_img = cv2.seamlessClone(
                            aligned_img, source_img, mask_uint8, center, cv2.NORMAL_CLONE
                        )
                    else:
                        # Fallback to alpha blending if mask is empty
                        raise ValueError("Mask has no non-zero values")
                else:
                    # Fallback to alpha blending if moments are zero
                    raise ValueError("Mask moments are zero")
            except Exception as e:
                print(f"Poisson blending failed: {str(e)}, falling back to alpha blending")
                # Fall back to alpha blending
                mask_float = aligned_mask.astype(float) / 255.0
                mask_float_3d = np.stack([mask_float] * 3, axis=2)
                result_img = source_img * (1 - mask_float_3d) + aligned_img * mask_float_3d
        elif blend_mode == "feathered":
            result_img = self._feathered_blend(
                source_img, aligned_img, 
                aligned_mask, 
                blend_extent
            )
        
        # Preserve edges if requested
        if preserve_edges:
            result_img = self._preserve_image_edges(
                source_img, result_img, 
                aligned_mask
            )
        
        # Save the result
        cv2.imwrite(output_path, result_img)
        
        # Create comparison image for debugging
        if debug_dir:
            comparison = np.hstack((source_img, aligned_img, result_img))
            cv2.imwrite(os.path.join(debug_dir, f"comparison_{os.path.basename(output_path)}"), comparison)
        
        return True

    def _find_source_mask(self, source_path):
        """Find the mask for a source image."""
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

    def _align_masks(self, source_mask, processed_mask, source_img, processed_img, alignment_method, debug_dir=None):
        """
        Align masks and images based on the specified method.
        
        Args:
            source_mask: Mask from source image
            processed_mask: Mask from processed image
            source_img: Source image
            processed_img: Processed image
            alignment_method: Method to use for alignment
            debug_dir: Directory to save debug visualizations
        
        Returns:
            tuple: (aligned_mask, aligned_image)
        """
        def clean_mask(mask, threshold=50):
            """
            Convert pixels below threshold to black
            
            Args:
                mask: Input grayscale mask
                threshold: Pixel intensity threshold (0-255)
                    - Pixels below this will be set to black (0)
                    - Pixels above will be preserved
            
            Returns:
                Cleaned mask with darker pixels removed
            """
            # Create a copy of the mask
            cleaned_mask = mask.copy()
            
            # Convert pixels below threshold to black
            cleaned_mask[cleaned_mask < threshold] = 0
            
            return cleaned_mask
        
        # Clean source and processed masks
        source_mask_cleaned = clean_mask(source_mask)
        processed_mask_cleaned = clean_mask(processed_mask)
        
        # Debug: Save cleaned masks if debug directory is provided
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "source_mask_cleaned.png"), source_mask_cleaned)
            cv2.imwrite(os.path.join(debug_dir, "processed_mask_cleaned.png"), processed_mask_cleaned)
        
        # Continue with alignment using cleaned masks
        # Binary threshold cleaned masks if needed
        _, source_mask_bin = cv2.threshold(source_mask_cleaned, 127, 255, cv2.THRESH_BINARY)
        _, processed_mask_bin = cv2.threshold(processed_mask_cleaned, 127, 255, cv2.THRESH_BINARY)
        
        # Make copies to modify
        aligned_mask = processed_mask.copy()
        aligned_img = processed_img.copy()
        
        # Centroid alignment
        if alignment_method == "centroid":
            # Calculate centroids
            source_moments = cv2.moments(source_mask_bin)
            processed_moments = cv2.moments(processed_mask_bin)
            
            if source_moments["m00"] > 0 and processed_moments["m00"] > 0:
                source_cx = int(source_moments["m10"] / source_moments["m00"])
                source_cy = int(source_moments["m01"] / source_moments["m00"])
                processed_cx = int(processed_moments["m10"] / processed_moments["m00"])
                processed_cy = int(processed_moments["m01"] / processed_moments["m00"])
                
                # Calculate shift
                dx = source_cx - processed_cx
                dy = source_cy - processed_cy
                
                # Apply shift
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                aligned_mask = cv2.warpAffine(processed_mask, M, (processed_mask.shape[1], processed_mask.shape[0]))
                aligned_img = cv2.warpAffine(processed_img, M, (processed_img.shape[1], processed_img.shape[0]))
        
        # Contour-based alignment (top point alignment)
        elif alignment_method == "contour":
            source_points = np.argwhere(source_mask_bin > 0)
            processed_points = np.argwhere(processed_mask_bin > 0)
            
            if len(source_points) > 0 and len(processed_points) > 0:
                # Find the top point
                source_top_y = source_points[:, 0].min()
                source_top_indices = np.where(source_points[:, 0] == source_top_y)[0]
                source_top_x = np.median(source_points[source_top_indices, 1])
                
                processed_top_y = processed_points[:, 0].min()
                processed_top_indices = np.where(processed_points[:, 0] == processed_top_y)[0]
                processed_top_x = np.median(processed_points[processed_top_indices, 1])
                
                # Calculate shift
                dx = int(source_top_x - processed_top_x)
                dy = int(source_top_y - processed_top_y)
                
                # Apply shift
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                aligned_mask = cv2.warpAffine(processed_mask, M, (processed_mask.shape[1], processed_mask.shape[0]))
                aligned_img = cv2.warpAffine(processed_img, M, (processed_img.shape[1], processed_img.shape[0]))
        
        # Bounding box alignment
        elif alignment_method == "bbox":
            source_contours, _ = cv2.findContours(source_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            processed_contours, _ = cv2.findContours(processed_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if source_contours and processed_contours:
                source_contour = max(source_contours, key=cv2.contourArea)
                processed_contour = max(processed_contours, key=cv2.contourArea)
                
                source_x, source_y, source_w, source_h = cv2.boundingRect(source_contour)
                processed_x, processed_y, processed_w, processed_h = cv2.boundingRect(processed_contour)
                
                # Calculate shifts to align top-left corners
                dx = source_x - processed_x
                dy = source_y - processed_y
                
                # Apply shift
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                aligned_mask = cv2.warpAffine(processed_mask, M, (processed_mask.shape[1], processed_mask.shape[0]))
                aligned_img = cv2.warpAffine(processed_img, M, (processed_img.shape[1], processed_img.shape[0]))
        
        # Intersection Over Union (IoU) alignment
        elif alignment_method == "iou":
            best_iou = 0
            best_mask = processed_mask.copy()
            best_img = processed_img.copy()
            
            max_shift = 20  # pixels
            for dx in range(-max_shift, max_shift + 1, 2):
                for dy in range(-max_shift, max_shift + 1, 2):
                    # Create shifted mask and image
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    shifted_mask = cv2.warpAffine(processed_mask, M, (processed_mask.shape[1], processed_mask.shape[0]))
                    shifted_img = cv2.warpAffine(processed_img, M, (processed_img.shape[1], processed_img.shape[0]))
                    
                    # Calculate IoU
                    intersection = np.logical_and(source_mask_bin > 0, shifted_mask > 0).sum()
                    union = np.logical_or(source_mask_bin > 0, shifted_mask > 0).sum()
                    iou = intersection / union if union > 0 else 0
                    
                    # Update best if improved
                    if iou > best_iou:
                        best_iou = iou
                        best_mask = shifted_mask
                        best_img = shifted_img
            
            aligned_mask = best_mask
            aligned_img = best_img
        
        # Debug visualization
        if debug_dir:
            # Source mask in red, aligned mask in green
            mask_viz = np.zeros((source_mask_bin.shape[0], source_mask_bin.shape[1], 3), dtype=np.uint8)
            mask_viz[source_mask_bin > 0] = [0, 0, 255]  # Red for source mask
            mask_viz[aligned_mask > 0] = [0, 255, 0]  # Green for aligned mask
            cv2.imwrite(os.path.join(debug_dir, "mask_alignment_viz.png"), mask_viz)
        
        return aligned_mask, aligned_img

    def _alpha_blend(self, source_img, processed_img, mask, blend_extent=0):
        """
        Perform alpha blending with optional feathering.
        
        Args:
            source_img: Original source image
            processed_img: Processed image to blend
            mask: Blending mask
            blend_extent: Extent of feathering (0 = no feathering)
        
        Returns:
            numpy.ndarray: Blended image
        """
        # Create alpha mask
        mask_float = mask.astype(float) / 255.0
        
        # Apply feathering if blend_extent > 0
        if blend_extent > 0:
            # Create feathering kernel
            kernel = np.ones((blend_extent, blend_extent), np.uint8)
        
        # Create dilation and border regions
        dilated = cv2.dilate(mask, kernel, iterations=1)
        border = dilated & ~mask
        
        # Create distance map for feathering
        dist = cv2.distanceTransform(~border, cv2.DIST_L2, 3)
        dist[dist > blend_extent] = blend_extent
        
        # Normalize distances
        feather = dist / blend_extent
        
        # Create alpha mask with feathering
        mask_float = mask.astype(float) / 255.0
        mask_float[border > 0] = 1.0 - feather[border > 0]
        
        # Create 3-channel mask
        mask_float_3d = np.stack([mask_float] * 3, axis=2)
        
        # Apply blending
        result_img = source_img * (1 - mask_float_3d) + processed_img * mask_float_3d
        
        return np.clip(result_img, 0, 255).astype(np.uint8)

    def _poisson_blend(self, source_img, processed_img, mask):
        """
        Perform Poisson blending.
        
        Args:
            source_img: Original source image
            processed_img: Processed image to blend
            mask: Blending mask
        
        Returns:
            numpy.ndarray: Blended image
        """
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
            else:
                # Fallback to alpha blending
                mask_float = mask.astype(float) / 255.0
                mask_float_3d = np.stack([mask_float] * 3, axis=2)
                result_img = source_img * (1 - mask_float_3d) + processed_img * mask_float_3d
        except Exception as e:
            print(f"Poisson blending failed: {str(e)}")
            # Fallback to alpha blending
            mask_float = mask.astype(float) / 255.0
            mask_float_3d = np.stack([mask_float] * 3, axis=2)
            result_img = source_img * (1 - mask_float_3d) + processed_img * mask_float_3d
        
        return np.clip(result_img, 0, 255).astype(np.uint8)

    def _feathered_blend(self, source_img, processed_img, mask, blend_extent=5):
        """
        Perform feathered blending with gradual transition.
        
        Args:
            source_img: Original source image
            processed_img: Processed image to blend
            mask: Blending mask
            blend_extent: Extent of feathering
        
        Returns:
            numpy.ndarray: Blended image
        """
        # Convert mask to binary
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Create distance transforms
        dist_inside = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)
        dist_outside = cv2.distanceTransform(255 - binary_mask, cv2.DIST_L2, 3)
        
        # Create alpha values based on distance
        alpha = np.ones_like(dist_inside, dtype=float)
        
        # Inside mask: fade from 1.0 at center to 0.5 at border
        fade_inside = np.clip(dist_inside / blend_extent, 0, 1)
        alpha = 0.5 + 0.5 * fade_inside
        
        # Outside mask: fade from 0.5 at border to 0.0 outside
        fade_outside = np.clip(1.0 - dist_outside / blend_extent, 0, 1)
        alpha = alpha * (binary_mask / 255.0) + fade_outside * (1 - binary_mask / 255.0) * 0.5
        
        # Create 3-channel alpha
        alpha_3d = np.stack([alpha] * 3, axis=2)
        
        # Blend images
        result_img = source_img * (1 - alpha_3d) + processed_img * alpha_3d
        
        return np.clip(result_img, 0, 255).astype(np.uint8)

    def _preserve_image_edges(self, source_img, result_img, mask):
        """
        Preserve original image edges outside the mask region.
        
        Args:
            source_img: Original source image
            result_img: Blended result image
            mask: Blending mask
        
        Returns:
            numpy.ndarray: Result image with preserved edges
        """
        # Detect edges in source image
        gray_source = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY) if len(source_img.shape) == 3 else source_img
        edges = cv2.Canny(gray_source, 50, 150)
        
        # Dilate edges to make them more prominent
        edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        # Only preserve edges outside the mask
        edge_mask = edge_mask & ~mask
        
        # Convert edge mask to 3 channels
        edge_mask_3d = np.stack([edge_mask / 255.0] * 3, axis=2)
        
        # Keep original pixel values at edges
        preserved_result = source_img * edge_mask_3d + result_img * (1 - edge_mask_3d)
        
        return np.clip(preserved_result, 0, 255).astype(np.uint8)
    
    def _extend_bangs_area(self, mask, extend_pixels=30, forehead_ratio=0.3, min_opacity=0.7):
        """
        Extend the mask downward in the bangs/forehead area with higher opacity.
        
        Args:
            mask: The binary mask image
            extend_pixels: How many pixels to extend downward
            forehead_ratio: What portion of the width to consider as forehead (centered)
            min_opacity: Minimum opacity value at the edges of extension (0.0-1.0)
            
        Returns:
            numpy.ndarray: Extended mask
        """
        if mask is None:
            return None
            
        # Create a copy of the mask to modify
        extended_mask = mask.copy()
        
        # Find the top points of the mask
        mask_points = np.argwhere(mask > 127)
        if len(mask_points) == 0:
            return mask  # Empty mask, nothing to extend
        
        # Find the top boundary of the mask
        top_y_values = {}
        height, width = mask.shape[:2]
        
        # For each column, find the topmost pixel
        for y, x in mask_points:
            if x not in top_y_values or y < top_y_values[x]:
                top_y_values[x] = y
        
        # Calculate the forehead region (center portion of width)
        center_x = width // 2
        forehead_half_width = int(width * forehead_ratio / 2)
        forehead_left = max(0, center_x - forehead_half_width)
        forehead_right = min(width, center_x + forehead_half_width)
        
        # Extend the mask downward in the forehead region
        for x in range(forehead_left, forehead_right):
            if x in top_y_values:
                # Get the topmost y for this column
                top_y = top_y_values[x]
                
                # Extend downward by extend_pixels, but don't go out of bounds
                extend_to_y = min(height, top_y + extend_pixels)
                
                # Create a gradually decreasing alpha value with a minimum opacity
                for y in range(top_y, extend_to_y):
                    # Calculate fade factor (1.0 at top, decreasing to min_opacity)
                    progress = (y - top_y) / float(extend_pixels)
                    fade = 1.0 - (progress * (1.0 - min_opacity))
                    fade_value = int(255 * fade)
                    
                    # Don't overwrite existing mask pixels with lower values
                    if extended_mask[y, x] < fade_value:
                        extended_mask[y, x] = fade_value
        
        # Apply a small amount of blur to smooth the extended edges
        extended_mask = cv2.GaussianBlur(extended_mask, (3, 3), 0)
        
        return extended_mask