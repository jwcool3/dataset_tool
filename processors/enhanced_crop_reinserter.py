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
        
        # If handling different masks, get configuration settings
        alignment_method = self.app.reinsert_alignment_method.get()
        blend_mode = self.app.reinsert_blend_mode.get()
        blend_extent = self.app.reinsert_blend_extent.get()
        preserve_edges = self.app.reinsert_preserve_edges.get()
        
        # Get vertical bias and soft edge width
        vertical_bias = 0
        soft_edge_width = 15
        
        if hasattr(self.app, 'vertical_alignment_bias'):
            vertical_bias = self.app.vertical_alignment_bias.get()
        
        if hasattr(self.app, 'soft_edge_width'):
            soft_edge_width = self.app.soft_edge_width.get()
        
        print(f"Using config settings: alignment={alignment_method}, blend={blend_mode}, extent={blend_extent}")
        print(f"Vertical Bias: {vertical_bias}, Soft Edge Width: {soft_edge_width}")  # Debug print
        
        # Align mask and image if not using "none" alignment
        aligned_mask = mask_resized.copy()
        aligned_img = processed_img_resized.copy()
        
        if alignment_method != "none":
            # Perform the alignment based on the selected method
            # THIS IS THE KEY FIX: properly unpack the two return values
            aligned_mask, aligned_img = self._align_masks(
                source_mask, mask_resized, 
                source_img, processed_img_resized, 
                alignment_method, 
                debug_dir,
                vertical_bias,
                soft_edge_width
            )
        
        # Blending stage
        if blend_mode == "alpha":
            result_img = self._alpha_blend(
                source_img, aligned_img, 
                aligned_mask, 
                blend_extent
            )
        elif blend_mode == "poisson":
            result_img = self._poisson_blend(
                source_img, aligned_img, 
                aligned_mask
            )
        elif blend_mode == "feathered":
            result_img = self._feathered_blend(
                source_img, aligned_img, 
                aligned_mask, 
                blend_extent
            )
        else:
            # Fallback to alpha blend if unknown mode
            result_img = self._alpha_blend(
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

    def _align_masks(self, source_mask, processed_mask, source_img, processed_img, alignment_method, debug_dir=None, vertical_bias=0, soft_edge_width=15):
        """
        Enhanced mask alignment with vertical bias and soft edge handling.
        
        Args:
            source_mask: Binary mask of source image
            processed_mask: Binary mask of processed image 
            source_img: Source image
            processed_img: Processed image
            alignment_method: Method to use for alignment
            debug_dir: Directory for debug output (optional)
            vertical_bias: Vertical adjustment bias (default: 0)
            soft_edge_width: Width of soft edge (default: 15)
        
        Returns:
            tuple: (aligned_mask, aligned_img) - The aligned mask and image
        """
        # Ensure source_mask is not None
        if source_mask is None:
            return processed_mask, processed_img
        
        # Binary threshold both masks
        _, source_mask_bin = cv2.threshold(source_mask, 127, 255, cv2.THRESH_BINARY)
        _, processed_mask_bin = cv2.threshold(processed_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Make copies to modify
        aligned_mask = processed_mask.copy()
        aligned_img = processed_img.copy()
        
        # Alignment methods with improved vertical handling
        if alignment_method in ["centroid", "landmarks", "contour", "bbox"]:
            try:
                # Get the vertical bias from the UI if available
                if hasattr(self.app, 'vertical_alignment_bias'):
                    vertical_bias = self.app.vertical_alignment_bias.get()
                
                # Default dx, dy for safety
                dx, dy = 0, vertical_bias
                
                # Compute alignment based on selected method
                if alignment_method == "centroid":
                    source_moments = cv2.moments(source_mask_bin)
                    processed_moments = cv2.moments(processed_mask_bin)
                    
                    if source_moments["m00"] > 0 and processed_moments["m00"] > 0:
                        source_cx = int(source_moments["m10"] / source_moments["m00"])
                        source_cy = int(source_moments["m01"] / source_moments["m00"])
                        processed_cx = int(processed_moments["m10"] / processed_moments["m00"])
                        processed_cy = int(processed_moments["m01"] / processed_moments["m00"])
                        
                        dx = source_cx - processed_cx
                        dy = source_cy - processed_cy + vertical_bias
                
                elif alignment_method in ["landmarks", "contour"]:
                    source_points = np.argwhere(source_mask_bin > 0)
                    processed_points = np.argwhere(processed_mask_bin > 0)
                    
                    if len(source_points) > 0 and len(processed_points) > 0:
                        # Find the top points of each mask (for hair alignment)
                        source_top = source_points[:, 0].min()
                        processed_top = processed_points[:, 0].min()
                        
                        # For horizontal alignment, use median x at the top
                        source_top_x = np.median(source_points[source_points[:, 0] == source_top, 1])
                        processed_top_x = np.median(processed_points[processed_points[:, 0] == processed_top, 1])
                        
                        dx = int(source_top_x - processed_top_x)
                        dy = source_top - processed_top + vertical_bias
                
                elif alignment_method == "bbox":
                    source_contours, _ = cv2.findContours(source_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    processed_contours, _ = cv2.findContours(processed_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if source_contours and processed_contours:
                        source_x, source_y, _, _ = cv2.boundingRect(max(source_contours, key=cv2.contourArea))
                        processed_x, processed_y, _, _ = cv2.boundingRect(max(processed_contours, key=cv2.contourArea))
                        
                        dx = source_x - processed_x
                        dy = source_y - processed_y + vertical_bias
                
                # Apply transformation
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                aligned_mask = cv2.warpAffine(processed_mask, M, (processed_mask.shape[1], processed_mask.shape[0]))
                aligned_img = cv2.warpAffine(processed_img, M, (processed_img.shape[1], processed_img.shape[0]))
                
                # Save debug visualization if debug directory is provided
                if debug_dir:
                    try:
                        # Create color visualizations of the masks
                        source_vis = np.zeros((source_mask.shape[0], source_mask.shape[1], 3), dtype=np.uint8)
                        source_vis[source_mask_bin > 0] = [0, 255, 0]  # Green for source mask
                        
                        aligned_vis = np.zeros((aligned_mask.shape[0], aligned_mask.shape[1], 3), dtype=np.uint8)
                        aligned_vis[aligned_mask > 0] = [0, 0, 255]  # Red for aligned mask
                        
                        # Combine visualizations
                        overlay = source_vis + aligned_vis
                        
                        # Save visualization
                        cv2.imwrite(os.path.join(debug_dir, "mask_alignment.png"), overlay)
                        cv2.imwrite(os.path.join(debug_dir, "aligned_mask.png"), aligned_mask)
                        cv2.imwrite(os.path.join(debug_dir, "aligned_image.png"), aligned_img)
                    except Exception as e:
                        print(f"Error saving debug visualization: {e}")
            
            except Exception as e:
                print(f"Alignment error: {e}")
                import traceback
                traceback.print_exc()
        
        # Now create a soft-edged version of the aligned mask if needed
        if soft_edge_width > 0:
            try:
                # Get the soft edge width from the UI if available
                if hasattr(self.app, 'soft_edge_width'):
                    soft_edge_width = max(1, self.app.soft_edge_width.get())
                
                # Create kernel for morphology operations
                kernel = np.ones((max(1, soft_edge_width // 3), max(1, soft_edge_width // 3)), np.uint8)
                
                # Create inner and outer mask regions
                _, binary_mask = cv2.threshold(aligned_mask, 127, 255, cv2.THRESH_BINARY)
                dilated = cv2.dilate(binary_mask, kernel, iterations=1)
                eroded = cv2.erode(binary_mask, kernel, iterations=1)
                
                # Create transition region
                border = dilated & ~eroded
                
                # Create float mask starting with 1.0 in eroded region
                soft_mask = np.zeros_like(binary_mask, dtype=np.float32)
                soft_mask[eroded > 0] = 1.0
                
                # Apply distance-based feathering in border region
                if np.any(border):
                    dist = cv2.distanceTransform(border, cv2.DIST_L2, 3)
                    max_dist = np.max(dist) if np.max(dist) > 0 else 1.0
                    normalized_dist = dist / max_dist
                    soft_mask[border > 0] = 1.0 - normalized_dist[border > 0]
                
                # Convert back to uint8 for return
                aligned_mask = (soft_mask * 255).astype(np.uint8)
                
                # Save debug visualization if debug directory is provided
                if debug_dir:
                    try:
                        cv2.imwrite(os.path.join(debug_dir, "soft_mask.png"), aligned_mask)
                        
                        # Create visualization of soft mask
                        soft_vis = np.zeros((aligned_mask.shape[0], aligned_mask.shape[1], 3), dtype=np.uint8)
                        # Use heat map coloring: red (1.0) to blue (0.0)
                        soft_vis[:,:,0] = (soft_mask * 255).astype(np.uint8)  # B
                        soft_vis[:,:,2] = ((1.0 - soft_mask) * 255).astype(np.uint8)  # R
                        cv2.imwrite(os.path.join(debug_dir, "soft_mask_visualization.png"), soft_vis)
                    except Exception as e:
                        print(f"Error saving soft mask visualization: {e}")
            
            except Exception as e:
                print(f"Soft mask creation error: {e}")
                import traceback
                traceback.print_exc()
        
        # Return the aligned mask and image
        return aligned_mask, aligned_img
        # Compute vertical bias dynamically
        def compute_vertical_bias(source_mask, processed_mask):
            """
            Compute an intelligent vertical bias based on mask characteristics.
            
            Args:
                source_mask: Binary mask of source image
                processed_mask: Binary mask of processed image
            
            Returns:
                int: Recommended vertical shift
            """
            source_points = np.argwhere(source_mask > 0)
            processed_points = np.argwhere(processed_mask > 0)
            
            if len(source_points) == 0 or len(processed_points) == 0:
                return 0
            
            source_top = source_points[:, 0].min()
            processed_top = processed_points[:, 0].min()
            
            # Compute vertical difference with additional upward bias
            vertical_diff = source_top - processed_top
            upward_bias = 10  # Configurable upward shift
            
            return vertical_diff - upward_bias
        
        # Soft mask edge function
        def soft_mask_edge(mask, feather_pixels=15):
            """
            Create a soft-edged mask with gradual transition.
            
            Args:
                mask: Input binary mask
                feather_pixels: Width of soft edge transition
            
            Returns:
                numpy.ndarray: Soft-edged mask with float values
            """
            kernel = np.ones((feather_pixels, feather_pixels), np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=1)
            eroded = cv2.erode(mask, kernel, iterations=1)
            
            soft_mask = np.zeros_like(mask, dtype=np.float32)
            soft_mask[eroded > 0] = 1.0
            
            border = dilated & ~eroded
            dist = cv2.distanceTransform(~border, cv2.DIST_L2, 5)
            dist_normalized = dist / feather_pixels
            soft_mask[border > 0] = 1.0 - dist_normalized[border > 0]
            
            return soft_mask
        
        # Compute intelligent vertical bias
        vertical_bias = compute_vertical_bias(source_mask_bin, processed_mask_bin)
        
        # Make copies to modify
        aligned_mask = processed_mask.copy()
        aligned_img = processed_img.copy()
        
        # Alignment methods with improved vertical handling
        if alignment_method in ["centroid", "landmarks", "contour", "bbox"]:
            try:
                # Compute moments or contours
                if alignment_method == "centroid":
                    source_moments = cv2.moments(source_mask_bin)
                    processed_moments = cv2.moments(processed_mask_bin)
                    
                    source_cx = int(source_moments["m10"] / source_moments["m00"])
                    source_cy = int(source_moments["m01"] / source_moments["m00"])
                    processed_cx = int(processed_moments["m10"] / processed_moments["m00"])
                    processed_cy = int(processed_moments["m01"] / processed_moments["m00"])
                    
                    dx = source_cx - processed_cx
                    dy = source_cy - processed_cy + vertical_bias
                
                elif alignment_method in ["landmarks", "contour"]:
                    source_points = np.argwhere(source_mask_bin > 0)
                    processed_points = np.argwhere(processed_mask_bin > 0)
                    
                    source_top_x = np.median(source_points[source_points[:, 0].min() == source_points[:, 0], 1])
                    processed_top_x = np.median(processed_points[processed_points[:, 0].min() == processed_points[:, 0], 1])
                    
                    dx = int(source_top_x - processed_top_x)
                    dy = vertical_bias
                
                elif alignment_method == "bbox":
                    source_contours, _ = cv2.findContours(source_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    processed_contours, _ = cv2.findContours(processed_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    source_x, _, _, _ = cv2.boundingRect(max(source_contours, key=cv2.contourArea))
                    processed_x, _, _, _ = cv2.boundingRect(max(processed_contours, key=cv2.contourArea))
                    
                    dx = source_x - processed_x
                    dy = vertical_bias
                
                # Apply transformation
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                aligned_mask = cv2.warpAffine(processed_mask, M, (processed_mask.shape[1], processed_mask.shape[0]))
                aligned_img = cv2.warpAffine(processed_img, M, (processed_img.shape[1], processed_img.shape[0]))
            
            except Exception as e:
                print(f"Alignment error: {e}")
        
        # Apply soft mask edge
        aligned_soft_mask = soft_mask_edge(aligned_mask)
        
        # Debug visualization
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "aligned_soft_mask.png"), (aligned_soft_mask * 255).astype(np.uint8))
        
        return aligned_soft_mask, aligned_img
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