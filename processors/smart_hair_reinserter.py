"""
Smart Hair Reinserter for Dataset Preparation Tool
Specialized processor for reinserting AI-processed hair into original images with precise control.
Optimized for hair masks with advanced blending and alignment features.
"""

import os
import cv2
import numpy as np
import json
from skimage.transform import resize as skimage_resize

class SmartHairReinserter:
    """Specialized processor for reinserting AI-processed hair into original images."""
    
    def __init__(self, app):
        """
        Initialize smart hair reinserter.
        
        Args:
            app: The main application with shared variables and UI controls
        """
        self.app = app
    
    def reinsert_hair(self, input_dir, output_dir):
        """
        Reinsert AI-processed hair back into original images with smart alignment and blending.
        
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
        
        # Log current settings for reference
        print(f"Smart Hair Reinsertion: Input Dir: {input_dir}")
        print(f"Smart Hair Reinsertion: Source Dir: {self.app.source_images_dir.get()}")
        print(f"Smart Hair Reinsertion: Output Dir: {reinsert_output_dir}")
        print(f"Smart Hair Reinsertion: Mask-only mode: {self.app.reinsert_mask_only.get()}")
        print(f"Smart Hair Reinsertion: Vertical Bias: {self.app.vertical_alignment_bias.get()}")
        print(f"Smart Hair Reinsertion: Soft Edge Width: {self.app.soft_edge_width.get()}")
        
        # Find all processed images and their corresponding masks
        processed_images = []
        
        for root, dirs, files in os.walk(input_dir):
            # Skip if this is a 'masks' directory
            if os.path.basename(root).lower() == "masks":
                continue
                
            # Find image files
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    processed_path = os.path.join(root, file)
                    
                    # Look for corresponding mask in masks subdirectory
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
                    
                    # Add to processing list with metadata
                    if mask_path:
                        processed_images.append({
                            'processed_path': processed_path,
                            'mask_path': mask_path,
                            'filename': file
                        })
        
        # If no images with masks were found, report and return
        if not processed_images:
            self.app.status_label.config(text="No processed images with masks found in input directory.")
            return False
        
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
            
            # Get current file information
            processed_path = img_data['processed_path']
            mask_path = img_data['mask_path']
            filename = img_data['filename']
            
            try:
                # Match processed image to a source image
                source_filename = self._match_source_image(filename, source_images)
                
                if not source_filename:
                    self.app.status_label.config(text=f"Source image not found for {filename}")
                    failed_count += 1
                    continue
                
                source_path = source_images[source_filename]
                
                # Check for a source mask
                source_mask_path = self._find_source_mask_path(source_path)
                
                # Perform smart hair reinsertion
                output_path = os.path.join(reinsert_output_dir, f"reinserted_{filename}")
                
                success = self._reinsert_hair_smart(
                    source_path=source_path,
                    processed_path=processed_path,
                    mask_path=mask_path,
                    source_mask_path=source_mask_path,
                    output_path=output_path,
                    debug_dir=debug_dir
                )
                
                if success:
                    processed_count += 1
                else:
                    failed_count += 1
                
            except Exception as e:
                self.app.status_label.config(text=f"Error processing {filename}: {str(e)}")
                print(f"Error in smart_reinsert_hair: {str(e)}")
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
            self.app.status_label.config(text=f"Smart hair reinsertion completed. Processed {processed_count} images. Failed: {failed_count}.")
        else:
            self.app.status_label.config(text=f"Smart hair reinsertion completed. Successfully processed {processed_count} images.")
        
        self.app.progress_bar['value'] = 100
        return processed_count > 0
    
    def _match_source_image(self, processed_filename, source_images):
        """
        Match processed image filename to source image with multiple strategies.
        
        Args:
            processed_filename: Filename of the processed image
            source_images: Dictionary of source images
            
        Returns:
            str: Matched source filename or None if no match found
        """
        # Try direct filename match
        if processed_filename in source_images:
            return processed_filename
        
        # Try matching base name without extension
        base_name = os.path.splitext(processed_filename)[0]
        for source_file in source_images:
            source_base = os.path.splitext(source_file)[0]
            if source_base == base_name:
                return source_file
        
        # Try parsing crop info JSON file for original image reference
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
        
        # Try numeric pattern matching
        import re
        numbers = re.findall(r'\d+', base_name)
        if numbers:
            # Use the last number in the filename for matching
            target_number = numbers[-1]
            for source_file in source_images:
                source_numbers = re.findall(r'\d+', os.path.splitext(source_file)[0])
                if source_numbers and source_numbers[-1] == target_number:
                    return source_file
        
        # If only one source image exists, use it as a fallback
        if len(source_images) == 1:
            return list(source_images.keys())[0]
        
        # No match found
        print(f"No source image match found for {processed_filename}")
        return None
    
    def _find_source_mask_path(self, source_path):
        """
        Find the mask for a source image.
        
        Args:
            source_path: Path to the source image
            
        Returns:
            str: Path to source mask if found, None otherwise
        """
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
    
    def _reinsert_hair_smart(self, source_path, processed_path, mask_path, source_mask_path, output_path, debug_dir=None):
        """
        Reinsert AI-processed hair into original image with smart alignment and blending.
        
        Args:
            source_path: Path to original source image
            processed_path: Path to AI-processed image with hair modification
            mask_path: Path to hair mask for processed image
            source_mask_path: Path to hair mask for source image (optional)
            output_path: Path to save the result
            debug_dir: Directory to save debug visualizations (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Load images
        source_img = cv2.imread(source_path)
        processed_img = cv2.imread(processed_path)
        
        if source_img is None or processed_img is None:
            print(f"Failed to load source or processed image")
            return False
        
        # Load mask
        if mask_path is None:
            print("Mask path is required")
            return False
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load mask")
            return False
        
        # Load source mask if available
        source_mask = None
        if source_mask_path and os.path.exists(source_mask_path):
            source_mask = cv2.imread(source_mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Get dimensions
        source_h, source_w = source_img.shape[:2]
        processed_h, processed_w = processed_img.shape[:2]
        mask_h, mask_w = mask.shape[:2]
        
        # Save original images for debug purposes
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "source_original.png"), source_img)
            cv2.imwrite(os.path.join(debug_dir, "processed_original.png"), processed_img)
            cv2.imwrite(os.path.join(debug_dir, "mask_original.png"), mask)
            if source_mask is not None:
                cv2.imwrite(os.path.join(debug_dir, "source_mask_original.png"), source_mask)
        
        # Resize processed image and mask to match source if dimensions differ
        if source_w != processed_w or source_h != processed_h:
            processed_img = cv2.resize(processed_img, (source_w, source_h), 
                                    interpolation=cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, (source_w, source_h), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Ensure source mask is also properly sized
        if source_mask is not None and (source_mask.shape[0] != source_h or source_mask.shape[1] != source_w):
            source_mask = cv2.resize(source_mask, (source_w, source_h), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # If we're handling different masks between source and processed images
        if self.app.reinsert_handle_different_masks.get() and source_mask is not None:
            result_img = self._apply_smart_hair_alignment(
                source_img, 
                processed_img, 
                mask, 
                source_mask,
                debug_dir
            )
        else:
            # Use basic alpha blending if not handling different masks or no source mask
            result_img = self._apply_feathered_alpha_blend(
                source_img,
                processed_img,
                mask,
                self.app.soft_edge_width.get()
            )
        
        # Save the result
        cv2.imwrite(output_path, result_img)
        
        # Create comparison image for debugging
        if debug_dir:
            # Create a directory for comparison images
            comparisons_dir = os.path.join(debug_dir, "comparisons")
            os.makedirs(comparisons_dir, exist_ok=True)
            
            # Create a side-by-side comparison
            comparison = np.hstack((source_img, result_img))
            cv2.imwrite(os.path.join(comparisons_dir, f"comparison_{os.path.basename(output_path)}"), comparison)
            
            # If we have enough vertical space, add processed image for a 3-way comparison
            if source_h >= source_w * 2:  # If the height allows for stacking
                three_way = np.hstack((source_img, processed_img, result_img))
                cv2.imwrite(os.path.join(comparisons_dir, f"three_way_{os.path.basename(output_path)}"), three_way)
        
        return True
    
    def _apply_smart_hair_alignment(self, source_img, processed_img, processed_mask, source_mask, debug_dir=None):
        """
        Apply smart hair alignment between source and processed images, optimized for hair masks.
        
        Args:
            source_img: Original source image
            processed_img: Processed image with hair modification
            processed_mask: Mask for processed image
            source_mask: Mask for source image
            debug_dir: Directory to save debug visualizations
            
        Returns:
            numpy.ndarray: Result image with aligned hair
        """
        # Ensure we have binary masks
        _, proc_mask_binary = cv2.threshold(processed_mask, 127, 255, cv2.THRESH_BINARY)
        _, source_mask_binary = cv2.threshold(source_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Get vertical alignment bias from UI
        vertical_bias = self.app.vertical_alignment_bias.get()
        
        # STEP 1: Find vertical offset - optimize for hair by giving bias to upward alignment
        # (Hair typically looks better aligned by top rather than center)
        
        # Find top points of masks (hair typically starts from top)
        proc_points = np.argwhere(proc_mask_binary > 0)
        source_points = np.argwhere(source_mask_binary > 0)
        
        if len(proc_points) == 0 or len(source_points) == 0:
            print("Empty mask detected, using standard alpha blend")
            return self._apply_feathered_alpha_blend(
                source_img, 
                processed_img, 
                processed_mask,
                self.app.soft_edge_width.get()
            )
        
        # Get top rows of hair in each mask
        proc_top = proc_points[:, 0].min()
        source_top = source_points[:, 0].min()
        
        # Calculate vertical alignment with bias
        dy = (source_top - proc_top) - vertical_bias
        
        # STEP 2: Find horizontal alignment
        # Use center of mass in top third of hair for better hair alignment
        
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
        
        # STEP 3: Apply alignment transformation
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        
        # Warp processed image and mask
        aligned_img = cv2.warpAffine(processed_img, M, (processed_img.shape[1], processed_img.shape[0]))
        aligned_mask = cv2.warpAffine(processed_mask, M, (processed_mask.shape[1], processed_mask.shape[0]))
        
        # Save alignment debug images
        if debug_dir:
            # Draw lines on source and processed images to show alignment
            source_debug = source_img.copy()
            proc_debug = processed_img.copy()
            aligned_debug = aligned_img.copy()
            
            # Draw horizontal line at top of hair
            cv2.line(source_debug, (0, source_top), (source_img.shape[1], source_top), (0, 255, 0), 2)
            cv2.line(proc_debug, (0, proc_top), (proc_debug.shape[1], proc_top), (0, 0, 255), 2)
            
            # Draw vertical line at center of mass
            if proc_top_moments["m00"] > 0 and source_top_moments["m00"] > 0:
                cv2.line(source_debug, (source_top_cx, 0), (source_top_cx, source_debug.shape[0]), (0, 255, 0), 2)
                cv2.line(proc_debug, (proc_top_cx, 0), (proc_top_cx, proc_debug.shape[0]), (0, 0, 255), 2)
            
            # Save debug images
            cv2.imwrite(os.path.join(debug_dir, "source_alignment_debug.png"), source_debug)
            cv2.imwrite(os.path.join(debug_dir, "processed_alignment_debug.png"), proc_debug)
            cv2.imwrite(os.path.join(debug_dir, "aligned_image_debug.png"), aligned_debug)
            cv2.imwrite(os.path.join(debug_dir, "aligned_mask_debug.png"), aligned_mask)
            
            # Create visualization of the alignment
            # Create a colored visualization of mask alignment
            mask_viz = np.zeros((source_img.shape[0], source_img.shape[1], 3), dtype=np.uint8)
            mask_viz[source_mask_binary > 0] = [0, 255, 0]  # Source mask in green
            mask_viz[aligned_mask > 0] = [0, 0, 255]  # Aligned mask in red
            # Overlap will appear as cyan
            cv2.imwrite(os.path.join(debug_dir, "mask_alignment_viz.png"), mask_viz)
        
        # STEP 4: Apply soft-edge blending for more natural hair transitions
        soft_edge_width = self.app.soft_edge_width.get()
        
        # Create the final blended result
        result_img = self._apply_feathered_alpha_blend(source_img, aligned_img, aligned_mask, soft_edge_width)
        
        return result_img
    
    def _apply_feathered_alpha_blend(self, source_img, processed_img, mask, feather_pixels=15):
        """
        Apply alpha blending with feathered edges for smoother transitions.
        
        Args:
            source_img: Original source image
            processed_img: Processed image (possibly aligned)
            mask: Binary mask for blending
            feather_pixels: Width of feathered edge in pixels
            
        Returns:
            numpy.ndarray: Blended result image
        """
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
        
        # Apply selective color correction to better match processed hair to source
        # (Using color statistics from mask regions)
        
        # Apply the final blend
        result_img = source_img * (1.0 - alpha_mask_3d) + processed_img * alpha_mask_3d
        
        return np.clip(result_img, 0, 255).astype(np.uint8)