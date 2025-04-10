"""
Enhanced Crop Reinserter for Dataset Preparation Tool
Specialized for handling resolution differences between source and processed images.
"""

import os
import cv2
import numpy as np
import re
import json

from .alignment_utils import AlignmentUtils
from .blending_utils import BlendingUtils
from .hair_utils import HairUtils


class EnhancedCropReinserter:
    """Reinserts processed regions back into original images, handling resolution differences."""
    
    def __init__(self, app):
        """
        Initialize enhanced crop reinserter.
        
        Args:
            app: The main application with shared variables and UI controls
        """
        self.app = app
        
        # Initialize utility classes
        self.alignment_utils = AlignmentUtils(app)
        self.blending_utils = BlendingUtils(app)
        self.hair_utils = HairUtils(app)
    
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
        
    def _reinsert_with_resolution_handling(self, source_path, processed_path, mask_path, output_path, debug_dir=None):
        """
        Reinsert a processed image region into the source image with enhanced mask alignment,
        with better error handling for missing or empty masks.
        
        Args:
            source_path: Path to source (original) image
            processed_path: Path to processed image
            mask_path: Path to mask (if available)
            output_path: Path to save the result
            debug_dir: Directory to save debug visualizations (if enabled)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Load images with error checking
        source_img = cv2.imread(source_path)
        processed_img = cv2.imread(processed_path)
        
        if source_img is None:
            print(f"Failed to load source image: {source_path}")
            return False
        
        if processed_img is None:
            print(f"Failed to load processed image: {processed_path}")
            return False
        
        # Get the source mask (if available)
        source_mask = self._find_source_mask(source_path)
        print(f"Source mask found: {source_mask is not None}")
        
        # Get dimensions
        source_h, source_w = source_img.shape[:2]
        processed_h, processed_w = processed_img.shape[:2]
        
        # Get facial landmarks from source image
        source_landmarks = self.alignment_utils.get_landmarks(source_img)
        if source_landmarks is not None:
            print("Successfully detected landmarks in source image")
        else:
            print("No landmarks detected in source image")
        
        # Initialize mask with error handling
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to load mask: {mask_path}")
                # Create a fallback mask that covers the entire processed image
                mask = np.ones((processed_h, processed_w), dtype=np.uint8) * 255
                print("Created a fallback full-image mask")
        else:
            print(f"No mask found at {mask_path}")
            # Create a fallback mask covering the entire processed image
            mask = np.ones((processed_h, processed_w), dtype=np.uint8) * 255
            print("Created a fallback full-image mask")
        
        # Handle bangs-only mode
        if self.app.use_bangs_only.get():
            if debug_dir:
                # Save the original mask
                cv2.imwrite(os.path.join(debug_dir, "original_mask.png"), mask)
                
            # If we have a source mask, use it for bangs isolation instead
            if source_mask is not None:
                print("Using source mask for bangs isolation")
                bangs_mask = self.hair_utils.isolate_bangs_region(source_mask, source_landmarks)
                
                if debug_dir:
                    cv2.imwrite(os.path.join(debug_dir, "isolated_bangs_mask.png"), bangs_mask)
                    
                    # Create a visualization of the mask overlay using the source mask
                    if bangs_mask.shape[0] != source_h or bangs_mask.shape[1] != source_w:
                        bangs_mask_resized = cv2.resize(bangs_mask, (source_w, source_h), 
                                                    interpolation=cv2.INTER_LINEAR)
                    else:
                        bangs_mask_resized = bangs_mask
                    
                    # Create visualization of the mask overlay using the resized mask
                    mask_viz = np.zeros((source_h, source_w, 3), dtype=np.uint8)
                    mask_viz[bangs_mask_resized > 0] = [0, 255, 0]  # Green for bangs mask
                    # Overlay on source image with 50% opacity
                    overlay = cv2.addWeighted(source_img, 0.7, mask_viz, 0.3, 0)
                    cv2.imwrite(os.path.join(debug_dir, "bangs_overlay_viz.png"), overlay)
                    
                # Update the mask for further processing
                mask = bangs_mask
            else:
                print("No source mask found, using processed mask for bangs isolation")
                bangs_mask = self.hair_utils.isolate_bangs_region(mask, source_landmarks)
                
                if debug_dir:
                    cv2.imwrite(os.path.join(debug_dir, "isolated_bangs_mask.png"), bangs_mask)
                    
                    # Create visualization with the same safety checks
                    if bangs_mask.shape[0] != source_h or bangs_mask.shape[1] != source_w:
                        bangs_mask_resized = cv2.resize(bangs_mask, (source_w, source_h), 
                                                    interpolation=cv2.INTER_LINEAR)
                    else:
                        bangs_mask_resized = bangs_mask
                    
                    mask_viz = np.zeros((source_h, source_w, 3), dtype=np.uint8)
                    mask_viz[bangs_mask_resized > 0] = [0, 255, 0]  # Green for bangs mask
                    # Overlay on source image with 50% opacity
                    overlay = cv2.addWeighted(source_img, 0.7, mask_viz, 0.3, 0)
                    cv2.imwrite(os.path.join(debug_dir, "bangs_overlay_viz.png"), overlay)
                
                # Update the mask for further processing
                mask = bangs_mask

        # Apply bangs extension if enabled
        if self.app.extend_bangs.get() and mask is not None:
            print("Extending mask in bangs/forehead area")
            extension_amount = self.app.bangs_extension_amount.get()
            width_ratio = self.app.bangs_width_ratio.get()
            
            # Save pre-extension mask if debugging
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "pre_extension_mask.png"), mask)
                # Save mask visualization
                mask_viz = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                mask_viz[mask > 0] = [0, 255, 0]  # Green for original mask
                cv2.imwrite(os.path.join(debug_dir, "pre_extension_mask_viz.png"), mask_viz)
            
            print(f"Input mask shape: {mask.shape}, non-zero pixels: {np.count_nonzero(mask)}")
            
            # For larger extensions, use a smaller min_opacity to ensure visibility
            min_opacity_value = max(0.5, 0.9 - (extension_amount / 100.0))
            print(f"Using min_opacity: {min_opacity_value} for extension_amount: {extension_amount}")
            
            extended_mask = self.hair_utils.extend_bangs_area(
                mask, 
                extend_pixels=extension_amount,
                forehead_ratio=width_ratio,
                min_opacity=min_opacity_value,  # Dynamic min_opacity based on extension amount
                source_landmarks=source_landmarks  # Pass source landmarks
            )
            print(f"Output mask shape: {extended_mask.shape}, non-zero pixels: {np.count_nonzero(extended_mask)}")
            # Assign extended mask back to mask
            mask = extended_mask
            
            # Save debug visualization of extension
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "extended_bangs_mask.png"), mask)
                
                # Create better visualizations to see the difference
                mask_viz = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                mask_viz[mask > 0] = [0, 0, 255]  # Blue for extended mask
                cv2.imwrite(os.path.join(debug_dir, "extended_bangs_mask_viz.png"), mask_viz)
                
                # Resize the mask to match source dimensions before visualization
                if mask.shape[0] != source_h or mask.shape[1] != source_w:
                    mask_resized_for_viz = cv2.resize(mask, (source_w, source_h), 
                                                interpolation=cv2.INTER_LINEAR)
                else:
                    mask_resized_for_viz = mask
                
                # Create extension visualization
                ext_viz = np.zeros((source_h, source_w, 3), dtype=np.uint8)
                ext_viz[mask_resized_for_viz > 0] = [0, 0, 255]  # Red for extended mask
                # Overlay on source image
                ext_overlay = cv2.addWeighted(source_img, 0.7, ext_viz, 0.3, 0)
                cv2.imwrite(os.path.join(debug_dir, "extension_overlay_viz.png"), ext_overlay)
                
        # Error check before resizing
        if mask is None or mask.size == 0 or mask.shape[0] == 0 or mask.shape[1] == 0:
            print("Error: Mask is empty or invalid after processing")
            # Create a fallback mask
            mask = np.ones((processed_h, processed_w), dtype=np.uint8) * 255
            print("Created a new fallback mask")
        
        # Resize processed image and mask to match source if dimensions differ
        if source_w != processed_w or source_h != processed_h:
            print(f"Resizing processed image from {processed_w}x{processed_h} to {source_w}x{source_h}")
            try:
                processed_img_resized = cv2.resize(processed_img, (source_w, source_h), 
                                                interpolation=cv2.INTER_LANCZOS4)
                mask_resized = cv2.resize(mask, (source_w, source_h), 
                                        interpolation=cv2.INTER_LINEAR)
            except cv2.error as e:
                print(f"Error during resize: {str(e)}")
                return False
        else:
            processed_img_resized = processed_img
            mask_resized = mask
        
        # For alignment, always use the source landmarks as reference
        # Don't try to detect landmarks in the processed image (it's just hair)
        aligned_mask = mask_resized.copy()
        aligned_img = processed_img_resized.copy()
        
        # Apply landmark-based alignment if selected
        if self.app.reinsert_alignment_method.get() == "landmarks" and source_landmarks is not None:
            print("Using source landmarks for alignment")
            # Create a geometric estimation of where the bangs should be positioned
            estimated_bangs_position = self.hair_utils.estimate_bangs_position(source_landmarks, source_h, source_w)
            
            # Apply a simple offset based on the estimated position
            offset_x = estimated_bangs_position['center_x'] - source_w // 2
            offset_y = estimated_bangs_position['top_y'] - 20  # Place slightly above eyebrows
            
            # Create and apply transformation
            M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
            aligned_mask = cv2.warpAffine(mask_resized, M, (source_w, source_h))
            aligned_img = cv2.warpAffine(processed_img_resized, M, (source_w, source_h))
            
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "landmark_aligned.png"), aligned_img)
                
                # Save debug visualization of landmarks and alignment
                debug_img = source_img.copy()
                for i, (x, y) in enumerate(source_landmarks):
                    cv2.circle(debug_img, (int(x), int(y)), 2, (0, 255, 0), -1)
                    if i in [0, 16, 17, 26]:  # Key points for bangs alignment
                        cv2.circle(debug_img, (int(x), int(y)), 4, (0, 0, 255), -1)
                cv2.imwrite(os.path.join(debug_dir, "source_landmarks_debug.png"), debug_img)
        
        # Debug: Save original and resized images
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "source_original.png"), source_img)
            cv2.imwrite(os.path.join(debug_dir, "processed_original.png"), processed_img)
            cv2.imwrite(os.path.join(debug_dir, "mask_original.png"), mask)
            
            if source_w != processed_w or source_h != processed_h:
                cv2.imwrite(os.path.join(debug_dir, "processed_resized.png"), processed_img_resized)
                cv2.imwrite(os.path.join(debug_dir, "mask_resized.png"), mask_resized)
        
        # Get manual offset values
        manual_offset_x = self.app.reinsert_manual_offset_x.get()
        manual_offset_y = self.app.reinsert_manual_offset_y.get()

        # Get manual scaling values
        scale_x = self.app.reinsert_manual_scale_x.get()
        scale_y = self.app.reinsert_manual_scale_y.get()

        # Apply scaling correctly without affecting position
        if scale_x != 1.0 or scale_y != 1.0:
            print(f"Applying manual scaling: X={scale_x}, Y={scale_y}")
            
            # Determine the anchor point (top center of the mask)
            if np.any(aligned_mask > 0):
                # Find top edge of mask for Y scaling anchor
                non_zero_y = np.where(np.any(aligned_mask > 0, axis=1))[0]
                if len(non_zero_y) > 0:
                    top_y = non_zero_y[0]
                    # Find center of mask horizontally
                    non_zero_x = np.where(aligned_mask[top_y, :] > 0)[0]
                    if len(non_zero_x) > 0:
                        center_x = (np.min(non_zero_x) + np.max(non_zero_x)) // 2
                    else:
                        center_x = source_w // 2
                else:
                    top_y = 0
                    center_x = source_w // 2
            else:
                top_y = 0
                center_x = source_w // 2
            
            print(f"Scaling anchor point: ({center_x}, {top_y})")
            
            # Create a transformation matrix that maintains the top point's position
            M = np.float32([
                [scale_x, 0, center_x * (1 - scale_x)],
                [0, scale_y, top_y * (1 - scale_y)]  # This keeps the top fixed during scaling
            ])
            
            # Apply to both processed image and mask
            aligned_img = cv2.warpAffine(
                aligned_img, M, (source_w, source_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            aligned_mask = cv2.warpAffine(
                aligned_mask, M, (source_w, source_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # Save scaled versions for debugging
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "scaled_processed.png"), aligned_img)
                cv2.imwrite(os.path.join(debug_dir, "scaled_mask.png"), aligned_mask)
        
        # Apply manual offset AFTER scaling
        if manual_offset_x != 0 or manual_offset_y != 0:
            print(f"Applying manual offset: X={manual_offset_x}, Y={manual_offset_y}")
            
            # Create transformation matrix for the offset
            M = np.float32([[1, 0, manual_offset_x], [0, 1, manual_offset_y]])
            
            # Apply to both processed image and mask
            aligned_img = cv2.warpAffine(
                aligned_img, M, (source_w, source_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            aligned_mask = cv2.warpAffine(
                aligned_mask, M, (source_w, source_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # Save offset versions for debugging
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "offset_processed.png"), aligned_img)
                cv2.imwrite(os.path.join(debug_dir, "offset_mask.png"), aligned_mask)
        
        # If we're not handling different masks, simply use standard blending
        if not self.app.reinsert_handle_different_masks.get() or source_mask is None:
            # Basic alpha blending
            mask_float = aligned_mask.astype(float) / 255.0
            mask_float_3d = np.stack([mask_float] * 3, axis=2)
            result_img = source_img * (1 - mask_float_3d) + aligned_img * mask_float_3d
            result_img = np.clip(result_img, 0, 255).astype(np.uint8)
            
            cv2.imwrite(output_path, result_img)
            return True
        
        # For more complex blending, use the chosen method
        alignment_method = self.app.reinsert_alignment_method.get()
        blend_mode = self.app.reinsert_blend_mode.get()
        blend_extent = self.app.reinsert_blend_extent.get()
        preserve_edges = self.app.reinsert_preserve_edges.get()
        
        # Align masks if not using "none" alignment
        aligned_mask = mask_resized.copy()
        aligned_img = processed_img_resized.copy()
        
        if alignment_method != "none":
            # Perform the alignment based on the selected method
            aligned_mask, aligned_img = self.alignment_utils.align_masks(
                source_mask, mask_resized, 
                source_img, processed_img_resized, 
                alignment_method, 
                debug_dir
            )
        
        # Apply more advanced alignment with landmarks if selected
        if self.app.reinsert_alignment_method.get() == "landmarks":
            if self.alignment_utils.face_detector is not None and self.alignment_utils.landmark_predictor is not None:
                # Only use source landmarks (already detected at beginning of method)
                if source_landmarks is not None:
                    print("Using source landmarks for alignment")
                    
                    # Use either translation-only or full transform based on settings
                    if self.app.use_translation_only.get():
                        # Create estimated landmarks for processed image
                        h, w = source_img.shape[:2]
                        ph, pw = processed_img_resized.shape[:2]
                        
                        print("Using source landmarks for positioning only")
                        
                        # Estimate where bangs should be positioned based on face landmarks
                        bangs_position = self.hair_utils.estimate_bangs_position(source_landmarks, h, w)
                        
                        # Calculate the center of the processed image
                        proc_center_x = pw // 2
                        proc_center_y = ph // 4  # Position toward the top quarter
                        
                        # Calculate the offset to move the processed image
                        offset_x = bangs_position['center_x'] - proc_center_x
                        offset_y = bangs_position['top_y'] - proc_center_y
                        
                        # Create and apply the transformation
                        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                        aligned_mask = cv2.warpAffine(mask_resized, M, (w, h), flags=cv2.INTER_LINEAR)
                        aligned_img = cv2.warpAffine(processed_img_resized, M, (w, h), flags=cv2.INTER_LANCZOS4)
                    else:
                        # For full transform, we still need some estimate of processed landmarks
                        # Create a simplified set of landmarks for the processed image
                        # based on the proportions of the source landmarks
                        print("Estimating processed landmarks for transform")
                        h, w = source_img.shape[:2]
                        ph, pw = processed_img_resized.shape[:2]
                        
                        # Create a simplified set of landmarks for the processed image
                        processed_landmarks = []
                        # Use the source landmarks as reference, but adjusted to the processed image proportions
                        for x, y in source_landmarks:
                            # Scale to processed image dimensions
                            scaled_x = int(x * (pw / w))
                            scaled_y = int(y * (ph / h))
                            processed_landmarks.append((scaled_x, scaled_y))
                        
                        # Now use both sets of landmarks for the full transform
                        aligned_mask, aligned_img = self.alignment_utils.apply_landmark_transform(
                            source_img, processed_img_resized, source_mask, mask_resized,
                            source_landmarks, processed_landmarks, debug_dir
                        )
                    
                    # Save additional debug visualization
                    if debug_dir:
                        source_vis = source_img.copy()
                        for i, (x, y) in enumerate(source_landmarks):
                            cv2.circle(source_vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                        
                        landmark_debug = np.hstack((source_vis, aligned_img))
                        cv2.imwrite(os.path.join(debug_dir, "landmark_alignment_process.png"), landmark_debug)
                else:
                    print("Could not detect landmarks in source image, falling back to default alignment")
                    # Use existing alignment method...
                    aligned_mask, aligned_img = self.alignment_utils.align_masks(
                        source_mask, mask_resized, 
                        source_img, processed_img_resized, 
                        alignment_method, 
                        debug_dir
                    )
            else:
                print("Landmark detection not available, falling back to default alignment")
                # Use existing alignment method...
                aligned_mask, aligned_img = self.alignment_utils.align_masks(
                    source_mask, mask_resized, 
                    source_img, processed_img_resized, 
                    alignment_method, 
                    debug_dir
                )
        else:
            # Use existing alignment method...
            aligned_mask, aligned_img = self.alignment_utils.align_masks(
                source_mask, mask_resized, 
                source_img, processed_img_resized, 
                alignment_method, 
                debug_dir
            )