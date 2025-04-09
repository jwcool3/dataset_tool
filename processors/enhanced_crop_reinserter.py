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
import dlib 


class EnhancedCropReinserter:
    """Reinserts processed regions back into original images, handling resolution differences."""
    
    def __init__(self, app):
        """
        Initialize enhanced crop reinserter.
        
        Args:
            app: The main application with shared variables and UI controls
        """
        self.app = app
        
        # Initialize face detection if dlib is available
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            
            # Look for the shape predictor file in a few common locations
            predictor_paths = [
                os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "shape_predictor_68_face_landmarks.dat"),
                os.path.join(os.path.expanduser("~"), ".dataset_preparation_tool", "models", "shape_predictor_68_face_landmarks.dat")
            ]
            
            self.landmark_predictor = None
            for path in predictor_paths:
                if os.path.exists(path):
                    self.landmark_predictor = dlib.shape_predictor(path)
                    print(f"Found landmark predictor at: {path}")
                    break
                    
            if self.landmark_predictor is None:
                print("WARNING: Facial landmark predictor file not found. Landmark-based alignment will not be available.")
                print("Please download shape_predictor_68_face_landmarks.dat and place it in the processors directory.")
        except Exception as e:
            print(f"Could not initialize facial landmark detection: {str(e)}")
            self.face_detector = None
            self.landmark_predictor = None
    
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
        
        # Get dimensions
        source_h, source_w = source_img.shape[:2]
        processed_h, processed_w = processed_img.shape[:2]
        
        # MODIFICATION: Only detect face in the SOURCE image
        source_landmarks = None
        if hasattr(self, 'face_detector') and self.face_detector is not None:
            print("Detecting face in source image...")
            source_landmarks = self._get_landmarks(source_img)
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
        
        # Replace this section in the debug visualization code:
        if self.app.use_bangs_only.get() and debug_dir:
            # Save the original mask
            cv2.imwrite(os.path.join(debug_dir, "original_mask.png"), mask)
            
            # After isolating bangs
            bangs_mask = self._isolate_bangs_region(mask, source_landmarks)
            cv2.imwrite(os.path.join(debug_dir, "isolated_bangs_mask.png"), bangs_mask)
            
            # IMPORTANT FIX: Resize the bangs mask to match source dimensions before visualization
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
            # Regular bangs-only mode without debug
            if self.app.use_bangs_only.get():
                print("Using bangs-only mode")
                bangs_mask = self._isolate_bangs_region(mask, source_landmarks)
                mask = bangs_mask

        # Apply bangs extension if enabled
        if self.app.extend_bangs.get() and mask is not None:
            print("Extending mask in bangs/forehead area")
            extension_amount = self.app.bangs_extension_amount.get()
            width_ratio = self.app.bangs_width_ratio.get()
            
            # Save pre-extension mask if debugging
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "pre_extension_mask.png"), mask)
            
            # Pass source_landmarks to the extension function
            mask = self._extend_bangs_area(
                mask, 
                extend_pixels=extension_amount,
                forehead_ratio=width_ratio,
                min_opacity=self.app.bangs_min_opacity.get(),
                source_landmarks=source_landmarks  # Pass source landmarks
            )
            
            # Save debug visualization of extension
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "extended_bangs_mask.png"), mask)
                
                # IMPORTANT FIX: Resize the mask to match source dimensions before visualization
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
        
        # Get the source mask (if available)
        source_mask = self._find_source_mask(source_path)
        
        # For alignment, always use the source landmarks as reference
        # Don't try to detect landmarks in the processed image (it's just hair)
        aligned_mask = mask_resized.copy()
        aligned_img = processed_img_resized.copy()
        
        # MODIFICATION: Change the landmark-based alignment to use source landmarks only
        if self.app.reinsert_alignment_method.get() == "landmarks" and source_landmarks is not None:
            print("Using source landmarks for alignment")
            # Create a geometric estimation of where the bangs should be positioned
            estimated_bangs_position = self._estimate_bangs_position(source_landmarks, source_h, source_w)
            
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

            processed_img_resized = cv2.resize(processed_img, (source_w, source_h), 
                                        interpolation=cv2.INTER_LANCZOS4)
            mask_resized = cv2.resize(mask, (source_w, source_h), 
                                interpolation=cv2.INTER_LINEAR)  # Changed from INTER_NEAREST for smoother results
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



        # Inside _reinsert_with_resolution_handling method, before applying blending:
        if self.app.reinsert_alignment_method.get() == "landmarks":
            if hasattr(self, 'face_detector') and self.face_detector is not None and \
            hasattr(self, 'landmark_predictor') and self.landmark_predictor is not None:
                
                # Get landmarks for both images
                source_landmarks = self._get_landmarks(source_img)
                processed_landmarks = self._get_landmarks(processed_img_resized)
                
                if source_landmarks is not None and processed_landmarks is not None:
                    # Determine which alignment method to use
                    if self.app.use_translation_only.get():
                        print("Using translation-only landmark-based alignment")
                        aligned_mask, aligned_img = self._apply_translation_only_transform(
                            source_img, processed_img_resized, source_mask, mask_resized, 
                            source_landmarks, processed_landmarks, debug_dir
                        )
                    else:
                        print("Using full transform landmark-based alignment")
                        aligned_mask, aligned_img = self._apply_landmark_transform(
                            source_img, processed_img_resized, source_mask, mask_resized, 
                            source_landmarks, processed_landmarks, debug_dir
                        )
                    
                    # Save additional debug visualization
                    if debug_dir:
                        landmark_debug = np.hstack((source_img, processed_img_resized, aligned_img))
                        cv2.imwrite(os.path.join(debug_dir, "landmark_alignment_process.png"), landmark_debug)
                else:
                    print("Could not detect landmarks in one or both images, falling back to default alignment")
                    # Use existing alignment method...
                    aligned_mask, aligned_img = self._align_masks(
                        source_mask, mask_resized, 
                        source_img, processed_img_resized, 
                        alignment_method, 
                        debug_dir
                    )
            else:
                print("Landmark detection not available, falling back to default alignment")
                # Use existing alignment method...
                aligned_mask, aligned_img = self._align_masks(
                    source_mask, mask_resized, 
                    source_img, processed_img_resized, 
                    alignment_method, 
                    debug_dir
                )
        else:
            # Use existing alignment method...
            aligned_mask, aligned_img = self._align_masks(
                source_mask, mask_resized, 
                source_img, processed_img_resized, 
                alignment_method, 
                debug_dir
            )
        # Preserve hair parting if option enabled
        if self.app.preserve_hair_parting.get():  # Add this option to UI
            # Detect landmarks for source and processed images
            source_landmarks = self._get_landmarks(source_img)
            processed_landmarks = self._get_landmarks(aligned_img)
            
            # Detect parting in source and processed hair
            source_parting, _ = self._detect_hair_parting(source_mask, source_landmarks)
            processed_parting, _ = self._detect_hair_parting(aligned_mask, processed_landmarks)
            
            # If partings detected, blend them
            if source_parting is not None and processed_parting is not None:
                # Create a blended parting that preserves the original direction
                blended_parting = cv2.addWeighted(source_parting, 0.7, processed_parting, 0.3, 0)
                
                # Apply blended parting to aligned mask
                _, aligned_mask = self._detect_hair_parting(aligned_mask, None, blended_parting)
                
                if debug_dir:
                    cv2.imwrite(os.path.join(debug_dir, "parting_preserved.png"), aligned_mask)
        
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
        # Ensure source_mask is valid
        if source_mask is None:
            print("Warning: source_mask is None, using processed mask without alignment")
            return processed_mask, processed_img
            
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
            # Get dimensions of source image for visualization
            h, w = source_img.shape[:2]
            mask_viz = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Resize masks if needed
            if source_mask_bin.shape[0] != h or source_mask_bin.shape[1] != w:
                source_mask_bin_resized = cv2.resize(source_mask_bin, (w, h), 
                                                interpolation=cv2.INTER_NEAREST)
            else:
                source_mask_bin_resized = source_mask_bin
                
            if aligned_mask.shape[0] != h or aligned_mask.shape[1] != w:
                aligned_mask_resized = cv2.resize(aligned_mask, (w, h), 
                                            interpolation=cv2.INTER_NEAREST)
            else:
                aligned_mask_resized = aligned_mask
            
            # Use the resized masks for visualization
            mask_viz[source_mask_bin_resized > 0] = [0, 0, 255]  # Red for source mask
            mask_viz[aligned_mask_resized > 0] = [0, 255, 0]     # Green for aligned mask
            
            cv2.imwrite(os.path.join(debug_dir, "mask_alignment_viz.png"), mask_viz)
        
        # Make sure to return the aligned mask and image
        return aligned_mask, aligned_img

    def _alpha_blend(self, source_img, processed_img, mask, blend_extent=8):
        """
        Enhanced alpha blending with better feathering and color correction.
        
        Args:
            source_img: Original source image
            processed_img: Processed image to blend
            mask: Blending mask
            blend_extent: Extent of feathering (pixels)
        
        Returns:
            numpy.ndarray: Blended image
        """
        # Create alpha mask with proper type conversion
        mask_float = mask.astype(np.float32) / 255.0
        
        # Apply feathering if blend_extent > 0
        if blend_extent > 0:
            # Create feathering kernel
            kernel = np.ones((blend_extent, blend_extent), np.uint8)
            
            # Create dilation and border regions
            mask_binary = (mask > 127).astype(np.uint8) * 255
            dilated = cv2.dilate(mask_binary, kernel, iterations=1)
            border = cv2.bitwise_and(dilated, cv2.bitwise_not(mask_binary))
            
            # Create distance map for feathering
            dist = cv2.distanceTransform(border, cv2.DIST_L2, 3)
            dist = np.clip(dist, 0, blend_extent)
            
            # Normalize distances (1.0 at mask edge, 0.0 at outer edge)
            feather = dist / blend_extent
            
            # Apply feathering to the mask
            # For border pixels, we want a gradient from mask_float to 0
            feathered_mask = mask_float.copy()
            border_indices = (border > 0)
            feathered_mask[border_indices] = feather[border_indices]
            
            # Use the feathered mask for blending
            mask_float = feathered_mask
        
        # Apply color correction at the boundary
        if blend_extent > 0:
            # Get blurred versions of source and processed images
            source_blur = cv2.GaussianBlur(source_img.astype(np.float32), (21, 21), 0)
            processed_blur = cv2.GaussianBlur(processed_img.astype(np.float32), (21, 21), 0)
            
            # Create a tight mask for the actual hair region (not the feathered part)
            core_mask = (mask > 200).astype(np.float32)
            
            # For the border region, calculate color adjustment
            border_float = (border > 0).astype(np.float32)
            border_float_3d = np.stack([border_float] * 3, axis=2)
            
            # Calculate color ratio (processed / source) for adjustment
            # Add epsilon to avoid division by zero
            color_ratio = np.divide(
                processed_blur + 1.0, 
                source_blur + 1.0,
                out=np.ones_like(processed_blur),
                where=(source_blur > 10.0) & (border_float_3d > 0)
            )
            
            # Apply color adjustment to processed image
            # in the border region to match the source color tone
            adjusted_processed = processed_img.astype(np.float32).copy()
            adjusted_processed = np.divide(
                adjusted_processed,
                color_ratio,
                out=adjusted_processed,
                where=border_float_3d > 0
            )
            
            # Apply adjustment with a gradient
            processed_blend = processed_img * (1 - border_float_3d) + adjusted_processed * border_float_3d
            processed_img = processed_blend.astype(np.float32)
        
        # Create 3-channel mask for blending
        mask_float_3d = np.stack([mask_float] * 3, axis=2)
        
        # Apply final blending
        result_img = source_img.astype(np.float32) * (1 - mask_float_3d) + processed_img * mask_float_3d
        
        # Add very slight contrast enhancement to the blended region to make it pop
        blend_region = (mask_float_3d > 0.1) & (mask_float_3d < 0.9)
        if np.any(blend_region):
            # Apply subtle contrast adjustment only to the blend border
            result_img = result_img.astype(np.float32)
            blended_part = result_img * blend_region
            contrast_enhanced = cv2.addWeighted(blended_part, 1.05, blended_part, 0, 0)
            result_img = result_img * (1 - blend_region) + contrast_enhanced * blend_region
        
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
    
    def _extend_bangs_area(self, mask, extend_pixels=30, forehead_ratio=0.3, min_opacity=0.7, source_landmarks=None):
        """
        Improved bangs extension with proper scaling for source landmarks.
        
        Args:
            mask: The binary mask image
            extend_pixels: How many pixels to extend downward
            forehead_ratio: What portion of the width to consider as forehead (centered)
            min_opacity: Minimum opacity value at the edges of extension (0.0-1.0)
            source_landmarks: Optional landmarks from source image for better positioning
                
        Returns:
            numpy.ndarray: Extended mask
        """
        # Basic error checking
        if mask is None:
            print("Input mask is None in extend_bangs_area")
            return None
        
        # Check if mask is valid
        if mask.size == 0 or mask.shape[0] == 0 or mask.shape[1] == 0:
            print("Input mask has invalid dimensions")
            return mask
        
        # Check if mask is empty (all zeros)
        if np.max(mask) == 0:
            print("Warning: Empty mask, nothing to extend")
            return mask
        
        # Create a copy of the mask to modify
        extended_mask = mask.copy()
        height, width = mask.shape[:2]
        
        # If source landmarks are available, use them to guide the extension
        if source_landmarks is not None and len(source_landmarks) >= 27:
            try:
                # Calculate estimated source image dimensions based on landmarks
                source_width_estimate = max([p[0] for p in source_landmarks[:17]]) - min([p[0] for p in source_landmarks[:17]])
                source_height_estimate = max([p[1] for p in source_landmarks]) - min([p[1] for p in source_landmarks])
                
                # Calculate scale factors
                scale_x = width / source_width_estimate
                scale_y = height / source_height_estimate
                
                # Scale the landmark coordinates
                eyebrow_points = source_landmarks[17:27]
                scaled_eyebrow_y = min([int(p[1] * scale_y) for p in eyebrow_points])
                
                # Make sure it's within bounds
                scaled_eyebrow_y = min(height-1, max(0, scaled_eyebrow_y))
                
                # Adjust extend_pixels based on eyebrow position
                # More extension if eyebrows are lower in the image
                eyebrow_ratio = scaled_eyebrow_y / height
                adjusted_extend = max(extend_pixels, int(extend_pixels * (1 + eyebrow_ratio)))
                
                print(f"Adjusted extend_pixels to {adjusted_extend} based on eyebrow position")
                extend_pixels = adjusted_extend
            except Exception as e:
                print(f"Error using landmarks for extension: {str(e)}")
        
        # Find the non-zero points in the mask
        mask_points = np.argwhere(mask > 0)
        if len(mask_points) == 0:
            print("Warning: No mask points found, nothing to extend")
            return mask
        
        # Calculate the forehead region (center portion of width)
        center_x = width // 2
        forehead_half_width = int(width * forehead_ratio / 2)
        forehead_left = max(0, center_x - forehead_half_width)
        forehead_right = min(width, center_x + forehead_half_width)
        
        # Find the topmost point of the mask
        top_y = np.min(mask_points[:, 0]) if len(mask_points) > 0 else 0
        
        print(f"Top Y position: {top_y}, extending by {extend_pixels} pixels")
        print(f"Forehead region: left={forehead_left}, right={forehead_right}, " +
            f"width={forehead_right-forehead_left}")
        
        # Create a more natural curved extension
        for x in range(forehead_left, forehead_right):
            # Calculate distance from center as a ratio (0.0 at center, 1.0 at edges)
            center_dist = abs(x - center_x) / (forehead_half_width + 1e-5)
            center_dist = min(1.0, center_dist)  # Cap at 1.0
            
            # Use cosine curve for natural falloff
            # More extension in center, less at edges
            extension_factor = np.cos(center_dist * np.pi / 2)
            current_extend = int(extend_pixels * extension_factor)
            
            # Find topmost non-zero pixel in this column
            col_points = np.where(mask[:, x] > 0)[0]
            col_top = np.min(col_points) if len(col_points) > 0 else top_y
            
            # Start extension from this point
            for y in range(col_top, min(height, col_top + current_extend)):
                # Skip if pixel already has a higher value
                if extended_mask[y, x] >= min_opacity * 255:
                    continue
                    
                # Calculate fade factor (1.0 at top, min_opacity at bottom)
                y_progress = (y - col_top) / float(max(1, current_extend))
                fade = 1.0 - (y_progress * (1.0 - min_opacity))
                fade_value = int(255 * fade)
                
                extended_mask[y, x] = fade_value
        
        # Apply a slight blur to create smoother transitions
        extended_mask = cv2.GaussianBlur(extended_mask, (3, 3), 0)
        
        return extended_mask

    def _improve_landmark_alignment(self, source_landmarks, processed_landmarks):
        """
        Calculate a robust transformation matrix based on facial landmarks
        with better error handling for incomplete landmarks.
        
        Args:
            source_landmarks: List of (x, y) landmarks from the source image
            processed_landmarks: List of (x, y) landmarks from the processed image
                
        Returns:
            tuple: (transformation_matrix, success_flag)
        """
        import numpy as np
        import cv2
        
        # Validate input landmarks
        if not source_landmarks or not processed_landmarks:
            print("Error: Missing landmarks for alignment")
            return None, False
        
        try:
            # Convert landmarks to numpy arrays if they aren't already
            source_points = np.array(source_landmarks)
            processed_points = np.array(processed_landmarks)
            
            # Check if we have enough landmarks
            if len(source_points) < 5 or len(processed_points) < 5:
                print(f"Not enough landmarks for alignment: source={len(source_points)}, processed={len(processed_points)}")
                return None, False
            
            # Get the minimum number of landmarks available in both arrays
            min_landmarks = min(len(source_points), len(processed_points))
            
            # Define key landmark indices that are robust for alignment
            # Only use indices that are guaranteed to be available
            if min_landmarks >= 68:  # Full facial landmarks available
                key_indices = [
                    0, 8, 16,           # Jaw line (chin and sides)
                    19, 24,             # Eyebrows
                    27, 30, 33,         # Nose
                    36, 39, 42, 45,     # Eyes
                    48, 54              # Mouth
                ]
            elif min_landmarks >= 27:  # At least has eyebrows and nose
                # Use a subset of landmarks
                key_indices = [
                    0, 8, 16,           # Face outline points if available
                    19, 24,             # Eyebrows
                    27                  # Nose bridge top
                ]
            else:
                # Use all available landmarks for very limited sets
                key_indices = list(range(min_landmarks))
            
            # Extract the selected landmarks, making sure not to go out of bounds
            source_key_points = []
            processed_key_points = []
            
            for i in key_indices:
                if i < len(source_points) and i < len(processed_points):
                    source_key_points.append(source_points[i])
                    processed_key_points.append(processed_points[i])
            
            # Convert to numpy arrays for the transformation calculation
            source_key_points = np.array(source_key_points, dtype=np.float32)
            processed_key_points = np.array(processed_key_points, dtype=np.float32)
            
            # Make sure we have enough points for a transformation
            if len(source_key_points) < 3:
                print("Not enough matched key points for alignment")
                return None, False
            
            # Estimate an affine transformation that allows for rotation, scaling, and translation
            transformation_matrix, inliers = cv2.estimateAffinePartial2D(
                processed_key_points, source_key_points, 
                method=cv2.RANSAC, 
                ransacReprojThreshold=3.0,
                confidence=0.99,
                maxIters=2000
            )
            
            # Check if we got a valid transformation
            if transformation_matrix is None or inliers is None or np.sum(inliers) < 3:
                print("Warning: Could not estimate a good transformation matrix, falling back to simpler alignment")
                return None, False
            
            # Debug info about the transformation
            print(f"Estimated transformation matrix with {np.sum(inliers)} inliers out of {len(source_key_points)} points")
            
            # Decompose the matrix to understand the transformation better
            scale_x = np.sqrt(transformation_matrix[0, 0]**2 + transformation_matrix[0, 1]**2)
            scale_y = np.sqrt(transformation_matrix[1, 0]**2 + transformation_matrix[1, 1]**2)
            theta = np.arctan2(transformation_matrix[0, 1], transformation_matrix[0, 0]) * 180 / np.pi
            tx, ty = transformation_matrix[0, 2], transformation_matrix[1, 2]
            
            print(f"Translation: ({tx:.2f}, {ty:.2f}), Rotation: {theta:.2f}°, Scale: ({scale_x:.2f}, {scale_y:.2f})")
            
            # If the transformation seems too extreme, limit it
            max_scale = 1.5
            min_scale = 0.5
            max_rotation = 30.0  # degrees
            
            if (scale_x > max_scale or scale_y > max_scale or 
                scale_x < min_scale or scale_y < min_scale or 
                abs(theta) > max_rotation):
                
                print("Warning: Limiting extreme transformation values")
                
                # Limit scaling
                scale_x = np.clip(scale_x, min_scale, max_scale)
                scale_y = np.clip(scale_y, min_scale, max_scale)
                
                # Limit rotation
                theta = np.clip(theta, -max_rotation, max_rotation)
                theta_rad = theta * np.pi / 180.0
                
                # Reconstruct the rotation/scaling part of the matrix
                transformation_matrix[0, 0] = scale_x * np.cos(theta_rad)
                transformation_matrix[0, 1] = scale_x * np.sin(theta_rad)
                transformation_matrix[1, 0] = -scale_y * np.sin(theta_rad)
                transformation_matrix[1, 1] = scale_y * np.cos(theta_rad)
            
            return transformation_matrix, True
            
        except Exception as e:
            print(f"Error in landmark alignment: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, False

    def _apply_landmark_transform(self, source_img, processed_img, source_mask, processed_mask, 
                                source_landmarks, processed_landmarks, debug_dir=None):
        """
        Apply improved landmark-based alignment to align processed image and mask with source.
        
        Args:
            source_img: Original source image
            processed_img: Processed image to align
            source_mask: Mask for source image (or None)
            processed_mask: Mask for processed image
            source_landmarks: Pre-computed landmarks for source image
            processed_landmarks: Pre-computed landmarks for processed image
            debug_dir: Directory to save debug visualizations
            
        Returns:
            tuple: (aligned_mask, aligned_image)
        """
        import numpy as np
        import cv2
        import os
        
        h, w = source_img.shape[:2]
        
        # Get the transformation matrix using improved landmark alignment
        transform_matrix, success = self._improve_landmark_alignment(source_landmarks, processed_landmarks)
        
        if not success or transform_matrix is None:
            print("Could not calculate transformation matrix. Falling back to default alignment.")
            # Return the unmodified inputs as fallback
            return processed_mask, processed_img
        
        # Apply the transformation to the processed image and mask
        aligned_img = cv2.warpAffine(
            processed_img, transform_matrix, (w, h), 
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT
        )
        
        aligned_mask = cv2.warpAffine(
            processed_mask, transform_matrix, (w, h), 
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_TRANSPARENT
        )
        
        # Apply manual offsets if specified
        manual_offset_x = self.app.reinsert_manual_offset_x.get()
        manual_offset_y = self.app.reinsert_manual_offset_y.get()
        
        if manual_offset_x != 0 or manual_offset_y != 0:
            print(f"Applying manual offset: X={manual_offset_x}, Y={manual_offset_y}")
            offset_matrix = np.float32([[1, 0, manual_offset_x], [0, 1, manual_offset_y]])
            
            aligned_img = cv2.warpAffine(
                aligned_img, offset_matrix, (w, h),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT
            )
            
            aligned_mask = cv2.warpAffine(
                aligned_mask, offset_matrix, (w, h),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_TRANSPARENT
            )
        
        # Apply manual scaling if specified
        scale_x = self.app.reinsert_manual_scale_x.get()
        scale_y = self.app.reinsert_manual_scale_y.get()
        
        if scale_x != 1.0 or scale_y != 1.0:
            print(f"Applying manual scaling: X={scale_x}, Y={scale_y}")
            
            # Calculate center for scaling
            center_x = w // 2
            center_y = h // 2
            
            # Create transformation matrix for scaling around the center
            scale_matrix = np.float32([
                [scale_x, 0, center_x * (1 - scale_x)],
                [0, scale_y, center_y * (1 - scale_y)]
            ])
            
            aligned_img = cv2.warpAffine(
                aligned_img, scale_matrix, (w, h),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT
            )
            
            aligned_mask = cv2.warpAffine(
                aligned_mask, scale_matrix, (w, h),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_TRANSPARENT
            )
        
        # Apply manual rotation if specified
        rotation_angle = self.app.reinsert_manual_rotation.get()
        if rotation_angle != 0:
            print(f"Applying manual rotation: {rotation_angle} degrees")
            
            # Calculate center of rotation
            center_x = w // 2
            center_y = h // 2
            
            # Get rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), -rotation_angle, 1.0)
            
            # Apply rotation
            aligned_img = cv2.warpAffine(
                aligned_img, rotation_matrix, (w, h),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT
            )
            
            aligned_mask = cv2.warpAffine(
                aligned_mask, rotation_matrix, (w, h),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_TRANSPARENT
            )
        
        # Save debug visualization
        if debug_dir:
            # Draw landmarks on images
            source_vis = source_img.copy()
            processed_vis = processed_img.copy()
            aligned_vis = aligned_img.copy()
            
            # Draw source landmarks
            for i, (x, y) in enumerate(source_landmarks):
                cv2.circle(source_vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                if i in [0, 8, 16, 27, 30, 36, 39, 42, 45, 48, 54]:  # Key points
                    cv2.circle(source_vis, (int(x), int(y)), 4, (255, 0, 0), -1)
                    
                    
            # Draw processed landmarks
            for i, (x, y) in enumerate(processed_landmarks):
                cv2.circle(processed_vis, (int(x), int(y)), 2, (0, 0, 255), -1)
                if i in [0, 8, 16, 27, 30, 36, 39, 42, 45, 48, 54]:  # Key points
                    cv2.circle(processed_vis, (int(x), int(y)), 4, (255, 0, 0), -1)
            
            # Create visualization of the alignment
            combined = np.hstack((source_vis, processed_vis, aligned_vis))
            cv2.imwrite(os.path.join(debug_dir, "improved_landmark_alignment.png"), combined)
            
            # Save the aligned mask
            cv2.imwrite(os.path.join(debug_dir, "aligned_mask_improved_landmarks.png"), aligned_mask)
            
            # Overlay visualization showing source and aligned masks
            mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
            if source_mask is not None:
                mask_overlay[source_mask > 127] = [0, 0, 255]  # Source mask in red
            mask_overlay[aligned_mask > 127] = [0, 255, 0]    # Aligned mask in green
            cv2.imwrite(os.path.join(debug_dir, "mask_alignment_comparison.png"), mask_overlay)
        
        return aligned_mask, aligned_img



    def _apply_translation_only_transform(self, source_img, processed_img, source_mask, processed_mask, 
                                source_landmarks, processed_landmarks, debug_dir=None):
        """
        Apply landmark-based alignment but only use translation component (no rotation).
        Enhanced error handling for when landmarks are incomplete.
        
        Args:
            source_img: Original source image
            processed_img: Processed image to align
            source_mask: Mask for source image (or None)
            processed_mask: Mask for processed image
            source_landmarks: Pre-computed landmarks for source image
            processed_landmarks: Pre-computed landmarks for processed image
            debug_dir: Directory to save debug visualizations
            
        Returns:
            tuple: (aligned_mask, aligned_image)
        """
        import numpy as np
        import cv2
        import os
        
        h, w = source_img.shape[:2]
        
        # Make sure we have landmarks to work with
        if source_landmarks is None or processed_landmarks is None:
            print("Missing landmarks, using direct alignment")
            return processed_mask, processed_img
        
        # Calculate centers of face landmarks
        source_center = np.mean(np.array(source_landmarks), axis=0)
        processed_center = np.mean(np.array(processed_landmarks), axis=0)
        
        # Just use translation between centers
        tx = source_center[0] - processed_center[0]
        ty = source_center[1] - processed_center[1]
        
        print(f"Using translation only: tx={tx:.2f}, ty={ty:.2f}")
        
        # Create a translation-only matrix
        translation_matrix = np.float32([
            [1, 0, tx],
            [0, 1, ty]
        ])
        
        # Apply translation-only transformation
        aligned_img = cv2.warpAffine(
            processed_img, translation_matrix, (w, h), 
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT
        )
        
        aligned_mask = cv2.warpAffine(
            processed_mask, translation_matrix, (w, h), 
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
        )
        
        # Apply manual offsets if specified
        manual_offset_x = self.app.reinsert_manual_offset_x.get()
        manual_offset_y = self.app.reinsert_manual_offset_y.get()
        
        if manual_offset_x != 0 or manual_offset_y != 0:
            print(f"Applying manual offset: X={manual_offset_x}, Y={manual_offset_y}")
            offset_matrix = np.float32([[1, 0, manual_offset_x], [0, 1, manual_offset_y]])
            
            aligned_img = cv2.warpAffine(
                aligned_img, offset_matrix, (w, h),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT
            )
            
            aligned_mask = cv2.warpAffine(
                aligned_mask, offset_matrix, (w, h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
            )
        
        # Save debug visualization
        if debug_dir:
            # Draw landmarks on images
            source_vis = source_img.copy()
            processed_vis = processed_img.copy()
            aligned_vis = aligned_img.copy()
            
            # Draw landmarks
            for i, (x, y) in enumerate(source_landmarks):
                cv2.circle(source_vis, (int(x), int(y)), 2, (0, 255, 0), -1)
            
            for i, (x, y) in enumerate(processed_landmarks):
                cv2.circle(processed_vis, (int(x), int(y)), 2, (0, 0, 255), -1)
            
            # Create visualizations
            cv2.imwrite(os.path.join(debug_dir, "source_landmarks.png"), source_vis)
            cv2.imwrite(os.path.join(debug_dir, "processed_landmarks.png"), processed_vis)
            cv2.imwrite(os.path.join(debug_dir, "aligned_translation_only.png"), aligned_vis)
            
            # Save the aligned mask
            cv2.imwrite(os.path.join(debug_dir, "aligned_mask_translation_only.png"), aligned_mask)
        
        return aligned_mask, aligned_img

    def _align_with_landmarks(self, source_img, processed_img, source_mask, processed_mask, 
                                source_landmarks, processed_landmarks, debug_dir=None):
        """
        Align processed hair mask and image based on facial landmarks, using translation only.
        This preserves the original orientation while correctly positioning the hair.
        
        Args:
            source_img: Original source image
            processed_img: Processed image with hair to insert
            source_mask: Mask for source image
            processed_mask: Mask for processed image
            source_landmarks: Pre-computed landmarks for source image
            processed_landmarks: Pre-computed landmarks for processed image
            debug_dir: Directory to save debug visualizations
            
        Returns:
            tuple: (aligned_mask, aligned_image)
        """
        import numpy as np
        import cv2
        import os
        
        # Validate input
        if source_landmarks is None or processed_landmarks is None:
            print("Could not detect landmarks in one or both images")
            return processed_mask, processed_img
        
        # Check if we should use translation-only or full alignment
        use_translation_only = self.app.use_translation_only.get()
        
        if use_translation_only:
            # Use the translation-only method
            aligned_mask, aligned_img = self._apply_translation_only_transform(
                source_img, processed_img, source_mask, processed_mask,
                source_landmarks, processed_landmarks, debug_dir
            )
            
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "aligned_translation_only.png"), aligned_img)
        else:
            # Use the full alignment method for comparison
            aligned_mask, aligned_img = self._apply_landmark_transform(
                source_img, processed_img, source_mask, processed_mask,
                source_landmarks, processed_landmarks, debug_dir
            )
            
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "aligned_full_transform.png"), aligned_img)
        
        return aligned_mask, aligned_img

    def _get_landmarks(self, image):
        """
        Enhanced face detection with reliable fallback options.
        """
        if self.face_detector is None or self.landmark_predictor is None:
            print("Facial landmark detection not available")
            return None
            
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # First try with default settings
        faces = self.face_detector(gray)
        
        # If no faces detected, try with upsampling
        if not faces:
            print("No faces detected with default settings, trying with upsampling")
            faces = self.face_detector(gray, 1)  # Upsample 1 time
        
        # If still no faces, try adjusting contrast
        if not faces:
            print("Trying contrast enhancement for face detection")
            # Apply contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            faces = self.face_detector(enhanced_gray, 1)
        
        # If we found faces, process the largest one
        if faces:
            # Use largest face by area
            largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Get landmarks
            try:
                shape = self.landmark_predictor(gray, largest_face)
                landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
                return landmarks
            except Exception as e:
                print(f"Error detecting landmarks: {str(e)}")
                # Fall through to geometric fallback
        
        # If still no faces, use a geometric approach as fallback
        print("No faces detected, using geometric fallback method")
        h, w = image.shape[:2]
        
        # Create a full set of 68 estimated facial landmarks based on image geometry
        landmarks = []
        
        # Jaw line (points 0-16)
        for i in range(17):
            angle = (i / 16.0) * np.pi
            radius = min(w, h) * 0.45
            x = int(w/2 - np.cos(angle) * radius)
            y = int(h/2 + np.sin(angle) * radius)
            landmarks.append((x, y))
        
        # Eyebrows (points 17-26)
        eyebrow_y = int(h * 0.3)
        for i in range(5):  # Left eyebrow
            x = int(w * (0.2 + i * 0.06))
            y = eyebrow_y
            landmarks.append((x, y))
        
        for i in range(5):  # Right eyebrow
            x = int(w * (0.8 - i * 0.06))
            y = eyebrow_y
            landmarks.append((x, y))
        
        # Nose (points 27-35)
        nose_bridge_top_y = int(h * 0.35)
        nose_tip_y = int(h * 0.5)
        nose_width = int(w * 0.1)
        
        # Nose bridge (vertical line from between eyebrows to tip)
        for i in range(4):
            x = int(w / 2)
            y = int(nose_bridge_top_y + (nose_tip_y - nose_bridge_top_y) * (i / 3.0))
            landmarks.append((x, y))
        
        # Nose base (horizontal line below tip)
        for i in range(5):
            x = int(w/2 + nose_width * (-1 + i/2.0))
            y = nose_tip_y + int(h * 0.02)
            landmarks.append((x, y))
        
        # Eyes (points 36-47)
        eye_y = int(h * 0.38)
        left_eye_x = int(w * 0.3)
        right_eye_x = int(w * 0.7)
        eye_width = int(w * 0.07)
        
        # Left eye (6 points)
        for i in range(6):
            angle = (i / 6.0) * 2 * np.pi
            x = int(left_eye_x + np.cos(angle) * eye_width)
            y = int(eye_y + np.sin(angle) * (eye_width * 0.4))
            landmarks.append((x, y))
        
        # Right eye (6 points)
        for i in range(6):
            angle = (i / 6.0) * 2 * np.pi
            x = int(right_eye_x + np.cos(angle) * eye_width)
            y = int(eye_y + np.sin(angle) * (eye_width * 0.4))
            landmarks.append((x, y))
        
        # Outer lips (points 48-59)
        mouth_y = int(h * 0.7)
        mouth_width = int(w * 0.3)
        mouth_height = int(h * 0.06)
        
        for i in range(12):
            angle = (i / 12.0) * 2 * np.pi
            x = int(w/2 + np.cos(angle) * mouth_width/2)
            y = int(mouth_y + np.sin(angle) * mouth_height/2)
            landmarks.append((x, y))
        
        # Inner lips (points 60-67)
        inner_mouth_width = mouth_width * 0.7
        inner_mouth_height = mouth_height * 0.7
        
        for i in range(8):
            angle = (i / 8.0) * 2 * np.pi
            x = int(w/2 + np.cos(angle) * inner_mouth_width/2)
            y = int(mouth_y + np.sin(angle) * inner_mouth_height/2)
            landmarks.append((x, y))
        
        return landmarks
    
    def _isolate_bangs_region(self, mask, landmarks=None):
        """
        Isolate only the bangs portion of a hair mask.
        
        Args:
            mask: The full hair mask
            landmarks: Optional facial landmarks for better bangs detection
                
        Returns:
            numpy.ndarray: Mask containing only the bangs region
        """
        if mask is None:
            return None
                
        # Create empty mask for bangs
        bangs_mask = np.zeros_like(mask)
        height, width = mask.shape[:2]
        
        # Simply extract the top portion of the mask - bangs are typically in the top 15-25% of hair
        top_percent = 0.25  # Take top 25% of the mask
        bangs_bottom = int(height * top_percent)
        
        # Find non-zero points in the mask to determine where the hair is
        non_zero_points = np.argwhere(mask > 0)
        if len(non_zero_points) == 0:
            print("Warning: Empty mask, nothing to isolate for bangs")
            return bangs_mask
        
        # Find the top-most point of the hair
        top_y = np.min(non_zero_points[:, 0])
        
        # Extract the top portion as bangs
        bangs_region = mask[top_y:min(top_y + bangs_bottom, height), :]
        bangs_mask[top_y:min(top_y + bangs_bottom, height), :] = bangs_region
        
        # Add a gradient for smoother transition
        gradient_height = int(bangs_bottom * 0.3)  # Bottom 30% of bangs has gradient
        gradient_start = min(top_y + bangs_bottom - gradient_height, height)
        
        for y in range(gradient_start, min(top_y + bangs_bottom, height)):
            fade_ratio = 1.0 - ((y - gradient_start) / float(max(1, min(top_y + bangs_bottom, height) - gradient_start)))
            if y < bangs_mask.shape[0]:  # Safety check
                bangs_mask[y, :] = (bangs_mask[y, :].astype(float) * fade_ratio).astype(np.uint8)
        
        # Apply slight Gaussian blur for smoother edges
        bangs_mask = cv2.GaussianBlur(bangs_mask, (3, 3), 0)
        
        return bangs_mask

    def _detect_hair_parting(self, mask, landmarks=None):
        """
        Detect and enhance the hair parting line in the mask.
        
        Args:
            mask: Hair mask image
            landmarks: Optional facial landmarks for guidance
            
        Returns:
            tuple: (parting_line, enhanced_mask)
        """
        # Create copy of mask
        enhanced_mask = mask.copy()
        
        # Ensure mask is grayscale
        if len(mask.shape) > 2:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask
        
        # Threshold mask to binary
        _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to find potential parting
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary_mask, kernel, iterations=2)
        
        # Find difference (potential parting lines)
        potential_parting = binary_mask - eroded
        
        # Use connected components to find the parting line
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(potential_parting)
        
        # If we have landmarks, use them to help identify the correct parting
        parting_line = None
        if landmarks is not None:
            # Find the center line of the face
            middle_points = [(landmarks[27][0], landmarks[27][1]),  # Nose bridge top
                            (landmarks[30][0], landmarks[30][1])]  # Nose tip
            
            # Create a vertical line approximating face midline
            face_midline_x = int(np.mean([p[0] for p in middle_points]))
            
            # Find component closest to this midline
            best_dist = float('inf')
            best_component = 0
            
            for i in range(1, num_labels):  # Skip background (0)
                if stats[i, cv2.CC_STAT_AREA] < 20:  # Skip very small components
                    continue
                    
                # Calculate distance to face midline
                component_x = centroids[i][0]
                dist = abs(component_x - face_midline_x)
                
                if dist < best_dist:
                    best_dist = dist
                    best_component = i
            
            if best_component > 0:
                parting_mask = np.zeros_like(binary_mask)
                parting_mask[labels == best_component] = 255
                parting_line = parting_mask
        else:
            # Without landmarks, use the longest thin connected component in the upper part
            best_length = 0
            best_component = 0
            
            for i in range(1, num_labels):  # Skip background (0)
                # Calculate length/width ratio
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                
                # Skip components that are too small or too square
                if area < 20 or min(width, height) == 0:
                    continue
                
                length_ratio = max(width, height) / (min(width, height) + 1)
                
                # Check if component is in upper half
                top = stats[i, cv2.CC_STAT_TOP]
                if top < mask.shape[0] / 2 and length_ratio > 3 and length_ratio > best_length:
                    best_length = length_ratio
                    best_component = i
            
            if best_component > 0:
                parting_mask = np.zeros_like(binary_mask)
                parting_mask[labels == best_component] = 255
                parting_line = parting_mask
        
        # If we found a parting line, enhance it in the mask
        if parting_line is not None:
            # Dilate the parting slightly to make it more visible
            parting_enhanced = cv2.dilate(parting_line, kernel, iterations=1)
            
            # Create a blend of the original mask with a gap for the parting
            parting_factor = 0.7  # Strength of the parting (lower value = more visible parting)
            enhanced_mask = enhanced_mask * (1 - parting_enhanced/255 * (1-parting_factor))
            enhanced_mask = enhanced_mask.astype(np.uint8)
        
        return parting_line, enhanced_mask

    def _create_face_protection_mask(self, mask, landmarks, protection_strength):
        """
        Create a mask that protects facial features from being covered by bangs.
        
        Args:
            mask: The original bangs mask
            landmarks: Facial landmarks
            protection_strength: Strength of the face protection (0.1 to 1.0)
            
        Returns:
            numpy.ndarray: Modified mask with face protection
        """
        if landmarks is None or len(landmarks) < 68:
            return mask
            
        # Create an empty mask for face protection
        protection_mask = np.zeros_like(mask, dtype=np.float32)
        
        # Get key facial feature points
        # Eyes
        left_eye = np.array(landmarks[36:42])
        right_eye = np.array(landmarks[42:48])
        
        # Eyebrows
        left_eyebrow = np.array(landmarks[17:22])
        right_eyebrow = np.array(landmarks[22:27])
        
        # Nose bridge
        nose_bridge = np.array(landmarks[27:31])
        
        # Create protection zones
        def create_protection_zone(points, radius, strength):
            center = np.mean(points, axis=0).astype(np.int32)
            y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            
            # Create a gradual falloff
            falloff = np.clip(1 - (dist / radius), 0, 1)
            falloff = falloff * strength
            
            return falloff
        
        # Add protection zones for each facial feature
        # Stronger protection for eyes
        protection_mask = np.maximum(protection_mask, 
                                   create_protection_zone(left_eye, radius=30, 
                                                       strength=protection_strength))
        protection_mask = np.maximum(protection_mask, 
                                   create_protection_zone(right_eye, radius=30, 
                                                       strength=protection_strength))
        
        # Medium protection for eyebrows
        protection_mask = np.maximum(protection_mask, 
                                   create_protection_zone(left_eyebrow, radius=25, 
                                                       strength=protection_strength * 0.8))
        protection_mask = np.maximum(protection_mask, 
                                   create_protection_zone(right_eyebrow, radius=25, 
                                                       strength=protection_strength * 0.8))
        
        # Light protection for nose bridge
        protection_mask = np.maximum(protection_mask, 
                                   create_protection_zone(nose_bridge, radius=20, 
                                                       strength=protection_strength * 0.6))
        
        # Smooth the protection mask
        protection_mask = cv2.GaussianBlur(protection_mask, (15, 15), 0)
        
        # Invert and apply the protection mask to the original mask
        protected_mask = mask.astype(np.float32) * (1 - protection_mask)
        
        # Convert back to uint8
        protected_mask = np.clip(protected_mask, 0, 255).astype(np.uint8)
        
        return protected_mask

    def _estimate_bangs_position(self, landmarks, height, width):
        """
        Estimates where bangs should be positioned based on facial landmarks.
        
        Args:
            landmarks: Facial landmarks from the source image
            height: Height of the image
            width: Width of the image
            
        Returns:
            dict: Contains estimated position information
        """
        if landmarks is None or len(landmarks) < 17:
            # If no landmarks, use geometric center and top 20%
            return {
                'center_x': width // 2,
                'top_y': int(height * 0.2),
                'width': int(width * 0.4),
                'height': int(height * 0.15)
            }
        
        try:
            # Get eyebrow positions
            if len(landmarks) >= 27:
                eyebrow_points = landmarks[17:27]  # Standard eyebrow landmarks
                eyebrow_y = min([p[1] for p in eyebrow_points])
                left_x = min([p[0] for p in eyebrow_points])
                right_x = max([p[0] for p in eyebrow_points])
            else:
                # Fallback if eyebrow landmarks aren't available
                # Use top of face
                top_points = [landmarks[i] for i in range(min(8, len(landmarks)))]
                eyebrow_y = min([p[1] for p in top_points]) + int(height * 0.1)
                left_x = min([p[0] for p in top_points])
                right_x = max([p[0] for p in top_points]) 
            
            # Calculate bangs parameters
            center_x = (left_x + right_x) // 2
            bangs_width = int((right_x - left_x) * 1.2)  # Make slightly wider than eyebrows
            top_y = max(0, eyebrow_y - int(height * 0.15))  # Place above eyebrows
            bangs_height = eyebrow_y - top_y
            
            return {
                'center_x': center_x,
                'top_y': top_y,
                'width': bangs_width,
                'height': bangs_height,
                'eyebrow_y': eyebrow_y,
                'left_x': left_x,
                'right_x': right_x
            }
        
        except Exception as e:
            print(f"Error estimating bangs position: {str(e)}")
            # Fallback to simple geometric positioning
            return {
                'center_x': width // 2,
                'top_y': int(height * 0.2),
                'width': int(width * 0.4),
                'height': int(height * 0.15)
            }