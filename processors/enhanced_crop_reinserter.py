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
        
        # Check if "bangs only" mode is enabled
        if self.app.use_bangs_only.get():
            print("Using bangs-only mode")
            # Get landmarks if available and face detector exists
            landmarks = None
            if hasattr(self, 'face_detector') and self.face_detector is not None:
                # Try to detect landmarks in the source image first (most reliable)
                landmarks = self._get_landmarks(source_img)
                if landmarks is None:
                    print("No faces detected in source image, trying processed image...")
                    # If that fails, try the processed image
                    landmarks = self._get_landmarks(processed_img)
                
                if landmarks is None:
                    print("No faces detected in either image, using geometric approach for bangs")
                else:
                    print(f"Face detected, using landmark-based approach for bangs")
            
            # Isolate just the bangs region
            print(f"Creating bangs-only mask with width ratio: {self.app.bangs_width_ratio.get()}, extension: {self.app.bangs_extension_amount.get()}")
            bangs_mask = self._isolate_bangs_region(mask, landmarks)
            
            # Replace the full mask with just the bangs
            mask = bangs_mask
            
            # Save debug image
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "bangs_only_mask.png"), mask)

        # Apply bangs extension if enabled (add after loading the mask but before any other mask processing)
        # Only apply if not in bangs-only mode (which already includes extension)
        if self.app.extend_bangs.get() and mask is not None and not self.app.use_bangs_only.get():
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
        
        # Special handling for bangs-only mode
        if self.app.use_bangs_only.get():
            # Get configuration settings
            blend_mode = self.app.reinsert_blend_mode.get()
            blend_extent = self.app.reinsert_blend_extent.get()
            preserve_edges = self.app.reinsert_preserve_edges.get()
            
            # First apply manual offset if needed
            manual_offset_x = self.app.reinsert_manual_offset_x.get()
            manual_offset_y = self.app.reinsert_manual_offset_y.get()
            
            if manual_offset_x != 0 or manual_offset_y != 0:
                print(f"Applying manual offset to bangs: X={manual_offset_x}, Y={manual_offset_y}")
                M = np.float32([[1, 0, manual_offset_x], [0, 1, manual_offset_y]])
                mask_resized = cv2.warpAffine(mask_resized, M, (mask_resized.shape[1], mask_resized.shape[0]))
                processed_img_resized = cv2.warpAffine(processed_img_resized, M, 
                                            (processed_img_resized.shape[1], processed_img_resized.shape[0]))
                
                if debug_dir:
                    cv2.imwrite(os.path.join(debug_dir, "bangs_offset_mask.png"), mask_resized)
            
            # Then apply manual scaling if needed
            scale_x = self.app.reinsert_manual_scale_x.get()
            scale_y = self.app.reinsert_manual_scale_y.get()
            
            if scale_x != 1.0 or scale_y != 1.0:
                print(f"Applying manual scaling to bangs: X={scale_x}, Y={scale_y}")
                h, w = processed_img_resized.shape[:2]
                center_x, center_y = w // 2, h // 2
                M = np.float32([
                    [scale_x, 0, center_x * (1 - scale_x)],
                    [0, scale_y, center_y * (1 - scale_y)]
                ])
                
                mask_resized = cv2.warpAffine(mask_resized, M, (w, h), flags=cv2.INTER_NEAREST)
                processed_img_resized = cv2.warpAffine(processed_img_resized, M, (w, h), flags=cv2.INTER_LANCZOS4)
                
                if debug_dir:
                    cv2.imwrite(os.path.join(debug_dir, "bangs_scaled_mask.png"), mask_resized)
            
            # Use our special bangs blending function that respects all settings
            result_img = self._blend_bangs_only(
                source_img, 
                processed_img_resized, 
                mask_resized,
                method=blend_mode,
                blend_extent=blend_extent,
                preserve_edges=preserve_edges,
                debug_dir=debug_dir
            )
            
            # Save the result
            cv2.imwrite(output_path, result_img)
            return True
        
        # Standard processing for non-bangs-only mode
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

    def _align_masks(self, source_mask, target_mask, source_img, target_img, method="centroid", debug_dir=None):
        """
        Align two masks using different methods.
        
        Args:
            source_mask: Source hair mask
            target_mask: Target hair mask
            source_img: Source image 
            target_img: Target image
            method: Alignment method ('none', 'centroid', 'bbox', 'landmarks', 'contour', or 'iou')
            debug_dir: Directory to save debug visualizations
            
        Returns:
            tuple: (aligned_mask, aligned_image)
        """
        if method == "none" or source_mask is None or target_mask is None:
            return target_mask, target_img
            
        print(f"Using {method} alignment method")
        
        # Convert masks to binary if needed
        if len(source_mask.shape) > 2:
            source_mask = cv2.cvtColor(source_mask, cv2.COLOR_BGR2GRAY)
        if len(target_mask.shape) > 2:
            target_mask = cv2.cvtColor(target_mask, cv2.COLOR_BGR2GRAY)
            
        # Thresholds for binary masks
        _, source_mask_bin = cv2.threshold(source_mask, 127, 255, cv2.THRESH_BINARY)
        _, target_mask_bin = cv2.threshold(target_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Save initial masks for debugging
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "source_mask_before_alignment.png"), source_mask)
            cv2.imwrite(os.path.join(debug_dir, "target_mask_before_alignment.png"), target_mask)
        
        # Initialize transformation matrices
        h, w = target_mask.shape[:2]
        M = np.float32([[1, 0, 0], [0, 1, 0]])  # Identity transform
        rotation_matrix = None
        
        # Get manual offset values from UI controls
        manual_offset_x = self.app.reinsert_manual_offset_x.get()
        manual_offset_y = self.app.reinsert_manual_offset_y.get()
        
        # Get manual scale values from UI controls
        manual_scale_x = self.app.reinsert_manual_scale_x.get()
        manual_scale_y = self.app.reinsert_manual_scale_y.get()
        
        # Get manual rotation value from UI controls
        manual_rotation = self.app.reinsert_manual_rotation.get() if hasattr(self.app, 'reinsert_manual_rotation') else 0.0
        
        # Alignment methods
        if method == "centroid":
            # Find centroids of both masks
            source_moments = cv2.moments(source_mask_bin)
            target_moments = cv2.moments(target_mask_bin)
            
            if source_moments["m00"] != 0 and target_moments["m00"] != 0:
                source_cx = int(source_moments["m10"] / source_moments["m00"])
                source_cy = int(source_moments["m01"] / source_moments["m00"])
                
                target_cx = int(target_moments["m10"] / target_moments["m00"])
                target_cy = int(target_moments["m01"] / target_moments["m00"])
                
                # Calculate offset to align centroids
                dx = source_cx - target_cx
                dy = source_cy - target_cy
                
                # Create transformation matrix for translation
                M = np.float32([[1, 0, dx], [0, 1, dy]])
            
        elif method == "bbox":
            # Find bounding boxes of both masks
            source_cnts, _ = cv2.findContours(source_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            target_cnts, _ = cv2.findContours(target_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if source_cnts and target_cnts:
                source_cnt = max(source_cnts, key=cv2.contourArea)
                target_cnt = max(target_cnts, key=cv2.contourArea)
                
                source_x, source_y, source_w, source_h = cv2.boundingRect(source_cnt)
                target_x, target_y, target_w, target_h = cv2.boundingRect(target_cnt)
                
                # Calculate center points of bounding boxes
                source_center = (source_x + source_w // 2, source_y + source_h // 2)
                target_center = (target_x + target_w // 2, target_y + target_h // 2)
                
                # Calculate offset to align centers
                dx = source_center[0] - target_center[0]
                dy = source_center[1] - target_center[1]
                
                # Calculate scale factors
                scale_x = source_w / max(1, target_w)
                scale_y = source_h / max(1, target_h)
                
                # Create transformation matrix for scaling and translation
                # First scale, then translate
                center_x, center_y = w // 2, h // 2
                scale_M = np.float32([
                    [scale_x, 0, center_x * (1 - scale_x)],
                    [0, scale_y, center_y * (1 - scale_y)]
                ])
                trans_M = np.float32([[1, 0, dx], [0, 1, dy]])
                
                # Apply scaling first
                target_mask = cv2.warpAffine(target_mask, scale_M, (w, h))
                target_img = cv2.warpAffine(target_img, scale_M, (w, h))
                
                # Then set translation matrix
                M = trans_M
                
        elif method == "landmarks":
            # Use facial landmarks for alignment (if available)
            if self.face_detector is not None and self.landmark_predictor is not None:
                # Get landmarks for both images
                source_landmarks = self._get_landmarks(source_img)
                target_landmarks = self._get_landmarks(target_img)
                
                if source_landmarks is not None and target_landmarks is not None:
                    # Calculate automatic scale and rotation
                    auto_scale_x, auto_scale_y, auto_rotation = self._calculate_face_scale_and_rotation(
                        source_landmarks, target_landmarks
                    )
                    
                    # Override manual controls with automatic values if translation-only mode is not enabled
                    if not hasattr(self.app, 'use_translation_only') or not self.app.use_translation_only.get():
                        # Use automatic scale and rotation with manual adjustments as fine-tuning
                        scale_x = auto_scale_x * manual_scale_x
                        scale_y = auto_scale_y * manual_scale_y
                        rotation = auto_rotation + manual_rotation
                    else:
                        # Use just manual values
                        scale_x = manual_scale_x
                        scale_y = manual_scale_y
                        rotation = manual_rotation
                    
                    # Find face centers
                    src_face_center = np.mean(source_landmarks, axis=0).astype(int)
                    target_face_center = np.mean(target_landmarks, axis=0).astype(int)
                    
                    # Calculate offset
                    dx = src_face_center[0] - target_face_center[0]
                    dy = src_face_center[1] - target_face_center[1]
                    
                    # Create a combined transformation matrix:
                    # 1. First translate to origin 
                    # 2. Then rotate
                    # 3. Then scale
                    # 4. Then translate back
                    # 5. Finally, add the offset

                    # Define center for rotation and scaling
                    center = (w // 2, h // 2)
                    
                    # Create rotation matrix
                    rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
                    
                    # Apply rotation to the mask and image
                    if abs(rotation) > 0.5:  # Only apply if rotation is significant
                        target_mask = cv2.warpAffine(target_mask, rotation_matrix, (w, h))
                        target_img = cv2.warpAffine(target_img, rotation_matrix, (w, h))
                    
                    # Create scaling matrix
                    scale_M = np.float32([
                        [scale_x, 0, center[0] * (1 - scale_x)],
                        [0, scale_y, center[1] * (1 - scale_y)]
                    ])
                    
                    # Apply scaling
                    if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:  # Only apply if scale is significant
                        target_mask = cv2.warpAffine(target_mask, scale_M, (w, h))
                        target_img = cv2.warpAffine(target_img, scale_M, (w, h))
                    
                    # Create final translation matrix
                    M = np.float32([[1, 0, dx + manual_offset_x], [0, 1, dy + manual_offset_y]])
                else:
                    # Fallback to centroid
                    print("Landmarks not detected, falling back to centroid alignment")
                    source_moments = cv2.moments(source_mask_bin)
                    target_moments = cv2.moments(target_mask_bin)
                    
                    if source_moments["m00"] != 0 and target_moments["m00"] != 0:
                        source_cx = int(source_moments["m10"] / source_moments["m00"])
                        source_cy = int(source_moments["m01"] / source_moments["m00"])
                        
                        target_cx = int(target_moments["m10"] / target_moments["m00"])
                        target_cy = int(target_moments["m01"] / target_moments["m00"])
                        
                        # Calculate offset to align centroids
                        dx = source_cx - target_cx + manual_offset_x
                        dy = source_cy - target_cy + manual_offset_y
                        
                        # Create transformation matrix for translation
                        M = np.float32([[1, 0, dx], [0, 1, dy]])
            else:
                print("Facial landmark detection not available - falling back to centroid alignment")
                # Fallback to centroid with manual controls
                source_moments = cv2.moments(source_mask_bin)
                target_moments = cv2.moments(target_mask_bin)
                
                if source_moments["m00"] != 0 and target_moments["m00"] != 0:
                    source_cx = int(source_moments["m10"] / source_moments["m00"])
                    source_cy = int(source_moments["m01"] / source_moments["m00"])
                    
                    target_cx = int(target_moments["m10"] / target_moments["m00"])
                    target_cy = int(target_moments["m01"] / target_moments["m00"])
                    
                    # Calculate offset to align centroids
                    dx = source_cx - target_cx + manual_offset_x
                    dy = source_cy - target_cy + manual_offset_y
                    
                    # Create transformation matrix for translation
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    
        elif method == "contour":
            # Use contour matching for alignment
            source_cnts, _ = cv2.findContours(source_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            target_cnts, _ = cv2.findContours(target_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if source_cnts and target_cnts:
                source_cnt = max(source_cnts, key=cv2.contourArea)
                target_cnt = max(target_cnts, key=cv2.contourArea)
                
                # Calculate centroids
                source_M = cv2.moments(source_cnt)
                target_M = cv2.moments(target_cnt)
                
                if source_M["m00"] != 0 and target_M["m00"] != 0:
                    source_cx = int(source_M["m10"] / source_M["m00"])
                    source_cy = int(source_M["m01"] / source_M["m00"])
                    
                    target_cx = int(target_M["m10"] / target_M["m00"])
                    target_cy = int(target_M["m01"] / target_M["m00"])
                    
                    # Calculate offset to align centroids
                    dx = source_cx - target_cx
                    dy = source_cy - target_cy
                    
                    # Create transformation matrix for translation
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    
                    # Also check size differences for scaling
                    source_area = cv2.contourArea(source_cnt)
                    target_area = cv2.contourArea(target_cnt)
                    
                    if target_area > 0:
                        scale_factor = np.sqrt(source_area / target_area)
                        scale_factor = min(max(scale_factor, 0.5), 2.0)  # Limit scaling
                        
                        # Create scaling matrix
                        center = (w // 2, h // 2)
                        scale_M = np.float32([
                            [scale_factor, 0, center[0] * (1 - scale_factor)],
                            [0, scale_factor, center[1] * (1 - scale_factor)]
                        ])
                        
                        # Apply scaling
                        target_mask = cv2.warpAffine(target_mask, scale_M, (w, h))
                        target_img = cv2.warpAffine(target_img, scale_M, (w, h))
        
        elif method == "iou":
            # Find the translation that maximizes IoU (Intersection over Union)
            best_iou = 0
            best_dx, best_dy = 0, 0
            
            # Simple grid search for optimal translation
            search_range = 40
            step = 5
            
            for dx in range(-search_range, search_range + 1, step):
                for dy in range(-search_range, search_range + 1, step):
                    # Create transformation matrix
                    test_M = np.float32([[1, 0, dx], [0, 1, dy]])
                    
                    # Apply transformation to target mask
                    shifted_mask = cv2.warpAffine(target_mask_bin, test_M, (w, h))
                    
                    # Calculate IoU
                    intersection = cv2.bitwise_and(source_mask_bin, shifted_mask)
                    union = cv2.bitwise_or(source_mask_bin, shifted_mask)
                    
                    # Count non-zero pixels
                    intersection_count = cv2.countNonZero(intersection)
                    union_count = cv2.countNonZero(union)
                    
                    if union_count > 0:
                        iou = intersection_count / union_count
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_dx = dx
                            best_dy = dy
            
            # Use the best translation
            M = np.float32([[1, 0, best_dx], [0, 1, best_dy]])
            
        # Apply final transformation to target mask and image
        aligned_mask = cv2.warpAffine(target_mask, M, (w, h))
        aligned_img = cv2.warpAffine(target_img, M, (w, h))
        
        # Save aligned masks for debugging
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "target_mask_after_alignment.png"), aligned_mask)
            cv2.imwrite(os.path.join(debug_dir, "target_img_after_alignment.png"), aligned_img)
            
            # Create visualization of alignment
            vis_img = source_img.copy()
            # Colorize masks for visualization
            color_source = cv2.cvtColor(source_mask, cv2.COLOR_GRAY2BGR)
            color_source[:,:,0] = 0  # Remove blue channel
            color_source[:,:,2] = 0  # Remove red channel
            
            color_aligned = cv2.cvtColor(aligned_mask, cv2.COLOR_GRAY2BGR)
            color_aligned[:,:,0] = 0  # Remove blue channel
            color_aligned[:,:,1] = 0  # Remove green channel
            
            # Overlay masks
            alpha = 0.5
            vis_img = cv2.addWeighted(vis_img, 1.0, color_source, alpha, 0)
            vis_img = cv2.addWeighted(vis_img, 1.0, color_aligned, alpha, 0)
            
            cv2.imwrite(os.path.join(debug_dir, "alignment_visualization.png"), vis_img)
        
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
            
            # Make sure mask has enough non-zero pixels to work with
            if cv2.countNonZero(mask_uint8) < 100:
                raise ValueError("Mask too small for Poisson blending")
            
            # Find center of mask
            moments = cv2.moments(mask_uint8)
            if moments["m00"] <= 0:
                raise ValueError("Invalid mask moments")
            
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            
            # Make sure center is within bounds and away from edges
            h, w = source_img.shape[:2]
            margin = 50  # Minimum distance from edges
            
            if center_x < margin or center_x >= w - margin or center_y < margin or center_y >= h - margin:
                print(f"Center point ({center_x}, {center_y}) too close to image boundaries")
                raise ValueError("Center point too close to image boundaries")
            
            center = (center_x, center_y)
            
            # Special handling for small masks or sparse masks
            nonzero_pixels = cv2.countNonZero(mask_uint8)
            total_pixels = mask_uint8.shape[0] * mask_uint8.shape[1]
            
            if nonzero_pixels < 1000 or nonzero_pixels / total_pixels < 0.01:
                print(f"Small mask detected ({nonzero_pixels} px), enhancing for Poisson blend")
                # For small masks, ensure they're dense enough by dilating
                kernel = np.ones((3, 3), np.uint8)
                mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
                # Increase contrast to ensure good blending
                mask_uint8 = cv2.normalize(mask_uint8, None, 100, 255, cv2.NORM_MINMAX)
            
            # Make extra sure the mask is properly filled
            mask_copy = mask_uint8.copy()
            
            # Check if processed and source images are the same size
            if source_img.shape != processed_img.shape:
                print("Source and processed images are different sizes")
                raise ValueError("Source and processed images must be the same size")
            
            # Make sure the mask doesn't extend beyond the image boundaries
            combined_mask = np.zeros((h+100, w+100), dtype=np.uint8)
            combined_mask[50:50+h, 50:50+w] = mask_copy
            
            # Shift center
            adjusted_center = (center_x + 50, center_y + 50)
            
            # Create padded versions of images
            padded_source = np.zeros((h+100, w+100, 3), dtype=np.uint8)
            padded_processed = np.zeros((h+100, w+100, 3), dtype=np.uint8)
            
            padded_source[50:50+h, 50:50+w] = source_img
            padded_processed[50:50+h, 50:50+w] = processed_img
            
            # Apply seamless cloning with mixed mode for better color preservation
            result_padded = cv2.seamlessClone(
                padded_processed, 
                padded_source, 
                combined_mask, 
                adjusted_center, 
                cv2.NORMAL_CLONE
            )
            
            # Extract the original region
            result_img = result_padded[50:50+h, 50:50+w]
            
            return np.clip(result_img, 0, 255).astype(np.uint8)
            
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
        
        # Ensure extend_pixels is an integer
        extend_pixels = int(extend_pixels)
        
        # Extend the mask downward in the forehead region
        for x in range(forehead_left, forehead_right):
            if x in top_y_values:
                # Get the topmost y for this column
                top_y = int(top_y_values[x])
                
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
    

    def _improve_landmark_alignment(self, source_landmarks, processed_landmarks):
        """
        Calculate a more robust transformation matrix based on facial landmarks
        to handle face rotation and position changes.
        
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
            
            # Select key landmark points that are robust for alignment
            # These points cover the face contour, eyes, nose, and mouth
            key_indices = [
                0, 8, 16,           # Jaw line (chin and sides)
                19, 24,             # Eyebrows
                27, 30, 33,         # Nose
                36, 39, 42, 45,     # Eyes
                48, 54              # Mouth
            ]
            
            # Extract the selected landmarks
            source_key_points = np.array([source_landmarks[i] for i in key_indices], dtype=np.float32)
            processed_key_points = np.array([processed_landmarks[i] for i in key_indices], dtype=np.float32)
            
            # Estimate an affine transformation that allows for rotation, scaling, and translation
            # Use RANSAC for robustness against outliers
            transformation_matrix, inliers = cv2.estimateAffinePartial2D(
                processed_key_points, source_key_points, 
                method=cv2.RANSAC, 
                ransacReprojThreshold=3.0,  # Maximum allowed reprojection error
                confidence=0.99,            # Confidence level
                maxIters=2000               # Maximum iterations
            )
            
            # Check if we got a valid transformation
            if transformation_matrix is None or inliers is None or np.sum(inliers) < 4:
                print("Warning: Could not estimate a good transformation matrix, falling back to simpler alignment")
                return None, False
            
            # Debug info about the transformation
            print(f"Estimated transformation matrix with {np.sum(inliers)} inliers out of {len(key_indices)} points")
            print(f"Transformation matrix:\n{transformation_matrix}")
            
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
        This preserves the original orientation of the processed image while still positioning it correctly.
        
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
        
        # Step 1: Calculate the full transformation matrix (for reference points)
        transform_matrix, success = self._improve_landmark_alignment(source_landmarks, processed_landmarks)
        
        if not success or transform_matrix is None:
            print("Could not calculate transformation matrix. Using simpler alignment.")
            # Calculate centers of face landmarks as fallback
            source_center = np.mean(np.array(source_landmarks), axis=0)
            processed_center = np.mean(np.array(processed_landmarks), axis=0)
            
            # Just use translation between centers
            tx = source_center[0] - processed_center[0]
            ty = source_center[1] - processed_center[1]
            
            # Create a translation-only matrix
            translation_matrix = np.float32([
                [1, 0, tx],
                [0, 1, ty]
            ])
        else:
            # Step 2: Extract only the translation component from the transformation
            # The translation components are in the last column of the matrix
            tx, ty = transform_matrix[0, 2], transform_matrix[1, 2]
            
            print(f"Full transformation matrix:\n{transform_matrix}")
            print(f"Using only translation components: tx={tx:.2f}, ty={ty:.2f}")
            
            # Create a translation-only matrix
            translation_matrix = np.float32([
                [1, 0, tx],
                [0, 1, ty]
            ])
        
        # Step 3: Apply translation-only transformation
        aligned_img = cv2.warpAffine(
            processed_img, translation_matrix, (w, h), 
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT
        )
        
        aligned_mask = cv2.warpAffine(
            processed_mask, translation_matrix, (w, h), 
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
            translation_only_vis = aligned_img.copy()
            
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
            
            # Draw source landmarks on the aligned image for comparison
            for i, (x, y) in enumerate(source_landmarks):
                if i in [0, 8, 16, 27, 30, 36, 39, 42, 45, 48, 54]:  # Key points only
                    cv2.circle(aligned_vis, (int(x), int(y)), 4, (255, 255, 0), -1)
            
            # Create comparison visualization
            full_comparison = np.hstack((source_vis, processed_vis, aligned_vis))
            cv2.imwrite(os.path.join(debug_dir, "translation_only_alignment.png"), full_comparison)
            
            # Save the aligned mask
            cv2.imwrite(os.path.join(debug_dir, "aligned_mask_translation_only.png"), aligned_mask)
            
            # Overlay visualization showing source and aligned masks
            mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
            if source_mask is not None:
                mask_overlay[source_mask > 127] = [0, 0, 255]  # Source mask in red
            mask_overlay[aligned_mask > 127] = [0, 255, 0]    # Aligned mask in green
            cv2.imwrite(os.path.join(debug_dir, "mask_alignment_translation_only.png"), mask_overlay)
        
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
        Get facial landmarks from an image.
        
        Args:
            image: Input image
            
        Returns:
            list: List of (x, y) landmarks or None if no face detected
        """
        if image is None or self.face_detector is None or self.landmark_predictor is None:
            return None
            
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # For smaller images, we shouldn't upsample as much to avoid false positives
            h, w = image.shape[:2]
            
            # Adaptive upsampling based on image size
            if max(h, w) < 300:
                # For very small images, use more upsampling to catch small faces
                faces = self.face_detector(gray, 2)
            elif max(h, w) < 600:
                # For medium-sized images, use standard upsampling
                faces = self.face_detector(gray, 1)
            else:
                # For large images, don't upsample
                faces = self.face_detector(gray, 0)
            
            if len(faces) == 0:
                print("No faces detected in image")
                return None
                
            # Get the largest face by area
            largest_face = faces[0]
            largest_area = (faces[0].right() - faces[0].left()) * (faces[0].bottom() - faces[0].top())
            
            for face in faces[1:]:
                area = (face.right() - face.left()) * (face.bottom() - face.top())
                if area > largest_area:
                    largest_face = face
                    largest_area = area
            
            # Get landmarks
            shape = self.landmark_predictor(gray, largest_face)
            landmarks = []
            
            for i in range(68):  # 68 landmarks
                x = shape.part(i).x
                y = shape.part(i).y
                landmarks.append((x, y))
                
            print(f"Detected face with {len(landmarks)} landmarks")
            return landmarks
            
        except Exception as e:
            print(f"Error detecting landmarks: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
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
        
        # Get configuration values from app settings
        width_ratio = self.app.bangs_width_ratio.get()
        extension_amount = self.app.bangs_extension_amount.get()
        min_opacity = self.app.bangs_min_opacity.get()
        
        # Method 1: Using landmarks if available
        if landmarks is not None:
            try:
                # Get eyebrow height (use top of eyebrows)
                eyebrow_points = landmarks[17:27]  # Eyebrow landmarks
                eyebrow_y = min([p[1] for p in eyebrow_points])
                
                # Get face width from landmarks
                left_temple = landmarks[0][0]  # Leftmost face point
                right_temple = landmarks[16][0]  # Rightmost face point
                face_width = right_temple - left_temple
                
                # Define forehead region above eyebrows, using width_ratio from config
                # Increase the width ratio slightly to ensure we capture enough of the bangs
                effective_width_ratio = max(width_ratio * 1.5, 0.3)  # Ensure it's at least 30% of face width
                forehead_width = int(face_width * effective_width_ratio)
                forehead_center = (left_temple + right_temple) // 2
                forehead_left = max(0, int(forehead_center - forehead_width // 2))
                forehead_right = min(width, int(forehead_center + forehead_width // 2))
                
                # Use extension_amount for vertical area calculation
                # Start higher above eyebrows and extend further down
                forehead_top = max(0, int(eyebrow_y - extension_amount * 1.5))
                forehead_bottom = min(height, int(eyebrow_y + extension_amount * 0.5))  # Extend more below eyebrows
                
                # Create region for forehead/bangs area
                bangs_region = np.zeros_like(mask)
                bangs_region[forehead_top:forehead_bottom, forehead_left:forehead_right] = 255
                
                # Intersect with the original mask to get only the bangs
                bangs_mask = cv2.bitwise_and(mask, bangs_region)
                
                # Check if we have enough non-zero pixels in the mask
                if cv2.countNonZero(bangs_mask) < 100:
                    print("Warning: Landmark-based approach produced too small a mask, trying to expand")
                    # Dilate the mask to ensure we have enough content
                    kernel = np.ones((5, 5), np.uint8)
                    bangs_mask = cv2.dilate(bangs_mask, kernel, iterations=2)
                    
                    # If still too small, use the geometric approach as fallback
                    if cv2.countNonZero(bangs_mask) < 100:
                        print("Falling back to geometric approach after landmark approach failed")
                        return self._isolate_bangs_region_geometric(mask, extension_amount, width_ratio, min_opacity)
                
                # Apply a feathering at the bottom for a more natural transition
                fade_height = int((forehead_bottom - forehead_top) * 0.4)
                fade_start = forehead_bottom - fade_height
                
                for y in range(fade_start, forehead_bottom):
                    fade_factor = 1.0 - ((y - fade_start) / float(max(1, fade_height)))
                    # Apply minimum opacity from config
                    adjusted_factor = min_opacity + (fade_factor * (1.0 - min_opacity))
                    bangs_mask_row = bangs_mask[y, :].astype(float) * adjusted_factor
                    bangs_mask[y, :] = bangs_mask_row.astype(np.uint8)
                
                # Apply a small blur to smooth the mask edges
                bangs_mask = cv2.GaussianBlur(bangs_mask, (3, 3), 0)
                
                # Force mask to have enough contrast to be visible
                if cv2.countNonZero(bangs_mask) > 0:
                    bangs_mask = cv2.normalize(bangs_mask, None, 100, 255, cv2.NORM_MINMAX)
                
                return bangs_mask
                
            except (IndexError, ValueError, TypeError) as e:
                print(f"Error using landmarks for bangs detection: {str(e)}")
                # Fall back to geometric method
                return self._isolate_bangs_region_geometric(mask, extension_amount, width_ratio, min_opacity)
        
        # If no landmarks, use geometric approach
        return self._isolate_bangs_region_geometric(mask, extension_amount, width_ratio, min_opacity)

    def _isolate_bangs_region_geometric(self, mask, extension_amount, width_ratio, min_opacity):
        """
        Isolate bangs using a geometric approach when landmarks are not available.
        This is extracted as a separate method for clarity and reuse.
        """
        height, width = mask.shape[:2]
        bangs_mask = np.zeros_like(mask)
        
        print("Using geometric approach for bangs detection (no facial landmarks)")
        
        # Find points in the mask
        mask_points = np.argwhere(mask > 127)
        if len(mask_points) == 0:
            return bangs_mask  # Empty mask, return empty
            
        # Find the topmost part of the mask
        top_y = np.min(mask_points[:, 0]) if len(mask_points) > 0 else 0
        
        # Find the vertical and horizontal extents of the mask
        bottom_y = np.max(mask_points[:, 0]) if len(mask_points) > 0 else height
        left_x = np.min(mask_points[:, 1]) if len(mask_points) > 0 else 0
        right_x = np.max(mask_points[:, 1]) if len(mask_points) > 0 else width
        
        # Calculate the mask center and total height
        mask_height = bottom_y - top_y
        mask_center_x = (left_x + right_x) // 2
        
        # Increase the bangs height for better coverage
        # Ensure all values are integers for slicing
        bangs_height = min(int(extension_amount * 1.5), int(mask_height * 0.4))  # Reduced from 0.6 to 0.4
        print(f"Calculated bangs height: {bangs_height} pixels (from top)")
        
        # Calculate bottom edge of bangs area - ensure integer
        bangs_bottom = min(height, int(top_y + bangs_height))
        top_y = int(top_y)  # Ensure top_y is also an integer
        
        # Create a temp mask that only contains the top portion
        top_region_mask = np.zeros_like(mask)
        top_region_mask[top_y:bangs_bottom, :] = mask[top_y:bangs_bottom, :]
        top_region_points = np.argwhere(top_region_mask > 127)
        
        if len(top_region_points) > 0:
            # Get actual width of top region
            top_left_x = np.min(top_region_points[:, 1])
            top_right_x = np.max(top_region_points[:, 1])
            top_width = top_right_x - top_left_x
            
            # Apply width_ratio to center portion - ensure integers
            adjusted_width = int(top_width * max(width_ratio * 1.2, 0.3))  # Reduced from 1.5 to 1.2
            center_x = int((top_left_x + top_right_x) // 2)
            
            # Ensure the width is not too narrow
            min_width = min(100, int(width * 0.15))  # Reduced from 150 to 100 and 0.25 to 0.15
            adjusted_width = max(adjusted_width, min_width)
            
            left_x = max(0, int(center_x - adjusted_width // 2))
            right_x = min(width, int(center_x + adjusted_width // 2))
            
            print(f"Bangs region: x={left_x}-{right_x}, y={top_y}-{bangs_bottom}")
        else:
            # Default to middle portion based on width_ratio if no points found in top region
            center_x = width // 2
            adjusted_width = max(int(width * width_ratio * 1.2), int(width * 0.15))  # Reduced from 1.5/0.25 to 1.2/0.15
            left_x = max(0, int(center_x - adjusted_width // 2))
            right_x = min(width, int(center_x + adjusted_width // 2))
            
            print(f"Using default bangs region: x={left_x}-{right_x}, y={top_y}-{bangs_bottom}")
        
        # IMPORTANT: Only apply the mask to the bangs region, not the entire mask!
        # Create a blank mask and only copy the specified region
        bangs_region_mask = np.zeros_like(mask)
        
        # Create a trapezoidal mask shape for the bangs
        for y in range(top_y, bangs_bottom):
            # Calculate width expansion ratio (wider at the bottom)
            progress = (y - top_y) / float(max(1, bangs_bottom - top_y))
            expansion = int((right_x - left_x) * 0.15 * progress)  # Reduced from 0.2 to 0.15
            
            x_start = max(0, int(left_x - expansion))
            x_end = min(width, int(right_x + expansion))
            
            # Copy mask at this row only in the specified region
            if y < mask.shape[0] and x_start < x_end:
                bangs_region_mask[y, x_start:x_end] = mask[y, x_start:x_end]
        
        # Use the correctly isolated region
        bangs_mask = bangs_region_mask.copy()
        
        # Add feathering at the bottom with min_opacity
        fade_height = int(bangs_height * 0.4)  # Increased fade height
        fade_start = bangs_bottom - fade_height
        
        for y in range(fade_start, bangs_bottom):
            if y >= mask.shape[0]:
                continue
            fade_factor = 1.0 - ((y - fade_start) / float(max(1, fade_height)))
            # Apply minimum opacity from config
            adjusted_factor = min_opacity + (fade_factor * (1.0 - min_opacity))
            if y < bangs_mask.shape[0]:
                bangs_mask[y, :] = (bangs_mask[y, :].astype(float) * adjusted_factor).astype(np.uint8)
        
        # Make sure the mask has enough content by dilating slightly
        if cv2.countNonZero(bangs_mask) < 200:
            kernel = np.ones((3, 3), np.uint8)  # Reduced from 5x5 to 3x3
            bangs_mask = cv2.dilate(bangs_mask, kernel, iterations=1)  # Reduced from 2 to 1
        
        # Count non-zero pixels for logging
        nonzero_pixels = cv2.countNonZero(bangs_mask)
        total_pixels = bangs_mask.shape[0] * bangs_mask.shape[1]
        mask_coverage = nonzero_pixels / total_pixels
        print(f"Bangs mask has {nonzero_pixels} non-zero pixels ({mask_coverage:.2%} coverage)")
        
        # Make sure we don't have too much coverage
        if mask_coverage > 0.1:  # If more than 10% of the image is covered
            print("Warning: Bangs mask coverage too high, reducing...")
            # Keep only the top half of the detected region
            reduced_bottom = top_y + (bangs_bottom - top_y) // 2
            temp_mask = np.zeros_like(bangs_mask)
            temp_mask[top_y:reduced_bottom, :] = bangs_mask[top_y:reduced_bottom, :]
            bangs_mask = temp_mask
            
            # Recount
            nonzero_pixels = cv2.countNonZero(bangs_mask)
            mask_coverage = nonzero_pixels / total_pixels
            print(f"Reduced bangs mask has {nonzero_pixels} non-zero pixels ({mask_coverage:.2%} coverage)")
        
        # Apply a small blur to smooth the mask edges
        bangs_mask = cv2.GaussianBlur(bangs_mask, (3, 3), 0)  # Reduced from 5x5 to 3x3
        
        # Force mask to have enough contrast to be visible
        if cv2.countNonZero(bangs_mask) > 0:
            bangs_mask = cv2.normalize(bangs_mask, None, 100, 255, cv2.NORM_MINMAX)
        
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

    def _calculate_face_scale_and_rotation(self, source_landmarks, processed_landmarks):
        """
        Calculate scale and rotation adjustments based on facial landmarks.
        
        Args:
            source_landmarks: Facial landmarks from source image
            processed_landmarks: Facial landmarks from processed image
            
        Returns:
            tuple: (scale_x, scale_y, rotation_angle_degrees)
        """
        if source_landmarks is None or processed_landmarks is None:
            return 1.0, 1.0, 0.0
            
        try:
            # Use eye landmarks for scale and rotation
            # Left eye: landmarks[36:42]
            # Right eye: landmarks[42:48]
            
            # Get eye centers
            def get_eye_center(landmarks, eye_indices):
                eye_points = landmarks[eye_indices[0]:eye_indices[1]]
                return np.mean(eye_points, axis=0)
            
            # Source image eyes
            src_left_eye = get_eye_center(source_landmarks, (36, 42))
            src_right_eye = get_eye_center(source_landmarks, (42, 48))
            
            # Processed image eyes
            proc_left_eye = get_eye_center(processed_landmarks, (36, 42))
            proc_right_eye = get_eye_center(processed_landmarks, (42, 48))
            
            # Calculate eye distances (inter-ocular distance)
            src_eye_distance = np.linalg.norm(src_right_eye - src_left_eye)
            proc_eye_distance = np.linalg.norm(proc_right_eye - proc_left_eye)
            
            # Calculate scale factor based on eye distance ratio
            if proc_eye_distance > 0 and src_eye_distance > 0:
                scale = src_eye_distance / proc_eye_distance
            else:
                scale = 1.0
                
            # Calculate face size using the face bounding box for vertical scale
            src_face_top = np.min(source_landmarks[:, 1])
            src_face_bottom = np.max(source_landmarks[:, 1])
            src_face_height = src_face_bottom - src_face_top
            
            proc_face_top = np.min(processed_landmarks[:, 1])
            proc_face_bottom = np.max(processed_landmarks[:, 1])
            proc_face_height = proc_face_bottom - proc_face_top
            
            # Calculate vertical scale
            if proc_face_height > 0 and src_face_height > 0:
                scale_y = src_face_height / proc_face_height
            else:
                scale_y = scale  # Use horizontal scale as fallback
            
            # Calculate rotation angle
            src_eye_angle = np.arctan2(src_right_eye[1] - src_left_eye[1], 
                                      src_right_eye[0] - src_left_eye[0])
            proc_eye_angle = np.arctan2(proc_right_eye[1] - proc_left_eye[1], 
                                       proc_right_eye[0] - proc_left_eye[0])
            
            # Calculate angle difference in degrees
            angle_diff = np.degrees(src_eye_angle - proc_eye_angle)
            
            # Normalize the angle to -180 to 180 range
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360
                
            return scale, scale_y, angle_diff
            
        except (IndexError, ValueError, ZeroDivisionError) as e:
            print(f"Error calculating face scale and rotation: {str(e)}")
            return 1.0, 1.0, 0.0

    def _blend_bangs_only(self, source_img, processed_img, mask, method="alpha", blend_extent=5, preserve_edges=True, debug_dir=None):
        """
        Special blending for bangs-only mode that respects all blending settings.
        
        Args:
            source_img: Original source image
            processed_img: Processed image with the new hair/bangs
            mask: The bangs-only mask
            method: Blending method ("alpha", "poisson", "feathered")
            blend_extent: Extent of feathering
            preserve_edges: Whether to preserve original image edges
            debug_dir: Debug directory for saving visualizations
            
        Returns:
            numpy.ndarray: Blended image
        """
        print(f"Blending bangs using method: {method}, extent: {blend_extent}, preserve_edges: {preserve_edges}")
        
        # Check if the mask is too small or sparse for effective Poisson blending
        nonzero_pixels = cv2.countNonZero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        mask_coverage = nonzero_pixels / total_pixels
        
        print(f"Mask has {nonzero_pixels} non-zero pixels ({mask_coverage:.2%} coverage)")
        
        # Save original mask for debugging
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "bangs_mask_before_blending.png"), mask)
            
        # Always force alpha blending for very small masks or very large masks
        if nonzero_pixels < 500 or mask_coverage < 0.005 or mask_coverage > 0.4:
            print(f"Warning: Mask size not ideal for {method} blending, forcing alpha blending with increased opacity")
            # Enhance the mask to make it more visible
            enhanced_mask = mask.copy()
            # Dilate slightly to increase coverage
            kernel = np.ones((3, 3), np.uint8)
            enhanced_mask = cv2.dilate(enhanced_mask, kernel, iterations=1)
            # Increase contrast
            enhanced_mask = cv2.normalize(enhanced_mask, None, 100, 255, cv2.NORM_MINMAX)
            
            # Use alpha blending with higher opacity
            result_img = self._alpha_blend(source_img, processed_img, enhanced_mask, blend_extent)
            
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "enhanced_bangs_mask.png"), enhanced_mask)
        else:
            try:
                # Choose the appropriate blending method
                if method == "alpha":
                    result_img = self._alpha_blend(source_img, processed_img, mask, blend_extent)
                elif method == "poisson":
                    try:
                        # For Poisson blending, make sure we have enough non-zero pixels
                        if nonzero_pixels < 1000 or mask_coverage < 0.01:
                            # For smaller masks, enhance slightly
                            enhanced_mask = cv2.normalize(mask, None, 100, 255, cv2.NORM_MINMAX)
                            result_img = self._alpha_blend(source_img, processed_img, enhanced_mask, blend_extent)
                        else:
                            # Before trying Poisson, check if mask is valid for seamless cloning
                            moments = cv2.moments(mask)
                            if moments["m00"] > 0:
                                center_x = int(moments["m10"] / moments["m00"])
                                center_y = int(moments["m01"] / moments["m00"])
                                
                                # Make sure center point is within safe boundaries
                                h, w = source_img.shape[:2]
                                min_distance = 50  # Minimum distance from edges
                                
                                if (min_distance <= center_x < w - min_distance and 
                                    min_distance <= center_y < h - min_distance):
                                    # Safe to try Poisson blending
                                    result_img = self._alpha_blend(source_img, processed_img, mask, blend_extent)
                                    # First create a backup result in case Poisson fails
                                    try:
                                        poisson_result = self._poisson_blend(source_img, processed_img, mask)
                                        # If it succeeded, use it
                                        result_img = poisson_result
                                    except Exception as e:
                                        print(f"Poisson blending failed for bangs: {str(e)}, using alpha blend")
                                else:
                                    print("Center point too close to edges, using alpha blending")
                                    result_img = self._alpha_blend(source_img, processed_img, mask, blend_extent)
                            else:
                                print("Invalid mask moments, using alpha blending")
                                result_img = self._alpha_blend(source_img, processed_img, mask, blend_extent)
                    except Exception as e:
                        print(f"Poisson blending failed for bangs: {str(e)}, falling back to alpha")
                        result_img = self._alpha_blend(source_img, processed_img, mask, blend_extent)
                elif method == "feathered":
                    result_img = self._feathered_blend(source_img, processed_img, mask, blend_extent)
                else:
                    # Default to alpha blending
                    result_img = self._alpha_blend(source_img, processed_img, mask, blend_extent)
            except Exception as e:
                print(f"Blending error: {str(e)}, using basic alpha blending")
                # Fallback to the simplest alpha blending
                mask_float = mask.astype(float) / 255.0
                mask_float_3d = np.stack([mask_float] * 3, axis=2)
                result_img = source_img * (1 - mask_float_3d) + processed_img * mask_float_3d
                result_img = np.clip(result_img, 0, 255).astype(np.uint8)
        
        # Preserve original edges if requested
        if preserve_edges:
            try:
                result_img = self._preserve_image_edges(source_img, result_img, mask)
            except Exception as e:
                print(f"Error preserving edges: {str(e)}")
            
        # Save debug image if directory provided
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "bangs_only_blended.png"), result_img)
            
            # Create a comparison image
            comparison = np.hstack((source_img, processed_img, result_img))
            cv2.imwrite(os.path.join(debug_dir, "bangs_only_comparison.png"), comparison)
            
            # Create before/after comparison
            before_after = np.vstack((source_img, result_img))
            cv2.imwrite(os.path.join(debug_dir, "bangs_before_after.png"), before_after)
        
        return result_img