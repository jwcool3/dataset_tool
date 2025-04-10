"""
Enhanced Crop Reinserter for Dataset Preparation Tool
Specialized for handling resolution differences between source and processed images.
"""

import os
import cv2
import numpy as np
import re
import json

# Import specialized modules
from processors.hair_processing.bangs_processor import isolate_bangs_region, extend_bangs_area, create_face_protection_mask
from processors.hair_processing.alignment import align_masks, apply_landmark_transform, apply_translation_only_transform
from processors.hair_processing.blending import blend_images, alpha_blend, poisson_blend, feathered_blend, preserve_image_edges
from processors.hair_processing.landmarks import get_landmarks, estimate_bangs_position


class EnhancedCropReinserter:
    """Reinserts processed regions back into original images, handling resolution differences."""
    
    def __init__(self, app):
        """
        Initialize enhanced crop reinserter.
        
        Args:
            app: The main application with shared variables and UI controls
        """
        self.app = app
        
        # Initialize face detection if needed
        try:
            import dlib
            self.face_detector = dlib.get_frontal_face_detector()
            
            # Look for the shape predictor file in common locations
            predictor_paths = [
                os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "shape_predictor_68_face_landmarks.dat"),
                os.path.join(os.path.expanduser("~"), ".dataset_preparation_tool", "models", "shape_predictor_68_face_landmarks.dat")
            ]
            
            self.landmark_predictor = None
            for path in predictor_paths:
                if os.path.exists(path):
                    print(f"Found landmark predictor at: {path}")
                    self.landmark_predictor = dlib.shape_predictor(path)
                    break
                    
            if self.landmark_predictor is None:
                print("WARNING: Facial landmark predictor file not found.")
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
        
        # Create a debug directory if in debug mode
        debug_dir = None
        if self.app.debug_mode.get():
            debug_dir = os.path.join(output_dir, "reinsert_debug")
            os.makedirs(debug_dir, exist_ok=True)
            print(f"Debug mode enabled, saving visualizations to: {debug_dir}")
        
        # Find all processed images and their corresponding masks
        processed_images = self._find_processed_images(input_dir)
        
        # Get source directory (original images)
        source_dir = self.app.source_images_dir.get()
        if not source_dir or not os.path.isdir(source_dir):
            self.app.status_label.config(text="Source images directory not set or invalid.")
            return False
        
        # Load all source images
        source_images = self._find_source_images(source_dir)
        
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
    
    def _find_processed_images(self, input_dir):
        """Find processed images and their corresponding masks in the input directory."""
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
                    mask_path = self._find_mask_for_image(root, file)
                    
                    # Add to processing list
                    processed_images.append({
                        'processed_path': processed_path,
                        'mask_path': mask_path,
                        'filename': file
                    })
                    
        return processed_images
    
    def _find_mask_for_image(self, root_dir, image_file):
        """Find the corresponding mask file for an image."""
        # Check if there's a masks subdirectory
        masks_dir = os.path.join(root_dir, "masks")
        if os.path.isdir(masks_dir):
            # Try exact filename match first
            potential_mask = os.path.join(masks_dir, image_file)
            if os.path.exists(potential_mask):
                return potential_mask
            
            # Try different extensions
            base_name = os.path.splitext(image_file)[0]
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_mask = os.path.join(masks_dir, base_name + ext)
                if os.path.exists(potential_mask):
                    return potential_mask
        
        return None
    
    def _find_source_images(self, source_dir):
        """Find all source images in the source directory."""
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
        # First, try a direct match - same filename
        if processed_filename in source_images:
            print(f"Found direct match for {processed_filename}")
            return processed_filename
        
        # Try to match by base name (ignoring extension)
        base_name = os.path.splitext(processed_filename)[0]
        for source_name, source_path in source_images.items():
            source_base = os.path.splitext(source_name)[0]
            if source_base == base_name:
                print(f"Found match by base name: {source_name}")
                return source_name
        
        # As a last resort, if there's only one source image, use that
        if len(source_images) == 1:
            source_name = next(iter(source_images))
            print(f"Only one source image found, using {source_name}")
            return source_name
        
        print(f"WARNING: No matching source image found for {processed_filename}")
        return None
    
    def _reinsert_with_resolution_handling(self, source_path, processed_path, mask_path, output_path, debug_dir=None):
        """
        Main function for reinserting processed image into source image with proper handling
        of masks, alignments, transformations, and blending.
        """
        # Debug output of parameters
        print(f"DEBUG: Bangs extension={self.app.extend_bangs.get()}, amount={self.app.bangs_extension_amount.get()}")
        print(f"DEBUG: Manual offset X={self.app.reinsert_manual_offset_x.get()}, Y={self.app.reinsert_manual_offset_y.get()}")
        print(f"DEBUG: Manual scale X={self.app.reinsert_manual_scale_x.get()}, Y={self.app.reinsert_manual_scale_y.get()}")
        
        # 1. Load source and processed images
        source_img = cv2.imread(source_path)
        processed_img = cv2.imread(processed_path)
        
        if source_img is None or processed_img is None:
            print(f"Failed to load images: {source_path} or {processed_path}")
            return False
        
        # 2. Find/create masks
        source_mask = self._find_source_mask(source_path)
        mask = self._load_mask(mask_path, processed_img.shape[:2])
        
        # Get source image dimensions
        source_h, source_w = source_img.shape[:2]
        
        # 3. Detect landmarks in source image (if available)
        source_landmarks = get_landmarks(source_img, self.face_detector, self.landmark_predictor) if hasattr(self, 'face_detector') else None
        
        # 4. Process bangs if bangs-only mode is enabled
        if self.app.use_bangs_only.get():
            # If we have a source mask, use it for isolation as it's more accurate
            if source_mask is not None:
                print("Using source mask for bangs isolation")
                mask = isolate_bangs_region(source_mask, source_landmarks)
            else:
                print("Using processed mask for bangs isolation")
                mask = isolate_bangs_region(mask, source_landmarks)
            
            # Debug visualization
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "isolated_bangs_mask.png"), mask)
        
        # 5. Apply bangs extension if enabled
        if self.app.extend_bangs.get() and mask is not None:
            extension_amount = self.app.bangs_extension_amount.get()
            width_ratio = self.app.bangs_width_ratio.get()
            min_opacity = max(0.5, 0.9 - (extension_amount / 100.0))
            
            # Save pre-extension mask if debugging
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "pre_extension_mask.png"), mask)
            
            # Extend the mask
            mask = extend_bangs_area(
                mask, 
                extend_pixels=extension_amount,
                forehead_ratio=width_ratio,
                min_opacity=min_opacity,
                source_landmarks=source_landmarks
            )
            
            # Save extended mask if debugging
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "extended_bangs_mask.png"), mask)
        
        # 6. Apply face protection if enabled
        if self.app.protect_face_from_bangs.get() and source_landmarks is not None:
            protection_strength = self.app.face_protection_strength.get()
            mask = create_face_protection_mask(mask, source_landmarks, protection_strength)
            
            # Save face-protected mask if debugging
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "face_protected_mask.png"), mask)
        
        # 7. Resize processed image and mask to match source dimensions
        if source_img.shape[:2] != processed_img.shape[:2] or source_img.shape[:2] != mask.shape[:2]:
            processed_img = cv2.resize(processed_img, (source_w, source_h), interpolation=cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, (source_w, source_h), interpolation=cv2.INTER_LINEAR)
        
        # 8. Apply transformations (alignment, scaling, rotation, offset)
        # Get transformation parameters
        params = {
            'align_method': self.app.reinsert_alignment_method.get(),
            'scale_x': self.app.reinsert_manual_scale_x.get(),
            'scale_y': self.app.reinsert_manual_scale_y.get(),
            'offset_x': self.app.reinsert_manual_offset_x.get(),
            'offset_y': self.app.reinsert_manual_offset_y.get(),
            'rotation': self.app.reinsert_manual_rotation.get() if hasattr(self.app, 'reinsert_manual_rotation') else 0.0,
            'use_translation_only': self.app.use_translation_only.get() if hasattr(self.app, 'use_translation_only') else True
        }
        
        # First align the mask if needed
        aligned_mask = mask
        aligned_img = processed_img
        
        if params['align_method'] != 'none':
            if params['align_method'] == 'landmarks' and source_landmarks is not None:
                # Create a set of processed landmarks based on geometric estimation
                processed_landmarks = get_landmarks(processed_img)
                
                # Apply landmark-based alignment
                if params['use_translation_only']:
                    aligned_mask, aligned_img = apply_translation_only_transform(
                        source_img, processed_img, source_mask, mask,
                        source_landmarks, processed_landmarks, params, debug_dir
                    )
                else:
                    aligned_mask, aligned_img = apply_landmark_transform(
                        source_img, processed_img, source_mask, mask,
                        source_landmarks, processed_landmarks, params, debug_dir
                    )
            else:
                # Use standard alignment methods
                aligned_mask, aligned_img = align_masks(
                    source_mask, mask, source_img, processed_img, 
                    params['align_method'], debug_dir
                )
        
        # 9. Apply blending
        blend_params = {
            'mode': self.app.reinsert_blend_mode.get(),
            'extent': self.app.reinsert_blend_extent.get(),
            'preserve_edges': self.app.reinsert_preserve_edges.get()
        }
        
        # Blend the images
        if self.app.reinsert_mask_only.get():
            # Mask-only mode: only blend the masked regions
            result_img = blend_images(source_img, aligned_img, aligned_mask, blend_params)
        else:
            # Regular mode: blend the entire image
            result_img = blend_images(source_img, aligned_img, aligned_mask, blend_params)
        
        # 10. Save the result
        cv2.imwrite(output_path, result_img)
        
        # Debug: Create comparison image
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
    
    def _load_mask(self, mask_path, shape):
        """Load mask with fallback to full image mask if not found."""
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return mask
        
        # Create a fallback mask covering the entire image
        h, w = shape
        return np.ones((h, w), dtype=np.uint8) * 255