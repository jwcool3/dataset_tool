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


    def _apply_transformation(self, img, mask, transformation_params, debug_dir=None, debug_prefix=None):
            """
            Apply a series of standard transformations to an image and its mask.
            
            Args:
                img: The image to transform
                mask: The mask to transform
                transformation_params: Dictionary containing transformation parameters:
                    - 'matrix': Optional affine transformation matrix (2x3 numpy array)
                    - 'translation': Optional (dx, dy) tuple for translation
                    - 'scale': Optional (scale_x, scale_y) tuple for scaling
                    - 'rotation': Optional rotation angle in degrees (around center)
                    - 'interpolation': Optional interpolation method for the image (default: cv2.INTER_LANCZOS4)
                    - 'mask_interpolation': Optional interpolation method for the mask (default: cv2.INTER_NEAREST)
                    - 'border_mode': Optional border mode (default: cv2.BORDER_TRANSPARENT)
                debug_dir: Optional directory to save debug images
                debug_prefix: Optional prefix for debug image filenames
                
            Returns:
                tuple: (transformed_img, transformed_mask)
            """
            # Get image dimensions
            h, w = img.shape[:2]
            
            # Create copies to modify
            transformed_img = img.copy()
            transformed_mask = mask.copy() if mask is not None else None
            
            # Set default interpolation and border mode if not specified
            img_interpolation = transformation_params.get('interpolation', cv2.INTER_LANCZOS4)
            mask_interpolation = transformation_params.get('mask_interpolation', cv2.INTER_NEAREST)
            border_mode = transformation_params.get('border_mode', cv2.BORDER_TRANSPARENT)
            
            # Apply affine transformation matrix if provided
            if 'matrix' in transformation_params and transformation_params['matrix'] is not None:
                matrix = transformation_params['matrix']
                
                transformed_img = cv2.warpAffine(
                    transformed_img, matrix, (w, h),
                    flags=img_interpolation,
                    borderMode=border_mode
                )
                
                if transformed_mask is not None:
                    transformed_mask = cv2.warpAffine(
                        transformed_mask, matrix, (w, h),
                        flags=mask_interpolation,
                        borderMode=border_mode
                    )
            
            # Apply translation if provided
            if 'translation' in transformation_params and transformation_params['translation'] is not None:
                dx, dy = transformation_params['translation']
                
                if dx != 0 or dy != 0:
                    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
                    
                    transformed_img = cv2.warpAffine(
                        transformed_img, translation_matrix, (w, h),
                        flags=img_interpolation,
                        borderMode=border_mode
                    )
                    
                    if transformed_mask is not None:
                        transformed_mask = cv2.warpAffine(
                            transformed_mask, translation_matrix, (w, h),
                            flags=mask_interpolation,
                            borderMode=border_mode
                        )
            
            # Apply scaling if provided
            if 'scale' in transformation_params and transformation_params['scale'] is not None:
                scale_x, scale_y = transformation_params['scale']
                
                if scale_x != 1.0 or scale_y != 1.0:
                    # Calculate center for scaling around center
                    center_x, center_y = w // 2, h // 2
                    
                    # Create transformation matrix for scaling
                    scale_matrix = np.float32([
                        [scale_x, 0, center_x * (1 - scale_x)],
                        [0, scale_y, center_y * (1 - scale_y)]
                    ])
                    
                    transformed_img = cv2.warpAffine(
                        transformed_img, scale_matrix, (w, h),
                        flags=img_interpolation,
                        borderMode=border_mode
                    )
                    
                    if transformed_mask is not None:
                        transformed_mask = cv2.warpAffine(
                            transformed_mask, scale_matrix, (w, h),
                            flags=mask_interpolation,
                            borderMode=border_mode
                        )
            
            # Apply rotation if provided
            if 'rotation' in transformation_params and transformation_params['rotation'] is not None:
                angle = transformation_params['rotation']
                
                if angle != 0:
                    # Calculate center of rotation (image center)
                    center_x, center_y = w // 2, h // 2
                    
                    # Get rotation matrix
                    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1.0)
                    
                    transformed_img = cv2.warpAffine(
                        transformed_img, rotation_matrix, (w, h),
                        flags=img_interpolation,
                        borderMode=border_mode
                    )
                    
                    if transformed_mask is not None:
                        transformed_mask = cv2.warpAffine(
                            transformed_mask, rotation_matrix, (w, h),
                            flags=mask_interpolation,
                            borderMode=border_mode
                        )
            
            # Save debug images if requested
            if debug_dir is not None and debug_prefix is not None:
                cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_img.png"), transformed_img)
                if transformed_mask is not None:
                    cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_mask.png"), transformed_mask)
            
            return transformed_img, transformed_mask
    

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

    def _get_landmarks(self, image):
        """
        Detect face and extract facial landmarks.
        
        Args:
            image: Input image
            
        Returns:
            list: List of (x, y) landmark coordinates or None if detection fails
        """
        # Check if detector and predictor are available
        if self.face_detector is None or self.landmark_predictor is None:
            print("Facial landmark detection not available")
            return None
            
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect faces
        try:
            faces = self.face_detector(gray)
            if not faces:
                print("No faces detected in image")
                return None
            
            # Get largest face
            largest_face = faces[0]
            largest_area = (largest_face.right() - largest_face.left()) * (largest_face.bottom() - largest_face.top())
            
            for face in faces[1:]:
                area = (face.right() - face.left()) * (face.bottom() - face.top())
                if area > largest_area:
                    largest_face = face
                    largest_area = area
            
            # Get landmarks for the face
            shape = self.landmark_predictor(gray, largest_face)
            
            # Convert landmarks to list of (x, y) coordinates
            landmarks = []
            for i in range(68):  # 68 landmarks in the standard model
                x = shape.part(i).x
                y = shape.part(i).y
                landmarks.append((x, y))
            
            return landmarks
        except Exception as e:
            print(f"Error detecting landmarks: {str(e)}")
            return None

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
            h, w = source_img.shape[:2]
            
            # Get the transformation matrix using improved landmark alignment
            transform_matrix, success = self._improve_landmark_alignment(source_landmarks, processed_landmarks)
            
            if not success or transform_matrix is None:
                print("Could not calculate transformation matrix. Falling back to default alignment.")
                # Return the unmodified inputs as fallback
                return processed_mask, processed_img
            
            # Get manual transformation parameters from the app
            manual_offset_x = self.app.reinsert_manual_offset_x.get()
            manual_offset_y = self.app.reinsert_manual_offset_y.get()
            scale_x = self.app.reinsert_manual_scale_x.get()
            scale_y = self.app.reinsert_manual_scale_y.get()
            rotation_angle = self.app.reinsert_manual_rotation.get()
            
            # Create a transformation parameters dictionary
            transformation_params = {
                'matrix': transform_matrix,
                'translation': (manual_offset_x, manual_offset_y) if manual_offset_x != 0 or manual_offset_y != 0 else None,
                'scale': (scale_x, scale_y) if scale_x != 1.0 or scale_y != 1.0 else None,
                'rotation': rotation_angle if rotation_angle != 0 else None
            }
            
            # Apply all transformations using the centralized method
            aligned_img, aligned_mask = self._apply_transformation(
                processed_img, 
                processed_mask, 
                transformation_params,
                debug_dir=debug_dir,
                debug_prefix='landmark_transform'
            )
            
            # Save debug visualization
            if debug_dir:
                # Draw landmarks on images for visualization
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
        h, w = source_img.shape[:2]
        
        # Step 1: Calculate the full transformation matrix (for reference points)
        transform_matrix, success = self._improve_landmark_alignment(source_landmarks, processed_landmarks)
        
        # Extract translation component
        if not success or transform_matrix is None:
            print("Could not calculate transformation matrix. Using simpler alignment.")
            # Calculate centers of face landmarks as fallback
            source_center = np.mean(np.array(source_landmarks), axis=0)
            processed_center = np.mean(np.array(processed_landmarks), axis=0)
            
            # Just use translation between centers
            tx = source_center[0] - processed_center[0]
            ty = source_center[1] - processed_center[1]
        else:
            # Extract only the translation component from the transformation
            tx, ty = transform_matrix[0, 2], transform_matrix[1, 2]
            print(f"Full transformation matrix:\n{transform_matrix}")
            print(f"Using only translation components: tx={tx:.2f}, ty={ty:.2f}")
        
        # Get manual transformation parameters from the app
        manual_offset_x = self.app.reinsert_manual_offset_x.get()
        manual_offset_y = self.app.reinsert_manual_offset_y.get()
        scale_x = self.app.reinsert_manual_scale_x.get()
        scale_y = self.app.reinsert_manual_scale_y.get()
        rotation_angle = self.app.reinsert_manual_rotation.get()
        
        # Create transformation parameters dictionary
        transformation_params = {
            'translation': (tx, ty),  # Apply the extracted translation
            'scale': (scale_x, scale_y) if scale_x != 1.0 or scale_y != 1.0 else None,
            'rotation': rotation_angle if rotation_angle != 0 else None
        }
        
        # Apply manual offset if any
        if manual_offset_x != 0 or manual_offset_y != 0:
            # Add manual offsets to the translation
            tx += manual_offset_x
            ty += manual_offset_y
            transformation_params['translation'] = (tx, ty)
        
        # Apply all transformations using the centralized method
        aligned_img, aligned_mask = self._apply_transformation(
            processed_img, 
            processed_mask, 
            transformation_params,
            debug_dir=debug_dir,
            debug_prefix='translation_only'
        )
        
        # Save debug visualization
        if debug_dir:
            # Draw landmarks on images for visualization
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
        
            # Create comparison visualization
            full_comparison = np.hstack((source_vis, processed_vis, aligned_vis))
            cv2.imwrite(os.path.join(debug_dir, "translation_only_alignment.png"), full_comparison)
            
            # Overlay visualization showing source and aligned masks
            mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
            if source_mask is not None:
                mask_overlay[source_mask > 127] = [0, 0, 255]  # Source mask in red
            mask_overlay[aligned_mask > 127] = [0, 255, 0]    # Aligned mask in green
            cv2.imwrite(os.path.join(debug_dir, "mask_alignment_translation_only.png"), mask_overlay)
        
        return aligned_mask, aligned_img
    

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
            
            # Default transformation parameters (no transformation)
            transformation_params = {}
            
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
                    
                    # Set translation parameters
                    transformation_params['translation'] = (dx, dy)
            
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
                    
                    # Set translation parameters
                    transformation_params['translation'] = (dx, dy)
            
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
                    
                    # Set translation parameters
                    transformation_params['translation'] = (dx, dy)
            
            # Intersection Over Union (IoU) alignment
            elif alignment_method == "iou":
                best_iou = 0
                best_dx = 0
                best_dy = 0
                
                max_shift = 20  # pixels
                for dx in range(-max_shift, max_shift + 1, 2):
                    for dy in range(-max_shift, max_shift + 1, 2):
                        # Create translation parameters
                        test_params = {'translation': (dx, dy)}
                        
                        # Apply translation to get shifted mask
                        _, shifted_mask = self._apply_transformation(
                            processed_img,
                            processed_mask,
                            test_params
                        )
                        
                        # Threshold the shifted mask for IoU calculation
                        _, shifted_mask_bin = cv2.threshold(shifted_mask, 127, 255, cv2.THRESH_BINARY)
                        
                        # Calculate IoU
                        intersection = np.logical_and(source_mask_bin > 0, shifted_mask_bin > 0).sum()
                        union = np.logical_or(source_mask_bin > 0, shifted_mask_bin > 0).sum()
                        iou = intersection / union if union > 0 else 0
                        
                        # Update best if improved
                        if iou > best_iou:
                            best_iou = iou
                            best_dx = dx
                            best_dy = dy
                
                # Set best translation parameters
                transformation_params['translation'] = (best_dx, best_dy)
            
            # Apply the determined transformation
            aligned_img, aligned_mask = self._apply_transformation(
                processed_img,
                processed_mask,
                transformation_params,
                debug_dir=debug_dir,
                debug_prefix=f'align_{alignment_method}'
            )
            

            # Ensure the aligned images have the correct dimensions (same as source image)
            h, w = source_img.shape[:2]
            if aligned_img.shape[:2] != (h, w):
                print(f"Warning: Aligned image dimensions don't match source. Resizing from {aligned_img.shape[1]}x{aligned_img.shape[0]} to {w}x{h}")
                aligned_img = cv2.resize(aligned_img, (w, h), interpolation=cv2.INTER_LANCZOS4)
                
            if aligned_mask.shape[:2] != (h, w):
                print(f"Warning: Aligned mask dimensions don't match source. Resizing from {aligned_mask.shape[1]}x{aligned_mask.shape[0]} to {w}x{h}")
                aligned_mask = cv2.resize(aligned_mask, (w, h), interpolation=cv2.INTER_NEAREST)


            # Debug visualization
            if debug_dir:
                # Source mask in red, aligned mask in green
                h, w = source_mask_bin.shape[:2]
                mask_viz = np.zeros((h, w, 3), dtype=np.uint8)
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

    def _isolate_bangs_region(self, mask, landmarks=None):
            """
            Isolate only the bangs portion of a hair mask with optional face protection.
            
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
            
            # Method 1: Using landmarks if available
            if landmarks is not None:
                try:
                    # Get eyebrow height (use top of eyebrows)
                    eyebrow_points = landmarks[17:27]  # Eyebrow landmarks
                    eyebrow_y = min([p[1] for p in eyebrow_points])
                    
                    # Get face width from landmarks
                    left_temple = landmarks[0][0]  # Leftmost face point
                    right_temple = landmarks[16][0]  # Rightmost face point
                    
                    # Define forehead region above eyebrows
                    forehead_top = max(0, eyebrow_y - int(height * 0.2))  # Region above eyebrows
                    forehead_bottom = eyebrow_y + int(height * 0.05)  # Slightly below eyebrows
                    forehead_left = max(0, left_temple - int(width * 0.05))
                    forehead_right = min(width, right_temple + int(width * 0.05))
                    
                    # Create region for forehead/bangs area
                    bangs_region = np.zeros_like(mask)
                    bangs_region[forehead_top:forehead_bottom, forehead_left:forehead_right] = 255
                    
                    # Intersect with the original mask to get only the bangs
                    bangs_mask = cv2.bitwise_and(mask, bangs_region)
                    
                    # Apply a feathering at the bottom for a more natural transition
                    fade_height = int((forehead_bottom - forehead_top) * 0.3)
                    fade_start = forehead_bottom - fade_height
                    
                    for y in range(fade_start, forehead_bottom):
                        fade_factor = 1.0 - ((y - fade_start) / float(fade_height))
                        bangs_mask_row = bangs_mask[y, :].astype(float) * fade_factor
                        bangs_mask[y, :] = bangs_mask_row.astype(np.uint8)
                    
                    # Apply face protection if enabled
                    if hasattr(self.app, 'protect_face_from_bangs') and self.app.protect_face_from_bangs.get():
                        protection_strength = self.app.face_protection_strength.get()
                        bangs_mask = self._create_face_protection_mask(
                            bangs_mask, 
                            landmarks, 
                            protection_strength
                        )
                        
                        if self.app.debug_mode.get():
                            debug_dir = os.path.join(os.path.dirname(os.path.dirname(mask)), "debug")
                            os.makedirs(debug_dir, exist_ok=True)
                            cv2.imwrite(
                                os.path.join(debug_dir, "face_protected_bangs.png"), 
                                bangs_mask
                            )
                    
                    return bangs_mask
                        
                except (IndexError, ValueError, TypeError) as e:
                    print(f"Error using landmarks for bangs detection: {str(e)}")
                    # Fall back to method 2
            
            # Method 2: Geometric approach (if landmarks not available or failed)
            # Find points in the mask
            mask_points = np.argwhere(mask > 127)
            if len(mask_points) == 0:
                return bangs_mask  # Empty mask, return empty
            
            # Find the topmost part of the mask
            top_y = np.min(mask_points[:, 0]) if len(mask_points) > 0 else 0
            
            # Define bangs height as a percentage of the image height
            bangs_height = int(height * 0.25)  # Top 25% of the mask
            bangs_bottom = min(height, top_y + bangs_height)
            
            # Find horizontal extent of mask at the top portion
            top_region_points = mask_points[mask_points[:, 0] <= bangs_bottom]
            if len(top_region_points) > 0:
                left_x = np.min(top_region_points[:, 1])
                right_x = np.max(top_region_points[:, 1])
            else:
                # Default to middle 60% if no points found
                left_x = int(width * 0.2)
                right_x = int(width * 0.8)
            
            # Create a trapezoidal mask shape for the bangs
            for y in range(top_y, bangs_bottom):
                # Calculate width expansion ratio (wider at the bottom)
                progress = (y - top_y) / float(max(1, bangs_bottom - top_y))
                expansion = int((right_x - left_x) * 0.1 * progress)
                
                x_start = max(0, left_x - expansion)
                x_end = min(width, right_x + expansion)
                
                # Copy mask at this row
                if y < mask.shape[0] and x_start < x_end:
                    bangs_mask[y, x_start:x_end] = mask[y, x_start:x_end]
            
            # Add feathering at the bottom
            fade_height = int(bangs_height * 0.3)
            fade_start = bangs_bottom - fade_height
            
            for y in range(fade_start, bangs_bottom):
                if y >= mask.shape[0]:
                    continue
                fade_factor = 1.0 - ((y - fade_start) / float(max(1, fade_height)))
                bangs_mask[y, :] = (bangs_mask[y, :].astype(float) * fade_factor).astype(np.uint8)
            
            return bangs_mask

    def _detect_hair_parting(self, mask, landmarks=None, parting_override=None):
            """
            Detect and enhance the hair parting line in the mask.
            
            Args:
                mask: Hair mask image
                landmarks: Optional facial landmarks for guidance
                parting_override: Optional pre-defined parting line to use
                
            Returns:
                tuple: (parting_line, enhanced_mask)
            """
            # Create copy of mask
            enhanced_mask = mask.copy()
            
            # If override is provided, use it directly
            if parting_override is not None:
                return parting_override, enhanced_mask
            
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
                    landmarks = self._get_landmarks(processed_img)
                
                # Isolate just the bangs region
                bangs_mask = self._isolate_bangs_region(mask, landmarks)
                
                # Replace the full mask with just the bangs
                mask = bangs_mask
                
                # Save debug image
                if debug_dir:
                    cv2.imwrite(os.path.join(debug_dir, "bangs_only_mask.png"), mask)

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
                    
                    # Directly use cv2.resize for resizing to ensure correct dimensions
                    processed_img_resized = cv2.resize(processed_img, (source_w, source_h), 
                                                    interpolation=cv2.INTER_LANCZOS4)
                    mask_resized = cv2.resize(mask, (source_w, source_h), 
                                            interpolation=cv2.INTER_NEAREST)
                    
                    # Save debug images if debug directory is provided
                    if debug_dir:
                        cv2.imwrite(os.path.join(debug_dir, "resized_img.png"), processed_img_resized)
                        cv2.imwrite(os.path.join(debug_dir, "resized_mask.png"), mask_resized)
                else:
                    processed_img_resized = processed_img
                    mask_resized = mask
            
            # Debug: Save original and resized images
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "source_original.png"), source_img)
                cv2.imwrite(os.path.join(debug_dir, "processed_original.png"), processed_img)
                cv2.imwrite(os.path.join(debug_dir, "mask_original.png"), mask)
            
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
            
            # Get manual transformation parameters
            manual_offset_x = self.app.reinsert_manual_offset_x.get()
            manual_offset_y = self.app.reinsert_manual_offset_y.get()
            scale_x = self.app.reinsert_manual_scale_x.get()
            scale_y = self.app.reinsert_manual_scale_y.get()
            rotation_angle = self.app.reinsert_manual_rotation.get()
            
            # Create manual transformation parameters dictionary
            manual_transform_params = {
                'translation': (manual_offset_x, manual_offset_y) if manual_offset_x != 0 or manual_offset_y != 0 else None,
                'scale': (scale_x, scale_y) if scale_x != 1.0 or scale_y != 1.0 else None,
                'rotation': rotation_angle if rotation_angle != 0 else None
            }
            
            # Apply manual transformations if any are set
            if any(value is not None for value in manual_transform_params.values()):
                # Apply all transformations at once using the centralized method
                processed_img_resized, mask_resized = self._apply_transformation(
                    processed_img_resized, 
                    mask_resized,
                    manual_transform_params,
                    debug_dir=debug_dir,
                    debug_prefix='manual_transforms'
                )
                
                # Save transformed versions for debugging
                if debug_dir:
                    cv2.imwrite(os.path.join(debug_dir, "manual_transformed_processed.png"), processed_img_resized)
                    cv2.imwrite(os.path.join(debug_dir, "manual_transformed_mask.png"), mask_resized)

            # If handling different masks, get configuration settings
            alignment_method = self.app.reinsert_alignment_method.get()
            blend_mode = self.app.reinsert_blend_mode.get()
            blend_extent = self.app.reinsert_blend_extent.get()
            preserve_edges = self.app.reinsert_preserve_edges.get()
            
            print(f"Using config settings: alignment={alignment_method}, blend={blend_mode}, extent={blend_extent}")
            
            # Align mask and image based on selected method
            aligned_mask = mask_resized.copy()
            aligned_img = processed_img_resized.copy()
            
            # With this updated version that handles missing faces better:
            if alignment_method == "landmarks":
                if hasattr(self, 'face_detector') and self.face_detector is not None and \
                hasattr(self, 'landmark_predictor') and self.landmark_predictor is not None:
                    
                    # Get landmarks for both images
                    source_landmarks = self._get_landmarks(source_img)
                    processed_landmarks = self._get_landmarks(processed_img_resized)
                    
                    # Check if we have source landmarks - that's the critical part
                    if source_landmarks is not None:
                        # If we don't have processed landmarks, we'll use a different approach
                        if processed_landmarks is None:
                            print("No face detected in processed image, using source-guided alignment")
                            # Use the source landmarks to guide the alignment with the mask
                            # Fall back to mask-based alignment but use source landmarks for guidance if possible
                            aligned_mask, aligned_img = self._align_with_source_landmarks(
                                source_img, processed_img_resized, source_mask, mask_resized,
                                source_landmarks, debug_dir
                            )
                        else:
                            # If we have both sets of landmarks, use normal landmark alignment
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
                    else:
                        print("No face detected in source image, falling back to mask-based alignment")
                        # Use mask-based alignment if no face in source image
                        aligned_mask, aligned_img = self._align_masks(
                            source_mask, mask_resized, 
                            source_img, processed_img_resized, 
                            "centroid",  # Default to centroid alignment when no faces detected
                            debug_dir
                        )
                else:
                    # Landmark detection not available
                    print("Landmark detection not available, falling back to mask-based alignment")
                    aligned_mask, aligned_img = self._align_masks(
                        source_mask, mask_resized, 
                        source_img, processed_img_resized, 
                        "centroid",  # Default to centroid alignment when landmarks unavailable
                        debug_dir
                    )



                
            # Preserve hair parting if option enabled
            if self.app.preserve_hair_parting.get():
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
    


    # Add this new method to work with source landmarks when processed image has no face:
    def _align_with_source_landmarks(self, source_img, processed_img, source_mask, processed_mask, 
                                source_landmarks, debug_dir=None):
        """
        Align processed image using only source landmarks when the processed image has no detectable face.
        This uses the source facial landmarks to guide the placement of the processed image's hair mask.
        
        Args:
            source_img: Original source image
            processed_img: Processed image to align (may not have a face)
            source_mask: Mask for source image
            processed_mask: Mask for processed image
            source_landmarks: Landmarks from the source image
            debug_dir: Directory to save debug visualizations
            
        Returns:
            tuple: (aligned_mask, aligned_image)
        """
        h, w = source_img.shape[:2]
        
        # Use the source mask and processed mask centroids for initial alignment
        if source_mask is not None and processed_mask is not None:
            # Calculate centroids
            source_moments = cv2.moments(source_mask)
            processed_moments = cv2.moments(processed_mask)
            
            # Initial translation based on mask centroids
            if source_moments["m00"] > 0 and processed_moments["m00"] > 0:
                source_cx = int(source_moments["m10"] / source_moments["m00"])
                source_cy = int(source_moments["m01"] / source_moments["m00"])
                processed_cx = int(processed_moments["m10"] / processed_moments["m00"])
                processed_cy = int(processed_moments["m01"] / processed_moments["m00"])
                
                # Calculate shift
                dx = source_cx - processed_cx
                dy = source_cy - processed_cy
            else:
                # Default to no shift if moments cannot be calculated
                dx, dy = 0, 0
        else:
            # Default to no shift if masks are not available
            dx, dy = 0, 0
        
        # Refine alignment using source face position
        # Get top of face region from landmarks to align hair position
        face_top = min([landmark[1] for landmark in source_landmarks[17:27]])  # Eyebrow region
        
        # Optional: Use hair position relative to face height
        # Assuming hair extends about 15% above eyebrows (adjust based on your data)
        face_height = source_landmarks[8][1] - face_top  # Distance from chin to eyebrows
        hair_offset = int(face_height * 0.15)  # Approximate hair extension above eyebrows
        
        # Create transformation parameters
        transformation_params = {
            'translation': (dx, dy)
        }
        
        # Apply the transformation
        aligned_img, aligned_mask = self._apply_transformation(
            processed_img, 
            processed_mask, 
            transformation_params,
            debug_dir=debug_dir,
            debug_prefix='source_landmark_guided'
        )
        
        # Ensure the aligned images have correct dimensions
        if aligned_img.shape[:2] != (h, w):
            aligned_img = cv2.resize(aligned_img, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        if aligned_mask.shape[:2] != (h, w):
            aligned_mask = cv2.resize(aligned_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Create debug visualization
        if debug_dir:
            # Visualize source landmarks
            vis_img = source_img.copy()
            for i, (x, y) in enumerate(source_landmarks):
                cv2.circle(vis_img, (int(x), int(y)), 2, (0, 255, 0), -1)
                if i in [17, 19, 24, 26]:  # Eyebrow points
                    cv2.circle(vis_img, (int(x), int(y)), 4, (255, 0, 0), -1)
            
            # Draw a line indicating the top of the hair region
            cv2.line(vis_img, (0, face_top - hair_offset), (w, face_top - hair_offset), (0, 0, 255), 2)
            
            # Save the visualization
            cv2.imwrite(os.path.join(debug_dir, "source_landmark_guide.png"), vis_img)
            
            # Create comparison of alignment
            comparison = np.hstack((source_img, aligned_img))
            cv2.imwrite(os.path.join(debug_dir, "source_guided_alignment.png"), comparison)
            
            # Visualize masks
            mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Ensure source_mask has correct dimensions before using it as an index
            if source_mask is not None:
                # Resize source_mask if dimensions don't match
                if source_mask.shape[:2] != (h, w):
                    source_mask_resized = cv2.resize(source_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_vis[source_mask_resized > 127] = [0, 0, 255]  # Source mask in red
                else:
                    mask_vis[source_mask > 127] = [0, 0, 255]  # Source mask in red
                    
            mask_vis[aligned_mask > 127] = [0, 255, 0]    # Aligned mask in green
            cv2.imwrite(os.path.join(debug_dir, "source_guided_masks.png"), mask_vis)
        
        return aligned_mask, aligned_img