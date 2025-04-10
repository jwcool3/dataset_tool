"""
Alignment utilities for crop reinsertion.
Handles various alignment methods between source and processed images/masks.
"""

import os
import cv2
import numpy as np
import dlib

class AlignmentUtils:
    """Provides mask and image alignment functionality for crop reinsertion."""
    
    def __init__(self, app):
        """
        Initialize alignment utilities.
        
        Args:
            app: The main application with shared variables and UI controls
        """
        self.app = app
        
        # Initialize face detection if dlib is available
        self.face_detector = None
        self.landmark_predictor = None
        
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            
            # Look for the shape predictor file in a few common locations
            predictor_paths = [
                os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "shape_predictor_68_face_landmarks.dat"),
                os.path.join(os.path.expanduser("~"), ".dataset_preparation_tool", "models", "shape_predictor_68_face_landmarks.dat")
            ]
            
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
    
    def get_landmarks(self, image):
        """
        Enhanced face detection with better fallbacks and diagnostics.
        
        Args:
            image: Image to detect landmarks in
            
        Returns:
            list: Detected landmarks as (x, y) coordinates, or None if detection failed
        """
        if self.face_detector is None or self.landmark_predictor is None:
            print("Facial landmark detection not available")
            return None
            
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Print some diagnostics about the image
        print(f"Image shape for face detection: {image.shape}")
        print(f"Image intensity range: {np.min(gray)} to {np.max(gray)}")
        
        # Try detection with different upsample factors
        for upsample_factor in [0, 1, 2]:  # Try no upsampling, then 1x, then 2x
            print(f"Attempting face detection with upsample factor {upsample_factor}")
            faces = self.face_detector(gray, upsample_factor)
            
            if faces:
                print(f"Detected {len(faces)} faces with upsample factor {upsample_factor}")
                # Use largest face by area
                largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
                
                # Get landmarks
                try:
                    shape = self.landmark_predictor(gray, largest_face)
                    landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
                    print(f"Successfully extracted {len(landmarks)} landmarks")
                    return landmarks
                except Exception as e:
                    print(f"Error detecting landmarks: {str(e)}")
                    continue  # Try next upsample factor if landmarks failed
        
        # If we reach here, no detection succeeded
        print("No faces detected with any upsampling, using geometric fallback")
        
        # Create a full set of 68 estimated facial landmarks based on image geometry
        h, w = image.shape[:2]
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
        
        print("Using fallback geometric landmarks")
        return landmarks
    
    def align_masks(self, source_mask, processed_mask, source_img, processed_img, alignment_method, debug_dir=None):
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
        
        # Binary threshold cleaned masks
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
    
    def improve_landmark_alignment(self, source_landmarks, processed_landmarks):
        """
        Calculate a robust transformation matrix based on facial landmarks
        with better error handling for incomplete landmarks.
        
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
    
    def apply_landmark_transform(self, source_img, processed_img, source_mask, processed_mask, 
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
        transform_matrix, success = self.improve_landmark_alignment(source_landmarks, processed_landmarks)
        
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
        rotation_angle = getattr(self.app, 'reinsert_manual_rotation', lambda: 0.0)()
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
    
    def apply_translation_only_transform(self, source_img, processed_img, source_mask, processed_mask, 
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
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_TRANSPARENT
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
    
    def align_with_landmarks(self, source_img, processed_img, source_mask, processed_mask, 
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
        # Validate input
        if source_landmarks is None or processed_landmarks is None:
            print("Could not detect landmarks in one or both images")
            return processed_mask, processed_img
        
        # Check if we should use translation-only or full alignment
        use_translation_only = self.app.use_translation_only.get()
        
        if use_translation_only:
            # Use the translation-only method
            aligned_mask, aligned_img = self.apply_translation_only_transform(
                source_img, processed_img, source_mask, processed_mask,
                source_landmarks, processed_landmarks, debug_dir
            )
            
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "aligned_translation_only.png"), aligned_img)
        else:
            # Use the full alignment method for comparison
            aligned_mask, aligned_img = self.apply_landmark_transform(
                source_img, processed_img, source_mask, processed_mask,
                source_landmarks, processed_landmarks, debug_dir
            )
            
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "aligned_full_transform.png"), aligned_img)
        
        return aligned_mask, aligned_img