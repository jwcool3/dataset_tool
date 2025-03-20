"""
Enhanced Mask Alignment Module for Crop Reinserter
Handles different masks between source and processed images with better alignment and blending.
"""

import os
import cv2
import numpy as np
from skimage.transform import resize as skimage_resize
from skimage.measure import label, regionprops
import dlib  # For facial landmark detection (if available)

class MaskAlignmentHandler:
    """Handles alignment and blending of different masks between source and processed images."""
    
    def __init__(self, debug_dir=None):
        """
        Initialize the mask alignment handler.
        
        Args:
            debug_dir: Directory to save debug visualizations (if enabled)
        """
        self.debug_dir = debug_dir
        self.face_detector = None
        self.landmark_predictor = None
        
        # Try to load facial landmark detector if available
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            # Check if the predictor file exists
            predictor_path = os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")
            if os.path.exists(predictor_path):
                self.landmark_predictor = dlib.shape_predictor(predictor_path)
                print("Facial landmark detector loaded successfully")
            else:
                print("Facial landmark predictor file not found")
        except Exception as e:
            print(f"Facial landmark detection not available: {str(e)}")
    
    def align_masks(self, source_img, processed_img, source_mask=None, processed_mask=None, 
                    alignment_method="centroid", blend_mode="alpha", blend_extent=5, 
                    preserve_original_edges=True):
        """
        Align and blend images using masks with various options.
        
        Args:
            source_img: Original source image
            processed_img: Processed image to insert
            source_mask: Binary mask for source image (optional)
            processed_mask: Binary mask for processed image (required)
            alignment_method: Method to align masks ('centroid', 'bbox', 'landmarks')
            blend_mode: Blending mode ('alpha', 'poisson', 'feathered')
            blend_extent: Extent of blending at edges (pixels)
            preserve_original_edges: Whether to preserve original image edges
            
        Returns:
            tuple: (aligned_result, aligned_mask)
        """
        # Ensure images are the same size
        if source_img.shape[:2] != processed_img.shape[:2]:
            processed_img = cv2.resize(processed_img, (source_img.shape[1], source_img.shape[0]), 
                                    interpolation=cv2.INTER_LANCZOS4)
        
        # Handle case when source mask isn't provided
        if source_mask is None:
            # Use processed mask as source mask
            source_mask = processed_mask.copy() if processed_mask is not None else None
        
        # Ensure masks are binary
        if processed_mask is not None:
            _, processed_mask = cv2.threshold(processed_mask, 127, 255, cv2.THRESH_BINARY)
        else:
            # Create a default mask covering the whole image if none provided
            processed_mask = np.ones(source_img.shape[:2], dtype=np.uint8) * 255
        
        if source_mask is not None and source_mask.shape[:2] != source_img.shape[:2]:
            source_mask = cv2.resize(source_mask, (source_img.shape[1], source_img.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
            _, source_mask = cv2.threshold(source_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Save debug images
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "source_img.png"), source_img)
            cv2.imwrite(os.path.join(self.debug_dir, "processed_img.png"), processed_img)
            cv2.imwrite(os.path.join(self.debug_dir, "source_mask.png"), source_mask)
            cv2.imwrite(os.path.join(self.debug_dir, "processed_mask.png"), processed_mask)
        
        # Calculate mask alignment
        aligned_mask = self._align_mask_by_method(source_mask, processed_mask, alignment_method)
        
        # Apply blending based on selected mode
        result_img = self._apply_blending(source_img, processed_img, aligned_mask, 
                                          blend_mode, blend_extent, preserve_original_edges)
        
        # Debug visualizations
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "aligned_mask.png"), aligned_mask)
            cv2.imwrite(os.path.join(self.debug_dir, "blended_result.png"), result_img)
            
            # Create visualization of mask alignment
            mask_overlay = np.zeros((source_img.shape[0], source_img.shape[1], 3), dtype=np.uint8)
            mask_overlay[source_mask > 0] = [0, 0, 255]  # Source mask in red
            mask_overlay[aligned_mask > 0] = [0, 255, 0]  # Aligned mask in green
            # Overlap appears as yellow
            cv2.imwrite(os.path.join(self.debug_dir, "mask_alignment_viz.png"), mask_overlay)
        
        return result_img, aligned_mask
    
    def _align_mask_by_method(self, source_mask, processed_mask, method):
        """
        Align the processed mask to better match the source mask.
        
        Args:
            source_mask: Binary mask for source image
            processed_mask: Binary mask for processed image
            method: Alignment method ('centroid', 'bbox', 'landmarks')
            
        Returns:
            numpy.ndarray: Aligned mask
        """
        aligned_mask = processed_mask.copy()
        
        if method == "none":
            # No alignment, just use the processed mask as is
            return aligned_mask
        
        if method == "centroid":
            # Align by matching centroids of the masks
            aligned_mask = self._align_by_centroid(source_mask, processed_mask)
        
        elif method == "bbox":
            # Align by matching bounding boxes
            aligned_mask = self._align_by_bbox(source_mask, processed_mask)
        
        elif method == "landmarks":
            # Align using facial landmarks
            aligned_mask = self._align_by_landmarks(source_mask, processed_mask)
        
        elif method == "contour":
            # Align by matching contours
            aligned_mask = self._align_by_contour(source_mask, processed_mask)
        
        elif method == "iou":
            # Align by maximizing Intersection over Union
            aligned_mask = self._align_by_iou(source_mask, processed_mask)
        
        # Apply morphological operations to smooth the mask
        aligned_mask = cv2.morphologyEx(aligned_mask, cv2.MORPH_CLOSE, 
                                      np.ones((5, 5), np.uint8))
        
        return aligned_mask
    
    def _align_by_centroid(self, source_mask, processed_mask):
        """Align masks by matching their centroids."""
        # Calculate centroids
        source_props = regionprops(label(source_mask > 0))
        processed_props = regionprops(label(processed_mask > 0))
        
        if not source_props or not processed_props:
            return processed_mask
        
        # Get centroids
        source_y, source_x = source_props[0].centroid
        processed_y, processed_x = processed_props[0].centroid
        
        # Calculate shift
        dy = int(source_y - processed_y)
        dx = int(source_x - processed_x)
        
        # Create aligned mask by shifting the processed mask
        aligned_mask = np.zeros_like(processed_mask)
        
        # Calculate bounds for shifted mask
        h, w = processed_mask.shape
        y1, y2 = max(0, dy), min(h, h + dy)
        x1, x2 = max(0, dx), min(w, w + dx)
        
        # Calculate corresponding regions in original mask
        y1_src, y2_src = max(0, -dy), min(h, h - dy)
        x1_src, x2_src = max(0, -dx), min(w, w - dx)
        
        # Copy the mask to the new position
        aligned_mask[y1:y2, x1:x2] = processed_mask[y1_src:y2_src, x1_src:x2_src]
        
        return aligned_mask
    
    def _align_by_bbox(self, source_mask, processed_mask):
        """Align masks by matching their bounding boxes."""
        # Find bounding boxes of the masks
        source_contours, _ = cv2.findContours(source_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        processed_contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not source_contours or not processed_contours:
            return processed_mask
        
        # Get largest contour by area
        source_contour = max(source_contours, key=cv2.contourArea)
        processed_contour = max(processed_contours, key=cv2.contourArea)
        
        # Get bounding boxes
        x_src, y_src, w_src, h_src = cv2.boundingRect(source_contour)
        x_proc, y_proc, w_proc, h_proc = cv2.boundingRect(processed_contour)
        
        # Calculate scaling factors to match bounding box size
        scale_x = w_src / w_proc if w_proc > 0 else 1
        scale_y = h_src / h_proc if h_proc > 0 else 1
        
        # Create aligned mask by transforming the processed mask
        aligned_mask = np.zeros_like(processed_mask)
        
        # Calculate target position
        target_x = x_src
        target_y = y_src
        
        # Resize the mask to match the source bounding box
        if scale_x != 1 or scale_y != 1:
            # Extract the mask region
            mask_roi = processed_mask[y_proc:y_proc+h_proc, x_proc:x_proc+w_proc]
            if mask_roi.size > 0:
                # Resize to match source bounding box
                resized_roi = cv2.resize(mask_roi, (w_src, h_src), interpolation=cv2.INTER_NEAREST)
                
                # Place in aligned mask
                try:
                    aligned_mask[y_src:y_src+h_src, x_src:x_src+w_src] = resized_roi
                except ValueError:
                    # Handle cases where the bounds might exceed image dimensions
                    h_place = min(h_src, aligned_mask.shape[0] - y_src)
                    w_place = min(w_src, aligned_mask.shape[1] - x_src)
                    aligned_mask[y_src:y_src+h_place, x_src:x_src+w_place] = resized_roi[:h_place, :w_place]
        else:
            # Just shift the mask
            dy = y_src - y_proc
            dx = x_src - x_proc
            
            # Calculate bounds for shifted mask
            h, w = processed_mask.shape
            y1, y2 = max(0, dy), min(h, h + dy)
            x1, x2 = max(0, dx), min(w, w + dx)
            
            # Calculate corresponding regions in original mask
            y1_src, y2_src = max(0, -dy), min(h, h - dy)
            x1_src, x2_src = max(0, -dx), min(w, w - dx)
            
            # Copy the mask to the new position
            aligned_mask[y1:y2, x1:x2] = processed_mask[y1_src:y2_src, x1_src:x2_src]
        
        return aligned_mask
    
    def _align_by_landmarks(self, source_mask, processed_mask):
        """Align masks using facial landmarks (if available)."""
        if self.face_detector is None or self.landmark_predictor is None:
            print("Facial landmark detection not available, falling back to bbox alignment")
            return self._align_by_bbox(source_mask, processed_mask)
        
        # For landmark detection, we need grayscale images
        # Since we only have masks, we'll use masks directly for now
        source_landmarks = self._get_landmarks_from_mask(source_mask)
        processed_landmarks = self._get_landmarks_from_mask(processed_mask)
        
        if source_landmarks is None or processed_landmarks is None:
            print("Landmarks not detected, falling back to bbox alignment")
            return self._align_by_bbox(source_mask, processed_mask)
        
        # TODO: Implement proper landmark alignment
        # For now, just align by top of head (highest point in mask)
        source_top = np.argwhere(source_mask > 0)[:, 0].min() if np.any(source_mask > 0) else 0
        processed_top = np.argwhere(processed_mask > 0)[:, 0].min() if np.any(processed_mask > 0) else 0
        
        # Calculate shift to align tops
        dy = source_top - processed_top
        
        # Create aligned mask by shifting vertically
        aligned_mask = np.zeros_like(processed_mask)
        
        # Calculate bounds for shifted mask
        h, w = processed_mask.shape
        y1, y2 = max(0, dy), min(h, h + dy)
        
        # Calculate corresponding regions in original mask
        y1_src, y2_src = max(0, -dy), min(h, h - dy)
        
        # Copy the mask to the new position
        aligned_mask[y1:y2, :] = processed_mask[y1_src:y2_src, :]
        
        return aligned_mask
    
    def _get_landmarks_from_mask(self, mask):
        """Attempt to extract landmarks from a mask."""
        # This is a placeholder for full landmark detection
        # In a real implementation, this would use face recognition on the actual images
        # For now, just return the contour points as "landmarks"
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour.reshape(-1, 2)
    
    def _align_by_contour(self, source_mask, processed_mask):
        """Align masks by matching their contours."""
        # Find contours
        source_contours, _ = cv2.findContours(source_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        processed_contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not source_contours or not processed_contours:
            return processed_mask
        
        # Get largest contours
        source_contour = max(source_contours, key=cv2.contourArea)
        processed_contour = max(processed_contours, key=cv2.contourArea)
        
        # Simplify contours to key points
        source_approx = cv2.approxPolyDP(source_contour, 0.02 * cv2.arcLength(source_contour, True), True)
        processed_approx = cv2.approxPolyDP(processed_contour, 0.02 * cv2.arcLength(processed_contour, True), True)
        
        # For simple alignment, just match the top points of contours
        source_top = tuple(source_contour[source_contour[:, :, 1].argmin()][0])
        processed_top = tuple(processed_contour[processed_contour[:, :, 1].argmin()][0])
        
        # Calculate shift
        dx = source_top[0] - processed_top[0]
        dy = source_top[1] - processed_top[1]
        
        # Create transformation matrix
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        
        # Apply the shift
        aligned_mask = cv2.warpAffine(processed_mask, M, (processed_mask.shape[1], processed_mask.shape[0]))
        
        return aligned_mask
    
    def _align_by_iou(self, source_mask, processed_mask):
        """Align masks by finding position that maximizes Intersection over Union."""
        best_iou = 0
        best_mask = processed_mask.copy()
        
        # Define search range for x and y shifts
        max_shift = 20  # pixels
        step = 2  # pixels
        
        for dy in range(-max_shift, max_shift + 1, step):
            for dx in range(-max_shift, max_shift + 1, step):
                # Create shifted mask
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted_mask = cv2.warpAffine(processed_mask, M, 
                                            (processed_mask.shape[1], processed_mask.shape[0]))
                
                # Calculate IoU
                intersection = np.logical_and(source_mask > 0, shifted_mask > 0).sum()
                union = np.logical_or(source_mask > 0, shifted_mask > 0).sum()
                iou = intersection / union if union > 0 else 0
                
                # Update best if improved
                if iou > best_iou:
                    best_iou = iou
                    best_mask = shifted_mask
        
        print(f"Best IOU: {best_iou:.4f}")
        return best_mask
    
    def _apply_blending(self, source_img, processed_img, mask, 
                      blend_mode="alpha", blend_extent=5, preserve_edges=True):
        """
        Apply blending between source and processed images.
        
        Args:
            source_img: Original source image
            processed_img: Processed image to insert
            mask: Binary mask for blending
            blend_mode: Blending mode ('alpha', 'poisson', 'feathered')
            blend_extent: Extent of blending at edges (pixels)
            preserve_edges: Whether to preserve original image edges
            
        Returns:
            numpy.ndarray: Blended result
        """
        # Start with a copy of the source image
        result = source_img.copy()
        
        if blend_mode == "alpha":
            # Simple alpha blending
            mask_float = mask.astype(float) / 255.0
            
            # Create feathered mask if requested
            if blend_extent > 0:
                # Dilate and erode to get border region
                kernel = np.ones((blend_extent, blend_extent), np.uint8)
                dilated = cv2.dilate(mask, kernel, iterations=1)
                border = dilated & ~mask  # Border pixels
                
                # Create distance map from border
                dist = cv2.distanceTransform(~border, cv2.DIST_L2, 3)
                dist[dist > blend_extent] = blend_extent
                
                # Normalize distances to create feathered mask
                feather = dist / blend_extent
                
                # Apply feathering to the mask
                mask_float = mask.astype(float) / 255.0
                
                # Where we have border pixels, apply feathering
                mask_float[border > 0] = 1.0 - feather[border > 0]
            
            # Expand to 3 channels for RGB
            mask_float_3d = np.stack([mask_float] * 3, axis=2)
            
            # Apply blending
            result = source_img * (1 - mask_float_3d) + processed_img * mask_float_3d
            
            # Convert back to uint8
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        elif blend_mode == "poisson":
            # Poisson blending for seamless integration
            try:
                # Convert mask to format needed by seamlessClone
                mask_uint8 = mask.astype(np.uint8)
                
                # Find center of mask
                moments = cv2.moments(mask_uint8)
                if moments["m00"] > 0:
                    center_x = int(moments["m10"] / moments["m00"])
                    center_y = int(moments["m01"] / moments["m00"])
                    center = (center_x, center_y)
                    
                    # Apply seamless cloning
                    result = cv2.seamlessClone(processed_img, source_img, mask_uint8, 
                                           center, cv2.NORMAL_CLONE)
                else:
                    print("Mask is empty, falling back to alpha blending")
                    # Fall back to alpha blending
                    mask_float = mask.astype(float) / 255.0
                    mask_float_3d = np.stack([mask_float] * 3, axis=2)
                    result = source_img * (1 - mask_float_3d) + processed_img * mask_float_3d
                    result = np.clip(result, 0, 255).astype(np.uint8)
            except Exception as e:
                print(f"Poisson blending failed: {str(e)}, falling back to alpha blending")
                # Fall back to alpha blending
                mask_float = mask.astype(float) / 255.0
                mask_float_3d = np.stack([mask_float] * 3, axis=2)
                result = source_img * (1 - mask_float_3d) + processed_img * mask_float_3d
                result = np.clip(result, 0, 255).astype(np.uint8)
        
        elif blend_mode == "feathered":
            # Feathered blending with specified extent
            # Create distance map from border
            mask_binary = (mask > 127).astype(np.uint8) * 255
            dist_inside = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 3)
            dist_outside = cv2.distanceTransform(255 - mask_binary, cv2.DIST_L2, 3)
            
            # Create alpha values based on distance
            alpha = np.ones_like(dist_inside)
            
            # Inside mask: fade from 1.0 at center to 0.5 at border
            fade_inside = np.clip(dist_inside / blend_extent, 0, 1)
            alpha = 0.5 + 0.5 * fade_inside
            
            # Outside mask: fade from 0.5 at border to 0.0 outside
            fade_outside = np.clip(1.0 - dist_outside / blend_extent, 0, 1)
            alpha = alpha * mask_binary / 255.0 + fade_outside * (1 - mask_binary / 255.0) * 0.5
            
            # Apply blending
            alpha_3d = np.stack([alpha] * 3, axis=2)
            result = source_img * (1 - alpha_3d) + processed_img * alpha_3d
            
            # Convert back to uint8
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Preserve original edges if requested
        if preserve_edges and blend_extent > 0:
            # Detect edges in source image
            gray_source = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_source, 50, 150)
            
            # Dilate edges to make them more prominent
            edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            
            # Only preserve edges outside the mask
            edge_mask = edge_mask & ~mask_binary if 'mask_binary' in locals() else edge_mask & ~mask
            
            # Convert edge mask to 3 channels
            edge_mask_3d = np.stack([edge_mask / 255.0] * 3, axis=2)
            
            # Keep original pixel values at edges
            result = source_img * edge_mask_3d + result * (1 - edge_mask_3d)
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result