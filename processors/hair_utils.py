"""
Hair-specific utilities for crop reinsertion.
Handles bangs detection, extension, and manipulation.
"""

import os
import cv2
import numpy as np
import dlib

class HairUtils:
    """Provides hair-specific functionality for crop reinsertion."""
    
    def __init__(self, app):
        """
        Initialize hair utilities.
        
        Args:
            app: The main application with shared variables and UI controls
        """
        self.app = app
    
    def isolate_bangs_region(self, mask, landmarks=None):
        """
        Improved bangs isolation with better hairline curve detection.
        
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
        
        # Find non-zero points in the mask to determine where the hair is
        non_zero_points = np.argwhere(mask > 0)
        if len(non_zero_points) == 0:
            print("Warning: Empty mask, nothing to isolate for bangs")
            return bangs_mask
        
        # Find the top-most point of the hair
        top_y = np.min(non_zero_points[:, 0])
        
        # Adjust top portion percentage based on the presence of landmarks
        if landmarks is not None and len(landmarks) >= 27:
            # Use more of the mask when we have landmarks for better guidance
            top_percent = 0.4  # Take top 40% of the mask - increased from 30%
            
            # Use eyebrow position to help determine how much to isolate
            eyebrow_points = landmarks[17:27]
            if len(eyebrow_points) > 0:
                eyebrow_y = min([p[1] for p in eyebrow_points])
                # If eyebrows are low in the image, take more of the mask
                eyebrow_ratio = eyebrow_y / height
                if eyebrow_ratio > 0.3:  # Eyebrows are lower down
                    top_percent = 0.5  # Take even more of the mask
                print(f"Using eyebrow position ratio {eyebrow_ratio:.2f} to set top_percent to {top_percent:.2f}")
        else:
            # Default to 35% when we don't have landmarks - increased from 25%
            top_percent = 0.35
        
        print(f"Isolating top {top_percent*100:.1f}% of mask as bangs region")
        bangs_bottom = int(height * top_percent)
        
        # Create a more natural curved bottom edge if landmarks are available
        if landmarks is not None and len(landmarks) >= 27:
            # Use eyebrow landmarks to create a curved bottom line
            eyebrow_points = landmarks[17:27]  # Eyebrow landmarks
            
            try:
                # Get eyebrow y-positions
                eyebrow_ys = [p[1] for p in eyebrow_points]
                eyebrow_top = max(0, int(min(eyebrow_ys)) - 10)  # Slight padding above eyebrows
                
                # Calculate a more natural curved boundary using the eyebrow shape
                x_points = np.arange(width)
                y_points = np.ones(width, dtype=int) * (top_y + bangs_bottom)
                
                # Form a curved bottom edge based on eyebrow position
                if len(eyebrow_points) >= 5:
                    # Get x positions of eyebrow points
                    eyebrow_xs = [int(p[0]) for p in eyebrow_points]
                    eyebrow_ys = [int(p[1]) for p in eyebrow_points]
                    
                    # Ensure we have enough points for interpolation
                    if len(eyebrow_xs) >= 3:
                        # Create a quadratic curve following eyebrow shape
                        from scipy.interpolate import interp1d
                        
                        # Sort points by x coordinate for proper interpolation
                        sorted_indices = np.argsort(eyebrow_xs)
                        sorted_xs = np.array(eyebrow_xs)[sorted_indices]
                        sorted_ys = np.array(eyebrow_ys)[sorted_indices]
                        
                        # Create interpolation function (use quadratic if possible)
                        try:
                            if len(sorted_xs) >= 3:
                                f = interp1d(sorted_xs, sorted_ys, kind='quadratic', 
                                            bounds_error=False, fill_value='extrapolate')
                            else:
                                f = interp1d(sorted_xs, sorted_ys, kind='linear', 
                                            bounds_error=False, fill_value='extrapolate')
                            
                            # Generate points along the curve
                            for x in range(width):
                                if x >= 0 and x < width:
                                    # Calculate y position based on interpolated curve
                                    # Adjust to be above eyebrows
                                    curve_y = int(f(x)) - 15  # 15 pixels above eyebrows
                                    
                                    # Ensure it's within reasonable bounds 
                                    # (not too high or too low)
                                    max_y = top_y + bangs_bottom
                                    curve_y = min(max_y, max(top_y, curve_y))
                                    
                                    # Update the y-position for this x-coordinate
                                    y_points[x] = curve_y
                        except Exception as e:
                            print(f"Error creating curved boundary: {str(e)}")
                            # Continue with default approach
                
                # Apply the calculated boundary
                for x in range(width):
                    # Get the boundary y-position for this column
                    boundary_y = y_points[x]
                    
                    # Apply the mask from top to boundary
                    if x < width and top_y < boundary_y and boundary_y < height:
                        bangs_mask[top_y:boundary_y, x] = mask[top_y:boundary_y, x]
            
            except Exception as e:
                print(f"Error creating curved bangs boundary: {str(e)}")
                # Fall back to original approach
                bangs_region = mask[top_y:min(top_y + bangs_bottom, height), :]
                bangs_mask[top_y:min(top_y + bangs_bottom, height), :] = bangs_region
        else:
            # Original approach without landmarks - straight cutoff
            bangs_region = mask[top_y:min(top_y + bangs_bottom, height), :]
            bangs_mask[top_y:min(top_y + bangs_bottom, height), :] = bangs_region
        
        # Add a gradient for smoother transition
        gradient_height = int(bangs_bottom * 0.4)  # Bottom 40% of bangs has gradient
        gradient_start = min(top_y + bangs_bottom - gradient_height, height)
        
        for y in range(gradient_start, min(top_y + bangs_bottom, height)):
            fade_ratio = 1.0 - ((y - gradient_start) / float(max(1, min(top_y + bangs_bottom, height) - gradient_start)))
            fade_ratio = fade_ratio ** 0.8  # Adjust power for more gradual falloff
            if y < bangs_mask.shape[0]:  # Safety check
                bangs_mask[y, :] = (bangs_mask[y, :].astype(float) * fade_ratio).astype(np.uint8)
        
        # Apply slight Gaussian blur for smoother edges
        bangs_mask = cv2.GaussianBlur(bangs_mask, (5, 5), 0)
        
        return bangs_mask
    
    def extend_bangs_area(self, mask, extend_pixels=30, forehead_ratio=0.3, min_opacity=0.7, source_landmarks=None):
        """
        Extends the isolated bangs mask directly downward following the mask's contours.
        
        Args:
            mask: The isolated bangs mask to extend
            extend_pixels: How many pixels to extend downward
            forehead_ratio: Not used, kept for backward compatibility 
            min_opacity: Minimum opacity at the furthest point of extension
            source_landmarks: Optional landmarks for face-aware extension
            
        Returns:
            numpy.ndarray: Extended bangs mask
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
        
        print(f"Extending bangs mask downward by {extend_pixels} pixels")
        
        # Keep a copy of the original isolated bangs mask
        original_mask = mask.copy()
        height, width = mask.shape[:2]
        
        # Create the output mask - initialize with the original mask
        extended_mask = original_mask.copy()
        
        # If using landmarks and they're available, adjust extension amount
        if source_landmarks is not None and len(source_landmarks) >= 27:
            try:
                # Get eyebrow positions for scaling
                eyebrow_points = source_landmarks[17:27]
                eyebrow_y = min([int(p[1]) for p in eyebrow_points])
                face_height = source_landmarks[8][1] - eyebrow_y  # Chin to eyebrows
                
                # Scale extension based on face proportions
                extend_factor = max(1.0, min(2.0, face_height / 180))
                adjusted_extend = int(extend_pixels * extend_factor)
                
                print(f"Face height: {face_height}, adjusting extension from {extend_pixels} to {adjusted_extend}")
                extend_pixels = adjusted_extend
            except Exception as e:
                print(f"Error adjusting extension with landmarks: {str(e)}")
        
        # IMPORTANT - Make sure extend_pixels is a reasonable value to avoid tiny extensions
        extend_pixels = max(10, extend_pixels)  # Ensure minimum extension of 10 pixels
        print(f"Final extension amount: {extend_pixels} pixels")
        
        # Calculate gradient falloff power based on extension amount
        # For larger extensions, use a gentler falloff to make the extension more visible
        if extend_pixels >= 70:
            falloff_power = 0.3  # Very gentle falloff for very large extensions
        elif extend_pixels >= 50:
            falloff_power = 0.4  # Gentle falloff for large extensions
        elif extend_pixels >= 30:
            falloff_power = 0.5  # Medium falloff for medium extensions
        else:
            falloff_power = 0.7  # Steeper falloff for small extensions
            
        print(f"Using falloff power: {falloff_power} for extension amount: {extend_pixels}")
        
        # For very large extensions, also allow a small upward extension to better blend with hair
        allow_upward = (extend_pixels >= 60)
        upward_pixels = min(10, extend_pixels // 10) if allow_upward else 0
        if upward_pixels > 0:
            print(f"Allowing {upward_pixels} pixels of upward extension for smoother blending")
        
        # Find bottom boundary of mask for each column
        bottom_boundary = np.zeros(width, dtype=int)
        for x in range(width):
            column_pixels = np.where(original_mask[:, x] > 0)[0]
            if len(column_pixels) > 0:
                bottom_boundary[x] = np.max(column_pixels)
        
        # Initialize top_boundary regardless of whether we're doing upward extension
        top_boundary = np.ones(width, dtype=int) * height
        
        # If we're doing upward extension, also find the top boundary of the mask
        if upward_pixels > 0:
            for x in range(width):
                    column_pixels = np.where(original_mask[:, x] > 0)[0]
                    if len(column_pixels) > 0:
                        top_boundary[x] = np.min(column_pixels)
        
        # For each column in the mask:
        pixels_extended = 0
        for x in range(width):
            # Find the bottom edge of the mask in this column
            bottom_y = bottom_boundary[x]
            
            # If this column has no mask pixels, skip it
            if bottom_y <= 0:
                continue
            
            # Get the original value at the bottom edge
            original_value = original_mask[bottom_y, x]
            
            # Skip if value is too low (nearly transparent)
            if original_value < 20:
                continue
                
            # Calculate how far to extend downward - use the full extend_pixels value
            extension_amount = min(extend_pixels, height - bottom_y - 1)
            if extension_amount <= 0:
                continue
                
            # Apply the extension with gradient falloff
            for y_offset in range(1, extension_amount + 1):
                y = bottom_y + y_offset
                
                # Skip if we'd go beyond the image
                if y >= height:
                        break
            
                # Calculate falloff ratio (1.0 at top, decreasing downward)
                # Use non-linear falloff for more natural appearance
                # The falloff_power controls the rate of fading - smaller = slower fade
                falloff = 1.0 - (y_offset / extension_amount) ** falloff_power
                
                # Calculate opacity based on original value and falloff
                # Preserve the original opacity pattern but fade out
                opacity = int(original_value * max(min_opacity, falloff))
                
                # Add slight noise for natural appearance
                if self.app.use_bangs_only.get():
                    noise_factor = 0.05  # 5% noise
                    noise = (((x * 17 + y * 31) % 10) - 5) / 100.0
                    opacity = int(np.clip(opacity * (1.0 + noise * noise_factor), 0, 255))
                
                # Set the pixel in the extended mask and count the extension
                if opacity > 0:
                    extended_mask[y, x] = opacity
                    pixels_extended += 1
            
            # Handle upward extension if enabled
            if upward_pixels > 0:
                top_y = top_boundary[x]
                # Skip if top is already at the top of the image
                if top_y <= 0 or top_y >= height:
                    continue
                
                # Get the value at the top edge
                top_value = original_mask[top_y, x]
                if top_value < 20:
                    continue
                    
                # Calculate how far to extend upward
                upward_amount = min(upward_pixels, top_y)
                if upward_amount <= 0:
                    continue
                    
                # Apply the upward extension with gradient falloff
                for y_offset in range(1, upward_amount + 1):
                    y = top_y - y_offset
                    
                    # Skip if we'd go beyond the image
                    if y < 0:
                        break
                    
                    # More aggressive falloff for upward extension
                    falloff = 1.0 - (y_offset / upward_amount) ** 0.9
                    
                    # Reduce opacity more for upward extension
                    opacity = int(top_value * falloff * 0.7)
                    
                    # Add slight noise
                    if self.app.use_bangs_only.get():
                        noise = (((x * 17 + y * 31) % 10) - 5) / 100.0
                        opacity = int(np.clip(opacity * (1.0 + noise * 0.03), 0, 255))
                    
                    # Set the pixel and count the extension
                    if opacity > 0:
                        extended_mask[y, x] = opacity
                        pixels_extended += 1
        
        print(f"Extended {pixels_extended} pixels downward from the original mask")
        
        # Apply face protection if enabled
        if self.app.protect_face_from_bangs.get() and source_landmarks is not None:
            protection_strength = self.app.face_protection_strength.get()
            extended_mask = self.create_face_protection_mask(extended_mask, source_landmarks, protection_strength)
        
        # Smooth the result slightly for natural appearance
        return cv2.GaussianBlur(extended_mask, (3, 3), 0)
    
    def create_face_protection_mask(self, mask, landmarks, protection_strength):
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
    
    def estimate_bangs_position(self, landmarks, height, width):
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
    
    def detect_hair_parting(self, mask, landmarks=None, custom_parting=None):
        """
        Detect and enhance the hair parting line in the mask.
        
        Args:
            mask: Hair mask image
            landmarks: Optional facial landmarks for guidance
            custom_parting: Optional custom parting mask to use instead of detection
            
        Returns:
            tuple: (parting_line, enhanced_mask)
        """
        # Create copy of mask
        enhanced_mask = mask.copy()
        
        # If a custom parting is provided, use it directly
        if custom_parting is not None:
            if custom_parting.shape != mask.shape:
                custom_parting = cv2.resize(custom_parting, (mask.shape[1], mask.shape[0]), 
                                          interpolation=cv2.INTER_LINEAR)
            return custom_parting, enhanced_mask
        
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
    
    def create_hairline_curve(self, width, height, extend_pixels, forehead_left, forehead_right, center_x, top_y):
        """
        Creates a simple downward-only extension curve with natural falloff.
        
        Note: This is maintained for backward compatibility but the main extension
        is now performed directly within extend_bangs_area.
        
        Args:
            width: Image width
            height: Image height
            extend_pixels: Pixels to extend downward
            forehead_left: Left boundary of forehead
            forehead_right: Right boundary of forehead
            center_x: Center X position
            top_y: Top Y position of extension
            
        Returns:
            numpy.ndarray: Created extension curve
        """
        curve = np.zeros((height, width), dtype=np.uint8)
        
        # Enhanced extension height
        extension_height = int(extend_pixels * 1.5)
        
        # Width of the area to extend
        area_width = forehead_right - forehead_left
        
        # For each column in the target area
        for x in range(max(0, forehead_left), min(width, forehead_right + 1)):
            # Calculate normalized horizontal distance from center
            x_dist = 2.0 * abs(x - center_x) / area_width if area_width > 0 else 0
            
            # Calculate how far down to extend based on distance from center
            # The center extends farthest, edges extend less
            local_extend = extension_height * (1.0 - 0.4 * x_dist**2)
            
            # Fill the extension with a gradient
            for y in range(top_y, min(height, int(top_y + local_extend))):
                # Calculate normalized position in the extension
                y_norm = (y - top_y) / local_extend if local_extend > 0 else 1.0
                
                # Apply falloff based on vertical position
                opacity = int(255 * (1.0 - y_norm**1.2))
                
                # Add some noise for more natural look
                noise = (((x * 13 + y * 29) % 10) - 5) / 150.0
                opacity = int(max(0, min(255, opacity + opacity * noise)))
                
                curve[y, x] = opacity
            
        # Apply a slight blur for smoother transitions
        curve = cv2.GaussianBlur(curve, (5, 5), 0)
        
        return curve