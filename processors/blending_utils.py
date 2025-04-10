"""
Blending utilities for crop reinsertion.
Handles various blending methods between source and processed images/masks.
"""

import os
import cv2
import numpy as np

class BlendingUtils:
    """Provides image blending functionality for crop reinsertion."""
    
    def __init__(self, app):
        """
        Initialize blending utilities.
        
        Args:
            app: The main application with shared variables and UI controls
        """
        self.app = app
    
    def alpha_blend(self, source_img, processed_img, mask, blend_extent=8):
        """
        Enhanced alpha blending with better feathering, color correction, and harmonization.
        
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
        
        # Apply improved color harmonization
        # Step 1: Color correction at the boundary
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
        
        # Step 2: Global color harmonization - match color statistics
        # Extract color statistics from the source hair (if visible)
        source_hair_mask = np.zeros_like(mask_binary)
        
        # Try to find hair in source image outside the mask
        # This assumes areas outside the mask may still be hair
        try:
            # Convert to HSV for better hair detection
            source_hsv = cv2.cvtColor(source_img, cv2.COLOR_BGR2HSV)
            
            # Sample area just outside the mask where hair likely exists
            # Dilate the mask to get the surrounding area
            sample_area = cv2.dilate(mask_binary, np.ones((15, 15), np.uint8), iterations=1)
            sample_area = cv2.bitwise_xor(sample_area, mask_binary)
            
            # Check if sample area exists
            if np.sum(sample_area) > 0:
                # Extract hair color from the source using HSV color range typical for hair
                h, s, v = cv2.split(source_hsv)
                
                # Define hair-like HSV ranges (works for many hair colors)
                # This is a general heuristic that may need adaptation
                if np.sum(mask_binary) > 0:
                    # Use a more focused sampling of where we know hair is
                    # Find mean HSV values in the mask region in processed image
                    processed_hsv = cv2.cvtColor(processed_img.astype(np.uint8), cv2.COLOR_BGR2HSV)
                    
                    # Get mean HSV values in the processed hair region
                    processed_hair_h = processed_hsv[:,:,0][mask_binary > 0]
                    processed_hair_s = processed_hsv[:,:,1][mask_binary > 0]
                    processed_hair_v = processed_hsv[:,:,2][mask_binary > 0]
                    
                    if len(processed_hair_h) > 0:
                        mean_h = np.mean(processed_hair_h)
                        mean_s = np.mean(processed_hair_s)
                        mean_v = np.mean(processed_hair_v)
                        
                        # Create range around the mean values
                        h_range = 20  # Hue range (adjust as needed)
                        s_range = 50  # Saturation range
                        v_range = 50  # Value range
                        
                        h_min = max(0, mean_h - h_range)
                        h_max = min(180, mean_h + h_range)
                        s_min = max(0, mean_s - s_range)
                        s_max = min(255, mean_s + s_range)
                        v_min = max(0, mean_v - v_range)
                        v_max = min(255, mean_v + v_range)
                        
                        # Create mask for source hair based on these ranges
                        source_hair_mask = cv2.inRange(source_hsv, 
                                                   (h_min, s_min, v_min), 
                                                   (h_max, s_max, v_max))
                        
                        # Apply the sample area to limit to areas outside the bangs region
                        source_hair_mask = cv2.bitwise_and(source_hair_mask, sample_area)
                    else:
                        # Fallback to generic hair detection
                        source_hair_mask = sample_area.copy()
        except Exception as e:
            print(f"Error in hair color harmonization: {str(e)}")
            # Fallback - use the sample area directly
            source_hair_mask = sample_area.copy()
        
        # Apply color harmonization if we have enough hair pixels to sample
        if np.sum(source_hair_mask) > 100:  # Ensure we have enough pixels
            try:
                # Get source hair pixels
                source_hair_pixels = source_img[source_hair_mask > 0].reshape(-1, 3)
                
                # Get processed hair pixels
                processed_hair_pixels = processed_img[mask_binary > 0].reshape(-1, 3)
                
                if len(source_hair_pixels) > 0 and len(processed_hair_pixels) > 0:
                    # Calculate mean and std for both source and processed hair
                    source_mean = np.mean(source_hair_pixels, axis=0)
                    source_std = np.std(source_hair_pixels, axis=0)
                    
                    processed_mean = np.mean(processed_hair_pixels, axis=0)
                    processed_std = np.std(processed_hair_pixels, axis=0)
                    
                    # Create adjusted processed image with matched statistics
                    # This is a simplified color transfer
                    processed_adjusted = processed_img.astype(np.float32).copy()
                    
                    # Only apply to hair region
                    hair_region = (mask_float > 0.3)
                    
                    # Create 3-channel version of the hair region
                    hair_region_3d = np.stack([hair_region] * 3, axis=2)
                    
                    # Adjust each channel
                    for c in range(3):
                        # Skip if standard deviation is too small
                        if processed_std[c] < 1.0 or source_std[c] < 1.0:
                            continue
                            
                        # Apply color matching: scale = source_std / processed_std
                        scale = source_std[c] / processed_std[c]
                        
                        # Don't apply extreme scaling
                        scale = np.clip(scale, 0.5, 2.0)
                        
                        # Adjust the processed image: (x - mean) * scale + new_mean
                        channel = processed_adjusted[:,:,c]
                        adjusted = (channel - processed_mean[c]) * scale + source_mean[c]
                        
                        # Apply only to hair region with gradual transition
                        strength = 0.6  # Adjust strength of color transfer
                        channel[hair_region] = channel[hair_region] * (1 - strength) + adjusted[hair_region] * strength
                    
                    # Convert back to proper type
                    processed_img = processed_adjusted.astype(np.float32)
            except Exception as e:
                print(f"Error applying color harmonization: {str(e)}")
                # Continue with original processed image
        
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
    
    def poisson_blend(self, source_img, processed_img, mask):
        """
        Perform Poisson blending with improved error handling and boundary checks.
        
        Args:
            source_img: Original source image
            processed_img: Processed image to blend
            mask: Blending mask
            
        Returns:
            numpy.ndarray: Blended image
        """
        try:
            # Ensure mask is uint8 and matches image dimensions
            mask_uint8 = mask.astype(np.uint8)
            h, w = source_img.shape[:2]
            
            # Check if mask has any non-zero values
            if np.sum(mask_uint8) == 0:
                print("Mask is empty, falling back to alpha blending")
                raise ValueError("Empty mask")
                
            # Find center of mask with boundary protection
            moments = cv2.moments(mask_uint8)
            if moments["m00"] > 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
                
                # Ensure center is at least 1/4 of the image dimensions from any edge
                safe_margin = min(w, h) // 4
                center_x = max(safe_margin, min(w - safe_margin, center_x))
                center_y = max(safe_margin, min(h - safe_margin, center_y))
                
                center = (center_x, center_y)
                print(f"Using center point for Poisson blending: {center}")
                
                # Erode mask slightly to avoid boundary issues
                eroded_mask = cv2.erode(mask_uint8, np.ones((3, 3), np.uint8), iterations=1)
                
                # Ensure both images and mask are the same size
                if source_img.shape[:2] != processed_img.shape[:2] or source_img.shape[:2] != mask_uint8.shape[:2]:
                    print("Warning: Shape mismatch. Resizing processed image and mask to match source.")
                    processed_img = cv2.resize(processed_img, (w, h), interpolation=cv2.INTER_LANCZOS4)
                    eroded_mask = cv2.resize(eroded_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Apply seamless cloning with checked inputs
                if np.any(eroded_mask > 0):
                    # Debug visualization of the mask and center point
                    if self.app.debug_mode.get() and hasattr(self, 'debug_dir') and self.debug_dir:
                        debug_img = source_img.copy()
                        cv2.circle(debug_img, center, 5, (0, 0, 255), -1)  # Red dot at center
                        mask_viz = np.zeros_like(source_img)
                        mask_viz[eroded_mask > 0] = [0, 255, 0]  # Green for mask
                        blend_debug = cv2.addWeighted(debug_img, 0.7, mask_viz, 0.3, 0)
                        cv2.imwrite(os.path.join(self.debug_dir, "poisson_blend_debug.png"), blend_debug)
                    
                    result_img = cv2.seamlessClone(
                        processed_img, source_img, eroded_mask, center, cv2.NORMAL_CLONE
                    )
                else:
                    # Fallback to alpha blending
                    raise ValueError("Mask has no non-zero values after erosion")
            else:
                # Fallback to alpha blending
                raise ValueError("Mask moments are zero")
                
        except Exception as e:
            print(f"Poisson blending failed: {str(e)}")
            # Fall back to alpha blending
            mask_float = mask.astype(float) / 255.0
            mask_float_3d = np.stack([mask_float] * 3, axis=2)
            result_img = source_img * (1 - mask_float_3d) + processed_img * mask_float_3d
        
        return np.clip(result_img, 0, 255).astype(np.uint8)
    
    def feathered_blend(self, source_img, processed_img, mask, blend_extent=5):
        """
        Enhanced feathered blending with improved edge handling and transitions.
        
        Args:
            source_img: Original source image
            processed_img: Processed image to blend
            mask: Blending mask
            blend_extent: Extent of feathering
        
        Returns:
            numpy.ndarray: Blended image
        """
        # Convert mask to binary for processing
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Create distance transforms for inside and outside the mask
        dist_inside = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)
        dist_outside = cv2.distanceTransform(255 - binary_mask, cv2.DIST_L2, 3)
        
        # Normalize the distance transforms by the blend extent
        inside_blend_extent = blend_extent * 1.5  # Slightly larger extent inside the mask
        outside_blend_extent = blend_extent
        
        # Create alpha values based on distance with smoother transitions
        alpha = np.ones_like(dist_inside, dtype=float)
        
        # Apply smoothstep function for more natural transitions
        # smoothstep(x) = 3x² - 2x³ for x in [0,1], which gives a smoother S-curve
        def smoothstep(x):
            x = np.clip(x, 0, 1)
            return x * x * (3 - 2 * x)
        
        # Inside mask: fade from 1.0 at center to transition value at border
        inside_fade = np.clip(dist_inside / inside_blend_extent, 0, 1)
        # Apply smoothstep for more natural falloff
        inside_fade = smoothstep(inside_fade)
        # Map 0->0.5, 1->1.0
        inside_fade = 0.5 + 0.5 * inside_fade
        
        # Outside mask: fade from transition value at border to 0.0 outside
        outside_fade = np.clip(1.0 - dist_outside / outside_blend_extent, 0, 1)
        # Apply smoothstep for more natural falloff
        outside_fade = smoothstep(outside_fade)
        # Map 0->0.0, 1->0.5
        outside_fade = 0.5 * outside_fade
        
        # Combine the fades using the binary mask
        binary_mask_float = binary_mask.astype(float) / 255.0
        alpha = inside_fade * binary_mask_float + outside_fade * (1.0 - binary_mask_float)
        
        # For hair, we can enhance the blending by adding noise to the transition
        # This helps simulate the natural randomness of hair strands
        if self.app.use_bangs_only.get():
            # Create a noise pattern
            np.random.seed(42)  # For reproducibility
            noise = np.random.normal(0, 0.05, alpha.shape).astype(np.float32)
            
            # Only apply noise to the transition region
            transition_region = (alpha > 0.1) & (alpha < 0.9)
            alpha[transition_region] += noise[transition_region]
            alpha = np.clip(alpha, 0, 1)
        
        # Apply a slight blur to the alpha for smoother transitions
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        
        # Create 3-channel alpha
        alpha_3d = np.stack([alpha] * 3, axis=2)
        
        # Apply color matching at boundaries before blending
        # Create a narrow border region for color matching
        border_mask = np.zeros_like(binary_mask)
        border = cv2.dilate(binary_mask, np.ones((3, 3), np.uint8)) - binary_mask
        border_mask[border > 0] = 255
        
        # If we have source color and processed color at the boundary, adjust the processed color
        if np.sum(border_mask) > 0:
            try:
                # Convert to LAB color space for better color matching
                source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)
                processed_lab = cv2.cvtColor(processed_img.astype(np.uint8), cv2.COLOR_BGR2LAB)
                
                # Sample the colors at the boundary
                source_border = source_lab[border_mask > 0]
                processed_border = processed_lab[border_mask > 0]
                
                if len(source_border) > 0 and len(processed_border) > 0:
                    # Calculate mean color difference
                    source_mean = np.mean(source_border, axis=0)
                    processed_mean = np.mean(processed_border, axis=0)
                    color_diff = source_mean - processed_mean
                    
                    # Apply a gradual color correction
                    # Create a gradient mask where correction is strongest at border and fades out
                    correction_mask = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)
                    correction_mask = np.clip(1.0 - correction_mask / (blend_extent * 2), 0, 1)
                    
                    # Apply the correction to the LAB image
                    for i in range(3):
                        processed_lab[:,:,i] = np.clip(
                            processed_lab[:,:,i] + color_diff[i] * correction_mask * 0.7,  # 70% strength
                            0, 255 if i == 0 else 255
                        )
                    
                    # Convert back to BGR
                    processed_img = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2BGR)
            except Exception as e:
                print(f"Error in color matching: {str(e)}")
        
        # Apply edge-aware blending for better hair strand preservation
        source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY) if len(source_img.shape) == 3 else source_img
        processed_gray = cv2.cvtColor(processed_img.astype(np.uint8), cv2.COLOR_BGR2GRAY) if len(processed_img.shape) == 3 else processed_img
        
        # Detect edges in both images
        source_edges = cv2.Canny(source_gray, 50, 150)
        processed_edges = cv2.Canny(processed_gray, 50, 150)
        
        # Create an edge-aware alpha mask that preserves hair detail
        edge_mask = np.zeros_like(alpha)
        # Strengthen alpha where processed image has edges (hair strands)
        edge_mask[processed_edges > 0] = 0.2  # Boost by 20%
        # Weaken alpha where source image has strong edges we want to preserve
        edge_mask[source_edges > 0] = -0.1  # Reduce by 10%
        
        # Apply the edge mask to the transition region
        transition_region = (alpha > 0.2) & (alpha < 0.8)
        alpha[transition_region] += edge_mask[transition_region]
        alpha = np.clip(alpha, 0, 1)
        
        # Final smoothing of the alpha mask
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        
        # Update 3D alpha with the enhanced version
        alpha_3d = np.stack([alpha] * 3, axis=2)
        
        # Blend images
        result_img = source_img * (1 - alpha_3d) + processed_img * alpha_3d
        
        return np.clip(result_img, 0, 255).astype(np.uint8)
    
    def improved_hybrid_blend(self, source_img, processed_img, mask, blend_extent=5, debug_dir=None):
        """
        Enhanced hybrid blending method that uses Poisson blending for better 
        positioning and natural integration with the source image.
        
        Args:
            source_img: Original source image
            processed_img: Processed image to insert (the bangs) - already aligned
            mask: Blending mask (already aligned)
            blend_extent: Blend extent from settings
            debug_dir: Directory to save debug visualizations
        
        Returns:
            numpy.ndarray: Blended result
        """
        # Ensure mask is in the correct format and preserve the original mask values
        # DON'T convert to binary - this is the key change to maintain positioning consistency
        mask_uint8 = mask.astype(np.uint8)
        
        # Create a debug visualization if enabled
        if debug_dir:
            mask_viz = np.zeros_like(source_img)
            mask_viz[mask_uint8 > 0] = [0, 255, 0]  # Green for mask
            mask_overlay = cv2.addWeighted(source_img, 0.7, mask_viz, 0.3, 0)
            cv2.imwrite(os.path.join(debug_dir, "hybrid_mask_overlay.png"), mask_overlay)
            
            # Save the input images for reference
            cv2.imwrite(os.path.join(debug_dir, "hybrid_source_img.png"), source_img)
            cv2.imwrite(os.path.join(debug_dir, "hybrid_processed_img.png"), processed_img)
            cv2.imwrite(os.path.join(debug_dir, "hybrid_mask.png"), mask_uint8)
        
        # Start with a basic alpha blend as a fallback
        feathered_mask = mask_uint8.copy().astype(np.float32) / 255.0
        
        # Apply a slight blur for smoother edges
        if blend_extent > 0:
            kernel_size = max(3, min(blend_extent, 7))
            feathered_mask = cv2.GaussianBlur(feathered_mask, (kernel_size, kernel_size), 0)
        
        # Create a 3-channel mask for blending
        feathered_mask_3d = np.stack([feathered_mask] * 3, axis=2)
        
        # Initialize result with a simple alpha blend
        result_img = source_img * (1 - feathered_mask_3d) + processed_img * feathered_mask_3d
        
        # Apply Poisson blending - this is now the primary method
        if np.sum(mask_uint8) > 100:  # Make sure we have enough pixels
            try:
                # Erode mask slightly to avoid boundary issues - exactly like in the Poisson blend method
                eroded_mask = cv2.erode(mask_uint8, np.ones((3, 3), np.uint8), iterations=1)
                
                # Save the eroded mask if debugging is enabled
                if debug_dir:
                    eroded_viz = np.zeros_like(source_img)
                    eroded_viz[eroded_mask > 0] = [0, 255, 0]  # Green for eroded mask
                    cv2.imwrite(os.path.join(debug_dir, "hybrid_eroded_mask.png"), eroded_mask)
                    cv2.imwrite(os.path.join(debug_dir, "hybrid_eroded_viz.png"), eroded_viz)
                
                # Find center of mask with boundary protection - exactly like in Poisson blend
                moments = cv2.moments(mask_uint8)  # Use the non-eroded mask for consistent positioning with Poisson
                if moments["m00"] > 0:
                    center_x = int(moments["m10"] / moments["m00"])
                    center_y = int(moments["m01"] / moments["m00"])
                    
                    # Ensure center is within safe boundaries - same as Poisson blend
                    h, w = source_img.shape[:2]
                    min_distance = min(w, h) // 4
                    center_x = max(min_distance, min(w - min_distance, center_x))
                    center_y = max(min_distance, min(h - min_distance, center_y))
                    center = (center_x, center_y)
                    
                    print(f"Using center point for Hybrid blending: {center}")
                    
                    # Use eroded_mask for seamless cloning just like the Poisson blend
                    poisson_result = cv2.seamlessClone(
                        processed_img, source_img, eroded_mask, center, cv2.NORMAL_CLONE
                    )
                    
                    # Save the poisson result for debugging
                    if debug_dir:
                        debug_img = source_img.copy()
                        cv2.circle(debug_img, center, 5, (0, 0, 255), -1)  # Red dot at center
                        mask_viz = np.zeros_like(source_img)
                        mask_viz[eroded_mask > 0] = [0, 255, 0]  # Green for mask
                        blend_debug = cv2.addWeighted(debug_img, 0.7, mask_viz, 0.3, 0)
                        cv2.imwrite(os.path.join(debug_dir, "hybrid_poisson_debug.png"), blend_debug)
                        cv2.imwrite(os.path.join(debug_dir, "hybrid_poisson_result.png"), poisson_result)
                    
                    # Use the Poisson result directly as our final result
                    result_img = poisson_result.astype(np.float32)
                    
                    # Create an outer edge mask for a subtle blending with the original
                    # Use the original mask_uint8 (not eroded) for the outer edge calculation
                    outer_edge = cv2.dilate(mask_uint8, np.ones((3, 3), np.uint8), iterations=1) - mask_uint8
                    if np.any(outer_edge):
                        outer_edge_float = outer_edge.astype(np.float32) / 255.0
                        outer_edge_float = cv2.GaussianBlur(outer_edge_float, (3, 3), 0)
                        outer_edge_3d = np.stack([outer_edge_float] * 3, axis=2)
                        
                        # Apply a very subtle blend at the outer edge only
                        result_img = result_img * (1 - outer_edge_3d * 0.2) + source_img * (outer_edge_3d * 0.2)
                else:
                    # Fallback to alpha blending if moments are zero
                    raise ValueError("Mask moments are zero")
            except Exception as e:
                print(f"Poisson blending failed: {str(e)}, using alpha blend")
        
        # Add subtle detail enhancement to make hair strands more visible
        if self.app.use_bangs_only.get():
            # Focus enhancement only on the bangs area
            hair_area = mask_uint8 > 0
            hair_area_3d = np.stack([hair_area] * 3, axis=2)
            
            # Create a detail layer using unsharp mask
            result_uint8 = np.clip(result_img, 0, 255).astype(np.uint8)
            blurred = cv2.GaussianBlur(result_uint8, (0, 0), 1.5)
            detail_layer = cv2.addWeighted(result_uint8, 1.5, blurred, -0.5, 0)
            
            # Apply detail enhancement only to hair area with a mask
            result_img = np.where(
                hair_area_3d, 
                cv2.addWeighted(result_uint8, 0.8, detail_layer, 0.2, 0).astype(np.float32),
                result_img
            )
        
        # Save the final result for debugging
        if debug_dir:
            final_result = np.clip(result_img, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(debug_dir, "hybrid_final_result.png"), final_result)
        
        return np.clip(result_img, 0, 255).astype(np.uint8)
    
    def preserve_image_edges(self, source_img, result_img, mask):
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