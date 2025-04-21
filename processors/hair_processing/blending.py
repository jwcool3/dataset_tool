"""
Blending utilities for hair mask processing.
Includes different blending modes for combining source and processed images.
"""

import cv2
import numpy as np

def alpha_blend(source_img, processed_img, mask, blend_extent=8):
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

def poisson_blend(source_img, processed_img, mask):
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

def feathered_blend(source_img, processed_img, mask, blend_extent=5):
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
    
    # Add a slight blur to the alpha for smoother transitions
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    
    # Create 3-channel alpha
    alpha_3d = np.stack([alpha] * 3, axis=2)
    
    # Blend images
    result_img = source_img * (1 - alpha_3d) + processed_img * alpha_3d
    
    return np.clip(result_img, 0, 255).astype(np.uint8)

def preserve_image_edges(source_img, result_img, mask):
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

def blend_images(source_img, processed_img, mask, blend_params=None):
    """
    Apply the appropriate blending method based on parameters.
    
    Args:
        source_img: Original source image
        processed_img: Processed image to blend
        mask: Blending mask
        blend_params: Dictionary of blending parameters
        
    Returns:
        numpy.ndarray: Blended image
    """
    # Parse parameters with defaults
    if blend_params is None:
        blend_params = {}
        
    blend_mode = blend_params.get('mode', 'alpha')
    blend_extent = blend_params.get('extent', 5)
    preserve_edges = blend_params.get('preserve_edges', True)
    
    # Apply the selected blending method
    if blend_mode == 'poisson':
        result_img = poisson_blend(source_img, processed_img, mask)
    elif blend_mode == 'feathered':
        result_img = feathered_blend(source_img, processed_img, mask, blend_extent)
    else:  # Default to alpha
        result_img = alpha_blend(source_img, processed_img, mask, blend_extent)
    
    # Preserve edges if requested
    if preserve_edges:
        result_img = preserve_image_edges(source_img, result_img, mask)
    
    return result_img