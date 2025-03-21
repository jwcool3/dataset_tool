"""
Hair Processing Utilities for Dataset Preparation Tool
Specialized utilities for processing and blending hair regions in images.
"""

import cv2
import numpy as np
from skimage import color, exposure

def match_hair_color(source_img, processed_img, source_mask, processed_mask, strength=0.5):
    """
    Match the color characteristics of processed hair to better blend with source hair.
    
    Args:
        source_img: Original source image
        processed_img: Processed image with modified hair
        source_mask: Mask for source hair region
        processed_mask: Mask for processed hair region
        strength: Strength of color correction (0.0-1.0)
        
    Returns:
        numpy.ndarray: Color-corrected processed image
    """
    # Ensure we have binary masks
    _, source_mask_bin = cv2.threshold(source_mask, 127, 255, cv2.THRESH_BINARY)
    _, processed_mask_bin = cv2.threshold(processed_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Check if masks have content
    if np.sum(source_mask_bin) == 0 or np.sum(processed_mask_bin) == 0:
        return processed_img
    
    # Extract only hair regions
    source_hair = cv2.bitwise_and(source_img, source_img, mask=source_mask_bin)
    processed_hair = cv2.bitwise_and(processed_img, processed_img, mask=processed_mask_bin)
    
    # Convert to LAB color space for better color matching
    source_lab = cv2.cvtColor(source_hair, cv2.COLOR_BGR2LAB)
    processed_lab = cv2.cvtColor(processed_hair, cv2.COLOR_BGR2LAB)
    
    # Calculate color statistics for each channel
    source_stats = []
    processed_stats = []
    
    for i in range(3):  # L, A, B channels
        # Get non-zero pixels (only hair pixels)
        source_channel = source_lab[:,:,i][source_mask_bin > 0]
        processed_channel = processed_lab[:,:,i][processed_mask_bin > 0]
        
        if len(source_channel) == 0 or len(processed_channel) == 0:
            continue
        
        # Calculate statistics
        source_mean = np.mean(source_channel)
        source_std = np.std(source_channel)
        processed_mean = np.mean(processed_channel)
        processed_std = np.std(processed_channel)
        
        source_stats.append((source_mean, source_std))
        processed_stats.append((processed_mean, processed_std))
    
    # Create a corrected version of the processed image
    corrected_img = processed_img.copy()
    corrected_lab = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2LAB)
    
    # Apply color correction (scale and shift) for each channel
    for i in range(3):
        if i >= len(source_stats) or i >= len(processed_stats):
            continue
            
        source_mean, source_std = source_stats[i]
        processed_mean, processed_std = processed_stats[i]
        
        # Skip if std is zero (avoid division by zero)
        if processed_std == 0:
            continue
        
        # Calculate scaling factor and shift
        scale = source_std / processed_std
        shift = source_mean - processed_mean * scale
        
        # Apply transformation to the hair region only
        channel = corrected_lab[:,:,i]
        mask_float = processed_mask_bin.astype(float) / 255.0
        
        # Interpolate between original and corrected based on strength
        corrected_channel = channel * scale + shift * mask_float
        
        # Blend with original based on strength
        channel[processed_mask_bin > 0] = (
            channel[processed_mask_bin > 0] * (1 - strength) + 
            corrected_channel[processed_mask_bin > 0] * strength
        )
    
    # Convert back to BGR
    corrected_img = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
    
    return corrected_img

def create_hair_specific_mask(mask, focus_on_top=True, padding=10):
    """
    Create a specialized mask for hair processing that prioritizes the top part.
    
    Args:
        mask: Binary mask of hair region
        focus_on_top: Whether to prioritize the top portion of hair
        padding: Padding around mask edges
        
    Returns:
        numpy.ndarray: Specialized hair mask
    """
    # Ensure mask is binary
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find top of hair
    hair_points = np.argwhere(binary_mask > 0)
    if len(hair_points) == 0:
        return binary_mask
    
    hair_top = hair_points[:, 0].min()
    hair_bottom = hair_points[:, 0].max()
    hair_height = hair_bottom - hair_top
    
    # Create a modified mask that emphasizes the top portion
    if focus_on_top and hair_height > 50:  # Only apply for taller hair
        # Define the upper third of the hair
        upper_third_cutoff = hair_top + hair_height // 3
        
        # Create a weighted mask
        weighted_mask = np.zeros_like(binary_mask, dtype=np.float32)
        
        # Set upper third to full weight
        weighted_mask[hair_top:upper_third_cutoff, :] = binary_mask[hair_top:upper_third_cutoff, :].astype(np.float32)
        
        # Create a gradient for the rest (weight decreases as we go down)
        for y in range(upper_third_cutoff, hair_bottom + 1):
            # Calculate weight based on position (1.0 at upper_third_cutoff, approaching 0.5 at bottom)
            weight = 1.0 - 0.5 * (y - upper_third_cutoff) / (hair_bottom - upper_third_cutoff + 1)
            weighted_mask[y, :] = binary_mask[y, :].astype(np.float32) * weight
        
        # Normalize and convert back to binary
        weighted_mask = (weighted_mask * 255).astype(np.uint8)
        _, weighted_binary = cv2.threshold(weighted_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Add padding if requested
        if padding > 0:
            kernel = np.ones((padding, padding), np.uint8)
            padded_mask = cv2.dilate(weighted_binary, kernel, iterations=1)
            return padded_mask
        
        return weighted_binary
    
    # If not focusing on top, just add padding if requested
    if padding > 0:
        kernel = np.ones((padding, padding), np.uint8)
        padded_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        return padded_mask
    
    return binary_mask

def analyze_hair_shape(hair_mask):
    """
    Analyze the shape characteristics of hair in a mask.
    
    Args:
        hair_mask: Binary mask of hair region
        
    Returns:
        dict: Dictionary of hair shape characteristics
    """
    # Ensure mask is binary
    _, binary_mask = cv2.threshold(hair_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {
            'is_valid': False,
            'hair_type': 'unknown'
        }
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate basic shape metrics
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Calculate shape descriptors
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    aspect_ratio = float(w) / h if h > 0 else 0
    extent = float(area) / (w * h) if w * h > 0 else 0
    
    # Determine hair type based on shape
    hair_type = 'unknown'
    is_long = aspect_ratio < 0.8 and h > w
    is_wide = aspect_ratio > 1.2
    is_dense = extent > 0.7
    is_sparse = extent < 0.4
    is_round = circularity > 0.6
    
    if is_long and is_sparse:
        hair_type = 'long_flowing'
    elif is_long and is_dense:
        hair_type = 'straight_long'
    elif is_wide and is_dense:
        hair_type = 'wide_voluminous'
    elif is_round:
        hair_type = 'afro_curly'
    elif aspect_ratio < 0.5:  # Very tall and narrow
        hair_type = 'ponytail_updo'
    
    # Find top and sides of hair
    hair_points = np.argwhere(binary_mask > 0)
    hair_top = hair_points[:, 0].min() if len(hair_points) > 0 else 0
    
    # Analyze the hair outline
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    is_curly = solidity < 0.8
    
    return {
        'is_valid': True,
        'hair_type': hair_type,
        'is_long': is_long,
        'is_wide': is_wide,
        'is_dense': is_dense,
        'is_sparse': is_sparse,
        'is_round': is_round,
        'is_curly': is_curly,
        'top_position': hair_top,
        'width': w,
        'height': h,
        'aspect_ratio': aspect_ratio,
        'circularity': circularity,
        'solidity': solidity
    }

def generate_hair_mask_recommendations(source_analysis, processed_analysis):
    """
    Generate recommendations for hair alignment based on analyses.
    
    Args:
        source_analysis: Analysis of source hair
        processed_analysis: Analysis of processed hair
        
    Returns:
        dict: Recommended settings for alignment and blending
    """
    if not source_analysis['is_valid'] or not processed_analysis['is_valid']:
        return {
            'vertical_bias': 10,
            'soft_edge_width': 15,
            'alignment_method': 'centroid',
            'blend_mode': 'feathered',
            'confidence': 'low'
        }
    
    # Default recommendations
    recommendations = {
        'vertical_bias': 10,
        'soft_edge_width': 15,
        'alignment_method': 'centroid',
        'blend_mode': 'feathered',
        'confidence': 'medium'
    }
    
    # Adjust vertical bias based on hair type differences
    if source_analysis['is_long'] and not processed_analysis['is_long']:
        # Source is long but processed is not - align tops more closely
        recommendations['vertical_bias'] = 5
    elif not source_analysis['is_long'] and processed_analysis['is_long']:
        # Source is not long but processed is - shift processed hair up
        recommendations['vertical_bias'] = -15
    
    # Special case for updos/ponytails
    if source_analysis['hair_type'] == 'ponytail_updo' or processed_analysis['hair_type'] == 'ponytail_updo':
        recommendations['vertical_bias'] = -20
        recommendations['alignment_method'] = 'bbox'
    
    # Adjust soft edge width based on hair density
    if source_analysis['is_dense'] and processed_analysis['is_sparse']:
        # Dense source with sparse processed - wider soft edges
        recommendations['soft_edge_width'] = 20
    elif source_analysis['is_sparse'] and processed_analysis['is_dense']:
        # Sparse source with dense processed - narrower soft edges
        recommendations['soft_edge_width'] = 10
    
    # Adjust for curly vs straight hair
    if source_analysis['is_curly'] != processed_analysis['is_curly']:
        # Different curliness - feathered blending works better
        recommendations['blend_mode'] = 'feathered'
        recommendations['soft_edge_width'] = max(recommendations['soft_edge_width'], 18)
    
    # Confidence level
    if source_analysis['hair_type'] == processed_analysis['hair_type']:
        recommendations['confidence'] = 'high'
    elif source_analysis['hair_type'] == 'unknown' or processed_analysis['hair_type'] == 'unknown':
        recommendations['confidence'] = 'low'
    
    return recommendations