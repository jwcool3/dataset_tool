"""
Image Utility Functions for Dataset Preparation Tool
Common image processing functions used throughout the application.
"""

import os
import cv2
import numpy as np

def load_image_with_mask(directory):
    """
    Find and load a sample image and its corresponding mask from a directory.
    
    Args:
        directory: Directory to search for images and masks
        
    Returns:
        tuple: (image_path, mask_path) or (image_path, None) if no mask is found
    """
    if not os.path.isdir(directory):
        return None, None
    
    # Look for image files
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Skip if the file is in a 'masks' directory
                if os.path.basename(root).lower() != "masks":
                    image_files.append(os.path.join(root, file))
    
    if not image_files:
        return None, None
    
    # Load the first image
    sample_image_path = image_files[0]
    
    # Try to find a corresponding mask
    mask_path = None
    # First try: Check if there's a mask folder in the same directory
    image_dir = os.path.dirname(sample_image_path)
    mask_dir = os.path.join(image_dir, "masks")
    if os.path.isdir(mask_dir):
        potential_mask = os.path.join(mask_dir, os.path.basename(sample_image_path))
        if os.path.exists(potential_mask):
            mask_path = potential_mask
        else:
            # Try other extensions
            for ext in ['.png', '.jpg', '.jpeg']:
                alt_mask_path = os.path.join(mask_dir, os.path.splitext(os.path.basename(sample_image_path))[0] + ext)
                if os.path.exists(alt_mask_path):
                    mask_path = alt_mask_path
                    break
    
    return sample_image_path, mask_path

def crop_to_square(image, position="center"):
    """
    Crop a portrait (tall) image to make it square.
    
    Args:
        image: The input image (numpy array)
        position: Where to take the crop from - "top", "center", or "bottom"
        
    Returns:
        numpy.ndarray: Square cropped image
    """
    height, width = image.shape[:2]
    
    # Only process portrait images (height > width)
    if height <= width:
        return image
    
    # Calculate crop dimensions
    crop_height = width  # Make height equal to width for square
    
    # Determine start row based on position
    if position == "top":
        start_row = 0
    elif position == "bottom":
        start_row = height - crop_height
    else:  # "center" (default)
        start_row = (height - crop_height) // 2
    
    # Ensure start_row is within bounds
    start_row = max(0, min(start_row, height - crop_height))
    
    # Perform the crop
    cropped = image[start_row:start_row + crop_height, 0:width]
    return cropped

def add_padding_to_square(image, padding_color=(0,0,0), target_size=None):
    """
    Add padding to an image to make it square while keeping the original image centered.
    
    Args:
        image: The input image (numpy array)
        padding_color: RGB color tuple for the padding (default: black)
        target_size: Optional specific size for the output square
        
    Returns:
        numpy.ndarray: Square padded image
    """
    height, width = image.shape[:2]
    
    # If image is already square and no specific target size, return as is
    if height == width and target_size is None:
        return image
    
    # Determine square size
    if target_size is not None:
        canvas_size = target_size
    else:
        canvas_size = max(width, height)
    
    # Create square canvas
    square_image = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    square_image[:] = padding_color  # Fill with padding color
    
    # Calculate position to paste the original image (centered)
    x_offset = (canvas_size - width) // 2
    y_offset = (canvas_size - height) // 2
    
    # Paste original image onto the square canvas
    square_image[y_offset:y_offset+height, x_offset:x_offset+width] = image
    
    return square_image

def improve_mask_detection(mask_img, threshold=20):
    """
    Enhanced mask detection that ensures all relevant mask areas are captured.
    
    Args:
        mask_img: Original mask image
        threshold: Brightness threshold for detection
            
    Returns:
        tuple: (improved_mask, bbox) where bbox is (x, y, width, height)
    """
    # Convert to grayscale if not already
    if len(mask_img.shape) == 3:
        if mask_img.shape[2] == 4:  # RGBA
            if np.any(mask_img[:,:,3] > 0):
                gray_mask = mask_img[:,:,3]
            else:
                gray_mask = cv2.cvtColor(mask_img[:,:,:3], cv2.COLOR_BGR2GRAY)
        else:  # RGB
            gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_mask = mask_img.copy()
    
    # Apply threshold to create binary mask
    _, binary_mask = cv2.threshold(gray_mask, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to enhance the mask
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find all contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return full image bounds
    if not contours:
        h, w = mask_img.shape[:2]
        return binary_mask, (0, 0, w, h)
    
    # Create a mask that includes all contours
    combined_mask = np.zeros_like(binary_mask)
    
    # Filter contours by area - only keep significant ones
    min_area = binary_mask.shape[0] * binary_mask.shape[1] * 0.0005  # 0.05% of image area
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # If no valid contours after filtering, use all contours
    if not valid_contours:
        valid_contours = contours
    
    # Draw all valid contours on the combined mask
    cv2.drawContours(combined_mask, valid_contours, -1, 255, -1)
    
    # Now find the global bounding box of all contours
    x_coords = []
    y_coords = []
    
    for contour in valid_contours:
        for point in contour:
            x_coords.append(point[0][0])
            y_coords.append(point[0][1])
    
    # Calculate bounding box from min/max coordinates
    if x_coords and y_coords:
        min_x = max(0, min(x_coords))
        min_y = max(0, min(y_coords))
        max_x = min(binary_mask.shape[1], max(x_coords))
        max_y = min(binary_mask.shape[0], max(y_coords))
        
        # Create the bounding box
        bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
    else:
        # Fallback to full image if something went wrong
        h, w = mask_img.shape[:2]
        bbox = (0, 0, w, h)
    
    return combined_mask, bbox