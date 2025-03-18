"""
Mask Processor for Dataset Preparation Tool
Detects and crops around mask regions with padding.
"""

import os
import cv2
import numpy as np
from utils.image_utils import improve_mask_detection

class MaskProcessor:
    """Processes image-mask pairs to focus on the masked regions."""
    
    def __init__(self, app):
        """
        Initialize mask processor.
        
        Args:
            app: The main application with shared variables and UI controls
        """
        self.app = app
    
    def process_masks(self, input_dir, output_dir):
        """
        Process image-mask pairs to crop around masked regions.
        
        Args:
            input_dir: Input directory containing images and masks
            output_dir: Output directory for processed images
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create output directories
        crop_output_dir = os.path.join(output_dir, "cropped")
        os.makedirs(crop_output_dir, exist_ok=True)
        crop_masks_dir = os.path.join(crop_output_dir, "masks")
        os.makedirs(crop_masks_dir, exist_ok=True)
        
        # Create debug directory if debug mode is enabled
        debug_dir = None
        if self.app.debug_mode.get():
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
        
        # Find all image and mask pairs
        image_mask_pairs = self._find_image_mask_pairs(input_dir)
        
        if not image_mask_pairs:
            self.app.status_label.config(text="No image-mask pairs found.")
            return False
        
        total_pairs = len(image_mask_pairs)
        processed_count = 0
        
        # Get processing parameters
        threshold = 20  # Default brightness threshold
        padding_percent = int(self.app.fill_ratio.get())  # Use fill ratio as padding percentage
        min_size = 20  # Minimum bounding box dimension
        
        # Process each pair
        for idx, (image_path, mask_path) in enumerate(image_mask_pairs):
            if not self.app.processing:  # Check if processing was cancelled
                break
            
            # Get filenames for output
            output_name = f"{processed_count:04d}.png"
            output_image_path = os.path.join(crop_output_dir, output_name)
            output_mask_path = os.path.join(crop_masks_dir, output_name)
            
            # Process the image-mask pair
            success = self._process_image_pair(
                image_path, 
                mask_path, 
                output_image_path, 
                output_mask_path,
                threshold=threshold,
                padding_percent=padding_percent, 
                debug_dir=debug_dir,
                min_size=min_size
            )
            
            if success:
                processed_count += 1
            
            # Update progress
            progress = (idx + 1) / total_pairs * 100
            self.app.progress_bar['value'] = min(progress, 100)
            self.app.status_label.config(text=f"Processed {idx+1}/{total_pairs} image-mask pairs")
            self.app.root.update_idletasks()
        
        self.app.status_label.config(text=f"Mask region cropping completed. Processed {processed_count} pairs.")
        self.app.progress_bar['value'] = 100
        return True
    
    def _find_image_mask_pairs(self, directory):
        """
        Find all image and mask pairs in a directory.
        
        Args:
            directory: Directory to search for images and masks
            
        Returns:
            list: List of (image_path, mask_path) tuples
        """
        image_mask_pairs = []
        
        # Track all directories we've processed to avoid duplicates
        processed_dirs = set()
        
        # Function to find mask-image pairs in a given directory
        def find_pairs_in_directory(directory):
            # Skip if we've already processed this directory
            if directory in processed_dirs:
                return
            
            processed_dirs.add(directory)
            
            # Get all files and subdirectories
            try:
                all_items = os.listdir(directory)
            except (PermissionError, FileNotFoundError):
                return  # Skip directories we can't access
                
            # Check if this directory itself has a 'masks' subfolder
            mask_dir = os.path.join(directory, "masks")
            if os.path.isdir(mask_dir):
                # This directory contains images and has a masks subfolder
                for file in all_items:
                    file_path = os.path.join(directory, file)
                    # Skip subdirectories and only process image files
                    if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # Check for corresponding mask
                        mask_path = os.path.join(mask_dir, file)
                        if os.path.exists(mask_path):
                            image_mask_pairs.append((file_path, mask_path))
                        else:
                            # Try different extensions
                            for ext in ['.png', '.jpg', '.jpeg']:
                                alt_mask_path = os.path.join(mask_dir, os.path.splitext(file)[0] + ext)
                                if os.path.exists(alt_mask_path):
                                    image_mask_pairs.append((file_path, alt_mask_path))
                                    break
            
            # Now check if any subdirectories need to be processed
            for item in all_items:
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path) and item.lower() != "masks":  # Skip 'masks' directories
                    find_pairs_in_directory(item_path)  # Recursive call
        
        # Start finding pairs from the input directory
        find_pairs_in_directory(directory)
        return image_mask_pairs
    
    def _process_image_pair(self, image_path, mask_path, output_image_path, output_mask_path, 
                          threshold=20, padding_percent=30, debug_dir=None, min_size=20):
        """
        Process an image-mask pair to detect mask regions and zoom in to fill the entire image.
        
        Args:
            image_path: Path to the input image
            mask_path: Path to the input mask
            output_image_path: Path to save the processed image
            output_mask_path: Path to save the processed mask
            threshold: Brightness threshold for considering a pixel as part of the mask (0-255)
            padding_percent: Padding percentage around the detected region
            debug_dir: Directory to save debug visualizations (None to disable)
            min_size: Minimum size for bounding box in pixels
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            # Load images
            src_img = cv2.imread(image_path)
            mask_img = cv2.imread(mask_path)
            
            # Check if images were loaded correctly
            if src_img is None or mask_img is None:
                print(f"Error loading images: {image_path} or {mask_path}")
                return False
            
            # Get image dimensions
            src_height, src_width = src_img.shape[:2]
            mask_height, mask_width = mask_img.shape[:2]
            
            # Get filename for logging
            basename = os.path.basename(image_path)
            
            # Check if dimensions match, if not resize mask to match source
            if src_width != mask_width or src_height != mask_height:
                if debug_dir:
                    print(f"Resizing mask to match source dimensions: {src_width}x{src_height}")
                mask_img = cv2.resize(mask_img, (src_width, src_height), interpolation=cv2.INTER_NEAREST)
            
            # Convert mask to grayscale if it's not already
            if len(mask_img.shape) == 3:
                mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = mask_img
                    
            # Apply threshold to create binary mask
            _, binary_mask = cv2.threshold(mask_gray, threshold, 255, cv2.THRESH_BINARY)
            
            # Find contours of the mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If no contours found, use the full image (no processing needed)
            if not contours:
                print(f"No contours found in mask for {basename}, using full image")
                cv2.imwrite(output_image_path, src_img)
                cv2.imwrite(output_mask_path, mask_img)
                return True
            
            # Combine all significant contours
            combined_mask = np.zeros_like(binary_mask)
            
            # Filter contours by area - only keep significant ones (at least 0.05% of image area)
            min_area = binary_mask.shape[0] * binary_mask.shape[1] * 0.0005
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            # If no valid contours after filtering, use all contours
            if not valid_contours:
                valid_contours = contours
            
            # Draw all valid contours on the combined mask
            cv2.drawContours(combined_mask, valid_contours, -1, 255, -1)
            
            # Find the bounding box that encompasses all contours
            x_min, y_min = src_width, src_height
            x_max, y_max = 0, 0
            
            for contour in valid_contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            
            # Create the bounding box (x, y, width, height)
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            # Validate bounding box
            x, y, w, h = bbox
            if w < min_size or h < min_size:
                print(f"Warning: Bounding box too small for {basename}: {bbox}")
                # Use full image as fallback
                bbox = (0, 0, src_width, src_height)
                x, y, w, h = bbox
            
            # Add padding to bounding box (while ensuring we stay within image bounds)
            padding_x = int(w * padding_percent / 100)
            padding_y = int(h * padding_percent / 100)
            
            padded_x = max(0, x - padding_x)
            padded_y = max(0, y - padding_y)
            padded_width = min(src_width - padded_x, w + 2 * padding_x)
            padded_height = min(src_height - padded_y, h + 2 * padding_y)
            
            padded_bbox = (padded_x, padded_y, padded_width, padded_height)
            
            # If the padded bbox is already the full image, no processing needed
            if padded_x == 0 and padded_y == 0 and padded_width == src_width and padded_height == src_height:
                print(f"Padded bbox is already full image for {basename}, no processing needed")
                cv2.imwrite(output_image_path, src_img)
                cv2.imwrite(output_mask_path, mask_img)
                return True
                
            # Crop the images using the padded bounding box
            cropped_src = src_img[padded_y:padded_y+padded_height, padded_x:padded_x+padded_width].copy()
            cropped_mask = mask_img[padded_y:padded_y+padded_height, padded_x:padded_x+padded_width].copy()
            
            # Check if cropped images are valid
            if cropped_src.size == 0 or cropped_mask.size == 0:
                print(f"Warning: Cropping resulted in empty images for {basename}")
                # Use original images as fallback
                cv2.imwrite(output_image_path, src_img)
                cv2.imwrite(output_mask_path, mask_img)
                return True
                
            # Save debug images before resizing if debug directory is provided
            if debug_dir:
                # Create debug directory if it doesn't exist
                os.makedirs(debug_dir, exist_ok=True)
                
                # Create visualization with bounding boxes
                debug_img = src_img.copy()
                
                # Draw original bounding box (red)
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # Draw padded bounding box (green)
                cv2.rectangle(debug_img, (padded_x, padded_y), 
                            (padded_x + padded_width, padded_y + padded_height), 
                            (0, 255, 0), 2)
                
                # Save debug images
                cv2.imwrite(os.path.join(debug_dir, f"debug_bbox_{basename}"), debug_img)
                cv2.imwrite(os.path.join(debug_dir, f"cropped_before_resize_{basename}"), cropped_src)
                cv2.imwrite(os.path.join(debug_dir, f"cropped_mask_before_resize_{basename}"), cropped_mask)
                
            # Don't resize - just use the cropped region as-is
            final_src = cropped_src
            final_mask = cropped_mask
            # Save extra debug images after resizing
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, f"final_resized_{basename}"), final_src)
                cv2.imwrite(os.path.join(debug_dir, f"final_mask_resized_{basename}"), final_mask)
            
            # Save processed images
            cv2.imwrite(output_image_path, final_src)
            cv2.imwrite(output_mask_path, final_mask)
            
            return True
            
        except Exception as e:
            print(f"Unexpected error processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # For any uncaught exceptions, copy the original images to output
            try:
                if 'src_img' in locals() and src_img is not None:
                    cv2.imwrite(output_image_path, src_img)
                if 'mask_img' in locals() and mask_img is not None:
                    # Resize mask to match source if dimensions differ
                    if 'src_width' in locals() and 'src_height' in locals() and (mask_img.shape[1] != src_width or mask_img.shape[0] != src_height):
                        mask_img = cv2.resize(mask_img, (src_width, src_height), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(output_mask_path, mask_img)
            except Exception as save_error:
                print(f"Error saving fallback images: {str(save_error)}")
            
            return False