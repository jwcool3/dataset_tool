"""
Square Padder for Dataset Preparation Tool
Adds padding to images to make them square while preserving aspect ratio.
"""

import os
import cv2
import numpy as np
from utils.image_utils import add_padding_to_square

class SquarePadder:
    """Adds padding to images to make them square while preserving aspect ratio."""
    
    def __init__(self, app):
        """
        Initialize square padder.
        
        Args:
            app: The main application with shared variables and UI controls
        """
        self.app = app
    
    def add_square_padding(self, input_dir, output_dir):
        """
        Add padding to make images square (1:1 aspect ratio).
        
        Args:
            input_dir: Input directory containing images and masks
            output_dir: Output directory for padded images
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create output directories
        square_output_dir = os.path.join(output_dir, "square_padded")
        os.makedirs(square_output_dir, exist_ok=True)
        
        # We'll create the masks directory only if we find masks
        square_masks_dir = os.path.join(square_output_dir, "masks")
        
        # Find all images and possible mask pairs
        source_images = []     # All source images
        image_mask_pairs = []  # Only images with corresponding masks
        
        # First, find all images in the input directory (excluding masks subdirectories)
        for root, dirs, files in os.walk(input_dir):
            # Skip if this is a 'masks' directory
            if os.path.basename(root).lower() == "masks":
                continue
            
            # Process each image file
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    source_images.append(image_path)
        
        # Now, look for corresponding masks for each source image
        for image_path in source_images:
            image_dir = os.path.dirname(image_path)
            image_filename = os.path.basename(image_path)
            image_basename = os.path.splitext(image_filename)[0]
            
            # Check common mask locations
            mask_locations = [
                os.path.join(image_dir, "masks", image_filename),  # Same filename in masks subfolder
            ]
            
            # Try different extensions in the masks folder
            for ext in ['.png', '.jpg', '.jpeg']:
                mask_locations.append(os.path.join(image_dir, "masks", image_basename + ext))
            
            # Look for any matching mask
            mask_found = False
            for mask_path in mask_locations:
                if os.path.exists(mask_path):
                    image_mask_pairs.append((image_path, mask_path))
                    mask_found = True
                    break
            
            # If no mask found, it will only be in the source_images list
        
        # If we found any masks, create the masks output directory
        has_masks = len(image_mask_pairs) > 0
        if has_masks:
            os.makedirs(square_masks_dir, exist_ok=True)
        
        # Log what we found
        self.app.status_label.config(text=f"Found {len(source_images)} images, including {len(image_mask_pairs)} with masks")
        print(f"Found {len(source_images)} images, including {len(image_mask_pairs)} with masks")
        
        if not source_images:
            self.app.status_label.config(text="No images found to process.")
            return False
        
        total_images = len(source_images)
        processed_count = 0
        
        # Get padding color
        padding_color = self.app.padding_color.get()
        
        # Convert color name to RGB values
        if padding_color == "black":
            padding_color_rgb = (0, 0, 0)
        elif padding_color == "white":
            padding_color_rgb = (255, 255, 255)
        elif padding_color == "gray":
            padding_color_rgb = (128, 128, 128)
        else:
            padding_color_rgb = (0, 0, 0)  # Default to black
        
        # Process each image
        for idx, image_path in enumerate(source_images):
            if not self.app.processing:  # Check if processing was cancelled
                break
            
            try:
                # Load image
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"Error loading image: {image_path}")
                    continue
                
                # Get original dimensions
                h, w = image.shape[:2]
                
                # Determine square canvas size based on settings
                if self.app.use_source_resolution_padding.get():
                    # Use the larger dimension from the source image
                    canvas_size = max(w, h)
                else:
                    # Use the specified target size
                    canvas_size = self.app.square_target_size.get()
                
                # Add padding to make the image square
                square_image = add_padding_to_square(image, padding_color_rgb, canvas_size)
                
                # Generate output filename
                output_name = f"{processed_count:04d}.png"
                output_image_path = os.path.join(square_output_dir, output_name)
                
                # Save the padded image
                cv2.imwrite(output_image_path, square_image)
                
                # Check if this image has a corresponding mask
                if has_masks:
                    # Find this image in the image_mask_pairs list
                    mask_path = None
                    for src, mask in image_mask_pairs:
                        if src == image_path:
                            mask_path = mask
                            break
                    
                    # If we found a mask, process it too
                    if mask_path:
                        # Load mask
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        
                        if mask is not None:
                            # Create square mask canvas (filled with zeros/black)
                            square_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
                            
                            # Calculate position to paste the original image (centered)
                            x_offset = (canvas_size - w) // 2
                            y_offset = (canvas_size - h) // 2
                            
                            # Paste mask onto square canvas
                            square_mask[y_offset:y_offset+h, x_offset:x_offset+w] = mask
                            
                            # Save the padded mask
                            output_mask_path = os.path.join(square_masks_dir, output_name)
                            cv2.imwrite(output_mask_path, square_mask)
                
                processed_count += 1
                
                # Update progress
                progress = (idx + 1) / total_images * 100
                self.app.progress_bar['value'] = min(progress, 100)
                if idx % 10 == 0 or idx == total_images - 1:  # Update less frequently for smoother UI
                    self.app.status_label.config(text=f"Processed {idx+1}/{total_images} images for square padding")
                    self.app.root.update_idletasks()
                    
            except Exception as e:
                print(f"Error processing square padding for {image_path}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if has_masks:
            self.app.status_label.config(text=f"Square padding completed. Processed {processed_count} images including {len(image_mask_pairs)} with masks.")
        else:
            self.app.status_label.config(text=f"Square padding completed. Processed {processed_count} images.")
        
        self.app.progress_bar['value'] = 100
        return True