"""
Image Resizer for Dataset Preparation Tool
Resizes images and masks with various options (conditional resize, source resolution).
"""

import os
import cv2
from utils.image_utils import crop_to_square

class ImageResizer:
    """Resizes images and masks with various options."""
    
    def __init__(self, app):
        """
        Initialize image resizer.
        
        Args:
            app: The main application with shared variables and UI controls
        """
        self.app = app
    
    def resize_images(self, input_dir, output_dir):
        """
        Resize images and masks with various options.
        
        Args:
            input_dir: Input directory containing images and masks
            output_dir: Output directory for resized images
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create output directories
        resize_output_dir = os.path.join(output_dir, "resized")
        os.makedirs(resize_output_dir, exist_ok=True)
        resize_masks_dir = os.path.join(resize_output_dir, "masks")
        os.makedirs(resize_masks_dir, exist_ok=True)
        
        # Find all images (with or without masks)
        all_images = []  # List of (image_path, mask_path or None) tuples
        
        for root, dirs, files in os.walk(input_dir):
            # Skip if this is already a 'masks' directory
            if os.path.basename(root).lower() == "masks":
                continue
            
            # Check if there's a 'masks' subdirectory
            mask_dir = os.path.join(root, "masks")
            has_mask_dir = os.path.isdir(mask_dir)
            
            # Process each image file
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    mask_path = None
                    
                    # If a mask directory exists, check for a corresponding mask
                    if has_mask_dir:
                        # Try exact filename match
                        potential_mask = os.path.join(mask_dir, file)
                        if os.path.exists(potential_mask):
                            mask_path = potential_mask
                        else:
                            # Try different extensions
                            for ext in ['.png', '.jpg', '.jpeg']:
                                alt_mask_path = os.path.join(mask_dir, os.path.splitext(file)[0] + ext)
                                if os.path.exists(alt_mask_path):
                                    mask_path = alt_mask_path
                                    break
                    
                    # Add the image (with or without mask) to our list
                    all_images.append((image_path, mask_path))
        
        if not all_images:
            self.app.status_label.config(text="No images found to resize.")
            return False
        
        total_images = len(all_images)
        processed_count = 0
        processed_with_mask = 0
        portrait_cropped_count = 0
        
        # Process each image
        for idx, (image_path, mask_path) in enumerate(all_images):
            if not self.app.processing:  # Check if processing was cancelled
                break
            
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error loading image: {image_path}")
                    continue
                
                # Get original dimensions
                orig_height, orig_width = image.shape[:2]
                
                # Load mask if it exists
                mask = None
                if mask_path:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        print(f"Error loading mask: {mask_path}")
                        # Continue with just the image
                
                was_portrait = False
                # Handle portrait photo cropping FIRST, before any resizing
                if self.app.portrait_crop_enabled.get():
                    # Check if this is a portrait photo (height > width)
                    if orig_height > orig_width:
                        was_portrait = True
                        # Get crop position
                        crop_position = self.app.portrait_crop_position.get()
                        
                        # Crop the image to make it square
                        image = crop_to_square(image, position=crop_position)
                        
                        # Update dimensions after cropping
                        orig_height, orig_width = image.shape[:2]
                        
                        # Also crop the mask if it exists
                        if mask is not None:
                            mask = crop_to_square(mask, position=crop_position)
                        
                        portrait_cropped_count += 1
                        self.app.status_label.config(text=f"Cropped portrait photo: {os.path.basename(image_path)}")
                
                # Initialize resizing variables with defaults
                needs_resize = False
                target_width = orig_width  # Default to original size
                target_height = orig_height  # Default to original size
                
                # Determine if and how to resize based on settings
                if self.app.use_source_resolution.get():
                    # Just use original resolution (which may have been cropped if portrait)
                    needs_resize = False
                elif self.app.resize_if_larger.get():
                    # Only resize if image is larger than the threshold
                    max_width = self.app.max_width.get()
                    max_height = self.app.max_height.get()
                    
                    if orig_width > max_width or orig_height > max_height:
                        # Calculate new dimensions while preserving aspect ratio
                        width_ratio = max_width / orig_width
                        height_ratio = max_height / orig_height
                        
                        # Use the smaller ratio to ensure both dimensions fit within the max
                        scale_ratio = min(width_ratio, height_ratio)
                        
                        target_width = int(orig_width * scale_ratio)
                        target_height = int(orig_height * scale_ratio)
                        needs_resize = True
                        
                        self.app.status_label.config(text=f"Resizing oversized image: {orig_width}x{orig_height} → {target_width}x{target_height}")
                else:
                    # Always resize to fixed dimensions
                    target_width = self.app.output_width.get()
                    target_height = self.app.output_height.get()
                    needs_resize = True
                
                # Generate output name
                output_name = f"{processed_count:04d}.png"
                output_image_path = os.path.join(resize_output_dir, output_name)
                output_mask_path = os.path.join(resize_masks_dir, output_name) if mask_path else None
                
                # Process image (and mask if available)
                if needs_resize:
                    # Resize image
                    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                    cv2.imwrite(output_image_path, resized_image)
                    
                    # Resize mask if it exists
                    if mask is not None:
                        resized_mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
                        cv2.imwrite(output_mask_path, resized_mask)
                        processed_with_mask += 1
                else:
                    # Save without resizing (might still have been cropped if portrait)
                    cv2.imwrite(output_image_path, image)
                    
                    # Save mask if it exists
                    if mask is not None:
                        cv2.imwrite(output_mask_path, mask)
                        processed_with_mask += 1
                
                processed_count += 1
                
                # Update progress
                progress = (idx + 1) / total_images * 100
                self.app.progress_bar['value'] = min(progress, 100)
                
                # Update status less frequently to improve performance
                if idx % 10 == 0 or idx == total_images - 1:
                    status_text = f"Processed {idx+1}/{total_images} images"
                    if portrait_cropped_count > 0:
                        status_text += f" (cropped {portrait_cropped_count} portraits)"
                    if processed_with_mask > 0:
                        status_text += f" ({processed_with_mask} with masks)"
                    self.app.status_label.config(text=status_text)
                    self.app.root.update_idletasks()
                
            except Exception as e:
                error_msg = f"Error processing {image_path}: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
        
        # Final status update
        status_text = f"Resizing completed. Processed {processed_count} images"
        if portrait_cropped_count > 0:
            status_text += f", cropped {portrait_cropped_count} portraits"
        if processed_with_mask > 0:
            status_text += f", including {processed_with_mask} with masks"
        self.app.status_label.config(text=status_text)
        self.app.progress_bar['value'] = 100
        
        return True