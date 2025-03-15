"""
Crop Reinserter for Dataset Preparation Tool
Reinserts cropped images back into their original uncropped versions.
"""

import os
import cv2
import numpy as np
from utils.image_utils import add_padding_to_square

class CropReinserter:
    """Reinserts cropped images back into their original positions."""
    
    def __init__(self, app):
        """
        Initialize crop reinserter.
        
        Args:
            app: The main application with shared variables and UI controls
        """
        self.app = app
    
    def reinsert_crops(self, input_dir, output_dir):
        """
        Reinsert cropped images back into their original images.
        
        Args:
            input_dir: Input directory containing cropped images
            output_dir: Output directory for reinserted images
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create output directory for reinserted images
        reinsert_output_dir = os.path.join(output_dir, "reinserted")
        os.makedirs(reinsert_output_dir, exist_ok=True)
        
        # Find all cropped images
        cropped_images = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    cropped_images.append(os.path.join(root, file))
        
        if not cropped_images:
            self.app.status_label.config(text="No cropped images found.")
            return False
        
        # Get original images path
        original_dir = self.app.original_images_dir.get()
        if not original_dir or not os.path.isdir(original_dir):
            self.app.status_label.config(text="Original images directory not set or invalid.")
            return False
        
        # Get crop parameters
        padding_percent = self.app.reinsert_padding.get()
        
        # Process each cropped image
        total_images = len(cropped_images)
        processed_count = 0
        
        for idx, cropped_path in enumerate(cropped_images):
            if not self.app.processing:  # Check if processing was cancelled
                break
            
            # Get cropped image filename 
            cropped_filename = os.path.basename(cropped_path)
            
            # Find corresponding original image
            # This assumes the user knows which original image each crop came from
            original_path = self.app.selected_original_image.get()
            if not original_path or not os.path.exists(original_path):
                self.app.status_label.config(text=f"Original image not found for {cropped_filename}")
                continue
            
            # Process the reinsert operation
            try:
                # Load images
                cropped_img = cv2.imread(cropped_path)
                original_img = cv2.imread(original_path)
                
                if cropped_img is None or original_img is None:
                    self.app.status_label.config(text=f"Error loading images for {cropped_filename}")
                    continue
                
                # Get dimensions
                orig_height, orig_width = original_img.shape[:2]
                crop_height, crop_width = cropped_img.shape[:2]
                
                # Calculate crop position based on user parameters
                # This is where we would reverse engineer the crop position
                x_pos = self.app.crop_x_position.get()
                y_pos = self.app.crop_y_position.get()
                crop_width_orig = self.app.crop_width.get()
                crop_height_orig = self.app.crop_height.get()
                
                # Resize cropped image back to original crop dimensions if needed
                if crop_width != crop_width_orig or crop_height != crop_height_orig:
                    cropped_img = cv2.resize(cropped_img, (crop_width_orig, crop_height_orig), 
                                          interpolation=cv2.INTER_LANCZOS4)
                
                # Create a copy of the original image to modify
                result_img = original_img.copy()
                
                # Place the cropped image back into the original
                result_img[y_pos:y_pos+crop_height_orig, x_pos:x_pos+crop_width_orig] = cropped_img
                
                # Save the reinserted image
                output_path = os.path.join(reinsert_output_dir, f"reinserted_{cropped_filename}")
                cv2.imwrite(output_path, result_img)
                
                processed_count += 1
                
                # Update progress
                progress = (idx + 1) / total_images * 100
                self.app.progress_bar['value'] = min(progress, 100)
                self.app.status_label.config(text=f"Processed {idx+1}/{total_images} images")
                self.app.root.update_idletasks()
                
            except Exception as e:
                self.app.status_label.config(text=f"Error processing {cropped_filename}: {str(e)}")
                print(f"Error in reinsert_crops: {str(e)}")
                import traceback
                traceback.print_exc()
        
        self.app.status_label.config(text=f"Reinsertion completed. Processed {processed_count} images.")
        self.app.progress_bar['value'] = 100
        return True