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
        
        # Get original images directory
        original_dir = self.app.original_images_dir.get()
        if not original_dir or not os.path.isdir(original_dir):
            self.app.status_label.config(text="Original images directory not set or invalid.")
            return False
        
        # Load all original images with their filename patterns
        original_images = {}
        for file in os.listdir(original_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                original_images[file] = os.path.join(original_dir, file)
        
        # Process each cropped image
        total_images = len(cropped_images)
        processed_count = 0
        
        # Get batch processing options
        naming_pattern = self.app.reinsert_naming_pattern.get()
        crop_width = self.app.crop_width.get()
        crop_height = self.app.crop_height.get()
        
        for idx, cropped_path in enumerate(cropped_images):
            if not self.app.processing:  # Check if processing was cancelled
                break
            
            # Get cropped image filename and extract information from it
            cropped_filename = os.path.basename(cropped_path)
            cropped_basename, cropped_ext = os.path.splitext(cropped_filename)
            
            # Match to original image based on naming pattern
            original_filename = self._find_original_image(
                cropped_basename, 
                original_images,
                naming_pattern
            )
            
            if not original_filename:
                self.app.status_label.config(text=f"Original image not found for {cropped_filename}")
                continue
            
            original_path = original_images[original_filename]
            
            # Get crop position data (either from filename or from metadata file)
            x_pos, y_pos, width, height = self._get_crop_position(
                cropped_path, 
                cropped_basename,
                self.app.position_source.get()
            )
            
            # If no valid position data found, use defaults from UI
            if x_pos is None or y_pos is None:
                x_pos = self.app.crop_x_position.get()
                y_pos = self.app.crop_y_position.get()
                width = crop_width if width is None else width
                height = crop_height if height is None else height
            
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
                crop_height_curr, crop_width_curr = cropped_img.shape[:2]
                
                # Resize cropped image back to original crop dimensions if needed
                if crop_width_curr != width or crop_height_curr != height:
                    cropped_img = cv2.resize(cropped_img, (width, height), 
                                        interpolation=cv2.INTER_LANCZOS4)
                
                # Validate position
                if x_pos + width > orig_width or y_pos + height > orig_height:
                    self.app.status_label.config(text=f"Warning: Crop position out of bounds for {cropped_filename}")
                    # Adjust position to fit within bounds
                    x_pos = min(x_pos, orig_width - width)
                    y_pos = min(y_pos, orig_height - height)
                
                # Create a copy of the original image to modify
                result_img = original_img.copy()
                
                # Place the cropped image back into the original
                result_img[y_pos:y_pos+height, x_pos:x_pos+width] = cropped_img
                
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

    def _find_original_image(self, cropped_basename, original_images, naming_pattern):
        """
        Find the corresponding original image based on naming pattern.
        
        Args:
            cropped_basename: Base name of the cropped image (without extension)
            original_images: Dictionary of original image filenames
            naming_pattern: Pattern to derive original name from cropped name
            
        Returns:
            str: Filename of the matching original image, or None if not found
        """
        # Different naming pattern strategies:
        if naming_pattern == "prefix":
            # Example: crop_original.jpg -> original.jpg
            # Find the original where cropped has a prefix
            prefix_len = self.app.prefix_length.get()
            original_base = cropped_basename[prefix_len:] if len(cropped_basename) > prefix_len else cropped_basename
            
            # Find original with this base name and any extension
            for orig_name in original_images.keys():
                orig_base = os.path.splitext(orig_name)[0]
                if orig_base == original_base:
                    return orig_name
        
        elif naming_pattern == "suffix":
            # Example: original_crop.jpg -> original.jpg
            # Find the original where cropped has a suffix
            suffix_len = self.app.suffix_length.get()
            original_base = cropped_basename[:-suffix_len] if len(cropped_basename) > suffix_len else cropped_basename
            
            # Find original with this base name and any extension
            for orig_name in original_images.keys():
                orig_base = os.path.splitext(orig_name)[0]
                if orig_base == original_base:
                    return orig_name
        
        elif naming_pattern == "indexed":
            # Example: original_001.jpg comes from original.jpg
            # Remove numeric suffix
            import re
            original_base = re.sub(r'_\d+$', '', cropped_basename)
            
            # Find original with this base name and any extension
            for orig_name in original_images.keys():
                orig_base = os.path.splitext(orig_name)[0]
                if orig_base == original_base:
                    return orig_name
        
        elif naming_pattern == "metadata":
            # The original filename should be stored in metadata
            # This would be implemented with the _get_crop_position method
            # using the metadata for both position and original filename
            pass
        
        # If no match found, try exact match
        for orig_name in original_images.keys():
            orig_base = os.path.splitext(orig_name)[0]
            if orig_base == cropped_basename:
                return orig_name
        
        return None

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
        
        # Get original images directory
        original_dir = self.app.original_images_dir.get()
        if not original_dir or not os.path.isdir(original_dir):
            self.app.status_label.config(text="Original images directory not set or invalid.")
            return False
        
        # Load all original images with their filename patterns
        original_images = {}
        for file in os.listdir(original_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                original_images[file] = os.path.join(original_dir, file)
        
        # Process each cropped image
        total_images = len(cropped_images)
        processed_count = 0
        
        # Get batch processing options
        naming_pattern = self.app.reinsert_naming_pattern.get()
        crop_width = self.app.crop_width.get()
        crop_height = self.app.crop_height.get()
        
        for idx, cropped_path in enumerate(cropped_images):
            if not self.app.processing:  # Check if processing was cancelled
                break
            
            # Get cropped image filename and extract information from it
            cropped_filename = os.path.basename(cropped_path)
            cropped_basename, cropped_ext = os.path.splitext(cropped_filename)
            
            # Match to original image based on naming pattern
            original_filename = self._find_original_image(
                cropped_basename, 
                original_images,
                naming_pattern
            )
            
            if not original_filename:
                self.app.status_label.config(text=f"Original image not found for {cropped_filename}")
                continue
            
            original_path = original_images[original_filename]
            
            # Get crop position data (either from filename or from metadata file)
            x_pos, y_pos, width, height = self._get_crop_position(
                cropped_path, 
                cropped_basename,
                self.app.position_source.get()
            )
            
            # If no valid position data found, use defaults from UI
            if x_pos is None or y_pos is None:
                x_pos = self.app.crop_x_position.get()
                y_pos = self.app.crop_y_position.get()
                width = crop_width if width is None else width
                height = crop_height if height is None else height
            
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
                crop_height_curr, crop_width_curr = cropped_img.shape[:2]
                
                # Resize cropped image back to original crop dimensions if needed
                if crop_width_curr != width or crop_height_curr != height:
                    cropped_img = cv2.resize(cropped_img, (width, height), 
                                        interpolation=cv2.INTER_LANCZOS4)
                
                # Validate position
                if x_pos + width > orig_width or y_pos + height > orig_height:
                    self.app.status_label.config(text=f"Warning: Crop position out of bounds for {cropped_filename}")
                    # Adjust position to fit within bounds
                    x_pos = min(x_pos, orig_width - width)
                    y_pos = min(y_pos, orig_height - height)
                
                # Create a copy of the original image to modify
                result_img = original_img.copy()
                
                # Place the cropped image back into the original
                result_img[y_pos:y_pos+height, x_pos:x_pos+width] = cropped_img
                
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

    def _find_original_image(self, cropped_basename, original_images, naming_pattern):
        """
        Find the corresponding original image based on naming pattern.
        
        Args:
            cropped_basename: Base name of the cropped image (without extension)
            original_images: Dictionary of original image filenames
            naming_pattern: Pattern to derive original name from cropped name
            
        Returns:
            str: Filename of the matching original image, or None if not found
        """
        # Different naming pattern strategies:
        if naming_pattern == "prefix":
            # Example: crop_original.jpg -> original.jpg
            # Find the original where cropped has a prefix
            prefix_len = self.app.prefix_length.get()
            original_base = cropped_basename[prefix_len:] if len(cropped_basename) > prefix_len else cropped_basename
            
            # Find original with this base name and any extension
            for orig_name in original_images.keys():
                orig_base = os.path.splitext(orig_name)[0]
                if orig_base == original_base:
                    return orig_name
        
        elif naming_pattern == "suffix":
            # Example: original_crop.jpg -> original.jpg
            # Find the original where cropped has a suffix
            suffix_len = self.app.suffix_length.get()
            original_base = cropped_basename[:-suffix_len] if len(cropped_basename) > suffix_len else cropped_basename
            
            # Find original with this base name and any extension
            for orig_name in original_images.keys():
                orig_base = os.path.splitext(orig_name)[0]
                if orig_base == original_base:
                    return orig_name
        
        elif naming_pattern == "indexed":
            # Example: original_001.jpg comes from original.jpg
            # Remove numeric suffix
            import re
            original_base = re.sub(r'_\d+$', '', cropped_basename)
            
            # Find original with this base name and any extension
            for orig_name in original_images.keys():
                orig_base = os.path.splitext(orig_name)[0]
                if orig_base == original_base:
                    return orig_name
        
        elif naming_pattern == "metadata":
            # The original filename should be stored in metadata
            # This would be implemented with the _get_crop_position method
            # using the metadata for both position and original filename
            pass
        
        # If no match found, try exact match
        for orig_name in original_images.keys():
            orig_base = os.path.splitext(orig_name)[0]
            if orig_base == cropped_basename:
                return orig_name
        
        return None

    def _get_crop_position(self, cropped_path, cropped_basename, position_source):
        """
        Get crop position data from the specified source.
        
        Args:
            cropped_path: Path to the cropped image
            cropped_basename: Base name of the cropped image
            position_source: Source for position data ('filename', 'metadata', or 'defaults')
            
        Returns:
            tuple: (x, y, width, height) or (None, None, None, None) if not found
        """
        if position_source == "filename":
            # Extract coordinates from filename
            # Example format: original_x100_y200_w300_h400.jpg
            import re
            x_match = re.search(r'_x(\d+)', cropped_basename)
            y_match = re.search(r'_y(\d+)', cropped_basename)
            w_match = re.search(r'_w(\d+)', cropped_basename)
            h_match = re.search(r'_h(\d+)', cropped_basename)
            
            if x_match and y_match and w_match and h_match:
                x = int(x_match.group(1))
                y = int(y_match.group(1))
                w = int(w_match.group(1))
                h = int(h_match.group(1))
                return (x, y, w, h)
        
        elif position_source == "metadata":
            # Look for a metadata JSON file with the same basename
            metadata_path = os.path.splitext(cropped_path)[0] + ".json"
            if os.path.exists(metadata_path):
                try:
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    if 'crop_x' in metadata and 'crop_y' in metadata and 'crop_width' in metadata and 'crop_height' in metadata:
                        return (
                            metadata['crop_x'], 
                            metadata['crop_y'], 
                            metadata['crop_width'], 
                            metadata['crop_height']
                        )
                except Exception as e:
                    print(f"Error reading metadata for {cropped_path}: {str(e)}")
        
        # If we couldn't extract position data, return None values
        return (None, None, None, None)