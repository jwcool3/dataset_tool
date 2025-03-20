"""
Enhanced Crop Reinserter for Dataset Preparation Tool
Specialized for handling resolution differences between source and processed images.
"""

import os
import cv2
import numpy as np
import re
import json
from skimage.transform import resize as skimage_resize
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_ubyte, img_as_float

class EnhancedCropReinserter:
    """Reinserts processed regions back into original images, handling resolution differences."""
    
    def __init__(self, app):
        """
        Initialize enhanced crop reinserter.
        
        Args:
            app: The main application with shared variables and UI controls
        """
        self.app = app
    
    def reinsert_crops(self, input_dir, output_dir):
        """
        Reinsert processed regions back into original images with enhanced resolution handling.
        
        Args:
            input_dir: Input directory containing processed images
            output_dir: Output directory for reinserted images
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create output directory for reinserted images
        reinsert_output_dir = os.path.join(output_dir, "reinserted")
        os.makedirs(reinsert_output_dir, exist_ok=True)
        
        # Output debug information
        print(f"Enhanced Reinsertion: Input (Processed) Dir: {input_dir}")
        print(f"Enhanced Reinsertion: Source (Original) Dir: {self.app.source_images_dir.get()}")
        print(f"Enhanced Reinsertion: Output Dir: {reinsert_output_dir}")
        print(f"Enhanced Reinsertion: Mask-only mode: {self.app.reinsert_mask_only.get()}")
        
        # Create a debug directory for visualization if in debug mode
        debug_dir = None
        if self.app.debug_mode.get():
            debug_dir = os.path.join(output_dir, "reinsert_debug")
            os.makedirs(debug_dir, exist_ok=True)
            print(f"Debug mode enabled, saving visualizations to: {debug_dir}")
        
        # Find all processed images and corresponding masks
        processed_images = []
        
        for root, dirs, files in os.walk(input_dir):
            # Skip if this is a 'masks' directory - we'll handle masks separately
            if os.path.basename(root).lower() == "masks":
                continue
                
            # Find image files
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    processed_path = os.path.join(root, file)
                    
                    # Look for corresponding mask
                    mask_path = None
                    masks_dir = os.path.join(root, "masks")
                    if os.path.isdir(masks_dir):
                        # Try exact filename match first
                        potential_mask = os.path.join(masks_dir, file)
                        if os.path.exists(potential_mask):
                            mask_path = potential_mask
                        else:
                            # Try different extensions
                            base_name = os.path.splitext(file)[0]
                            for ext in ['.png', '.jpg', '.jpeg']:
                                potential_mask = os.path.join(masks_dir, base_name + ext)
                                if os.path.exists(potential_mask):
                                    mask_path = potential_mask
                                    break
                    
                    # Add to processing list
                    processed_images.append({
                        'processed_path': processed_path,
                        'mask_path': mask_path,
                        'filename': file
                    })
        
        # Get source directory (original images)
        source_dir = self.app.source_images_dir.get()
        if not source_dir or not os.path.isdir(source_dir):
            self.app.status_label.config(text="Source images directory not set or invalid.")
            return False
        
        # Load all source images
        source_images = {}
        for root, dirs, files in os.walk(source_dir):
            # Skip if this is a 'masks' directory
            if os.path.basename(root).lower() == "masks":
                continue
                
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    source_images[file] = os.path.join(root, file)
        
        # Process each image
        total_images = len(processed_images)
        processed_count = 0
        failed_count = 0
        
        for idx, img_data in enumerate(processed_images):
            if not self.app.processing:  # Check if processing was cancelled
                break
            
            processed_path = img_data['processed_path']
            mask_path = img_data['mask_path']
            filename = img_data['filename']
            
            try:
                # Match processed image to source image
                matched_source = self._match_source_image(filename, source_images)
                
                if not matched_source:
                    self.app.status_label.config(text=f"Source image not found for {filename}")
                    failed_count += 1
                    continue
                
                source_path = source_images[matched_source]
                
                # Perform the resolution-aware reinsertion
                success = self._reinsert_with_resolution_handling(
                    source_path,
                    processed_path,
                    mask_path,
                    os.path.join(reinsert_output_dir, f"reinserted_{filename}"),
                    debug_dir
                )
                
                if success:
                    processed_count += 1
                else:
                    failed_count += 1
                
            except Exception as e:
                self.app.status_label.config(text=f"Error processing {filename}: {str(e)}")
                print(f"Error in enhanced_reinsert_crops: {str(e)}")
                import traceback
                traceback.print_exc()
                failed_count += 1
                continue
            
            # Update progress
            progress = (idx + 1) / total_images * 100
            self.app.progress_bar['value'] = min(progress, 100)
            self.app.status_label.config(text=f"Processed {idx+1}/{total_images} images")
            self.app.root.update_idletasks()
        
        # Final status update
        if failed_count > 0:
            self.app.status_label.config(text=f"Enhanced reinsertion completed. Processed {processed_count} images. Failed: {failed_count}.")
        else:
            self.app.status_label.config(text=f"Enhanced reinsertion completed. Successfully processed {processed_count} images.")
        
        self.app.progress_bar['value'] = 100
        return processed_count > 0
    
    def _match_source_image(self, processed_filename, source_images):
        """
        Match processed image filename to source image using multiple strategies.
        
        Args:
            processed_filename: Filename of the processed image
            source_images: Dictionary of source images
            
        Returns:
            str: Matched source filename or None if no match found
        """
        # Try direct filename match first
        if processed_filename in source_images:
            return processed_filename
        
        # Extract base name without extension
        base_name = os.path.splitext(processed_filename)[0]
        
        # Strategy 1: Try different extensions with the same base name
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_match = base_name + ext
            if potential_match in source_images:
                return potential_match
        
        # Strategy 2: Check if source contains the base name
        for source_name in source_images:
            source_base = os.path.splitext(source_name)[0]
            if base_name in source_base or source_base in base_name:
                return source_name
        
        # Strategy 3: Try numerical matching
        numbers_in_processed = re.findall(r'\d+', base_name)
        if numbers_in_processed:
            # Use the last number in the filename
            number = numbers_in_processed[-1]
            for source_name in source_images:
                source_numbers = re.findall(r'\d+', os.path.splitext(source_name)[0])
                if source_numbers and source_numbers[-1] == number:
                    return source_name
        
        # Strategy 4: Check for JSON metadata file
        metadata_path = os.path.splitext(processed_filename)[0] + "_info.json"
        metadata_fullpath = os.path.join(os.path.dirname(self.app.input_dir.get()), metadata_path)
        if os.path.exists(metadata_fullpath):
            try:
                with open(metadata_fullpath, 'r') as f:
                    metadata = json.load(f)
                if 'source_image' in metadata:
                    source_name = metadata['source_image']
                    if source_name in source_images:
                        return source_name
            except Exception as e:
                print(f"Error reading metadata: {str(e)}")
        
        # If all else fails, use image similarity to find the best match
        # (This is computationally expensive, so we use it as a last resort)
        if self.app.debug_mode.get():  # Only do this in debug mode
            try:
                print("Attempting image similarity matching...")
                processed_img = cv2.imread(os.path.join(self.app.input_dir.get(), processed_filename))
                if processed_img is not None:
                    # Process a small version for speed
                    processed_small = cv2.resize(processed_img, (64, 64))
                    processed_gray = cv2.cvtColor(processed_small, cv2.COLOR_BGR2GRAY)
                    
                    best_score = -1
                    best_match = None
                    
                    for source_name, source_path in source_images.items():
                        source_img = cv2.imread(source_path)
                        if source_img is not None:
                            source_small = cv2.resize(source_img, (64, 64))
                            source_gray = cv2.cvtColor(source_small, cv2.COLOR_BGR2GRAY)
                            
                            # Calculate structural similarity
                            score = ssim(processed_gray, source_gray)
                            
                            if score > best_score:
                                best_score = score
                                best_match = source_name
                    
                    if best_score > 0.5:  # Threshold for a reasonably good match
                        print(f"Found match by image similarity: {best_match} (score: {best_score:.2f})")
                        return best_match
            except Exception as e:
                print(f"Error during image similarity matching: {str(e)}")
        
        # No match found
        return None
    
    def _reinsert_with_resolution_handling(self, source_path, processed_path, mask_path, output_path, debug_dir=None):
        """
        Reinsert a processed image region into the source image, handling resolution differences.
        
        Args:
            source_path: Path to source (original) image
            processed_path: Path to processed image
            mask_path: Path to mask (if available)
            output_path: Path to save the result
            debug_dir: Directory to save debug visualizations (if enabled)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Load images
        source_img = cv2.imread(source_path)
        processed_img = cv2.imread(processed_path)
        
        if source_img is None or processed_img is None:
            print(f"Failed to load source or processed image: {source_path} / {processed_path}")
            return False
        
        # Get dimensions
        source_h, source_w = source_img.shape[:2]
        processed_h, processed_w = processed_img.shape[:2]
        
        # Load and prepare mask
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to load mask: {mask_path}")
                # Try to create a simple mask (white rectangle) if no mask is available
                mask = np.ones((processed_h, processed_w), dtype=np.uint8) * 255
        else:
            # Create a simple mask (white rectangle) if no mask is available
            mask = np.ones((processed_h, processed_w), dtype=np.uint8) * 255
        
        # Check resolution difference and if we need resizing
        resolution_diff = (abs(source_w - processed_w) / source_w > 0.05 or 
                          abs(source_h - processed_h) / source_h > 0.05)
        
        # If dimensions are very different, resize processed image and mask to match source
        if resolution_diff:
            print(f"Resolution difference detected: Source {source_w}x{source_h}, Processed {processed_w}x{processed_h}")
            
            # Save original images for debugging if enabled
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "source_original.png"), source_img)
                cv2.imwrite(os.path.join(debug_dir, "processed_original.png"), processed_img)
                if mask is not None:
                    cv2.imwrite(os.path.join(debug_dir, "mask_original.png"), mask)
            
            # Approach 1: Resize processed image and mask to match source dimensions
            processed_img_resized = cv2.resize(processed_img, (source_w, source_h), 
                                           interpolation=cv2.INTER_LANCZOS4)
            mask_resized = cv2.resize(mask, (source_w, source_h), 
                                   interpolation=cv2.INTER_NEAREST)
            
            # Save resized images for debugging if enabled
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "processed_resized.png"), processed_img_resized)
                cv2.imwrite(os.path.join(debug_dir, "mask_resized.png"), mask_resized)
            
            # Create a floating point mask for blending (0.0 to 1.0)
            mask_float = mask_resized.astype(float) / 255.0
            
            # Create 3-channel mask for RGB images
            mask_float_3d = np.stack([mask_float] * 3, axis=2)
            
            # Perform blending with full resolution images
            # The mask controls which parts of the processed image replace the source
            result_img = source_img * (1 - mask_float_3d) + processed_img_resized * mask_float_3d
            
            # Convert back to uint8 for saving
            result_img = result_img.astype(np.uint8)
            
        else:
            # Similar resolutions - simple blending
            # Create a floating point mask for blending (0.0 to 1.0)
            mask_float = mask.astype(float) / 255.0
            
            # Create 3-channel mask for RGB images
            mask_float_3d = np.stack([mask_float] * 3, axis=2)
            
            # Perform blending
            result_img = source_img * (1 - mask_float_3d) + processed_img * mask_float_3d
            
            # Convert back to uint8 for saving
            result_img = result_img.astype(np.uint8)
        
        # Save the result
        cv2.imwrite(output_path, result_img)
        
        # Save side-by-side comparison if debug is enabled
        if debug_dir:
            # Resize for comparison if needed
            if source_img.shape != processed_img.shape:
                processed_img_display = cv2.resize(processed_img, (source_img.shape[1], source_img.shape[0]))
            else:
                processed_img_display = processed_img
                
            # Create side-by-side comparison
            comparison = np.hstack((source_img, processed_img_display, result_img))
            cv2.imwrite(os.path.join(debug_dir, f"comparison_{os.path.basename(output_path)}"), comparison)
            
            # Create visualization of the mask
            if mask is not None:
                # Resize mask for visualization if needed
                if mask.shape[:2] != source_img.shape[:2]:
                    mask_viz = cv2.resize(mask, (source_img.shape[1], source_img.shape[0]))
                else:
                    mask_viz = mask
                    
                # Create a colorized mask for better visualization
                mask_colored = cv2.cvtColor(mask_viz, cv2.COLOR_GRAY2BGR)
                # Make it more visible by using red channel
                mask_colored[:,:,0] = 0  # Zero out blue channel
                mask_colored[:,:,1] = 0  # Zero out green channel
                
                # Create a visualization with mask overlay
                mask_overlay = source_img.copy()
                mask_binary = (mask_viz > 127).astype(np.uint8) * 255
                mask_overlay[mask_binary > 0] = [0, 0, 255]  # Red color for mask regions
                
                # Save mask visualizations
                cv2.imwrite(os.path.join(debug_dir, f"mask_viz_{os.path.basename(output_path)}"), mask_colored)
                cv2.imwrite(os.path.join(debug_dir, f"mask_overlay_{os.path.basename(output_path)}"), mask_overlay)
        
        return True