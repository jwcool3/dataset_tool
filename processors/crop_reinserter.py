"""
Crop Reinserter for Dataset Preparation Tool
Reinserts cropped images back into their original uncropped versions.
"""

import os
import cv2
import numpy as np
import re
import json

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
        """
        # Create output directory for reinserted images
        reinsert_output_dir = os.path.join(output_dir, "reinserted")
        os.makedirs(reinsert_output_dir, exist_ok=True)
        
        # Output debug information
        print(f"Reinsertion: Input (Cropped) Dir: {input_dir}")
        print(f"Reinsertion: Source (Original) Dir: {self.app.source_images_dir.get()}")
        print(f"Reinsertion: Output Dir: {reinsert_output_dir}")
        
        # Find all cropped images
        cropped_images = []
        for root, _, files in os.walk(input_dir):
            # Skip if this is a 'masks' directory
            if os.path.basename(root).lower() == "masks":
                print(f"Skipping masks directory: {root}")
                continue
                
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    cropped_images.append(os.path.join(root, file))
        
        print(f"Found {len(cropped_images)} cropped images to process")
        
        # Get source directory (original images)
        source_dir = self.app.source_images_dir.get()
        if not source_dir or not os.path.isdir(source_dir):
            self.app.status_label.config(text="Source images directory not set or invalid.")
            return False
        
        # Load all source images (excluding files in any masks subdirectories)
        source_images = {}
        for root, dirs, files in os.walk(source_dir):
            # Skip if this is a 'masks' directory
            if os.path.basename(root).lower() == "masks":
                continue
                
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    source_images[file] = os.path.join(root, file)
        
        # Rest of the method remains the same...
        
        # Get reinsertion parameters
        padding_percent = self.app.reinsert_padding.get()
        match_method = self.app.reinsert_match_method.get()
        
        # Process each cropped image
        total_images = len(cropped_images)
        processed_count = 0
        failed_count = 0
        
        for idx, cropped_path in enumerate(cropped_images):
            if not self.app.processing:  # Check if processing was cancelled
                break
            
            # Get cropped image filename
            cropped_filename = os.path.basename(cropped_path)
            cropped_basename, cropped_ext = os.path.splitext(cropped_filename)
            
            # Inside the loop where you process each cropped image
            try:
                # Match cropped image to source image
                source_filename = self._match_source_image(cropped_basename, source_images, match_method)
                
                if not source_filename:
                    self.app.status_label.config(text=f"Source image not found for {cropped_filename}")
                    failed_count += 1
                    continue
                
                source_path = source_images[source_filename]
                
                # Load images
                cropped_img = cv2.imread(cropped_path)
                source_img = cv2.imread(source_path)
                
                if cropped_img is None or source_img is None:
                    self.app.status_label.config(text=f"Error loading images for {cropped_filename}")
                    failed_count += 1
                    continue
                
                # Find if there's a corresponding mask for this cropped image
                mask_path = None
                cropped_dir = os.path.dirname(cropped_path)
                
                # Check if there's a masks directory in the same directory as the cropped image
                masks_dir = os.path.join(cropped_dir, "masks")
                if os.path.isdir(masks_dir):
                    # Try with the same filename
                    potential_mask = os.path.join(masks_dir, cropped_filename)
                    if os.path.exists(potential_mask):
                        mask_path = potential_mask
                    else:
                        # Try with different extensions
                        base_name = os.path.splitext(cropped_filename)[0]
                        for ext in ['.png', '.jpg', '.jpeg']:
                            potential_mask = os.path.join(masks_dir, base_name + ext)
                            if os.path.exists(potential_mask):
                                mask_path = potential_mask
                                break
                
                # Get dimensions
                source_height, source_width = source_img.shape[:2]
                crop_height, crop_width = cropped_img.shape[:2]
                
                # Calculate crop position based on padding percent
                x_pos, y_pos, insert_width, insert_height = self._calculate_insertion_position(
                    source_img, 
                    cropped_img, 
                    padding_percent
                )
                
                # Create a copy of the source image to modify
                result_img = source_img.copy()
                
                # Resize cropped image if needed to fit calculated dimensions
                if crop_width != insert_width or crop_height != insert_height:
                    resized_crop = cv2.resize(cropped_img, (insert_width, insert_height), 
                                        interpolation=cv2.INTER_LANCZOS4)
                else:
                    resized_crop = cropped_img
                
                # Ensure coordinates are within bounds
                x_end = min(x_pos + insert_width, source_width)
                y_end = min(y_pos + insert_height, source_height)
                insert_width = x_end - x_pos
                insert_height = y_end - y_pos
                
                # Create a mask for blending (if available)
                blend_mask = None
                if mask_path and os.path.exists(mask_path):
                    blend_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if blend_mask is not None:
                        # Resize the mask to match the insertion dimensions
                        blend_mask = cv2.resize(blend_mask, (insert_width, insert_height), 
                                            interpolation=cv2.INTER_NEAREST)
                        # Normalize to 0-1 range for blending
                        blend_mask = blend_mask.astype(float) / 255.0
                
                # Insert the cropped image with blending if mask is available
                if blend_mask is not None:
                    # Extract the region where we'll insert
                    roi = result_img[y_pos:y_end, x_pos:x_end]
                    
                    # Use the mask to blend the cropped image with the source
                    for c in range(3):  # For each color channel
                        roi[:,:,c] = roi[:,:,c] * (1 - blend_mask) + resized_crop[:,:,c] * blend_mask
                else:
                    # Simple insertion without blending
                    result_img[y_pos:y_end, x_pos:x_end] = resized_crop[:insert_height, :insert_width]
                
                # Save the reinserted image
                output_path = os.path.join(reinsert_output_dir, f"reinserted_{cropped_filename}")
                cv2.imwrite(output_path, result_img)
                
                # Also save a comparison image for debugging
                comparison = np.hstack((source_img, result_img))
                cv2.imwrite(os.path.join(reinsert_output_dir, f"comparison_{cropped_filename}"), comparison)
                
                processed_count += 1
                # DEBUG - Show dimensions before insertion
                print(f"Insertion region: ({x_pos},{y_pos}) to ({x_end},{y_end})")
                print(f"Final insert dimensions: {insert_width}x{insert_height}")
                print(f"Resized crop dimensions: {resized_crop.shape}")

                # CRITICAL CHECK - Make sure we're not replacing the entire image
                if x_pos == 0 and y_pos == 0 and insert_width == source_width and insert_height == source_height:
                    print("WARNING: Insertion would replace entire image. Using center positioning instead.")
                    # Calculate a more reasonable position (center with 20% of image size)
                    center_width = min(source_width // 2, crop_width)
                    center_height = min(source_height // 2, crop_height)
                    x_pos = (source_width - center_width) // 2
                    y_pos = (source_height - center_height) // 2
                    x_end = x_pos + center_width
                    y_end = y_pos + center_height
                    insert_width = center_width
                    insert_height = center_height
                    # Resize crop to fit this region
                    resized_crop = cv2.resize(cropped_img, (insert_width, insert_height), interpolation=cv2.INTER_LANCZOS4)
                    print(f"New insertion region: ({x_pos},{y_pos}) to ({x_end},{y_end})")

                try:
                    # Actually perform the insertion
                    result_img[y_pos:y_end, x_pos:x_end] = resized_crop[:insert_height, :insert_width]
                    
                    # For debugging, also save a before/after comparison
                    comparison = np.hstack((source_img, result_img))
                    cv2.imwrite(os.path.join(reinsert_output_dir, f"comparison_{os.path.basename(cropped_filename)}"), comparison)
                except Exception as e:
                    print(f"ERROR during insertion: {str(e)}")
            except Exception as e:
                self.app.status_label.config(text=f"Error processing {cropped_filename}: {str(e)}")
                print(f"Error in reinsert_crops: {str(e)}")
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
            self.app.status_label.config(text=f"Reinsertion completed. Processed {processed_count} images. Failed: {failed_count}.")
        else:
            self.app.status_label.config(text=f"Reinsertion completed. Processed {processed_count} images.")
        
        self.app.progress_bar['value'] = 100
        return processed_count > 0
    
    def _match_source_image(self, cropped_basename, source_images, match_method):
        """
        Match cropped image to source image using the specified method.
        
        Args:
            cropped_basename: Base name of the cropped image
            source_images: Dictionary of source images
            match_method: Method to use for matching
            
        Returns:
            str: Filename of the matching source image, or None if not found
        """
        print(f"Trying to match: {cropped_basename} using method: {match_method}")
        print(f"Available source images: {list(source_images.keys())[:5]}...")
        
        result = None
        
        if match_method == "name_prefix":
            # Remove prefix (anything before first underscore)
            parts = cropped_basename.split('_', 1)
            if len(parts) > 1:
                # The source name is everything after the first underscore
                source_base = parts[1]
                for source_name in source_images:
                    source_basename = os.path.splitext(source_name)[0]
                    if source_basename == source_base:
                        result = source_name
                        break
        
        elif match_method == "name_suffix":
            # Remove suffix (anything after last underscore)
            parts = cropped_basename.rsplit('_', 1)
            if len(parts) > 1:
                # The source name is everything before the last underscore
                source_base = parts[0]
                for source_name in source_images:
                    source_basename = os.path.splitext(source_name)[0]
                    if source_basename == source_base:
                        result = source_name
                        break
        
        elif match_method == "metadata":
            # Check for a metadata JSON file with the same base name
            # We need the full path of a JSON file with the same basename
            metadata_path = os.path.join(os.path.dirname(self.app.input_dir.get()), f"{cropped_basename}.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    if 'source_image' in metadata:
                        result = metadata['source_image']
                except Exception as e:
                    print(f"Error reading metadata: {str(e)}")
        
        elif match_method == "numeric_match":
            # Extract numeric part of filename and match to same number in source
            numbers = re.findall(r'\d+', cropped_basename)
            if numbers:
                # Use the last number in the filename
                number = numbers[-1]
                for source_name in source_images:
                    source_numbers = re.findall(r'\d+', os.path.splitext(source_name)[0])
                    if source_numbers and source_numbers[-1] == number:
                        result = source_name
                        break
        
        # Fall back to exact match
        if not result:
            for source_name in source_images:
                source_basename = os.path.splitext(source_name)[0]
                if source_basename == cropped_basename:
                    result = source_name
                    break
        
        # If all else fails, try to find a source image that contains the cropped basename
        if not result:
            for source_name in source_images:
                source_basename = os.path.splitext(source_name)[0]
                if cropped_basename in source_basename or source_basename in cropped_basename:
                    result = source_name
                    break

        # Before returning, print the result
        if result:
            print(f"Match found: {result}")
        else:
            print(f"No match found for {cropped_basename}")
        
        return result
    

    
    def _calculate_insertion_position(self, source_img, cropped_img, padding_percent):
        """
        Calculate where to insert the cropped image in the source image.
        """
        source_height, source_width = source_img.shape[:2]
        crop_height, crop_width = cropped_img.shape[:2]
        
        # For automatic positioning, use feature matching
        if self.app.use_center_position.get():
            # Convert to grayscale for feature detection
            source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY) if len(source_img.shape) == 3 else source_img
            crop_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY) if len(cropped_img.shape) == 3 else cropped_img
            
            # Try using ORB feature detector and matcher
            try:
                # Create ORB detector
                orb = cv2.ORB_create(nfeatures=500)
                
                # Find keypoints and descriptors
                kp1, des1 = orb.detectAndCompute(source_gray, None)
                kp2, des2 = orb.detectAndCompute(crop_gray, None)
                
                if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                    # Create BF matcher
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    
                    # Match descriptors
                    matches = bf.match(des2, des1)
                    
                    # Sort by distance
                    matches = sorted(matches, key=lambda x: x.distance)
                    
                    # Use only good matches (first 10-20)
                    good_matches = matches[:min(20, len(matches))]
                    
                    if len(good_matches) >= 4:  # Need at least 4 points for homography
                        # Extract points
                        src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        
                        # Find homography
                        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        
                        if H is not None:
                            # Get corners of cropped image
                            h, w = crop_gray.shape
                            corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                            
                            # Transform corners to source image
                            transformed = cv2.perspectiveTransform(corners, H)
                            
                            # Get bounding box
                            x_min = max(0, int(np.min(transformed[:, 0, 0])))
                            y_min = max(0, int(np.min(transformed[:, 0, 1])))
                            x_max = min(source_width, int(np.max(transformed[:, 0, 0])))
                            y_max = min(source_height, int(np.max(transformed[:, 0, 1])))
                            
                            # Calculate dimensions
                            width = x_max - x_min
                            height = y_max - y_min
                            
                            # Ensure reasonable dimensions
                            if width > 20 and height > 20:
                                return x_min, y_min, width, height
            except Exception as e:
                print(f"Feature matching failed: {str(e)}")
            
            # If feature matching fails, fall back to template matching
            try:
                # Use template matching as a fallback
                result = cv2.matchTemplate(source_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > 0.3:  # Reasonable threshold
                    return max_loc[0], max_loc[1], crop_width, crop_height
            except Exception as e:
                print(f"Template matching failed: {str(e)}")
        
        # Fall back to center positioning
        x_pos = (source_width - crop_width) // 2
        y_pos = (source_height - crop_height) // 2
        
        return x_pos, y_pos, crop_width, crop_height