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
        
        Args:
            input_dir: Input directory containing cropped images
            output_dir: Output directory for reinserted images
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create output directory for reinserted images
        reinsert_output_dir = os.path.join(output_dir, "reinserted")
        os.makedirs(reinsert_output_dir, exist_ok=True)
        
        # Output debug information
        print(f"Reinsertion: Input (Cropped) Dir: {input_dir}")
        print(f"Reinsertion: Source (Original) Dir: {self.app.source_images_dir.get()}")
        print(f"Reinsertion: Output Dir: {reinsert_output_dir}")
        print(f"Reinsertion: Mask-only mode: {self.app.reinsert_mask_only.get()}")
        
        # Find all cropped images and respect subfolder structure
        cropped_images = []
        subfolders = set()
        
        for root, _, files in os.walk(input_dir):
            # Skip if this is a 'masks' directory
            if os.path.basename(root).lower() == "masks":
                print(f"Skipping masks directory: {root}")
                continue
                
            # Get the relative path from input_dir to this subfolder
            rel_path = os.path.relpath(root, input_dir)
            if rel_path != '.':
                subfolders.add(rel_path)
            
            # Find image files
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    cropped_images.append(os.path.join(root, file))
                    
                    # Also check for metadata JSON file
                    json_path = os.path.splitext(os.path.join(root, file))[0] + "_crop_info.json"
                    if os.path.exists(json_path):
                        print(f"Found metadata for {file}")
        
        print(f"Found {len(cropped_images)} cropped images to process across {len(subfolders) if subfolders else 1} subfolders")
        
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
        
        # Get reinsertion parameters
        padding_percent = self.app.reinsert_padding.get()
        match_method = self.app.reinsert_match_method.get()
        use_mask_only = self.app.reinsert_mask_only.get()
        
        # Process each cropped image
        total_images = len(cropped_images)
        processed_count = 0
        failed_count = 0
        
        for idx, cropped_path in enumerate(cropped_images):
            if not self.app.processing:  # Check if processing was cancelled
                break
            
            # Get cropped image filename
            cropped_filename = os.path.basename(cropped_path)
            cropped_basename, _ = os.path.splitext(cropped_filename)
            
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
                    resized_crop = cv2.resize(cropped_img, (insert_width, insert_height), 
                                         interpolation=cv2.INTER_LANCZOS4)
                    print(f"New insertion region: ({x_pos},{y_pos}) to ({x_end},{y_end})")
                
                # Handle mask reinsertion differently based on settings
                if use_mask_only and mask_path and os.path.exists(mask_path):
                    # Load the mask and make it binary
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        print(f"Error loading mask: {mask_path}")
                        # Fall back to regular insertion without mask
                        result_img[y_pos:y_end, x_pos:x_end] = resized_crop[:insert_height, :insert_width]
                    else:
                        # Resize mask to match cropped image if needed
                        if mask.shape != resized_crop.shape[:2]:
                            mask = cv2.resize(mask, (insert_width, insert_height), 
                                           interpolation=cv2.INTER_NEAREST)
                        
                        # Threshold mask to make it binary (0 or 255)
                        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                        
                        # Create a floating point mask for blending (0.0 to 1.0)
                        mask_float = binary_mask.astype(float) / 255.0
                        
                        # Create 3-channel mask for RGB images
                        mask_float_3d = np.stack([mask_float] * 3, axis=2)
                        
                        # Create a ROI (Region of Interest) in the source image
                        roi = result_img[y_pos:y_end, x_pos:x_end]
                        
                        # Make sure sizes match before blending (in case of boundary issues)
                        mask_height, mask_width = mask_float_3d.shape[:2]
                        crop_height, crop_width = resized_crop.shape[:2]
                        roi_height, roi_width = roi.shape[:2]
                        
                        # Use the minimum dimensions to ensure no out-of-bounds access
                        use_height = min(mask_height, crop_height, roi_height)
                        use_width = min(mask_width, crop_width, roi_width)
                        
                        # Trim all arrays to the same size
                        mask_float_3d = mask_float_3d[:use_height, :use_width]
                        resized_crop = resized_crop[:use_height, :use_width]
                        roi_view = roi[:use_height, :use_width]
                        
                        # Blend only the masked regions (where mask > 0)
                        blended_roi = roi_view * (1 - mask_float_3d) + resized_crop * mask_float_3d
                        
                        # Place the blended region back into the result image
                        result_img[y_pos:y_pos+use_height, x_pos:x_pos+use_width] = blended_roi
                else:
                    # Regular insertion (full crop replacement)
                    # Make sure dimensions match
                    insert_height = min(insert_height, resized_crop.shape[0])
                    insert_width = min(insert_width, resized_crop.shape[1])
                    
                    # Insert the cropped image
                    result_img[y_pos:y_pos+insert_height, x_pos:x_pos+insert_width] = resized_crop[:insert_height, :insert_width]
                
                # Save the reinserted image
                output_path = os.path.join(reinsert_output_dir, f"reinserted_{cropped_filename}")
                cv2.imwrite(output_path, result_img)
                
                # Also save a comparison image for debugging
                comparison = np.hstack((source_img, result_img))
                cv2.imwrite(os.path.join(reinsert_output_dir, f"comparison_{cropped_filename}"), comparison)
                
                processed_count += 1
                
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
        
        Args:
            source_img: Source image (numpy array)
            cropped_img: Cropped image (numpy array)
            padding_percent: Padding percentage
            
        Returns:
            tuple: (x_pos, y_pos, insert_width, insert_height)
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
                # Create ORB detector with more features for better matching
                orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, WTA_K=2)
                
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
                    
                    # Use only good matches (first 30)
                    good_matches = matches[:min(30, len(matches))]
                    
                    if len(good_matches) >= 10:  # Need sufficient points for reliable homography
                        # Extract points
                        src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        
                        # Find homography with RANSAC for robustness
                        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        
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
                            
                            print("Feature matching found position but dimensions were too small")
                        else:
                            print("Homography calculation failed")
                    else:
                        print(f"Not enough good matches: {len(good_matches)}")
                else:
                    print("No descriptors found in source or crop")
            
            except Exception as e:
                print(f"Feature matching failed: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # If feature matching fails, try template matching as fallback
            try:
                # Use template matching as a fallback
                # Resize cropped image if it's too large for effective template matching
                max_template_dimension = 300
                if crop_width > max_template_dimension or crop_height > max_template_dimension:
                    scale = min(max_template_dimension / crop_width, max_template_dimension / crop_height)
                    template_width = int(crop_width * scale)
                    template_height = int(crop_height * scale)
                    template = cv2.resize(crop_gray, (template_width, template_height), 
                                      interpolation=cv2.INTER_AREA)
                    
                    # If source is also large, resize it proportionately
                    if source_width > 1000 or source_height > 1000:
                        source_scale = scale * 1.5  # Scale slightly more to ensure template fits
                        scaled_source_width = int(source_width * source_scale)
                        scaled_source_height = int(source_height * source_scale)
                        scaled_source = cv2.resize(source_gray, (scaled_source_width, scaled_source_height), 
                                              interpolation=cv2.INTER_AREA)
                        
                        # Perform template matching
                        result = cv2.matchTemplate(scaled_source, template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)
                        
                        if max_val > 0.4:  # Threshold for a decent match
                            # Convert back to original coordinates
                            orig_x = int(max_loc[0] / source_scale)
                            orig_y = int(max_loc[1] / source_scale)
                            return orig_x, orig_y, crop_width, crop_height
                    else:
                        # Source is small enough to use directly
                        result = cv2.matchTemplate(source_gray, template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)
                        
                        if max_val > 0.4:
                            # Scale back to original crop size
                            return max_loc[0], max_loc[1], crop_width, crop_height
                else:
                    # Both images are small enough for direct template matching
                    result = cv2.matchTemplate(source_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val > 0.3:  # Lower threshold for direct matching
                        return max_loc[0], max_loc[1], crop_width, crop_height
                
                print(f"Template matching failed with correlation: {max_val:.2f}")
            
            except Exception as e:
                print(f"Template matching failed: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Fall back to center positioning if all else fails
        print("Falling back to center positioning")
        x_pos = (source_width - crop_width) // 2
        y_pos = (source_height - crop_height) // 2
        
        return x_pos, y_pos, crop_width, crop_height