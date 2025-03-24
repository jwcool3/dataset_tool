"""
Crop Exporter for Dataset Preparation Tool
Exports just the cropped mask regions (with optional transparency) as images or video.
"""

import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

class CropExporter:
    """Exports cropped mask regions as standalone images or video."""
    
    def __init__(self, app):
        """
        Initialize crop exporter.
        
        Args:
            app: The main application with shared variables and UI controls
        """
        self.app = app
    
    def export_cropped_areas(self, input_dir, output_dir):
        """
        Export just the cropped areas from mask regions.
        
        Args:
            input_dir: Input directory containing image-mask pairs
            output_dir: Output directory for exported crops
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create output directory for cropped areas
        crops_output_dir = os.path.join(output_dir, "cropped_only")
        os.makedirs(crops_output_dir, exist_ok=True)
        
        # Create a debug directory if debug mode is enabled
        debug_dir = None
        if self.app.debug_mode.get():
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
        
        # Find all image and mask pairs
        image_mask_pairs = []
        
        for root, dirs, files in os.walk(input_dir):
            # Skip if this is a 'masks' directory
            if os.path.basename(root).lower() == "masks":
                continue
                
            # Check if there's a 'masks' subdirectory
            masks_dir = os.path.join(root, "masks")
            if os.path.isdir(masks_dir):
                # Create corresponding output directory
                rel_path = os.path.relpath(root, input_dir)
                current_output_dir = os.path.join(crops_output_dir, rel_path)
                os.makedirs(current_output_dir, exist_ok=True)
                
                # Find image-mask pairs
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(root, file)
                        
                        # Look for corresponding mask with same name
                        mask_path = os.path.join(masks_dir, file)
                        if os.path.exists(mask_path):
                            image_mask_pairs.append({
                                'image_path': image_path,
                                'mask_path': mask_path,
                                'output_dir': current_output_dir,
                                'filename': file
                            })
                        else:
                            # Try different extensions
                            basename = os.path.splitext(file)[0]
                            for ext in ['.png', '.jpg', '.jpeg']:
                                alt_mask_path = os.path.join(masks_dir, basename + ext)
                                if os.path.exists(alt_mask_path):
                                    image_mask_pairs.append({
                                        'image_path': image_path,
                                        'mask_path': alt_mask_path,
                                        'output_dir': current_output_dir,
                                        'filename': file
                                    })
                                    break
        
        # Check if we found any image-mask pairs
        if not image_mask_pairs:
            self.app.status_label.config(text="No image-mask pairs found for export.")
            return False
        
        # Process all image-mask pairs
        total_pairs = len(image_mask_pairs)
        processed_count = 0
        
        # Track all exported images for video creation
        all_exported_images = []
        
        for idx, pair in enumerate(image_mask_pairs):
            if not self.app.processing:  # Check if processing was cancelled
                break
            
            try:
                # Load the image and mask
                image = cv2.imread(pair['image_path'])
                mask = cv2.imread(pair['mask_path'], cv2.IMREAD_GRAYSCALE)
                
                if image is None or mask is None:
                    print(f"Error loading image or mask: {pair['image_path']} or {pair['mask_path']}")
                    continue
                
                # Resize mask to match image if needed
                if image.shape[:2] != mask.shape[:2]:
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Create the output filename
                basename = os.path.splitext(pair['filename'])[0]
                
                # Choose the extension based on alpha setting
                ext = ".png" if self.app.export_with_alpha.get() else ".jpg"
                output_filename = basename + "_crop" + ext
                output_path = os.path.join(pair['output_dir'], output_filename)
                
                # Process the image based on whether we want alpha channel
                if self.app.export_with_alpha.get():
                    # Create an image with transparency
                    # Convert mask to binary
                    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    
                    # Add alpha channel to the image
                    b, g, r = cv2.split(image)
                    rgba = [b, g, r, binary_mask]
                    dst = cv2.merge(rgba, 4)
                    
                    # Save with transparency
                    cv2.imwrite(output_path, dst)
                else:
                    # Create a version with black background
                    # Convert mask to binary and create a 3-channel version
                    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    mask_3ch = cv2.merge([binary_mask, binary_mask, binary_mask])
                    
                    # Apply mask to original image (black background)
                    masked_image = cv2.bitwise_and(image, mask_3ch)
                    
                    # Save without transparency
                    cv2.imwrite(output_path, masked_image)
                
                # For debugging, save the original mask
                if debug_dir:
                    cv2.imwrite(os.path.join(debug_dir, f"mask_{basename}.png"), mask)
                
                # Add to list of exported images
                all_exported_images.append(output_path)
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {pair['filename']}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
            
            # Update progress
            progress = (idx + 1) / total_pairs * 100
            self.app.progress_bar['value'] = min(progress, 100)
            self.app.status_label.config(text=f"Exported {idx+1}/{total_pairs} cropped regions")
            self.app.root.update_idletasks()
        
        # Create video if requested
        if self.app.export_cropped_video.get() and all_exported_images:
            success = self._create_video_from_crops(all_exported_images, crops_output_dir)
            if success:
                self.app.status_label.config(text=f"Export completed. Created video from {processed_count} cropped regions.")
            else:
                self.app.status_label.config(text=f"Export completed with {processed_count} crops, but video creation failed.")
        else:
            self.app.status_label.config(text=f"Export completed. Processed {processed_count} cropped regions.")
        
        return processed_count > 0
    
    def _create_video_from_crops(self, image_paths, output_dir):
        """
        Create a video from the exported cropped images.
        
        Args:
            image_paths: List of paths to exported images
            output_dir: Directory to save the video
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Sort images to ensure proper sequence
        image_paths.sort()
        
        if not image_paths:
            return False
        
        try:
            # Load the first image to get dimensions
            first_img = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
            if first_img is None:
                print(f"Unable to read first image: {image_paths[0]}")
                return False
            
            # Get dimensions
            h, w = first_img.shape[:2]
            
            # Determine video format and codec
            video_format = self.app.cropped_video_format.get().lower()
            
            if video_format == "mp4":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif video_format == "avi":
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            elif video_format == "mov":
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            else:
                # Default to MP4
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_format = "mp4"
            
            # Create video output path
            video_path = os.path.join(output_dir, f"cropped_regions.{video_format}")
            
            # Create VideoWriter
            fps = self.app.cropped_video_fps.get()
            out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
            
            if not out.isOpened():
                print(f"Failed to open VideoWriter for {video_path}")
                return False
            
            # Add each image to the video
            total_frames = len(image_paths)
            self.app.status_label.config(text=f"Creating video from {total_frames} frames...")
            
            for i, img_path in enumerate(image_paths):
                if not self.app.processing:  # Check if processing was cancelled
                    break
                
                # Read image
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                
                if img is None:
                    print(f"Unable to read image: {img_path}")
                    continue
                
                # If image has alpha channel, blend with black background
                if img.shape[2] == 4:
                    # Split channels
                    b, g, r, a = cv2.split(img)
                    
                    # Normalize alpha to 0-1 range
                    alpha = a.astype(float) / 255.0
                    
                    # Create black background
                    black = np.zeros((h, w, 3), dtype=np.uint8)
                    
                    # Blend image with black background using alpha
                    r = (r * alpha).astype(np.uint8)
                    g = (g * alpha).astype(np.uint8)
                    b = (b * alpha).astype(np.uint8)
                    
                    # Merge channels
                    rgb = cv2.merge([b, g, r])
                else:
                    # Use image as is (assuming RGB)
                    rgb = img
                
                # Write frame to video
                out.write(rgb)
                
                # Update progress
                progress = (i + 1) / total_frames * 100
                self.app.progress_bar['value'] = min(progress, 100)
                if i % 10 == 0:  # Update UI every 10 frames to avoid slowdown
                    self.app.status_label.config(text=f"Creating video: {i+1}/{total_frames} frames")
                    self.app.root.update_idletasks()
            
            # Release video writer
            out.release()
            
            self.app.status_label.config(text=f"Video exported to: {video_path}")
            return True
            
        except Exception as e:
            print(f"Error creating video: {str(e)}")
            import traceback
            traceback.print_exc()
            return False