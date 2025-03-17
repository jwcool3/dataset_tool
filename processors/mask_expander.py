"""
Mask Expander for Dataset Preparation Tool
Expands mask regions in images or videos using dilation.
Based on the provided maskexpand.py implementation.
"""

import os
import cv2
import numpy as np
import glob

class MaskExpander:
    """Expands mask regions using dilation."""
    
    def __init__(self, app):
        """
        Initialize mask expander.
        
        Args:
            app: The main application with shared variables and UI controls
        """
        self.app = app
    
    def expand_masks(self, input_dir, output_dir):
        """
        Expand mask regions in images or videos.
        
        Args:
            input_dir: Input directory containing masks (images or videos)
            output_dir: Output directory for expanded masks
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create output directory for expanded masks
        expanded_output_dir = os.path.join(output_dir, "expanded_masks")
        os.makedirs(expanded_output_dir, exist_ok=True)
        
        # Get expansion parameters
        iterations = self.app.mask_expand_iterations.get()
        kernel_size = self.app.mask_expand_kernel_size.get()
        
        # Determine if we're processing videos or images
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        image_extensions = ['.png', '.jpg', '.jpeg']
        
        # Find all mask files (both videos and images)
        mask_files = []
        
        # First check for videos
        for ext in video_extensions:
            mask_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
            # Also check in masks subdirectory if it exists
            masks_dir = os.path.join(input_dir, "masks")
            if os.path.isdir(masks_dir):
                mask_files.extend(glob.glob(os.path.join(masks_dir, f"*{ext}")))
        
        # Then check for images
        for ext in image_extensions:
            mask_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
            # Also check in masks subdirectory if it exists
            masks_dir = os.path.join(input_dir, "masks")
            if os.path.isdir(masks_dir):
                mask_files.extend(glob.glob(os.path.join(masks_dir, f"*{ext}")))
        
        # Also scan any masks subdirectories in input_dir's subdirectories
        for root, dirs, _ in os.walk(input_dir):
            for dir in dirs:
                if dir.lower() == "masks":
                    mask_dir_path = os.path.join(root, dir)
                    for ext in image_extensions + video_extensions:
                        mask_files.extend(glob.glob(os.path.join(mask_dir_path, f"*{ext}")))
        
        if not mask_files:
            self.app.status_label.config(text="No mask files found.")
            return False
        
        # Create subdirectory for images if needed
        images_output_dir = os.path.join(expanded_output_dir, "images")
        os.makedirs(images_output_dir, exist_ok=True)
        
        # Create subdirectory for videos if needed
        videos_output_dir = os.path.join(expanded_output_dir, "videos")
        os.makedirs(videos_output_dir, exist_ok=True)
        
        # Process each mask file
        total_files = len(mask_files)
        processed_count = 0
        video_count = 0
        image_count = 0
        
        # Create kernel for dilation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        for idx, mask_path in enumerate(mask_files):
            if not self.app.processing:  # Check if processing was cancelled
                break
            
            # Get file extension
            _, ext = os.path.splitext(mask_path)
            ext = ext.lower()
            
            try:
                if ext in video_extensions:
                    # Process video mask
                    output_path = os.path.join(videos_output_dir, os.path.basename(mask_path))
                    success = self._expand_mask_video(mask_path, output_path, kernel, iterations)
                    if success:
                        video_count += 1
                else:
                    # Process image mask
                    output_path = os.path.join(images_output_dir, os.path.basename(mask_path))
                    success = self._expand_mask_image(mask_path, output_path, kernel, iterations)
                    if success:
                        image_count += 1
                
                if success:
                    processed_count += 1
                
            except Exception as e:
                self.app.status_label.config(text=f"Error processing {os.path.basename(mask_path)}: {str(e)}")
                print(f"Error in expand_masks: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
            
            # Update progress
            progress = (idx + 1) / total_files * 100
            self.app.progress_bar['value'] = min(progress, 100)
            
            # Update status periodically
            if idx % 5 == 0 or idx == total_files - 1:
                self.app.status_label.config(text=f"Processed {idx+1}/{total_files} mask files")
                self.app.root.update_idletasks()
        
        # Copy directory structure if requested
        if self.app.mask_expand_preserve_structure.get():
            self._copy_directory_structure(input_dir, expanded_output_dir)
        
        # Final status update
        self.app.status_label.config(text=f"Mask expansion completed. Processed {processed_count} files ({image_count} images, {video_count} videos).")
        self.app.progress_bar['value'] = 100
        return processed_count > 0
    
    def _expand_mask_image(self, input_path, output_path, kernel, iterations):
        """
        Expand mask regions in an image using dilation.
        
        Args:
            input_path: Path to input mask image
            output_path: Path to save expanded mask image
            kernel: Dilation kernel
            iterations: Number of dilation iterations
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Load the image
        image = cv2.imread(input_path)
        if image is None:
            return False
        
        # Convert to grayscale if it's not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply threshold to get binary mask
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Apply dilation to expand the mask
        expanded_mask = cv2.dilate(mask, kernel, iterations=iterations)
        
        # If original image was color, convert back to color
        if len(image.shape) == 3:
            expanded_mask = cv2.cvtColor(expanded_mask, cv2.COLOR_GRAY2BGR)
        
        # Save the expanded mask
        cv2.imwrite(output_path, expanded_mask)
        return True
    
    def _expand_mask_video(self, input_path, output_path, kernel, iterations):
        """
        Expand mask regions in a video using dilation.
        
        Args:
            input_path: Path to input mask video
            output_path: Path to save expanded mask video
            kernel: Dilation kernel
            iterations: Number of dilation iterations
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Open the input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False
        
        try:
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            frame_count = 0
            
            while cap.isOpened():
                if not self.app.processing:  # Check if processing was cancelled
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply threshold to get binary mask
                _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                
                # Apply dilation to expand the mask
                expanded_mask = cv2.dilate(mask, kernel, iterations=iterations)
                
                # Convert single channel mask back to 3 channels
                expanded_mask_colored = cv2.cvtColor(expanded_mask, cv2.COLOR_GRAY2BGR)
                
                # Write the frame to the output video
                out.write(expanded_mask_colored)
                
                frame_count += 1
                
                # Update progress within the video processing
                if total_frames > 0 and frame_count % 30 == 0:  # Update every 30 frames
                    self.app.status_label.config(text=f"Processing video: {frame_count}/{total_frames} frames")
                    self.app.root.update_idletasks()
            
            # Release resources
            out.release()
            return True
            
        except Exception as e:
            print(f"Error in _expand_mask_video: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            cap.release()
    
    def _copy_directory_structure(self, input_dir, output_dir):
        """
        Copy the directory structure from input to output.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
        """
        for root, dirs, _ in os.walk(input_dir):
            # Create relative path
            rel_path = os.path.relpath(root, input_dir)
            if rel_path == '.':
                continue  # Skip root directory
            
            # Create corresponding directory in output_dir
            target_dir = os.path.join(output_dir, rel_path)
            os.makedirs(target_dir, exist_ok=True)