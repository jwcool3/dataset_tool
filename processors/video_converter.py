"""
Video Converter for Dataset Preparation Tool
Converts image sequences to video files.
"""

import os
import cv2

class VideoConverter:
    """Converts image sequences to video files."""
    
    def __init__(self, app):
        """
        Initialize video converter.
        
        Args:
            app: The main application with shared variables and UI controls
        """
        self.app = app
    
    def convert_to_video(self, input_dir, output_dir):
        """
        Convert image sequences to video files.
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for videos
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create output directory for videos
        video_output_dir = os.path.join(output_dir, "videos")
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Check if we have masks to convert as well
        mask_dir = os.path.join(input_dir, "masks")
        has_masks = os.path.isdir(mask_dir) and os.listdir(mask_dir)
        
        # Prepare processing directories
        directories_to_process = [(input_dir, os.path.join(video_output_dir, "main.mp4"))]
        if has_masks:
            directories_to_process.append((mask_dir, os.path.join(video_output_dir, "masks.mp4")))
        
        # Set video FPS
        fps = self.app.video_fps.get()
        if fps <= 0:
            fps = 30.0  # Default FPS
        
        # Process each directory
        for dir_idx, (src_dir, output_video_path) in enumerate(directories_to_process):
            if not self.app.processing:  # Check if processing was cancelled
                break
            
            # Get all image files in the directory
            image_files = []
            for file in os.listdir(src_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(src_dir, file))
            
            if not image_files:
                self.app.status_label.config(text=f"No image files found in {src_dir}.")
                continue
            
            # Sort image files numerically/alphabetically
            image_files.sort()
            
            total_images = len(image_files)
            self.app.status_label.config(text=f"Converting {total_images} images to video...")
            
            try:
                # Get dimensions from first image
                first_image = cv2.imread(image_files[0])
                height, width = first_image.shape[:2]
                
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                
                # Process each image
                for idx, image_path in enumerate(image_files):
                    if not self.app.processing:  # Check if processing was cancelled
                        break
                    
                    # Read image
                    img = cv2.imread(image_path)
                    
                    # If this is a mask directory, convert to RGB if grayscale
                    if "mask" in src_dir.lower() and len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    
                    # Write frame to video
                    video_writer.write(img)
                    
                    # Update progress
                    progress = (dir_idx + (idx + 1) / total_images) / len(directories_to_process) * 100
                    self.app.progress_bar['value'] = min(progress, 100)
                    if idx % 20 == 0 or idx == total_images - 1:  # Update status less frequently
                        self.app.status_label.config(text=f"Converting {idx+1}/{total_images} images to video")
                        self.app.root.update_idletasks()
                
                # Release video writer
                video_writer.release()
                
                self.app.status_label.config(text=f"Video creation completed: {os.path.basename(output_video_path)}")
                
            except Exception as e:
                self.app.status_label.config(text=f"Error creating video: {str(e)}")
                print(f"Error creating video from {src_dir}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Clean up resources
                if 'video_writer' in locals():
                    video_writer.release()
        
        self.app.status_label.config(text="Video conversion completed.")
        self.app.progress_bar['value'] = 100
        return True