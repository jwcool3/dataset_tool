"""
Frame Extractor for Dataset Preparation Tool
Extracts frames from videos at specified intervals.
"""

import os
import cv2
import shutil
import uuid
import threading

class FrameExtractor:
    """Extracts frames from videos at specified intervals."""
    
    def __init__(self, app):
        """
        Initialize frame extractor.
        
        Args:
            app: The main application with shared variables and UI controls
        """
        self.app = app
        self.processing = False
    
    def extract_frames(self, input_dir, output_dir):
        """
        Extract frames from videos at specified intervals.
        
        Args:
            input_dir: Input directory containing videos
            output_dir: Output directory for extracted frames
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Create frames output directory
        frames_output_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_output_dir, exist_ok=True)
        
        # Look for video files in the input directory
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []
        
        for root, _, files in os.walk(input_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(os.path.join(root, file))
        
        if not video_files:
            self.app.status_label.config(text="No video files found.")
            return False
        
        # Extract mask video frames first if needed - to a completely separate directory
        mask_frames = []
        temp_mask_dir = None
        
        if self.app.use_mask_video.get() and self.app.mask_video_path.get():
            mask_video_path = self.app.mask_video_path.get()
            
            # Verify the mask video exists
            if not os.path.exists(mask_video_path):
                from tkinter import messagebox
                messagebox.showerror("Error", f"Mask video not found: {mask_video_path}")
                self.app.status_label.config(text="Mask video not found.")
                return False
            
            # Create a temporary directory OUTSIDE the frames directory
            # to ensure mask frames never mix with source frames
            temp_dir_name = f"_temp_mask_{uuid.uuid4().hex}"
            temp_mask_dir = os.path.join(output_dir, temp_dir_name)
            os.makedirs(temp_mask_dir, exist_ok=True)
            
            self.app.status_label.config(text=f"Extracting frames from mask video to temporary location...")
            
            # Extract mask frames to the temporary directory
            mask_frames = self._extract_frames_from_video(
                mask_video_path,
                temp_mask_dir,
                self.app.frame_rate.get(),
                self.app.use_source_resolution.get(),
                self.app.output_width.get(),
                self.app.output_height.get(),
                progress_start=0,
                progress_end=20
            )
            
            self.app.status_label.config(text=f"Extracted {len(mask_frames)} mask frames to temporary location.")
        
        # If using a mask video, exclude it from the source videos list
        if self.app.use_mask_video.get() and self.app.mask_video_path.get():
            mask_video_path = os.path.abspath(self.app.mask_video_path.get())
            # Create a new list without the mask video
            video_files = [v for v in video_files if os.path.abspath(v) != mask_video_path]
            # Update total_videos count
            if not video_files:
                from tkinter import messagebox
                messagebox.showerror("Error", "No source videos found after excluding mask video.")
                self.app.status_label.config(text="No source videos found.")
                # Clean up temp directory
                if temp_mask_dir and os.path.exists(temp_mask_dir):
                    shutil.rmtree(temp_mask_dir)
                return False
        
        # Calculate progress per video
        total_videos = len(video_files)
        progress_per_video = 80 / total_videos if self.app.use_mask_video.get() else 100 / total_videos
        
        # Process each source video
        for video_index, video_path in enumerate(video_files):
            if not self.app.processing:  # Check if processing was cancelled
                break
            
            # Calculate progress range for this video
            progress_start = 20 + video_index * progress_per_video if self.app.use_mask_video.get() else video_index * progress_per_video
            progress_end = 20 + (video_index + 1) * progress_per_video if self.app.use_mask_video.get() else (video_index + 1) * progress_per_video
            
            # Get video name without extension
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Create directory for this video's frames
            video_frames_dir = os.path.join(frames_output_dir, video_name)
            os.makedirs(video_frames_dir, exist_ok=True)
            
            # Update status
            self.app.status_label.config(text=f"Extracting frames from source video {video_index+1}/{total_videos}: {video_name}")
            
            # Extract frames from this source video
            extracted_frame_paths = self._extract_frames_from_video(
                video_path,
                video_frames_dir,
                self.app.frame_rate.get(),
                self.app.use_source_resolution.get(),
                self.app.output_width.get(),
                self.app.output_height.get(),
                progress_start=progress_start,
                progress_end=progress_end * 0.8
            )
            
            # If we have mask frames from a separate mask video, create masks subfolder and copy them
            if self.app.use_mask_video.get() and mask_frames and extracted_frame_paths:
                self.app.status_label.config(text=f"Copying mask frames for video {video_index+1}/{total_videos}: {video_name}")
                
                # Create masks subfolder for this source video
                masks_dir = os.path.join(video_frames_dir, "masks")
                os.makedirs(masks_dir, exist_ok=True)
                
                # Copy mask frames to match source frames
                self._copy_mask_frames(
                    mask_frames,
                    extracted_frame_paths,
                    masks_dir,
                    progress_start=progress_end * 0.8,
                    progress_end=progress_end
                )
        
        # Clean up temporary mask frames directory
        if temp_mask_dir and os.path.exists(temp_mask_dir):
            try:
                shutil.rmtree(temp_mask_dir)
                print(f"Cleaned up temporary mask frames directory: {temp_mask_dir}")
            except Exception as e:
                print(f"Warning: Could not remove temporary mask frames directory: {str(e)}")
        
        self.app.status_label.config(text="Frame extraction completed.")
        self.app.progress_bar['value'] = 100
        return True
    
    def _extract_frames_from_video(self, video_path, output_dir, frame_rate, use_source_resolution, 
                                   target_width, target_height, progress_start=0, progress_end=100):
        """
        Extract frames from a single video file.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save the extracted frames
            frame_rate: Target frame rate for extraction
            use_source_resolution: Whether to use source resolution
            target_width: Target width if not using source resolution
            target_height: Target height if not using source resolution
            progress_start: Starting percentage for progress bar
            progress_end: Ending percentage for progress bar
            
        Returns:
            list: Paths to the extracted frames
        """
        extracted_frames = []
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.app.status_label.config(text=f"Error: Could not open video {video_path}.")
            return extracted_frames
        
        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame interval based on user-specified frame rate
            interval = max(1, int(fps / frame_rate))
            
            # Initialize counters
            frame_number = 0
            saved_frame_number = 0
            
            # Extract frames
            while cap.isOpened():
                if not self.app.processing:  # Check if processing was cancelled
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_number % interval == 0:
                    # Resize frame if necessary
                    if not use_source_resolution:
                        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                    
                    output_path = os.path.join(output_dir, f"{saved_frame_number:04d}.jpg")
                    cv2.imwrite(output_path, frame)
                    extracted_frames.append(output_path)
                    saved_frame_number += 1
                
                frame_number += 1
                
                # Update progress
                if total_frames > 0:  # Avoid division by zero
                    progress = progress_start + (frame_number / total_frames) * (progress_end - progress_start)
                    self.app.progress_bar['value'] = min(progress, 100)
                    self.app.root.update_idletasks()
                    
        finally:
            # Release video capture
            cap.release()
        
        return extracted_frames
    
    def _copy_mask_frames(self, mask_frames, video_frames, masks_dir, 
                         progress_start=0, progress_end=100):
        """
        Copy mask frames to a video's masks subfolder, matching them to the video frames.
        
        Args:
            mask_frames: List of paths to mask frames
            video_frames: List of paths to video frames
            masks_dir: Directory to save the mask frames
            progress_start: Starting percentage for progress bar
            progress_end: Ending percentage for progress bar
        """
        if not mask_frames or not video_frames:
            return
        
        # Calculate how many mask frames to assign to each video frame
        num_mask_frames = len(mask_frames)
        num_video_frames = len(video_frames)
        
        # Handle the case where we have more or fewer mask frames than video frames
        if num_mask_frames != num_video_frames:
            self.app.status_label.config(text=(
                f"Note: Mask video has {num_mask_frames} frames while source video has {num_video_frames} frames. "
                "Adjusting mask frames to match."
            ))
        
        # Copy mask frames to match video frames
        for i, video_frame in enumerate(video_frames):
            if not self.app.processing:
                break
            
            # Calculate which mask frame to use (handle cases with different frame counts)
            if num_mask_frames == num_video_frames:
                # Direct 1:1 mapping
                mask_frame_index = i
            else:
                # Scale the index to map between different frame counts
                mask_frame_index = min(int(i * num_mask_frames / num_video_frames), num_mask_frames - 1)
            
            # Get source frame name
            video_frame_name = os.path.basename(video_frame)
            
            # Create the same filename in the masks folder
            mask_output_path = os.path.join(masks_dir, video_frame_name)
            
            # Copy the mask frame
            shutil.copy2(mask_frames[mask_frame_index], mask_output_path)
            
            # Update progress
            if i % 10 == 0 or i == len(video_frames) - 1:  # Update less frequently for performance
                progress = progress_start + (i / num_video_frames) * (progress_end - progress_start)
                self.app.progress_bar['value'] = min(progress, 100)
                self.app.status_label.config(text=f"Copied {i+1}/{num_video_frames} mask frames")
                self.app.root.update_idletasks()