"""
Gallery Manager for Dataset Preparation Tool
Handles finding and comparing images across different folders.
"""

import os
import cv2
import numpy as np
from collections import defaultdict

class GalleryManager:
    """Manages finding and comparing images across different folders."""
    
    def __init__(self):
        """Initialize the gallery manager."""
        self.reset()
    
    def reset(self):
        """Reset the gallery manager state."""
        self.all_images = {}  # Dictionary of folder -> list of images
        self.image_groups = defaultdict(list)  # Groups of images by filename
        self.version_folders = []  # List of folders containing image versions
    
    def scan_directory(self, root_dir):
        """
        Scan a directory for image versions across different subfolders.
        
        Args:
            root_dir: Root directory to scan
            
        Returns:
            dict: Information about found images and their grouping
        """
        self.reset()
        
        if not os.path.isdir(root_dir):
            return {
                'success': False,
                'message': f"Directory not found: {root_dir}",
                'groups': [],
                'folders': []
            }
        
        # Find all immediate subdirectories
        try:
            subdirs = [d for d in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]
        except Exception as e:
            return {
                'success': False,
                'message': f"Error reading directory: {str(e)}",
                'groups': [],
                'folders': []
            }
        
        if not subdirs:
            return {
                'success': False,
                'message': "No subdirectories found",
                'groups': [],
                'folders': []
            }
        
        # Scan each subdirectory for images
        for subdir in subdirs:
            subdir_path = os.path.join(root_dir, subdir)
            self._scan_subfolder(subdir, subdir_path)
        
        # Group images by filename across directories
        self._group_images()
        
        # Filter to only include images that appear in multiple directories
        shared_images = []
        for filename, versions in self.image_groups.items():
            # Count unique directories (not including masks)
            unique_dirs = set(img['folder'] for img in versions)
            if len(unique_dirs) > 1:
                shared_images.append({
                    'filename': filename,
                    'versions': versions,
                    'folders': list(unique_dirs)
                })
        
        # Sort by filename
        shared_images.sort(key=lambda x: x['filename'])
        
        return {
            'success': True,
            'message': f"Found {len(shared_images)} images across {len(subdirs)} folders",
            'groups': shared_images,
            'folders': subdirs
        }
    
    def _scan_subfolder(self, folder_name, folder_path):
        """
        Scan a subfolder for images and masks.
        
        Args:
            folder_name: Name of the folder (used as identifier)
            folder_path: Full path to the folder
        """
        # Initialize entry for this folder
        self.all_images[folder_name] = []
        
        # Scan for images directly in this folder
        try:
            direct_images = [f for f in os.listdir(folder_path) 
                           if os.path.isfile(os.path.join(folder_path, f)) and 
                           f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in direct_images:
                self.all_images[folder_name].append({
                    'path': os.path.join(folder_path, img_file),
                    'filename': img_file,
                    'folder': folder_name,
                    'is_mask': False
                })
        except Exception as e:
            print(f"Error scanning folder {folder_name}: {str(e)}")
        
        # Check for a masks subfolder
        masks_dir = os.path.join(folder_path, "masks")
        if os.path.isdir(masks_dir):
            try:
                mask_files = [f for f in os.listdir(masks_dir) 
                             if os.path.isfile(os.path.join(masks_dir, f)) and 
                             f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for mask_file in mask_files:
                    self.all_images[folder_name].append({
                        'path': os.path.join(masks_dir, mask_file),
                        'filename': mask_file,
                        'folder': folder_name,
                        'is_mask': True
                    })
            except Exception as e:
                print(f"Error scanning masks in {folder_name}: {str(e)}")
    
    def _group_images(self):
        """Group images by filename across all directories."""
        self.image_groups = defaultdict(list)
        for folder, images in self.all_images.items():
            for img_info in images:
                self.image_groups[img_info['filename']].append(img_info)
    
    def load_image_thumbnail(self, image_path, max_size=200):
        """
        Load an image and create a thumbnail.
        
        Args:
            image_path: Path to the image file
            max_size: Maximum dimension of the thumbnail
            
        Returns:
            tuple: (success, numpy array or error message)
        """
        try:
            # Load image with OpenCV
            img = cv2.imread(image_path)
            if img is None:
                return False, f"Failed to load image: {image_path}"
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get dimensions
            h, w = img_rgb.shape[:2]
            
            # Create thumbnail
            scale = min(max_size / w, max_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            thumbnail = cv2.resize(img_rgb, (new_w, new_h))
            
            # Return success with thumbnail and original dimensions
            return True, {
                'thumbnail': thumbnail,
                'width': w,
                'height': h
            }
            
        except Exception as e:
            return False, f"Error processing image: {str(e)}"
    
    def delete_image(self, image_path):
        """
        Delete an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            tuple: (success, message)
        """
        try:
            if not os.path.exists(image_path):
                return False, f"File not found: {image_path}"
            
            os.remove(image_path)
            return True, f"Successfully deleted: {os.path.basename(image_path)}"
            
        except Exception as e:
            return False, f"Error deleting file: {str(e)}"