"""
File Organizer for Dataset Preparation Tool
Organizes and renames files according to naming patterns.
"""

import os
import shutil

class FileOrganizer:
    """Organizes and renames files according to naming patterns."""
    
    def __init__(self, app):
        """
        Initialize file organizer.
        
        Args:
            app: The main application with shared variables and UI controls
        """
        self.app = app
    
    def organize_files(self, input_dir, output_dir):
        """
        Organize and rename files according to the naming pattern.
        
        Args:
            input_dir: Input directory containing images and masks
            output_dir: Output directory for organized files
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create output directories
        organized_output_dir = os.path.join(output_dir, "organized")
        os.makedirs(organized_output_dir, exist_ok=True)
        
        # Always create a masks directory since we'll scan for masks in multiple locations
        organized_masks_dir = os.path.join(organized_output_dir, "masks")
        os.makedirs(organized_masks_dir, exist_ok=True)
        
        # Find all image files and potential mask files with a recursive approach
        image_files = []
        all_mask_dirs = []
        
        # Track all directories we've processed to avoid duplicates
        processed_dirs = set()
        
        # Function to find mask-image pairs in a given directory
        def scan_directory(directory):
            # Skip if we've already processed this directory
            if directory in processed_dirs:
                return
            
            processed_dirs.add(directory)
            
            # Get all files and subdirectories
            try:
                all_items = os.listdir(directory)
            except (PermissionError, FileNotFoundError):
                return  # Skip directories we can't access
                
            # Check if this directory itself has a 'masks' subfolder
            mask_dir = os.path.join(directory, "masks")
            if os.path.isdir(mask_dir):
                all_mask_dirs.append(mask_dir)
            
            # Process all files in this directory
            for item in all_items:
                item_path = os.path.join(directory, item)
                
                # If it's a file and an image, add it to our list
                if os.path.isfile(item_path) and item.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(item_path)
                
                # If it's a directory (and not a masks folder), scan it recursively
                elif os.path.isdir(item_path) and os.path.basename(item_path).lower() != "masks":
                    scan_directory(item_path)
        
        # Start scanning from the input directory
        try:
            scan_directory(input_dir)
        except Exception as e:
            self.app.status_label.config(text=f"Error scanning directories: {str(e)}")
            print(f"Error scanning directories: {str(e)}")
            return False
        
        # Print summary of what we found
        print(f"Found {len(image_files)} images and {len(all_mask_dirs)} mask directories")
        
        if not image_files:
            self.app.status_label.config(text="No image files found to organize.")
            return False
        
        # Sort image files for consistent ordering
        image_files.sort()
        
        total_files = len(image_files)
        self.app.status_label.config(text=f"Found {total_files} images to organize.")
        self.app.root.update_idletasks()
        
        # Get naming pattern
        pattern = self.app.naming_pattern.get()
        if not pattern or "{index" not in pattern:
            # Default pattern if invalid
            pattern = "{index:04d}.png"
        
        # Create a lookup dictionary from original image paths to new filenames
        # This helps us track which images correspond to which masks
        path_to_newname = {}
        
        # Process each image
        for idx, image_path in enumerate(image_files):
            if not self.app.processing:  # Check if processing was cancelled
                break
            
            try:
                # Generate new filename based on pattern
                new_filename = pattern.format(index=idx)
                
                # Ensure the filename has an extension
                if not os.path.splitext(new_filename)[1]:
                    # If no extension in the pattern, use the original extension
                    orig_ext = os.path.splitext(image_path)[1]
                    new_filename += orig_ext if orig_ext else ".png"
                
                # Store the mapping from original path to new filename
                path_to_newname[image_path] = new_filename
                
                # Copy the image to the new location
                output_path = os.path.join(organized_output_dir, new_filename)
                shutil.copy2(image_path, output_path)
                
                # Update progress
                if idx % 10 == 0 or idx == total_files - 1:  # Update status less frequently to avoid UI slowdown
                    self.app.status_label.config(text=f"Copied {idx+1}/{total_files} source images")
                    self.app.progress_bar['value'] = min((idx + 1) / total_files * 50, 50)  # First 50% is copying images
                    self.app.root.update_idletasks()
                    
            except Exception as e:
                print(f"Error organizing file {image_path}: {str(e)}")
        
        # Now search for corresponding masks for each image
        self.app.status_label.config(text="Searching for corresponding masks...")
        self.app.progress_bar['value'] = 50
        self.app.root.update_idletasks()
        
        # Process each image to find its mask
        found_masks = 0
        for idx, image_path in enumerate(image_files):
            if not self.app.processing:  # Check if processing was cancelled
                break
            
            try:
                # Get the full and base filename
                image_basename = os.path.basename(image_path)
                image_basename_no_ext = os.path.splitext(image_basename)[0]
                
                # Get the parent directory of the image
                image_parent_dir = os.path.dirname(image_path)
                
                # First check for mask in the same directory but with a mask indicator
                mask_path = None
                
                # Standard locations to check for masks:
                # 1. Check "masks" subfolder in the parent directory
                parent_mask_dir = os.path.join(image_parent_dir, "masks")
                if os.path.isdir(parent_mask_dir):
                    # First try exact filename match
                    potential_mask = os.path.join(parent_mask_dir, image_basename)
                    if os.path.exists(potential_mask):
                        mask_path = potential_mask
                    else:
                        # Try with different extensions
                        for ext in ['.png', '.jpg', '.jpeg']:
                            alt_mask_path = os.path.join(parent_mask_dir, image_basename_no_ext + ext)
                            if os.path.exists(alt_mask_path):
                                mask_path = alt_mask_path
                                break
                
                # 2. If not found in parent's mask dir, try all other mask dirs we found
                if mask_path is None:
                    # Try to find in any other mask directory
                    for mask_dir in all_mask_dirs:
                        # Skip if we already checked this directory
                        if mask_dir == parent_mask_dir:
                            continue
                            
                        # Try exact match
                        potential_mask = os.path.join(mask_dir, image_basename)
                        if os.path.exists(potential_mask):
                            mask_path = potential_mask
                            break
                            
                        # Try variations of the filename with different extensions
                        for ext in ['.png', '.jpg', '.jpeg']:
                            alt_mask_path = os.path.join(mask_dir, image_basename_no_ext + ext)
                            if os.path.exists(alt_mask_path):
                                mask_path = alt_mask_path
                                break
                                
                        # If we found a mask, no need to check other directories
                        if mask_path:
                            break
                
                # If mask found, copy it to the masks output directory
                if mask_path and os.path.exists(mask_path):
                    # Get the new filename for this image
                    new_filename = path_to_newname[image_path]
                    
                    # Copy mask to the output masks directory with the same new filename
                    mask_output_path = os.path.join(organized_masks_dir, new_filename)
                    shutil.copy2(mask_path, mask_output_path)
                    found_masks += 1
                
                # Update progress for mask processing
                if idx % 10 == 0 or idx == total_files - 1:
                    self.app.status_label.config(text=f"Processed {idx+1}/{total_files} masks (found {found_masks})")
                    self.app.progress_bar['value'] = 50 + min((idx + 1) / total_files * 50, 50)  # Last 50% is copying masks
                    self.app.root.update_idletasks()
                    
            except Exception as e:
                print(f"Error finding mask for {image_path}: {str(e)}")
        
        # Final status update
        self.app.status_label.config(text=(
            f"Organization completed. Processed {total_files} images and found {found_masks} masks."
        ))
        self.app.progress_bar['value'] = 100
        
        return True