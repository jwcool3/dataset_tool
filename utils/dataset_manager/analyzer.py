"""
Dataset Analyzer for Dataset Preparation Tool
Handles analysis of dataset contents and quality.
"""

import os
import datetime
import json
import imagehash


class DatasetAnalyzer:
    """Analyzes datasets to provide insights and quality assessment."""
    
    def __init__(self, registry):
        """
        Initialize the dataset analyzer.
        
        Args:
            registry: DatasetRegistry instance
        """
        self.registry = registry
    
    def analyze_dataset(self, dataset_id):
        """
        Perform comprehensive analysis of a dataset.
        
        Args:
            dataset_id: Dataset ID to analyze
            
        Returns:
            dict: Analysis results
        """
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset or not os.path.isdir(dataset['path']):
            return None
            
        # Initialize results
        results = {
            "dataset_id": dataset_id,
            "dataset_name": dataset['name'],
            "path": dataset['path'],
            "file_stats": {},
            "image_stats": {},
            "folder_stats": {},
            "quality_issues": [],
            "duplicate_analysis": {}
        }
        
        # Get file statistics
        results["file_stats"] = self._analyze_files(dataset['path'])
        
        # Get image statistics (only if there are images)
        if results["file_stats"]["image_count"] > 0:
            results["image_stats"] = self._analyze_images(dataset['path'])
            
            # Check for image quality issues
            results["quality_issues"] = self._check_quality_issues(dataset['path'])
            
            # Find potential duplicates
            results["duplicate_analysis"] = self._find_duplicates(dataset['path'])
        
        # Analyze folder structure
        results["folder_stats"] = self._analyze_folders(dataset['path'])
        
        # Store analysis results
        timestamp = datetime.datetime.now().isoformat()
        
        # Update dataset with analysis timestamp
        attributes = dataset.get('attributes', {})
        if not attributes:
            attributes = {}
            
        attributes['last_analysis'] = timestamp
        self.registry.update_dataset(dataset_id, attributes=attributes)
        
        # Record the analysis operation
        self.registry.record_processing_step(
            dataset_id,
            "dataset_analysis",
            {"timestamp": timestamp}
        )
        
        return results
    
    def _analyze_files(self, path):
        """Analyze files in the dataset."""
        stats = {
            "total_count": 0,
            "image_count": 0,
            "video_count": 0,
            "other_count": 0,
            "total_size_bytes": 0,
            "image_size_bytes": 0,
            "video_size_bytes": 0,
            "other_size_bytes": 0,
            "formats": {},
            "largest_file": {"path": None, "size": 0},
            "smallest_file": {"path": None, "size": float('inf')},
        }
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
        
        # Scan all files
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                
                # Update total stats
                stats["total_count"] += 1
                stats["total_size_bytes"] += file_size
                
                # Update largest/smallest
                if file_size > stats["largest_file"]["size"]:
                    stats["largest_file"] = {"path": file_path, "size": file_size}
                    
                if file_size < stats["smallest_file"]["size"]:
                    stats["smallest_file"] = {"path": file_path, "size": file_size}
                
                # Categorize by file type
                ext = os.path.splitext(file)[1].lower()
                
                # Update format counts
                if ext not in stats["formats"]:
                    stats["formats"][ext] = {"count": 0, "size_bytes": 0}
                    
                stats["formats"][ext]["count"] += 1
                stats["formats"][ext]["size_bytes"] += file_size
                
                # Categorize as image, video, or other
                if ext in image_extensions:
                    stats["image_count"] += 1
                    stats["image_size_bytes"] += file_size
                elif ext in video_extensions:
                    stats["video_count"] += 1
                    stats["video_size_bytes"] += file_size
                else:
                    stats["other_count"] += 1
                    stats["other_size_bytes"] += file_size
        
        # Calculate averages
        if stats["total_count"] > 0:
            stats["avg_file_size"] = stats["total_size_bytes"] / stats["total_count"]
        else:
            stats["avg_file_size"] = 0
            
        if stats["image_count"] > 0:
            stats["avg_image_size"] = stats["image_size_bytes"] / stats["image_count"]
        else:
            stats["avg_image_size"] = 0
            
        if stats["video_count"] > 0:
            stats["avg_video_size"] = stats["video_size_bytes"] / stats["video_count"]
        else:
            stats["avg_video_size"] = 0
        
        # Handle case where no smallest file was found
        if stats["smallest_file"]["path"] is None:
            stats["smallest_file"] = {"path": None, "size": 0}
            
        return stats
    
    def _analyze_images(self, path, sample_size=500):
        """Analyze image properties in the dataset."""
        stats = {
            "dimensions": [],
            "aspect_ratios": [],
            "formats": {},
            "color_modes": {},
            "min_width": float('inf'),
            "max_width": 0,
            "min_height": float('inf'),
            "max_height": 0,
            "resolution_groups": {}
        }
        
        # Find image files
        image_files = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for root, _, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        
        # Limit sample size for performance
        if sample_size and len(image_files) > sample_size:
            import random
            random.shuffle(image_files)
            image_files = image_files[:sample_size]
        
        # Process images
        for file_path in image_files:
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    # Get dimensions
                    width, height = img.size
                    
                    # Update dimension stats
                    stats["dimensions"].append((width, height))
                    stats["min_width"] = min(stats["min_width"], width)
                    stats["max_width"] = max(stats["max_width"], width)
                    stats["min_height"] = min(stats["min_height"], height)
                    stats["max_height"] = max(stats["max_height"], height)
                    
                    # Calculate aspect ratio
                    aspect_ratio = width / height
                    stats["aspect_ratios"].append(aspect_ratio)
                    
                    # Update format stats
                    format_name = img.format or "Unknown"
                    if format_name not in stats["formats"]:
                        stats["formats"][format_name] = 0
                    stats["formats"][format_name] += 1
                    
                    # Update color mode stats
                    color_mode = img.mode
                    if color_mode not in stats["color_modes"]:
                        stats["color_modes"][color_mode] = 0
                    stats["color_modes"][color_mode] += 1
                    
                    # Group by resolution
                    res_key = f"{width}x{height}"
                    if res_key not in stats["resolution_groups"]:
                        stats["resolution_groups"][res_key] = {
                            "width": width,
                            "height": height,
                            "count": 0,
                            "aspect_ratio": aspect_ratio
                        }
                    stats["resolution_groups"][res_key]["count"] += 1
                    
            except Exception as e:
                print(f"Error analyzing image {file_path}: {str(e)}")
        
        # Calculate averages and distribution
        if stats["dimensions"]:
            avg_width = sum(w for w, _ in stats["dimensions"]) / len(stats["dimensions"])
            avg_height = sum(h for _, h in stats["dimensions"]) / len(stats["dimensions"])
            avg_aspect_ratio = sum(stats["aspect_ratios"]) / len(stats["aspect_ratios"])
            
            stats["avg_width"] = avg_width
            stats["avg_height"] = avg_height
            stats["avg_aspect_ratio"] = avg_aspect_ratio
            
            # Sort resolution groups by frequency
            stats["resolution_groups"] = dict(
                sorted(
                    stats["resolution_groups"].items(),
                    key=lambda x: x[1]["count"],
                    reverse=True
                )
            )
            
            # Find most common resolution
            common_res = list(stats["resolution_groups"].keys())[0]
            stats["most_common_resolution"] = {
                "dimensions": common_res,
                "count": stats["resolution_groups"][common_res]["count"],
                "percentage": stats["resolution_groups"][common_res]["count"] / len(stats["dimensions"]) * 100
            }
        
        # Clean up min/max if no images were processed
        if stats["min_width"] == float('inf'):
            stats["min_width"] = 0
        if stats["min_height"] == float('inf'):
            stats["min_height"] = 0
            
        return stats
    
    def _analyze_folders(self, path):
        """Analyze folder structure in the dataset."""
        stats = {
            "total_folders": 0,
            "max_depth": 0,
            "folders_by_level": {},
            "empty_folders": 0,
            "folder_sizes": {}
        }
        
        # Track root depth
        root_depth = path.count(os.sep)
        
        # Scan folders
        for root, dirs, files in os.walk(path):
            # Calculate depth
            depth = root.count(os.sep) - root_depth
            
            # Update total folders
            stats["total_folders"] += 1
            
            # Update max depth
            stats["max_depth"] = max(stats["max_depth"], depth)
            
            # Update folders by level
            if depth not in stats["folders_by_level"]:
                stats["folders_by_level"][depth] = 0
            stats["folders_by_level"][depth] += 1
            
            # Check if folder is empty
            if not dirs and not files:
                stats["empty_folders"] += 1
            
            # Calculate folder size
            folder_size = 0
            for file in files:
                file_path = os.path.join(root, file)
                folder_size += os.path.getsize(file_path)
                
            # Store folder size (only for main subfolders)
            if depth == 1:
                folder_name = os.path.basename(root)
                stats["folder_sizes"][folder_name] = folder_size
        
        return stats
    
    def _check_quality_issues(self, path, sample_size=500):
        """Check for quality issues in the dataset."""
        issues = []
        
        # Find image files
        image_files = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for root, _, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        
        # Limit sample size for performance
        if sample_size and len(image_files) > sample_size:
            import random
            random.shuffle(image_files)
            image_files = image_files[:sample_size]
        
        # Check for various issues
        for file_path in image_files:
            try:
                from PIL import Image
                img = Image.open(file_path)
                
                # Check image size
                width, height = img.size
                if width < 32 or height < 32:
                    issues.append({
                        "type": "small_image",
                        "path": file_path,
                        "details": {"width": width, "height": height}
                    })
                
                # Check if image can be fully loaded
                try:
                    img.load()
                except Exception as e:
                    issues.append({
                        "type": "corrupt_image",
                        "path": file_path,
                        "details": {"error": str(e)}
                    })
                    continue
                
                # Check if image is very dark or very bright
                try:
                    if img.mode == "RGB":
                        import numpy as np
                        img_array = np.array(img)
                        brightness = np.mean(img_array)
                        
                        if brightness < 30:
                            issues.append({
                                "type": "dark_image",
                                "path": file_path,
                                "details": {"brightness": float(brightness)}
                            })
                        elif brightness > 240:
                            issues.append({
                                "type": "bright_image",
                                "path": file_path,
                                "details": {"brightness": float(brightness)}
                            })
                except:
                    pass
                
                # Close image
                img.close()
                
            except Exception as e:
                # Failed to open image
                issues.append({
                    "type": "invalid_image",
                    "path": file_path,
                    "details": {"error": str(e)}
                })
        
        return issues
    
    def _find_duplicates(self, path, sample_size=1000, threshold=0.9):
        """Find potential duplicate images in the dataset."""
        from PIL import Image
        import numpy as np
        
        results = {
            "exact_duplicates": [],
            "similar_images": [],
            "analyzed_count": 0
        }
        
        # Find image files
        image_files = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for root, _, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        
        # Limit sample size for performance
        if sample_size and len(image_files) > sample_size:
            import random
            random.shuffle(image_files)
            image_files = image_files[:sample_size]
        
        results["analyzed_count"] = len(image_files)
        
        # Find exact duplicates based on file hash
        file_hashes = {}
        
        for file_path in image_files:
            try:
                # Calculate file hash
                import hashlib
                with open(file_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                # Check if hash exists
                if file_hash in file_hashes:
                    # Found an exact duplicate
                    results["exact_duplicates"].append({
                        "original": file_hashes[file_hash],
                        "duplicate": file_path
                    })
                else:
                    file_hashes[file_hash] = file_path
                    
            except Exception as e:
                print(f"Error hashing {file_path}: {str(e)}")
        
        # Find similar images using perceptual hash (if library available)
        try:

            
            # Calculate perceptual hashes
            image_hashes = []
            
            for file_path in image_files:
                try:
                    img = Image.open(file_path)
                    phash = imagehash.phash(img)
                    image_hashes.append((file_path, phash))
                    img.close()
                except:
                    pass
            
            # Compare hashes
            for i in range(len(image_hashes)):
                for j in range(i+1, len(image_hashes)):
                    path1, hash1 = image_hashes[i]
                    path2, hash2 = image_hashes[j]
                    
                    # Calculate hash difference (0 = identical, higher = more different)
                    difference = hash1 - hash2
                    
                    # Convert to similarity score (1.0 = identical, 0.0 = completely different)
                    # A good threshold is around 0.9 (90% similar)
                    similarity = 1.0 - (difference / 64.0)  # 64 bits in the hash
                    
                    if similarity >= threshold:
                        results["similar_images"].append({
                            "image1": path1,
                            "image2": path2,
                            "similarity": float(similarity)
                        })
        except ImportError:
            # imagehash library not available
            results["similar_images"] = []
        
        return results