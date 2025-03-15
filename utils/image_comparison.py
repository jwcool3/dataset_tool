"""
Image Comparison Utilities for Dataset Preparation Tool
Provides functions to compare images and identify outliers in groups.
"""
import os
import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import io
from skimage.metrics import structural_similarity as ssim

class ImageComparison:
    """Class for comparing images and identifying outliers."""
    
    def __init__(self):
        """Initialize the image comparison utility."""
        self.comparison_methods = {
            "histogram_correlation": self._compare_histogram_correlation,
            "histogram_chi": self._compare_histogram_chi,
            "histogram_intersection": self._compare_histogram_intersection,
            "histogram_bhattacharyya": self._compare_histogram_bhattacharyya,
            "ssim": self._compare_ssim,
            "mse": self._compare_mse
        }
    
    def compare_images(self, image_paths, method="histogram_correlation", resize=True, 
                      resize_dim=(256, 256), color_space="rgb"):
        """
        Compare a group of images using the specified method.
        
        Args:
            image_paths: List of paths to images to compare
            method: Comparison method to use
            resize: Whether to resize images before comparison
            resize_dim: Dimensions to resize to if resize=True
            color_space: Color space to use for comparison ('rgb', 'hsv', 'lab')
            
        Returns:
            dict: Results with distance matrix, outlier scores, and visualizations
        """
        if not image_paths:
            return {
                "success": False,
                "message": "No images provided for comparison"
            }
        
        if method not in self.comparison_methods:
            return {
                "success": False,
                "message": f"Unknown comparison method: {method}"
            }
        
        # Load and preprocess images
        images = []
        for path in image_paths:
            try:
                # Load image
                img = cv2.imread(path)
                if img is None:
                    print(f"Warning: Failed to load image {path}")
                    continue
                
                # Convert color space if needed
                if color_space == "hsv":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                elif color_space == "lab":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                else:  # Default to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize if requested
                if resize:
                    img = cv2.resize(img, resize_dim, interpolation=cv2.INTER_AREA)
                
                images.append(img)
                
            except Exception as e:
                print(f"Error processing image {path}: {str(e)}")
                continue
        
        if len(images) < 2:
            return {
                "success": False,
                "message": "Need at least 2 valid images for comparison"
            }
        
        # Compute pairwise distances using the selected method
        comparison_func = self.comparison_methods[method]
        distance_matrix = self._compute_distance_matrix(images, comparison_func)
        
        # Compute outlier scores (average distance to all other images)
        outlier_scores = np.mean(distance_matrix, axis=1)
        
        # Get indices of images sorted by outlier score (descending)
        sorted_indices = np.argsort(outlier_scores)[::-1]
        
        # Create result with original paths and scores
        result = {
            "success": True,
            "message": f"Compared {len(images)} images using {method}",
            "distance_matrix": distance_matrix,
            "outlier_scores": outlier_scores,
            "sorted_indices": sorted_indices,
            "paths": [image_paths[i] for i in range(len(images))],
            "ranked_paths": [image_paths[i] for i in sorted_indices if i < len(image_paths)]
        }
        
        return result
    
    def generate_comparison_visualization(self, images, distance_matrix, outlier_scores, 
                                        image_paths=None, max_images=10):
        """
        Generate visualization of image similarity and outliers.
        
        Args:
            images: List of image arrays
            distance_matrix: Matrix of distances between images
            outlier_scores: Outlier score for each image
            image_paths: Optional list of image paths (for labels)
            max_images: Maximum number of images to include in visualization
            
        Returns:
            PIL.ImageTk.PhotoImage: Visualization as a Tkinter-compatible image
        """
        # Limit to max_images
        n_images = min(len(images), max_images)
        
        # Sort images by outlier score (descending)
        sorted_indices = np.argsort(outlier_scores)[::-1][:n_images]
        
        # Create a figure with subplots for the heatmap and top outliers
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot distance heatmap (truncated to max_images)
        truncated_matrix = distance_matrix[sorted_indices][:, sorted_indices]
        heatmap = axes[0].imshow(truncated_matrix, cmap='viridis')
        axes[0].set_title('Image Distance Matrix')
        plt.colorbar(heatmap, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Add labels if image_paths is provided
        if image_paths:
            sorted_labels = [os.path.basename(image_paths[i]) for i in sorted_indices]
            axes[0].set_xticks(np.arange(n_images))
            axes[0].set_yticks(np.arange(n_images))
            axes[0].set_xticklabels(sorted_labels, rotation=90)
            axes[0].set_yticklabels(sorted_labels)
        
        # Plot outlier scores
        sorted_scores = outlier_scores[sorted_indices]
        bars = axes[1].bar(np.arange(n_images), sorted_scores)
        axes[1].set_title('Outlier Scores (Higher = More Different)')
        axes[1].set_xlabel('Image Index')
        axes[1].set_ylabel('Average Distance to Other Images')
        
        # Add labels if image_paths is provided
        if image_paths:
            axes[1].set_xticks(np.arange(n_images))
            axes[1].set_xticklabels(np.arange(n_images), rotation=90)
        
        plt.tight_layout()
        
        # Convert matplotlib figure to a Tkinter-compatible image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # Close the figure to free memory
        plt.close(fig)
        
        # Convert to PIL image and then to ImageTk
        pil_img = Image.open(buf)
        
        return pil_img
    
    def visualize_differences(self, img1, img2):
        """
        Create a visual representation of differences between two images.
        
        Args:
            img1: First image (numpy array)
            img2: Second image (numpy array)
            
        Returns:
            numpy.ndarray: Difference visualization
        """
        # Ensure images are in RGB
        if img1.shape[2] != 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        if img2.shape[2] != 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        
        # Resize images to match
        if img1.shape[:2] != img2.shape[:2]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
        
        # Calculate absolute difference
        diff = cv2.absdiff(img1, img2)
        
        # Enhance difference for visibility
        # Convert to HSV and increase saturation
        diff_hsv = cv2.cvtColor(diff, cv2.COLOR_RGB2HSV)
        diff_hsv[:, :, 1] = diff_hsv[:, :, 1] * 2  # Increase saturation
        diff_enhanced = cv2.cvtColor(diff_hsv, cv2.COLOR_HSV2RGB)
        
        # Create heatmap version
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        diff_heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
        diff_heatmap = cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Create a side-by-side comparison
        h, w = img1.shape[:2]
        comparison = np.zeros((h, w*4, 3), dtype=np.uint8)
        
        # First image
        comparison[:, 0:w] = img1
        
        # Second image
        comparison[:, w:w*2] = img2
        
        # Difference
        comparison[:, w*2:w*3] = diff_enhanced
        
        # Heatmap
        comparison[:, w*3:w*4] = diff_heatmap
        
        return comparison
    
    def _compute_distance_matrix(self, images, comparison_func):
        """
        Compute distance matrix between all pairs of images.
        
        Args:
            images: List of images to compare
            comparison_func: Function to use for comparison
            
        Returns:
            numpy.ndarray: Matrix of distances/similarities
        """
        n = len(images)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                # Compute distance (each comparison function returns distance, not similarity)
                distance = comparison_func(images[i], images[j])
                
                # Set symmetric values in the matrix
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        return distance_matrix
    
    def _compare_histogram_correlation(self, img1, img2):
        """
        Compare images using histogram correlation.
        Returns a distance value (1 - correlation, so 0 = identical, 2 = completely opposite).
        """
        hist1_b = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist1_g = cv2.calcHist([img1], [1], None, [256], [0, 256])
        hist1_r = cv2.calcHist([img1], [2], None, [256], [0, 256])
        
        hist2_b = cv2.calcHist([img2], [0], None, [256], [0, 256])
        hist2_g = cv2.calcHist([img2], [1], None, [256], [0, 256])
        hist2_r = cv2.calcHist([img2], [2], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1_b, hist1_b, 0, 1.0, cv2.NORM_MINMAX)
        cv2.normalize(hist1_g, hist1_g, 0, 1.0, cv2.NORM_MINMAX)
        cv2.normalize(hist1_r, hist1_r, 0, 1.0, cv2.NORM_MINMAX)
        
        cv2.normalize(hist2_b, hist2_b, 0, 1.0, cv2.NORM_MINMAX)
        cv2.normalize(hist2_g, hist2_g, 0, 1.0, cv2.NORM_MINMAX)
        cv2.normalize(hist2_r, hist2_r, 0, 1.0, cv2.NORM_MINMAX)
        
        # Calculate correlation for each channel and average
        corr_b = cv2.compareHist(hist1_b, hist2_b, cv2.HISTCMP_CORREL)
        corr_g = cv2.compareHist(hist1_g, hist2_g, cv2.HISTCMP_CORREL)
        corr_r = cv2.compareHist(hist1_r, hist2_r, cv2.HISTCMP_CORREL)
        
        # Average correlation across channels
        avg_corr = (corr_b + corr_g + corr_r) / 3.0
        
        # Convert to distance (1 - correlation)
        # Result range: 0 (identical) to 2 (completely opposite)
        return 1.0 - avg_corr
    
    def _compare_histogram_chi(self, img1, img2):
        """Compare histograms using Chi-Square distance."""
        hist1_b = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist1_g = cv2.calcHist([img1], [1], None, [256], [0, 256])
        hist1_r = cv2.calcHist([img1], [2], None, [256], [0, 256])
        
        hist2_b = cv2.calcHist([img2], [0], None, [256], [0, 256])
        hist2_g = cv2.calcHist([img2], [1], None, [256], [0, 256])
        hist2_r = cv2.calcHist([img2], [2], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1_b, hist1_b, 0, 1.0, cv2.NORM_MINMAX)
        cv2.normalize(hist1_g, hist1_g, 0, 1.0, cv2.NORM_MINMAX)
        cv2.normalize(hist1_r, hist1_r, 0, 1.0, cv2.NORM_MINMAX)
        
        cv2.normalize(hist2_b, hist2_b, 0, 1.0, cv2.NORM_MINMAX)
        cv2.normalize(hist2_g, hist2_g, 0, 1.0, cv2.NORM_MINMAX)
        cv2.normalize(hist2_r, hist2_r, 0, 1.0, cv2.NORM_MINMAX)
        
        # Calculate Chi-Square distance
        chi_b = cv2.compareHist(hist1_b, hist2_b, cv2.HISTCMP_CHISQR)
        chi_g = cv2.compareHist(hist1_g, hist2_g, cv2.HISTCMP_CHISQR)
        chi_r = cv2.compareHist(hist1_r, hist2_r, cv2.HISTCMP_CHISQR)
        
        # Normalize and average
        avg_chi = (chi_b + chi_g + chi_r) / 3.0
        
        # Limit to reasonable range (most values will be between 0-1)
        return min(1.0, avg_chi / 10.0)
    
    def _compare_histogram_intersection(self, img1, img2):
        """
        Compare histograms using intersection method.
        Return value is converted to distance (1 - intersection).
        """
        hist1_b = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist1_g = cv2.calcHist([img1], [1], None, [256], [0, 256])
        hist1_r = cv2.calcHist([img1], [2], None, [256], [0, 256])
        
        hist2_b = cv2.calcHist([img2], [0], None, [256], [0, 256])
        hist2_g = cv2.calcHist([img2], [1], None, [256], [0, 256])
        hist2_r = cv2.calcHist([img2], [2], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1_b, hist1_b, 0, 1.0, cv2.NORM_MINMAX)
        cv2.normalize(hist1_g, hist1_g, 0, 1.0, cv2.NORM_MINMAX)
        cv2.normalize(hist1_r, hist1_r, 0, 1.0, cv2.NORM_MINMAX)
        
        cv2.normalize(hist2_b, hist2_b, 0, 1.0, cv2.NORM_MINMAX)
        cv2.normalize(hist2_g, hist2_g, 0, 1.0, cv2.NORM_MINMAX)
        cv2.normalize(hist2_r, hist2_r, 0, 1.0, cv2.NORM_MINMAX)
        
        # Calculate intersection (higher means more similar)
        inter_b = cv2.compareHist(hist1_b, hist2_b, cv2.HISTCMP_INTERSECT)
        inter_g = cv2.compareHist(hist1_g, hist2_g, cv2.HISTCMP_INTERSECT)
        inter_r = cv2.compareHist(hist1_r, hist2_r, cv2.HISTCMP_INTERSECT)
        
        # Average intersection
        avg_inter = (inter_b + inter_g + inter_r) / 3.0
        
        # Convert to distance (1 - intersection)
        return 1.0 - avg_inter
    
    def _compare_histogram_bhattacharyya(self, img1, img2):
        """Compare histograms using Bhattacharyya distance."""
        hist1_b = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist1_g = cv2.calcHist([img1], [1], None, [256], [0, 256])
        hist1_r = cv2.calcHist([img1], [2], None, [256], [0, 256])
        
        hist2_b = cv2.calcHist([img2], [0], None, [256], [0, 256])
        hist2_g = cv2.calcHist([img2], [1], None, [256], [0, 256])
        hist2_r = cv2.calcHist([img2], [2], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1_b, hist1_b, 0, 1.0, cv2.NORM_MINMAX)
        cv2.normalize(hist1_g, hist1_g, 0, 1.0, cv2.NORM_MINMAX)
        cv2.normalize(hist1_r, hist1_r, 0, 1.0, cv2.NORM_MINMAX)
        
        cv2.normalize(hist2_b, hist2_b, 0, 1.0, cv2.NORM_MINMAX)
        cv2.normalize(hist2_g, hist2_g, 0, 1.0, cv2.NORM_MINMAX)
        cv2.normalize(hist2_r, hist2_r, 0, 1.0, cv2.NORM_MINMAX)
        
        # Calculate Bhattacharyya distance (higher means more different)
        bhatta_b = cv2.compareHist(hist1_b, hist2_b, cv2.HISTCMP_BHATTACHARYYA)
        bhatta_g = cv2.compareHist(hist1_g, hist2_g, cv2.HISTCMP_BHATTACHARYYA)
        bhatta_r = cv2.compareHist(hist1_r, hist2_r, cv2.HISTCMP_BHATTACHARYYA)
        
        # Average distance
        avg_dist = (bhatta_b + bhatta_g + bhatta_r) / 3.0
        
        return avg_dist
    
    def _compare_ssim(self, img1, img2):
        """
        Compare using structural similarity index (SSIM).
        Converted to distance (1 - SSIM) so 0 = identical, 1 = completely different.
        """
        # Ensure images are the same size
        if img1.shape[:2] != img2.shape[:2]:
            # Resize the second image to match the first
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
        
        # Calculate SSIM for each channel
        ssim_values = []
        for i in range(3):  # For each channel (R, G, B)
            ssim_value = ssim(img1[:,:,i], img2[:,:,i], data_range=255)
            ssim_values.append(ssim_value)
        
        # Average SSIM across channels
        avg_ssim = np.mean(ssim_values)
        
        # Convert to distance (1 - SSIM)
        return 1.0 - avg_ssim
    
    def _compare_mse(self, img1, img2):
        """
        Compare using Mean Squared Error.
        Normalized to a 0-1 range where 0 = identical, 1 = very different.
        """
        # Ensure images are the same size
        if img1.shape[:2] != img2.shape[:2]:
            # Resize the second image to match the first
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
        
        # Calculate MSE
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        
        # Normalize to 0-1 range (assuming typical values between 0-10000)
        # This is a rough heuristic; might need adjustment
        normalized_mse = min(1.0, mse / 10000.0)
        
        return normalized_mse