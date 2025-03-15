"""
Enhanced Image Comparison Utilities for Dataset Preparation Tool
Provides advanced functions to compare images and identify outliers in groups.
"""
import os
import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
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
        valid_paths = []
        
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
                valid_paths.append(path)
                
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
            "paths": valid_paths,
            "ranked_paths": [valid_paths[i] for i in sorted_indices],
            "comparison_method": method,
            "color_space": color_space
        }
        
        return result
    
    def generate_comparison_visualization(self, images, distance_matrix, outlier_scores, 
                                        image_paths=None, max_images=12):
        """
        Generate visualization of image similarity and outliers.
        
        Args:
            images: List of image arrays
            distance_matrix: Matrix of distances between images
            outlier_scores: Outlier score for each image
            image_paths: Optional list of image paths (for labels)
            max_images: Maximum number of images to include in visualization
            
        Returns:
            PIL.Image: Visualization as a PIL image
        """
        # Limit to max_images
        n_images = min(len(images), max_images)
        
        # Sort images by outlier score (descending)
        sorted_indices = np.argsort(outlier_scores)[::-1][:n_images]
        
        # Create a figure with subplots for the heatmap and top outliers
        fig = Figure(figsize=(12, 8), dpi=100)
        
        # Add a suptitle
        fig.suptitle("Image Similarity Analysis", fontsize=16)
        
        # Create a 2x1 grid layout
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
        
        # First subplot: heatmap of image similarities
        ax1 = fig.add_subplot(gs[0])
        
        # Plot distance heatmap (truncated to max_images)
        truncated_matrix = distance_matrix[sorted_indices][:, sorted_indices]
        im = ax1.imshow(truncated_matrix, cmap='viridis', interpolation='nearest')
        ax1.set_title('Image Distance Matrix (darker = more different)')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Distance')
        
        # Add labels if image_paths is provided
        if image_paths:
            sorted_labels = []
            for i in sorted_indices:
                path = image_paths[i]
                folder = os.path.basename(os.path.dirname(path))
                filename = os.path.basename(path)
                sorted_labels.append(f"{folder}/{filename[:10]}")
            
            ax1.set_xticks(np.arange(n_images))
            ax1.set_yticks(np.arange(n_images))
            ax1.set_xticklabels(sorted_labels, rotation=90, fontsize=8)
            ax1.set_yticklabels(sorted_labels, fontsize=8)
        
        # Add grid to help with readability
        ax1.grid(False)
        
        # Second subplot: outlier scores bar chart
        ax2 = fig.add_subplot(gs[1])
        
        # Get sorted outlier scores
        sorted_scores = outlier_scores[sorted_indices]
        
        # Plot bar chart
        bars = ax2.bar(np.arange(n_images), sorted_scores, color='skyblue')
        ax2.set_title('Outlier Scores (Higher score = more different from other images)')
        ax2.set_xlabel('Image Index')
        ax2.set_ylabel('Outlier Score')
        
        # Add labels if image_paths is provided
        if image_paths:
            ax2.set_xticks(np.arange(n_images))
            short_labels = [f"{i+1}: {os.path.basename(image_paths[sorted_indices[i]])[:10]}" 
                           for i in range(n_images)]
            ax2.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
        
        # Adjust layout
        fig.tight_layout()
        
        # Convert to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        
        # Close the figure to free memory
        plt.close(fig)
        
        # Convert to PIL image
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
        
        # Store original dimensions for reference
        orig_h1, orig_w1 = img1.shape[:2]
        orig_h2, orig_w2 = img2.shape[:2]
        
        # Resize images to match the smaller dimensions for comparison
        if img1.shape[:2] != img2.shape[:2]:
            # Determine which image is smaller
            if orig_h1 * orig_w1 < orig_h2 * orig_w2:
                target_size = (orig_w1, orig_h1)
                img2 = cv2.resize(img2, target_size, interpolation=cv2.INTER_AREA)
            else:
                target_size = (orig_w2, orig_h2)
                img1 = cv2.resize(img1, target_size, interpolation=cv2.INTER_AREA)
        
        # Get final dimensions
        h, w = img1.shape[:2]
        
        # Calculate absolute difference
        diff = cv2.absdiff(img1, img2)
        
        # Create a colored version of the difference (green highlights differences)
        diff_enhanced = np.zeros_like(diff)
        # Enhance green channel where differences are significant
        threshold = 30  # Threshold to consider a difference significant
        diff_mask = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY) > threshold
        diff_enhanced[diff_mask, 1] = 255  # Set green channel to max where differences are significant
        
        # Create a heatmap version
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        # Apply CLAHE to enhance contrast in the difference image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        diff_gray = clahe.apply(diff_gray)
        # Apply color map
        diff_heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
        diff_heatmap = cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Create a side-by-side comparison with labels
        # Add padding between images
        padding = 10
        
        # Create a larger canvas with padding
        comparison = np.zeros((h + 30, w*4 + padding*3, 3), dtype=np.uint8)
        comparison.fill(240)  # Light gray background
        
        # Labels for the views
        labels = ["Original 1", "Original 2", "Difference", "Heatmap"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (0, 0, 0)  # Black
        font_thickness = 2
        
        # Draw labels and images
        for i, (label, img) in enumerate(zip(labels, [img1, img2, diff_enhanced, diff_heatmap])):
            # Calculate position
            x_offset = i * (w + padding)
            
            # Add label
            label_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            label_x = x_offset + (w - label_size[0]) // 2
            cv2.putText(comparison, label, (label_x, 25), font, font_scale, font_color, font_thickness)
            
            # Add image
            comparison[30:30+h, x_offset:x_offset+w] = img
            
            # Add grid line (vertical separator)
            if i < 3:
                line_x = x_offset + w + padding // 2
                cv2.line(comparison, (line_x, 0), (line_x, h+30), (180, 180, 180), 1)
        
        return comparison
    
    def generate_detailed_report(self, img1, img2, path1, path2):
        """
        Generate a detailed report comparing two images.
        
        Args:
            img1: First image (numpy array)
            img2: Second image (numpy array)
            path1: Path of first image
            path2: Path of second image
            
        Returns:
            PIL.Image: Report as a PIL image
        """
        # Extract folder names for better display
        folder1 = os.path.basename(os.path.dirname(path1))
        folder2 = os.path.basename(os.path.dirname(path2))
        filename1 = os.path.basename(path1)
        filename2 = os.path.basename(path2)
        
        # Use combined folder/filename for display
        # Use combined folder/filename for display
        display_name1 = f"{folder1}/{filename1}"
        display_name2 = f"{folder2}/{filename2}"
        
        # Ensure images are in RGB
        if img1.shape[2] != 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        if img2.shape[2] != 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        
        # Resize if necessary
        if img1.shape[:2] != img2.shape[:2]:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            # Resize the larger image to match the smaller
            if h1*w1 > h2*w2:
                img1 = cv2.resize(img1, (w2, h2), interpolation=cv2.INTER_AREA)
            else:
                img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_AREA)
        
        # Get dimensions
        h, w = img1.shape[:2]
        
        # Calculate metrics
        metrics = {}
        # Histogram comparison
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
        
        # Calculate correlation
        metrics['correlation'] = {}
        metrics['correlation']['blue'] = cv2.compareHist(hist1_b, hist2_b, cv2.HISTCMP_CORREL)
        metrics['correlation']['green'] = cv2.compareHist(hist1_g, hist2_g, cv2.HISTCMP_CORREL)
        metrics['correlation']['red'] = cv2.compareHist(hist1_r, hist2_r, cv2.HISTCMP_CORREL)
        metrics['correlation']['avg'] = (metrics['correlation']['blue'] + 
                                      metrics['correlation']['green'] + 
                                      metrics['correlation']['red']) / 3.0
        
        # Calculate SSIM
        metrics['ssim'] = {}
        for i, color in enumerate(['blue', 'green', 'red']):
            metrics['ssim'][color] = ssim(img1[:,:,i], img2[:,:,i], data_range=255)
        metrics['ssim']['avg'] = np.mean(list(metrics['ssim'].values())[:3])  # Average of RGB
        
        # Calculate MSE
        metrics['mse'] = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        
        # Create detailed report figure
        fig = Figure(figsize=(10, 12), dpi=100)
        
        # Add title
        fig.suptitle(f"Detailed Image Comparison Report", fontsize=16)
        
        # Create a grid layout
        gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.5], width_ratios=[1, 1], hspace=0.4, wspace=0.3)
        
        # Image previews
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img1)
        ax1.set_title(f"Image 1: {display_name1}")
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img2)
        ax2.set_title(f"Image 2: {display_name2}")
        ax2.axis('off')
        
        # Difference visualizations
        diff = cv2.absdiff(img1, img2)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        diff_heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
        diff_heatmap = cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB)
        
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.imshow(diff)
        ax3.set_title("Absolute Difference")
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.imshow(diff_heatmap)
        ax4.set_title("Difference Heatmap")
        ax4.axis('off')
        
        # Histogram comparisons
        ax5 = fig.add_subplot(gs[2, 0])
        
        # Plot the histograms
        colors = ['blue', 'green', 'red']
        for i, color_name in enumerate(colors):
            ax5.plot(cv2.normalize(hist1_r if i == 2 else hist1_g if i == 1 else hist1_b, None, 0, 1, cv2.NORM_MINMAX), 
                    color=color_name, linestyle='-', alpha=0.7, label=f"Image 1 {color_name}")
            ax5.plot(cv2.normalize(hist2_r if i == 2 else hist2_g if i == 1 else hist2_b, None, 0, 1, cv2.NORM_MINMAX), 
                    color=color_name, linestyle='--', alpha=0.7, label=f"Image 2 {color_name}")
        
        ax5.set_title("Color Histograms Comparison")
        ax5.set_xlabel("Pixel Value")
        ax5.set_ylabel("Normalized Frequency")
        ax5.legend(loc='upper right', fontsize='x-small')
        
        # Metrics summary
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        metrics_text = (
            f"Similarity Metrics:\n\n"
            f"• Histogram Correlation:\n"
            f"  - Red: {metrics['correlation']['red']:.4f}\n"
            f"  - Green: {metrics['correlation']['green']:.4f}\n"
            f"  - Blue: {metrics['correlation']['blue']:.4f}\n"
            f"  - Average: {metrics['correlation']['avg']:.4f}\n\n"
            f"• Structural Similarity (SSIM):\n"
            f"  - Red: {metrics['ssim']['red']:.4f}\n"
            f"  - Green: {metrics['ssim']['green']:.4f}\n"
            f"  - Blue: {metrics['ssim']['blue']:.4f}\n"
            f"  - Average: {metrics['ssim']['avg']:.4f}\n\n"
            f"• Mean Squared Error: {metrics['mse']:.2f}\n"
            f"  (Lower values indicate higher similarity)"
        )
        
        ax6.text(0, 0.95, metrics_text, va='top', fontsize=10)
        
        # Summary and interpretation
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        # Interpret the results
        if metrics['correlation']['avg'] > 0.9 and metrics['ssim']['avg'] > 0.9:
            interpretation = "The images are VERY SIMILAR with only minor differences."
        elif metrics['correlation']['avg'] > 0.8 and metrics['ssim']['avg'] > 0.8:
            interpretation = "The images are SIMILAR with some noticeable differences."
        elif metrics['correlation']['avg'] > 0.6 and metrics['ssim']['avg'] > 0.6:
            interpretation = "The images have MODERATE differences."
        else:
            interpretation = "The images are SIGNIFICANTLY DIFFERENT."
        
        summary_text = (
            f"Summary: {interpretation}\n\n"
            f"• Histogram Correlation: {metrics['correlation']['avg']:.4f} (1.0 = identical, -1.0 = inverse)\n"
            f"• SSIM: {metrics['ssim']['avg']:.4f} (1.0 = identical, 0.0 = no similarity)\n"
            f"• Image Dimensions: {w}x{h} pixels"
        )
        
        ax7.text(0.5, 0.5, summary_text, va='center', ha='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.2))
        
        # Convert to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        
        # Close the figure to free memory
        plt.close(fig)
        
        # Convert to PIL image
        pil_img = Image.open(buf)
        return pil_img
    
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
    
    def find_outliers(self, image_paths, threshold=0.25, method="histogram_correlation"):
        """
        Find outlier images in a set based on their average dissimilarity.
        
        Args:
            image_paths: List of paths to images
            threshold: Outlier threshold (higher = more restrictive)
            method: Comparison method to use
            
        Returns:
            dict: Results with outlier paths and scores
        """
        # Compare all images
        result = self.compare_images(image_paths, method=method)
        
        if not result.get("success", False):
            return {
                "success": False,
                "message": result.get("message", "Comparison failed")
            }
        
        # Get outlier scores and threshold
        outlier_scores = result.get("outlier_scores", [])
        
        # Calculate adaptive threshold if needed
        if threshold is None or threshold <= 0:
            # Use mean + 1.5 * std as a typical outlier threshold
            mean_score = np.mean(outlier_scores)
            std_score = np.std(outlier_scores)
            threshold = mean_score + 1.5 * std_score
        
        # Find outliers
        outlier_indices = [i for i, score in enumerate(outlier_scores) if score > threshold]
        outlier_paths = [result["paths"][i] for i in outlier_indices]
        outlier_scores_filtered = [outlier_scores[i] for i in outlier_indices]
        
        # Sort by score (descending)
        sorted_indices = np.argsort(outlier_scores_filtered)[::-1]
        sorted_paths = [outlier_paths[i] for i in sorted_indices]
        sorted_scores = [outlier_scores_filtered[i] for i in sorted_indices]
        
        return {
            "success": True,
            "message": f"Found {len(sorted_paths)} outliers using threshold {threshold:.4f}",
            "outlier_paths": sorted_paths,
            "outlier_scores": sorted_scores,
            "threshold": threshold,
            "comparison_method": method
        }