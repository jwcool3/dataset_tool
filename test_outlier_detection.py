"""
Test script for the ImageComparison class
This can be used to verify that all methods are working correctly.
"""

import os
import sys
import cv2
import numpy as np
from utils.image_comparison import ImageComparison
from PIL import Image

def test_image_comparison():
    """Test the ImageComparison class functionality."""
    print("Testing ImageComparison class...")
    
    # Create test directory if it doesn't exist
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create some test images if they don't exist
    test_images = []
    
    # If no test images exist, create some
    if not os.listdir(test_dir):
        print("Creating test images...")
        
        # Create a few simple test images
        # 1. Plain red image
        red_img = np.zeros((100, 100, 3), dtype=np.uint8)
        red_img[:, :, 2] = 255  # Set red channel to max
        red_path = os.path.join(test_dir, "red.png")
        cv2.imwrite(red_path, red_img)
        test_images.append(red_path)
        
        # 2. Plain blue image
        blue_img = np.zeros((100, 100, 3), dtype=np.uint8)
        blue_img[:, :, 0] = 255  # Set blue channel to max
        blue_path = os.path.join(test_dir, "blue.png")
        cv2.imwrite(blue_path, blue_img)
        test_images.append(blue_path)
        
        # 3. Gradient image
        gradient_img = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            gradient_img[:, i, :] = i * 2.55  # Create gradient
        gradient_path = os.path.join(test_dir, "gradient.png")
        cv2.imwrite(gradient_path, gradient_img)
        test_images.append(gradient_path)
        
        # 4. Checkerboard image
        checkerboard = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(0, 100, 20):
            for j in range(0, 100, 20):
                checkerboard[i:i+10, j:j+10, :] = 255
        checkerboard_path = os.path.join(test_dir, "checkerboard.png")
        cv2.imwrite(checkerboard_path, checkerboard)
        test_images.append(checkerboard_path)
        
        # 5. Similar to red but slightly different
        red_similar = np.zeros((100, 100, 3), dtype=np.uint8)
        red_similar[:, :, 2] = 240  # Slightly darker red
        red_similar_path = os.path.join(test_dir, "red_similar.png")
        cv2.imwrite(red_similar_path, red_similar)
        test_images.append(red_similar_path)
    else:
        # Use existing test images
        print("Using existing test images...")
        for file in os.listdir(test_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join(test_dir, file))
    
    # Initialize ImageComparison instance
    comparator = ImageComparison()
    
    # Test basic comparison
    print("\nTesting basic comparison...")
    for method in comparator.comparison_methods:
        print(f"\nMethod: {method}")
        result = comparator.compare_images(test_images, method=method)
        
        if result.get("success", False):
            print(f"  - Successfully compared {len(test_images)} images")
            print(f"  - Distance matrix shape: {result['distance_matrix'].shape}")
            print(f"  - Outlier scores: {[f'{s:.4f}' for s in result['outlier_scores']]}")
            print(f"  - Ranked paths: {[os.path.basename(p) for p in result['ranked_paths']]}")
        else:
            print(f"  - Error: {result.get('message', 'Unknown error')}")
    
    # Test outlier detection
    print("\nTesting outlier detection...")
    for threshold in [0.1, 0.3, 0.5]:
        print(f"\nThreshold: {threshold}")
        result = comparator.find_outliers(test_images, threshold=threshold)
        
        if result.get("success", False):
            print(f"  - Found {len(result['outlier_paths'])} outliers")
            for i, (path, score) in enumerate(zip(result['outlier_paths'], result['outlier_scores'])):
                print(f"  - #{i+1}: {os.path.basename(path)} (score: {score:.4f})")
        else:
            print(f"  - Error: {result.get('message', 'Unknown error')}")
    
    # Test visualization functions
    print("\nTesting visualization functions...")
    
    # Test difference visualization
    if len(test_images) >= 2:
        print("\nTesting visualize_differences...")
        
        # Load two images
        img1 = cv2.imread(test_images[0])
        img2 = cv2.imread(test_images[1])
        
        if img1 is not None and img2 is not None:
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            # Generate difference visualization
            diff_vis = comparator.visualize_differences(img1_rgb, img2_rgb)
            
            # Save visualization
            diff_vis_path = os.path.join(test_dir, "difference_visualization.png")
            cv2.imwrite(diff_vis_path, diff_vis)
            print(f"  - Saved difference visualization to {diff_vis_path}")
        else:
            print("  - Error loading images for difference visualization")
    
    # Test detailed report generation
    if len(test_images) >= 2:
        print("\nTesting generate_detailed_report...")
        
        # Load two images
        img1 = cv2.imread(test_images[0])
        img2 = cv2.imread(test_images[1])
        
        if img1 is not None and img2 is not None:
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            # Generate detailed report
            filename1 = os.path.basename(test_images[0])
            filename2 = os.path.basename(test_images[1])
            try:
                report = comparator.generate_detailed_report(img1_rgb, img2_rgb, filename1, filename2)
                
                # Save report
                report_path = os.path.join(test_dir, "detailed_report.png")
                report.save(report_path)
                print(f"  - Saved detailed report to {report_path}")
            except Exception as e:
                print(f"  - Error generating detailed report: {str(e)}")
        else:
            print("  - Error loading images for detailed report")
    
    # Test comparison visualization
    if len(test_images) >= 3:
        print("\nTesting generate_comparison_visualization...")
        
        # First run a comparison to get the distance matrix and outlier scores
        result = comparator.compare_images(test_images)
        
        if result.get("success", False):
            # Load images
            images = []
            for path in test_images:
                img = cv2.imread(path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img_rgb)
            
            # Generate visualization
            try:
                vis = comparator.generate_comparison_visualization(
                    images, 
                    result["distance_matrix"], 
                    result["outlier_scores"], 
                    test_images
                )
                
                # Save visualization
                vis_path = os.path.join(test_dir, "comparison_visualization.png")
                vis.save(vis_path)
                print(f"  - Saved comparison visualization to {vis_path}")
            except Exception as e:
                print(f"  - Error generating comparison visualization: {str(e)}")
        else:
            print(f"  - Error in comparison: {result.get('message', 'Unknown error')}")
    
    print("\nTesting completed.")

if __name__ == "__main__":
    test_image_comparison()