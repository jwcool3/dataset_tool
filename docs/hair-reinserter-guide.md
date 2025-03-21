# Smart Hair Reinserter Guide

## Overview

The Smart Hair Reinserter is a specialized feature in the Dataset Preparation Tool designed specifically for hair replacement workflows. It provides intelligent alignment and blending capabilities to seamlessly incorporate AI-generated or modified hair into original images.

## Use Cases

- Replacing blonde hair with red/brown/black hair
- Improving the quality of AI-generated hairstyles
- Blending multiple hair styles or colors together
- Fixing inconsistent hair in AI-generated images

## Getting Started

### Basic Usage

1. **Input Directory Setup**:
   - Place your processed images (with modified hair) in your Input Directory
   - Ensure each processed image has a corresponding mask in a "masks" subdirectory
   - Set your Source Directory to the folder with your original images

2. **Enable Smart Hair Reinserter**:
   - In the Input/Output tab, check "Reinsert cropped images"
   - Enable "Use Smart Hair Reinserter (optimized for hair replacement)"
   - Click "Start Processing"

3. **Output**:
   - Reinserted images will be saved in the "reinserted" subdirectory of your Output Directory
   - If Debug Mode is enabled, you'll also get visualization images in the "reinsert_debug" subdirectory

## Advanced Features

### Vertical Alignment Bias

This controls how the processed hair aligns vertically with the original:

- **Positive values** (5-20): Move hair downward (good for keeping hair away from face)
- **Zero**: No vertical adjustment
- **Negative values** (-5 to -20): Move hair upward (good for updos, ponytails)

### Soft Edge Width

Controls the smoothness of the transition between original and processed hair:

- **Lower values** (5-10): Sharper edges, more defined boundary
- **Medium values** (10-20): Natural blending for most hairstyles
- **Higher values** (20-30): Very soft blending for wispy or thin hair

### Hair Presets

Use the quick preset buttons for common hair types:

- **Natural Hair**: Balanced settings for most real-world hair
- **Anime Hair**: Sharper edges with less feathering for animated-style hair
- **Updo/Ponytail**: Optimized for vertically-styled hair

### Hair Color Correction

When enabled, the Smart Hair Reinserter will automatically analyze and adjust the color characteristics of the processed hair to better match the source image's color profile. This helps maintain consistent lighting and color temperature.

## Troubleshooting

### Common Issues

1. **Hair Positioned Incorrectly**:
   - Adjust the Vertical Alignment Bias slider
   - Try different alignment methods in the Advanced options

2. **Unnatural Edges**:
   - Increase the Soft Edge Width for softer transitions
   - Try the "feathered" blend mode for gradual blending

3. **Color Mismatch**:
   - Ensure Hair Color Correction is enabled
   - Use Debug Mode to check alignment and color matching

4. **Missing Hair Sections**:
   - Check that your masks fully cover the hair region
   - Ensure source and processed images are properly paired

### Debug Visualizations

When Debug Mode is enabled, the following visualizations are saved:

- **source_original.png**: The original source image
- **processed_original.png**: The processed image before alignment
- **mask_original.png**: The hair mask used for processing
- **aligned_image_debug.png**: The processed image after alignment
- **mask_alignment_viz.png**: Visual representation of mask alignment
- **comparison_[filename].png**: Side-by-side comparison of source and result

## Advanced Configuration

For more complex hair replacements, you can fine-tune these advanced settings:

1. **Alignment Method**:
   - `centroid`: Balances overall hair position (default)
   - `bbox`: Better for rectangular hair regions
   - `contour`: Best for matching specific hair shapes

2. **Blend Mode**:
   - `alpha`: Simple transparency blending
   - `feathered`: Gradual transparency for natural transitions
   - `poisson`: Seamless blending that preserves textures

## Tips for Best Results

1. **Clean Hair Masks**:
   - Masks should contain only hair regions, not face or other features
   - Binary masks work best (pure black and white)

2. **Similar Hair Styles**:
   - The closer the hair shapes are between source and processed, the better the results
   - Long to short or curly to straight transitions may require manual adjustments

3. **Optimal Workflow**:
   - Process small batches and refine settings as needed
   - Save successful settings for similar images

4. **Color Consistency**:
   - Ensure source and processed images have similar lighting conditions
   - Use Hair Color Correction for automatic color matching

## Example Commands

For command-line usage:

```bash
python dataset_tool.py --input ./processed_hair --source ./original_images --output ./results --reinsert --smart-hair --vertical-bias 10 --soft-edge 15
```

This guide should help you get the most out of the Smart Hair Reinserter feature. Experiment with different settings to find what works best for your specific hair replacement needs.
