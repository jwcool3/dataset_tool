# Dataset Manager - User Guide

This guide provides practical instructions for using the Dataset Manager component of the Dataset Preparation Tool.

## Getting Started

### Accessing the Dataset Manager

There are multiple ways to access the Dataset Manager:

1. **Main Tab**: Click on the "Dataset Manager" tab in the main application
2. **Tools Menu**: Select Tools → Dataset Manager from the menu bar
3. **Integration Buttons**: Use the Dataset Manager buttons in the Input/Output tab

### Adding Your First Dataset

To add your first dataset to the registry:

1. Click the "Add Dataset" button in the Dataset Explorer panel
2. Enter a descriptive name for your dataset
3. Click "Browse..." to select the directory containing your dataset
4. Add an optional description and category
5. Click "Create Dataset"

The dataset will appear in the tree view on the left side of the Dataset Manager.

### Viewing Dataset Details

To view details about a dataset:

1. Click on the dataset in the tree view
2. The details panel on the right will show:
   - Basic information (name, path, file counts)
   - Description and tags
   - Thumbnails of dataset content
   - Statistics and quality information

## Dataset Operations

### Splitting a Dataset

To split a dataset into training, validation, and test sets:

1. Select the dataset in the tree view
2. Go to the "Split Dataset" tab in the operations panel
3. Adjust the split ratios (default: 70% training, 15% validation, 15% test)
4. Configure additional options:
   - Random seed for reproducibility
   - Enable stratified split if your dataset has class folders
5. Click "Split Selected Dataset"

The operation will create three new datasets (train, validation, test) as children of the original dataset.

### Merging Datasets

To merge multiple datasets into a single dataset:

1. Go to the "Merge Datasets" tab in the operations panel
2. Select the datasets you want to merge from the list (hold Ctrl to select multiple)
3. Enter a name for the merged dataset
4. Choose a merge method:
   - "copy": Creates copies of all files (uses more disk space)
   - "link": Creates links to original files (saves space but requires originals)
5. Click "Merge Selected Datasets"

The operation will create a new dataset containing all files from the source datasets.

### Filtering a Dataset

To create a filtered version of a dataset:

1. Select the dataset in the tree view
2. Go to the "Filter Dataset" tab in the operations panel
3. Configure filter criteria:
   - File type (e.g., .jpg, .png)
   - Minimum/maximum dimensions
   - Aspect ratio constraints
4. Enter a name for the filtered dataset
5. Click "Filter Selected Dataset"

The operation will create a new dataset containing only files that match your criteria.

### Exporting a Dataset

To export a dataset to a standard format:

1. Select the dataset in the tree view
2. Go to the "Export Dataset" tab in the operations panel
3. Choose an export format:
   - CSV: Simple inventory with metadata
   - COCO: Common Objects in Context format
   - YOLO: Format for YOLO object detection
   - Pascal VOC: XML annotation format
4. Specify an output path
5. Click "Export Selected Dataset"

The operation will create a new directory with the dataset in the selected format.

## Dataset Analysis

### Running Analysis

To analyze a dataset:

1. Select the dataset in the tree view
2. Go to the "Analysis" tab in the operations panel
3. Configure analysis options:
   - File Statistics: Basic file information
   - Image Properties: Analysis of image dimensions, formats, etc.
   - Find Quality Issues: Identify problematic images
   - Find Duplicate Images: Detect exact and similar duplicates
4. Adjust the sample size for performance (larger samples take longer)
5. Click "Analyze Selected Dataset"

The results will appear in the tabs below the analysis options.

### Interpreting Analysis Results

The analysis results are organized into several tabs:

1. **Files**: Statistics about file counts, sizes, and formats
2. **Images**: Information about image dimensions, aspect ratios, and color modes
3. **Issues**: List of potential quality issues (corrupt, small, dark images, etc.)
4. **Duplicates**: List of exact duplicates and similar images

Each tab provides detailed information and, where applicable, options to view the files.

## Advanced Features

### Tagging Datasets

Tags help you organize and find datasets:

1. Select a dataset in the tree view
2. In the details panel, find the "Tags" section
3. Enter a tag name in the text field
4. Click "Add Tag"

You can later filter datasets by tag using the search box in the Dataset Explorer.

### Managing Dataset Relationships

The Dataset Manager tracks relationships between datasets:

- **Parent-Child**: Datasets can have parent-child relationships (e.g., original and processed)
- **Hierarchy**: The tree view shows the dataset hierarchy
- **Lineage**: You can see which operations created which datasets

When adding a new dataset, you can specify its parent to maintain these relationships.

### Integration with Processing

The Dataset Manager integrates with the main processing pipeline:

1. **Automatic Registration**: Processing results can be automatically added as datasets
2. **Process from Dataset**: You can send a dataset directly to processing
3. **Gallery Integration**: You can add the current gallery view as a dataset

To enable automatic registration, check the "Automatically add processing results to dataset registry" option in the Input/Output tab.

### Batch Operations

For batch operations on multiple datasets:

1. Use the merge functionality to combine datasets
2. Use the filter functionality with custom criteria
3. Export multiple datasets to the same format for comparison

## Tips and Best Practices

### Naming Conventions

Adopt consistent naming conventions for datasets:

- Use descriptive names that indicate content and purpose
- Include version numbers for iterations
- Indicate processing steps in the name (e.g., "faces_cropped")

### Tagging Strategy

Develop a tagging strategy for better organization:

- Use tags for dataset status (raw, processed, verified)
- Use tags for content type (faces, objects, scenes)
- Use tags for project association (project1, project2)

### Regular Analysis

Analyze datasets regularly to catch issues:

- Run analysis on new datasets before processing
- Look for quality issues that might affect model training
- Identify and remove duplicate images that could bias results

### Disk Space Management

Manage disk space efficiently:

- Use the "link" method when merging large datasets
- Delete intermediate datasets that are no longer needed
- Export only the datasets you need to share

## Troubleshooting

### Common Issues and Solutions

**Dataset not appearing in tree view:**
- Click the "Refresh" button in the toolbar
- Check that the path exists and is accessible

**Analysis taking too long:**
- Reduce the sample size in the analysis options
- Run analysis on a subset of the dataset

**Operations failing:**
- Check for sufficient disk space
- Ensure you have write permissions for the output directory
- Look for error messages in the status area

**Image previews not loading:**
- Ensure the image files are valid
- Check that you have the Pillow library installed
- Try refreshing the dataset selection

### Getting Help

For more help with the Dataset Manager:

1. Click on "Dataset Manager Help" in the Help menu
2. Read the detailed help information for each feature
3. Check for tooltips and hints in the UI