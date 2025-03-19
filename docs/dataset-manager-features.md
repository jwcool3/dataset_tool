# Dataset Manager - Detailed Feature Analysis

This document provides an in-depth analysis of the features offered by the Dataset Manager component, including technical implementation details and use cases.

## Dataset Registry System

### Database Management

The Dataset Registry uses SQLite for persistent storage of dataset metadata:

```sql
CREATE TABLE datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    description TEXT,
    created_date TEXT NOT NULL,
    modified_date TEXT NOT NULL,
    category TEXT,
    file_count INTEGER DEFAULT 0,
    image_count INTEGER DEFAULT 0,
    video_count INTEGER DEFAULT 0,
    parent_id INTEGER,
    processing_history TEXT,
    attributes TEXT,
    FOREIGN KEY(parent_id) REFERENCES datasets(id)
)
```

**Key Features**:
- **Hierarchical relationships**: Parent-child relationships track dataset lineage
- **Processing history**: JSON-encoded history of operations performed on a dataset
- **Dynamic attributes**: Flexible JSON-encoded attributes for additional metadata
- **File statistics**: Cached counts for quick summary information

### Tagging System

Datasets can be tagged for easy filtering and categorization:

```sql
CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
)

CREATE TABLE dataset_tags (
    dataset_id INTEGER,
    tag_id INTEGER,
    PRIMARY KEY(dataset_id, tag_id),
    FOREIGN KEY(dataset_id) REFERENCES datasets(id),
    FOREIGN KEY(tag_id) REFERENCES tags(id)
)
```

**Implementation Details**:
- Tags are stored globally and shared across datasets
- Many-to-many relationship between datasets and tags
- Efficient filtering by tag through SQL joins

### Dataset Metadata Tracking

The registry automatically collects and updates metadata:

- **File counts**: Total, image, and video counts
- **Creation/modification dates**: ISO format timestamps
- **Categories**: User-defined categories (Training, Validation, etc.)
- **Processing lineage**: Parent-child relationships between datasets

## Dataset Operations

### Dataset Splitting

Splits a dataset into training, validation, and test sets:

**Features**:
- **Configurable ratios**: Adjustable percentages for each split
- **Random or stratified**: Option for class-balanced splits
- **Directory preservation**: Maintains subdirectory structure
- **Reproducibility**: Random seed for consistent results

**Implementation**:
```python
def split_dataset(self, dataset_id, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                 random_seed=42, stratify_by=None):
    # Verification and setup
    # ...
    
    # Choose split method based on stratification
    if stratify_by:
        return self._split_stratified(...)
    else:
        return self._split_random(...)
```

**How stratified splitting works**:
1. Identifies class folders in the dataset
2. For each class, splits files according to specified ratios
3. Maintains class distribution across all splits
4. Creates new datasets with the same folder structure

### Dataset Merging

Combines multiple datasets into a single unified dataset:

**Features**:
- **Multiple source datasets**: Can merge any number of datasets
- **Method options**: Copy files or create symbolic links
- **Structure preservation**: Maintains directory structure of source datasets
- **Conflict handling**: Manages duplicate filenames

**Implementation**:
```python
def merge_datasets(self, dataset_ids, merged_name=None, merge_method="copy"):
    # Validation and setup
    # ...
    
    # Process each source dataset
    for dataset in datasets:
        source_path = dataset['path']
        
        # Copy or link files while preserving structure
        for root, dirs, files in os.walk(source_path):
            rel_path = os.path.relpath(root, source_path)
            # ...
```

**Merge methods**:
- **Copy**: Creates physical copies of all files (more space, but independent)
- **Link**: Creates symbolic links to original files (space-efficient, but dependent)

### Dataset Filtering

Creates a new dataset by filtering an existing one based on criteria:

**Filter criteria**:
- **File extension**: Filter by file type (.jpg, .png, etc.)
- **Image dimensions**: Min/max width and height
- **Aspect ratio**: Min/max aspect ratio
- **File size**: Min/max file size

**Implementation**:
```python
def _apply_filters(self, file_path, criteria):
    # Check file extension
    if 'extension' in criteria:
        # ...
    
    # Check file size
    if 'min_size' in criteria or 'max_size' in criteria:
        # ...
    
    # Check image dimensions
    if any(k in criteria for k in ['min_width', 'min_height', 'max_width', 'max_height']):
        # ...
    
    # Check aspect ratio
    if 'min_aspect_ratio' in criteria or 'max_aspect_ratio' in criteria:
        # ...
```

### Dataset Exporting

Exports datasets to standard formats for use with other tools:

**Supported formats**:
- **CSV**: Simple inventory with metadata
- **COCO**: Common Objects in Context format
- **YOLO**: Format for YOLO object detection
- **Pascal VOC**: XML annotation format

**Implementation details**:
- Creates appropriate directory structure for each format
- Converts images to required format if needed
- Generates metadata files (annotations.json, dataset.yaml, etc.)
- Creates placeholder annotations when needed

## Dataset Analysis System

### File Statistics Analysis

Collects comprehensive statistics about files in the dataset:

**Metrics**:
- **File counts**: Total, image, video, and other files
- **File sizes**: Total, average, and by type
- **File formats**: Counts and sizes by extension
- **Folder structure**: Depth, subfolder counts, etc.

**Implementation**:
```python
def _analyze_files(self, path):
    stats = {
        "total_count": 0,
        "image_count": 0,
        "video_count": 0,
        # More fields...
    }
    
    # Process all files
    for root, _, files in os.walk(path):
        for file in files:
            # Update statistics
            # ...
```

### Image Analysis

Analyzes image properties in detail:

**Metrics**:
- **Dimensions**: Width, height, min/max/avg
- **Aspect ratios**: Distribution and statistics
- **Resolution groups**: Grouping by common resolutions
- **Color modes**: RGB, grayscale, CMYK, etc.

**Implementation**:
```python
def _analyze_images(self, path, sample_size=500):
    stats = {
        "dimensions": [],
        "aspect_ratios": [],
        "formats": {},
        "color_modes": {},
        # More fields...
    }
    
    # Sample images for analysis
    # ...
    
    # Process each image
    for file_path in image_files:
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                # Collect image properties
                # ...
```

### Quality Issue Detection

Identifies potential problems in the dataset:

**Issue types**:
- **Corrupt images**: Files that can't be properly loaded
- **Small images**: Images below a size threshold
- **Dark/bright images**: Images with extreme brightness values
- **Invalid files**: Files with incorrect formats or headers

**Implementation**:
```python
def _check_quality_issues(self, path, sample_size=500):
    issues = []
    
    # Find and sample image files
    # ...
    
    # Check for various issues
    for file_path in image_files:
        try:
            # Check image properties
            # ...
            
            # Record issues found
            if condition:
                issues.append({
                    "type": "issue_type",
                    "path": file_path,
                    "details": {...}
                })
        except:
            # Record loading errors
            # ...
```

### Duplicate Detection

Finds duplicate or very similar images:

**Detection methods**:
- **Exact duplicates**: Identified by file hash
- **Similar images**: Identified by perceptual hash
- **Configurable threshold**: Adjustable similarity threshold

**Implementation**:
```python
def _find_duplicates(self, path, sample_size=1000, threshold=0.9):
    results = {
        "exact_duplicates": [],
        "similar_images": [],
        "analyzed_count": 0
    }
    
    # Find exact duplicates using file hash
    # ...
    
    # Find similar images using perceptual hash
    try:
        # Calculate perceptual hashes
        # ...
        
        # Compare hashes and find similarities
        for i in range(len(image_hashes)):
            for j in range(i+1, len(image_hashes)):
                # Calculate similarity
                # ...
                
                if similarity >= threshold:
                    results["similar_images"].append({...})
    except:
        # Handle case where libraries aren't available
        # ...
```

## User Interface Components

### Dataset Explorer

Provides a visual interface for browsing and managing datasets:

**Components**:
- **Tree view**: Hierarchical display of datasets with parent-child relationships
- **Search/filter**: Ability to search and filter datasets
- **Details panel**: Shows comprehensive information about selected dataset
- **Action buttons**: Common actions like add, edit, delete

**Implementation details**:
- Uses Tkinter's ttk.Treeview for hierarchical display
- Lazy loading of dataset details for performance
- Responsive layout with resizable panels

### Dataset Preview

Provides visual preview of dataset contents:

**Features**:
- **Thumbnail grid**: Visual preview of dataset images
- **Statistics display**: Summary statistics in a readable format
- **Quality issues display**: List of potential problems
- **Duplicate display**: Shows duplicate or similar images

**Implementation details**:
- Loads a limited sample of images for performance
- Uses PIL for thumbnail generation
- Provides tooltips and navigation aids

### Operation Interfaces

Provides interfaces for configuring and executing operations:

**Implementation pattern**:
1. User selects a dataset in the explorer
2. Configures operation parameters in the right panel
3. Clicks the action button to execute
4. Operation runs in a background thread
5. Results are displayed and new datasets are registered

**Threading model**:
- Long-running operations execute in separate threads
- UI remains responsive during processing
- Thread-safe updates to the UI via event queue

## Integration with Main Application

### Input/Output Tab Integration

Connects the dataset manager with the input/output tab:

**Features**:
- Button to add current directory as dataset
- Button to browse datasets
- Option to automatically add processing results as datasets

### Processing Pipeline Integration

Integrates with the main processing pipeline:

**Implementation**:
- Wraps the original processing method
- Automatically registers output directories as datasets
- Preserves relationships between input and output datasets

### Gallery Integration

Integrates with the gallery view:

**Features**:
- Button to add current gallery view as dataset
- Dialog to select which directory to add
- Automatic navigation to the new dataset

## Standalone Mode

The system can function as a standalone dataset manager:

**Implementation**:
```python
def create_standalone_dataset_manager():
    import tkinter as tk
    from tkinter import ttk
    
    root = tk.Tk()
    root.title("Dataset Manager")
    root.geometry("1200x800")
    
    # Create minimal app object
    # ...
    
    # Initialize Dataset Manager
    # ...
    
    # Add menus and UI elements
    # ...
    
    return root, app
```

**Features**:
- Full dataset management functionality
- Simplified UI focused on dataset operations
- Can be launched independently of the main application

## Help System

Provides contextual help for dataset management:

**Implementation**:
```python
def _show_dataset_manager_help(app):
    # Create help dialog
    help_dialog = tk.Toplevel(app.root)
    # ...
    
    # Help content
    help_text = """Dataset Manager Help
    
    The Dataset Manager helps you organize, track, and manipulate multiple datasets...
    """
    
    # Display help text
    # ...
```

**Content areas**:
- Core features explanation
- Getting started guide
- Best practices
- Operation-specific help
