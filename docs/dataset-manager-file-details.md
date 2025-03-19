# Dataset Manager - File-by-File Analysis

This document provides a detailed breakdown of each file in the Dataset Preparation Tool, focusing on the Dataset Manager component.

## Core Application Files

### `main.py`

**Purpose**: Main entry point for the application

**Key Functions**:
- `check_dependencies()`: Verifies required packages (opencv-python, pillow, numpy)
- `main()`: Launches the application with proper error handling

**Flow**:
1. Checks for required dependencies
2. Shows error if packages are missing
3. Initializes and runs the main application
4. Catches and displays any startup errors

### `app.py`

**Purpose**: Initializes the main application window

**Key Classes**:
- `DatasetPreparationApp`: Main application class

**Key Functions**:
- `__init__()`: Initializes the application window
- `run()`: Starts the application main loop

**Implementation Details**:
- Sets window size, title, and position
- Centers the window on screen
- Initializes the main UI components

### `setup.py`

**Purpose**: Creates the required directory structure

**Key Functions**:
- `create_directory_structure()`: Creates folders and initializes empty `__init__.py` files

**Implementation Details**:
- Creates directories: `ui`, `ui/tabs`, `processors`, `utils`
- Adds `__init__.py` files with docstrings

## Dataset Manager Files

### `utils/dataset_manager/__init__.py`

**Purpose**: Exports the main components of the dataset manager module

**Key Exports**:
- `DatasetRegistry`: Database manager for datasets
- `DatasetExplorer`: UI component for browsing datasets
- `DatasetOperations`: Implementation of dataset operations
- `DatasetAnalyzer`: Analyzer for dataset statistics and quality
- `DatasetManagerTab`: Main UI tab component
- `add_dataset_manager_to_app()`: Integration function
- `initialize_dataset_manager()`: Initialization function

### `utils/dataset_manager.py`

**Purpose**: Main integration point for adding the dataset manager to the application

**Key Functions**:
- `add_dataset_manager_to_app()`: Adds dataset manager to the main application
- `_add_integration_hooks()`: Adds hooks to connect with existing functionality
- `_add_input_output_integration()`: Integrates with the input/output tab
- `_add_processing_integration()`: Adds hooks to the processing pipeline
- `_add_gallery_integration()`: Integrates with the gallery tab
- `_add_current_directory()`: Adds the current input directory as a dataset
- `create_standalone_dataset_manager()`: Creates a standalone version of the dataset manager

**Implementation Details**:
- Adds UI elements to interact with the dataset manager
- Wraps processing functions to automatically add outputs as datasets
- Provides helper dialogs for adding datasets from different sources

### `utils/dataset_manager/registry.py`

**Purpose**: Handles database operations for tracking datasets

**Key Class**:
- `DatasetRegistry`: Manages the dataset catalog and metadata storage

**Key Functions**:
- `_initialize_database()`: Creates or connects to the SQLite database
- `add_dataset()`: Adds a new dataset to the registry
- `update_dataset()`: Updates an existing dataset
- `get_dataset()`: Retrieves a dataset by ID
- `get_datasets()`: Gets datasets matching criteria
- `delete_dataset()`: Deletes a dataset from the registry
- `add_tag()`, `remove_tag()`, `get_tags()`: Tag management functions
- `record_processing_step()`: Records operations in the dataset history

**Database Structure**:
- `datasets` table: Stores dataset metadata
- `tags` table: Stores available tags
- `dataset_tags` table: Maps tags to datasets

### `utils/dataset_manager/explorer.py`

**Purpose**: UI component for browsing and managing datasets

**Key Class**:
- `DatasetExplorer`: UI component for exploring datasets

**Key Functions**:
- `_create_layout()`: Creates the main UI layout
- `_create_toolbar()`: Creates the toolbar with actions
- `_create_tree_view()`: Creates the tree view for datasets
- `_create_details_panel()`: Creates the panel for dataset details
- `refresh_datasets()`: Refreshes the dataset list
- `_on_dataset_select()`: Handles dataset selection events
- `_update_details()`: Updates the details panel with dataset info
- `_load_preview()`: Loads dataset preview content
- `_load_thumbnails()`: Displays thumbnail images from the dataset
- `_add_dataset()`: Shows dialog to add a new dataset
- `_edit_dataset()`: Shows dialog to edit a dataset
- `_delete_dataset()`: Deletes a dataset

**UI Components**:
- Tree view for dataset browsing
- Details panel with tabs for info, previews, and statistics
- Thumbnail grid for dataset content preview
- Forms for adding and editing datasets

### `utils/dataset_manager/operations.py`

**Purpose**: Implements dataset operations like splitting and merging

**Key Class**:
- `DatasetOperations`: Handles operations on datasets

**Key Functions**:
- `split_dataset()`: Splits a dataset into train/validation/test sets
- `_split_random()`: Performs a random split
- `_split_stratified()`: Performs a stratified split
- `merge_datasets()`: Merges multiple datasets into one
- `filter_dataset()`: Filters a dataset based on criteria
- `_apply_filters()`: Applies filter criteria to a file
- `export_dataset()`: Exports a dataset to a standard format
- `_export_to_csv()`, `_export_to_coco()`, `_export_to_yolo()`, `_export_to_voc()`: Format-specific export functions

**Implementation Details**:
- File operations preserve directory structure
- Each operation creates new datasets in the registry
- Operations record their actions in dataset history
- Supports multiple export formats for different ML frameworks

### `utils/dataset_manager/analyzer.py`

**Purpose**: Analyzes datasets to provide insights and quality assessment

**Key Class**:
- `DatasetAnalyzer`: Analyzes datasets for statistics and issues

**Key Functions**:
- `analyze_dataset()`: Performs comprehensive analysis of a dataset
- `_analyze_files()`: Analyzes file statistics
- `_analyze_images()`: Analyzes image properties
- `_analyze_folders()`: Analyzes folder structure
- `_check_quality_issues()`: Checks for quality issues
- `_find_duplicates()`: Finds duplicate or similar images

**Analysis Features**:
- File statistics (counts, sizes, formats)
- Image statistics (dimensions, aspect ratios, color modes)
- Quality issues detection (corrupt, dark, small images)
- Duplicate image detection using perceptual hashing

### `utils/dataset_manager/integration.py`

**Purpose**: Integrates the dataset manager with the main application

**Key Functions**:
- `add_dataset_manager_to_app()`: Adds dataset manager to the application
- `initialize_dataset_manager()`: Initializes the dataset manager
- `_add_integration_hooks()`: Adds hooks to existing functionality
- Various integration helper functions for different parts of the app

**Implementation Details**:
- Adds menu items and buttons
- Connects with input/output tab
- Adds hooks to processing pipeline
- Provides help documentation

### `ui/tabs/dataset_manager_tab.py`

**Purpose**: Main UI tab for the dataset manager

**Key Class**:
- `DatasetManagerTab`: Dataset Manager tab for the main application

**Key Functions**:
- `_create_ui()`: Creates the UI elements for the tab
- `_create_operations_ui()`: Creates the UI for dataset operations
- UI creation functions for each operation type:
  - `_create_split_ui()`: UI for dataset splitting
  - `_create_merge_ui()`: UI for dataset merging
  - `_create_filter_ui()`: UI for dataset filtering
  - `_create_export_ui()`: UI for dataset exporting
  - `_create_analysis_ui()`: UI for dataset analysis
- Operation execution functions:
  - `_split_dataset()`: Executes dataset split
  - `_merge_datasets()`: Executes dataset merge
  - `_filter_dataset()`: Executes dataset filtering
  - `_export_dataset()`: Executes dataset export
  - `_analyze_dataset()`: Executes dataset analysis
- Result display functions:
  - `_display_file_stats()`: Displays file statistics
  - `_display_image_stats()`: Displays image statistics
  - `_display_quality_issues()`: Displays quality issues
  - `_display_duplicates()`: Displays duplicate information

**UI Structure**:
- PanedWindow with explorer on left, operations on right
- Notebook with tabs for different operations
- Forms for configuring operations
- Result displays for analysis and operations

## Component Integration

The files work together in the following way:

1. `main.py` launches the application
2. `app.py` initializes the main window and UI
3. `utils/dataset_manager.py` adds the dataset manager to the app
   - Creates the `DatasetRegistry` from `registry.py`
   - Initializes the `DatasetManagerTab` from `dataset_manager_tab.py`
   - Sets up integration hooks with existing functionality
4. The `DatasetManagerTab` creates:
   - `DatasetExplorer` from `explorer.py` for browsing datasets
   - `DatasetOperations` from `operations.py` for dataset operations
   - `DatasetAnalyzer` from `analyzer.py` for dataset analysis
5. User interactions with the UI trigger functions in these components
6. Operations record their results in the registry and create new datasets

## Key Workflows

### Dataset Registration Workflow
1. User clicks "Add Dataset" in the explorer
2. `_add_dataset()` in `explorer.py` shows the dialog
3. User inputs dataset information
4. `_create_dataset()` calls `registry.add_dataset()`
5. `registry.py` adds the dataset to the database
6. `refresh_datasets()` updates the UI

### Dataset Split Workflow
1. User selects a dataset and configures split options
2. `_split_dataset()` in `dataset_manager_tab.py` is called
3. It calls `operations.split_dataset()` in a separate thread
4. `operations.py` performs the split and creates new datasets
5. Results are registered in the database
6. UI is updated to show the new datasets

### Analysis Workflow
1. User selects a dataset and analysis options
2. `_analyze_dataset()` in `dataset_manager_tab.py` is called
3. It calls `analyzer.analyze_dataset()` in a separate thread
4. `analyzer.py` analyzes the dataset and returns results
5. Results are displayed in the UI
6. Analysis record is added to the dataset history
