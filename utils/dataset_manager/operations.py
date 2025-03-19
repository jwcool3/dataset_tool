"""
Dataset Operations for Dataset Preparation Tool
Handles operations like splitting, merging, and filtering datasets.
"""

import os
import shutil
import json
import random
import datetime
import random

class DatasetOperations:
    """Handles operations that can be performed on datasets."""
    
    def __init__(self, registry):
        """
        Initialize dataset operations.
        
        Args:
            registry: DatasetRegistry instance
        """
        self.registry = registry
    
    def split_dataset(self, dataset_id, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                     random_seed=42, stratify_by=None):
        """
        Split a dataset into training, validation, and test sets.
        
        Args:
            dataset_id: ID of the dataset to split
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_seed: Random seed for reproducibility
            stratify_by: Optional folder name to stratify by (assumes class folders)
            
        Returns:
            tuple: (train_dataset_id, val_dataset_id, test_dataset_id) or None if failed
        """
        # Verify ratios sum to approximately 1
        if not 0.99 <= train_ratio + val_ratio + test_ratio <= 1.01:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Get dataset info
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            return None
            
        source_path = dataset['path']
        if not os.path.isdir(source_path):
            return None
            
        # Create output directories
        dataset_name = dataset['name']
        parent_dir = os.path.dirname(source_path)
        
        train_dir = os.path.join(parent_dir, f"{dataset_name}_train")
        val_dir = os.path.join(parent_dir, f"{dataset_name}_val")
        test_dir = os.path.join(parent_dir, f"{dataset_name}_test")
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Configure random number generator
        random.seed(random_seed)
        
        # Get list of files or folders (depending on stratification)
        if stratify_by:
            # Stratified split by folder
            return self._split_stratified(
                dataset_id, source_path, train_dir, val_dir, test_dir,
                train_ratio, val_ratio, test_ratio
            )
        else:
            # Simple random split
            return self._split_random(
                dataset_id, source_path, train_dir, val_dir, test_dir,
                train_ratio, val_ratio, test_ratio
            )
    
    def _split_random(self, dataset_id, source_path, train_dir, val_dir, test_dir,
                     train_ratio, val_ratio, test_ratio):
        """Perform a random split of the dataset."""
        # Get all image files
        image_files = []
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    src_path = os.path.join(root, file)
                    # Create relative path from source_path
                    rel_path = os.path.relpath(src_path, source_path)
                    image_files.append(rel_path)
        
        if not image_files:
            return None
            
        # Shuffle files
        random.shuffle(image_files)
        
        # Split files according to ratios
        n_files = len(image_files)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train+n_val]
        test_files = image_files[n_train+n_val:]
        
        # Copy files to respective directories
        self._copy_files(source_path, train_dir, train_files)
        self._copy_files(source_path, val_dir, val_files)
        self._copy_files(source_path, test_dir, test_files)
        
        # Register new datasets
        dataset = self.registry.get_dataset(dataset_id)
        train_id = self.registry.add_dataset(
            name=f"{dataset['name']}_train",
            path=train_dir,
            description=f"Training split of {dataset['name']}",
            category="Training",
            parent_id=dataset_id,
            attributes={"split_type": "train", "split_ratio": train_ratio}
        )
        
        val_id = self.registry.add_dataset(
            name=f"{dataset['name']}_val",
            path=val_dir,
            description=f"Validation split of {dataset['name']}",
            category="Validation",
            parent_id=dataset_id,
            attributes={"split_type": "val", "split_ratio": val_ratio}
        )
        
        test_id = self.registry.add_dataset(
            name=f"{dataset['name']}_test",
            path=test_dir,
            description=f"Test split of {dataset['name']}",
            category="Testing",
            parent_id=dataset_id,
            attributes={"split_type": "test", "split_ratio": test_ratio}
        )
        
        # Add processing record to parent dataset
        self.registry.record_processing_step(
            dataset_id,
            "dataset_split",
            {
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "stratify": False,
                "train_count": len(train_files),
                "val_count": len(val_files),
                "test_count": len(test_files)
            }
        )
        
        return (train_id, val_id, test_id)
    
    def _split_stratified(self, dataset_id, source_path, train_dir, val_dir, test_dir,
                        train_ratio, val_ratio, test_ratio):
        """Perform a stratified split preserving class distribution."""
        # Get all subfolders (assuming they are class folders)
        class_folders = []
        for item in os.listdir(source_path):
            item_path = os.path.join(source_path, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                class_folders.append(item)
        
        if not class_folders:
            # No class folders found, fall back to random split
            return self._split_random(
                dataset_id, source_path, train_dir, val_dir, test_dir,
                train_ratio, val_ratio, test_ratio
            )
        
        # Process each class folder and maintain the same split ratio
        for class_folder in class_folders:
            # Create output class folders
            train_class_dir = os.path.join(train_dir, class_folder)
            val_class_dir = os.path.join(val_dir, class_folder)
            test_class_dir = os.path.join(test_dir, class_folder)
            
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)
            
            # Get all images in this class
            class_path = os.path.join(source_path, class_folder)
            class_files = []
            
            for file in os.listdir(class_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    class_files.append(file)
            
            # Shuffle files
            random.shuffle(class_files)
            
            # Split files according to ratios
            n_files = len(class_files)
            n_train = int(n_files * train_ratio)
            n_val = int(n_files * val_ratio)
            
            train_files = class_files[:n_train]
            val_files = class_files[n_train:n_train+n_val]
            test_files = class_files[n_train+n_val:]
            
            # Copy files to respective directories
            for file in train_files:
                src = os.path.join(class_path, file)
                dst = os.path.join(train_class_dir, file)
                shutil.copy2(src, dst)
            
            for file in val_files:
                src = os.path.join(class_path, file)
                dst = os.path.join(val_class_dir, file)
                shutil.copy2(src, dst)
            
            for file in test_files:
                src = os.path.join(class_path, file)
                dst = os.path.join(test_class_dir, file)
                shutil.copy2(src, dst)
        
        # Register new datasets
        dataset = self.registry.get_dataset(dataset_id)
        train_id = self.registry.add_dataset(
            name=f"{dataset['name']}_train",
            path=train_dir,
            description=f"Training split of {dataset['name']} (stratified)",
            category="Training",
            parent_id=dataset_id,
            attributes={"split_type": "train", "split_ratio": train_ratio, "stratified": True}
        )
        
        val_id = self.registry.add_dataset(
            name=f"{dataset['name']}_val",
            path=val_dir,
            description=f"Validation split of {dataset['name']} (stratified)",
            category="Validation",
            parent_id=dataset_id,
            attributes={"split_type": "val", "split_ratio": val_ratio, "stratified": True}
        )
        
        test_id = self.registry.add_dataset(
            name=f"{dataset['name']}_test",
            path=test_dir,
            description=f"Test split of {dataset['name']} (stratified)",
            category="Testing",
            parent_id=dataset_id,
            attributes={"split_type": "test", "split_ratio": test_ratio, "stratified": True}
        )
        
        # Add processing record to parent dataset
        self.registry.record_processing_step(
            dataset_id,
            "dataset_split",
            {
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "stratify": True,
                "class_count": len(class_folders)
            }
        )
        
        return (train_id, val_id, test_id)
    
    def _copy_files(self, source_base, dest_base, file_list):
        """Copy files from source to destination, preserving directory structure."""
        for rel_path in file_list:
            source_path = os.path.join(source_base, rel_path)
            dest_path = os.path.join(dest_base, rel_path)
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Copy the file
            shutil.copy2(source_path, dest_path)
    
    def merge_datasets(self, dataset_ids, merged_name=None, merge_method="copy"):
        """
        Merge multiple datasets into a new dataset.
        
        Args:
            dataset_ids: List of dataset IDs to merge
            merged_name: Name for the merged dataset (default: auto-generated)
            merge_method: Method to use ("copy" or "link")
            
        Returns:
            int: ID of the merged dataset or None if failed
        """
        if not dataset_ids or len(dataset_ids) < 2:
            return None
            
        # Get all datasets
        datasets = []
        for dataset_id in dataset_ids:
            dataset = self.registry.get_dataset(dataset_id)
            if dataset and os.path.isdir(dataset['path']):
                datasets.append(dataset)
        
        if len(datasets) < 2:
            return None
            
        # Create merged name if not provided
        if not merged_name:
            merged_name = f"Merged_{len(datasets)}_datasets"
            
        # Create output directory
        parent_dir = os.path.dirname(datasets[0]['path'])
        merged_dir = os.path.join(parent_dir, merged_name)
        
        if os.path.exists(merged_dir):
            # Add timestamp to make unique
            import time
            timestamp = int(time.time())
            merged_dir = f"{merged_dir}_{timestamp}"
            
        os.makedirs(merged_dir, exist_ok=True)
        
        # Copy or link files from each dataset
        for dataset in datasets:
            source_path = dataset['path']
            
            for root, dirs, files in os.walk(source_path):
                # Create relative path from source
                rel_path = os.path.relpath(root, source_path)
                
                # Create corresponding directory in merged_dir
                if rel_path != '.':
                    dest_dir = os.path.join(merged_dir, rel_path)
                    os.makedirs(dest_dir, exist_ok=True)
                else:
                    dest_dir = merged_dir
                
                # Process each file
                for file in files:
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(dest_dir, file)
                    
                    if merge_method == "copy":
                        shutil.copy2(src_file, dst_file)
                    elif merge_method == "link":
                        # Create a symbolic link instead of copying
                        if os.path.exists(dst_file):
                            os.remove(dst_file)
                        os.symlink(os.path.abspath(src_file), dst_file)
                    else:
                        # Invalid method
                        shutil.rmtree(merged_dir)
                        return None
        
        # Create the merged dataset
        merged_id = self.registry.add_dataset(
            name=merged_name,
            path=merged_dir,
            description=f"Merged dataset from {len(datasets)} source datasets",
            category="Merged",
            attributes={"source_datasets": [d['id'] for d in datasets]}
        )
        
        # Record the merge operation in each source dataset
        for dataset in datasets:
            self.registry.record_processing_step(
                dataset['id'],
                "dataset_merge",
                {"merged_dataset_id": merged_id, "merge_method": merge_method}
            )
        
        return merged_id
    
    def filter_dataset(self, dataset_id, filter_criteria, output_name=None):
        """
        Create a new dataset by filtering an existing one.
        
        Args:
            dataset_id: Source dataset ID
            filter_criteria: Dict of filter criteria (e.g., {"min_width": 800, "extension": ".jpg"})
            output_name: Name for the filtered dataset
            
        Returns:
            int: ID of the filtered dataset or None if failed
        """
        # Get source dataset
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset or not os.path.isdir(dataset['path']):
            return None
            
        # Create output name if not provided
        if not output_name:
            output_name = f"{dataset['name']}_filtered"
            
        # Create output directory
        parent_dir = os.path.dirname(dataset['path'])
        output_dir = os.path.join(parent_dir, output_name)
        
        if os.path.exists(output_dir):
            # Add timestamp to make unique
            import time
            timestamp = int(time.time())
            output_dir = f"{output_dir}_{timestamp}"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Apply filters
        source_path = dataset['path']
        filtered_files = []
        
        for root, _, files in os.walk(source_path):
            for file in files:
                src_file = os.path.join(root, file)
                
                # Apply filters
                if self._apply_filters(src_file, filter_criteria):
                    # File passes all filters
                    rel_path = os.path.relpath(root, source_path)
                    if rel_path != '.':
                        dest_dir = os.path.join(output_dir, rel_path)
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_file = os.path.join(dest_dir, file)
                    else:
                        dest_file = os.path.join(output_dir, file)
                    
                    # Copy the file
                    shutil.copy2(src_file, dest_file)
                    filtered_files.append(dest_file)
        
        if not filtered_files:
            # No files passed the filters
            shutil.rmtree(output_dir)
            return None
        
        # Create the filtered dataset
        filtered_id = self.registry.add_dataset(
            name=output_name,
            path=output_dir,
            description=f"Filtered version of {dataset['name']}",
            category=dataset['category'],
            parent_id=dataset_id,
            attributes={"filter_criteria": filter_criteria}
        )
        
        # Record the filter operation
        self.registry.record_processing_step(
            dataset_id,
            "dataset_filter",
            {
                "filter_criteria": filter_criteria,
                "filtered_dataset_id": filtered_id,
                "filtered_file_count": len(filtered_files)
            }
        )
        
        return filtered_id
    
    def _apply_filters(self, file_path, criteria):
        """Apply filter criteria to a file."""
        # Check file extension
        if 'extension' in criteria:
            ext = os.path.splitext(file_path)[1].lower()
            if criteria['extension'] == "all":
                # All extensions are allowed, so continue with other checks
                pass
            elif isinstance(criteria['extension'], list):
                if ext not in [e.lower() for e in criteria['extension']]:
                    return False
            elif ext != criteria['extension'].lower():
                return False
        
        # Check file size
        if 'min_size' in criteria or 'max_size' in criteria:
            file_size = os.path.getsize(file_path)
            
            if 'min_size' in criteria and file_size < criteria['min_size']:
                return False
                
            if 'max_size' in criteria and file_size > criteria['max_size']:
                return False
        
        # Check image dimensions
        if any(k in criteria for k in ['min_width', 'min_height', 'max_width', 'max_height']):
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    width, height = img.size
                    
                    if 'min_width' in criteria and width < criteria['min_width']:
                        return False
                        
                    if 'min_height' in criteria and height < criteria['min_height']:
                        return False
                        
                    if 'max_width' in criteria and criteria['max_width'] > 0 and width > criteria['max_width']:
                        return False
                        
                    if 'max_height' in criteria and criteria['max_height'] > 0 and height > criteria['max_height']:
                        return False
            except Exception:
                # Not an image or error opening it
                return False
        
        # Check aspect ratio
        if 'min_aspect_ratio' in criteria or 'max_aspect_ratio' in criteria:
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    width, height = img.size
                    
                    # Avoid division by zero
                    if height == 0:
                        return False
                        
                    aspect_ratio = width / height
                    
                    if 'min_aspect_ratio' in criteria and criteria['min_aspect_ratio'] > 0 and aspect_ratio < criteria['min_aspect_ratio']:
                        return False
                        
                    if 'max_aspect_ratio' in criteria and criteria['max_aspect_ratio'] > 0 and aspect_ratio > criteria['max_aspect_ratio']:
                        return False
            except Exception:
                return False
        
        # All filters passed
        return True
    
    def export_dataset(self, dataset_id, format_type, output_path=None):
        """
        Export a dataset to a specified format.
        
        Args:
            dataset_id: Source dataset ID
            format_type: Format type ("coco", "yolo", "voc", "csv")
            output_path: Output path (default: auto-generated)
            
        Returns:
            str: Path to the exported dataset or None if failed
        """
        # Get source dataset
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset or not os.path.isdir(dataset['path']):
            return None
            
        # Create output path if not provided
        if not output_path:
            parent_dir = os.path.dirname(dataset['path'])
            output_path = os.path.join(parent_dir, f"{dataset['name']}_{format_type}")
            
        os.makedirs(output_path, exist_ok=True)
        
        # Export based on format type
        try:
            if format_type.lower() == "csv":
                self._export_to_csv(dataset, output_path)
            elif format_type.lower() == "coco":
                self._export_to_coco(dataset, output_path)
            elif format_type.lower() == "yolo":
                self._export_to_yolo(dataset, output_path)
            elif format_type.lower() == "voc":
                self._export_to_voc(dataset, output_path)
            else:
                # Unsupported format
                return None
                
            # Record the export operation
            self.registry.record_processing_step(
                dataset_id,
                "dataset_export",
                {"format": format_type, "output_path": output_path}
            )
            
            return output_path
            
        except Exception as e:
            print(f"Export error: {str(e)}")
            return None
    
    def _export_to_csv(self, dataset, output_path):
        """Export dataset to CSV format."""
        import csv
        
        # Create CSV file
        csv_path = os.path.join(output_path, "dataset_inventory.csv")
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['file_path', 'width', 'height', 'format', 'size_bytes', 'folder'])
            
            # Process all files
            source_path = dataset['path']
            for root, _, files in os.walk(source_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, source_path)
                        folder = os.path.relpath(root, source_path)
                        
                        try:
                            # Get file info
                            size_bytes = os.path.getsize(file_path)
                            
                            # Get image dimensions
                            from PIL import Image
                            with Image.open(file_path) as img:
                                width, height = img.size
                                format_name = img.format
                                
                            # Write to CSV
                            writer.writerow([rel_path, width, height, format_name, size_bytes, folder])
                            
                        except Exception as e:
                            print(f"Error processing {file_path}: {str(e)}")
    
    def _export_to_coco(self, dataset, output_path):
        """Export dataset to COCO format (simplified, without annotations)."""
        import json
        
        # Create COCO dataset structure
        coco_data = {
            "info": {
                "description": dataset['description'] or "Exported dataset",
                "url": "",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "contributor": "Dataset Manager",
                "date_created": datetime.datetime.now().isoformat()
            },
            "licenses": [
                {
                    "url": "https://creativecommons.org/licenses/by/4.0/",
                    "id": 1,
                    "name": "Attribution 4.0 International (CC BY 4.0)"
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Process all image files
        source_path = dataset['path']
        image_id = 0
        
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, source_path)
                    
                    try:
                        # Get image dimensions
                        from PIL import Image
                        with Image.open(file_path) as img:
                            width, height = img.size
                        
                        # Add image to COCO format
                        coco_data["images"].append({
                            "id": image_id,
                            "width": width,
                            "height": height,
                            "file_name": rel_path,
                            "license": 1,
                            "date_captured": datetime.datetime.fromtimestamp(
                                os.path.getmtime(file_path)
                            ).isoformat()
                        })
                        
                        # Copy image to output directory
                        target_dir = os.path.join(output_path, os.path.dirname(rel_path))
                        os.makedirs(target_dir, exist_ok=True)
                        shutil.copy2(file_path, os.path.join(output_path, rel_path))
                        
                        image_id += 1
                        
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
        
        # Write COCO JSON file
        with open(os.path.join(output_path, "annotations.json"), 'w') as f:
            json.dump(coco_data, f, indent=2)
    
    def _export_to_yolo(self, dataset, output_path):
        """Export dataset to YOLO format (simplified, without annotations)."""
        # Create directory structure
        images_dir = os.path.join(output_path, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Create empty labels directory
        labels_dir = os.path.join(output_path, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        
        # Create dataset.yaml
        yaml_content = f"""
# YOLO dataset configuration
path: {output_path}
train: images/train
val: images/val
test: images/test

# Classes
nc: 0  # number of classes
names: []  # class names
"""
        
        with open(os.path.join(output_path, "dataset.yaml"), 'w') as f:
            f.write(yaml_content)
        
        # Create train/val/test splits
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(images_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
        
        # Process all image files (put all in train for now)
        source_path = dataset['path']
        
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    file_path = os.path.join(root, file)
                    
                    # Copy to train directory
                    shutil.copy2(file_path, os.path.join(images_dir, "train", file))
    
    def _export_to_voc(self, dataset, output_path):
        """Export dataset to Pascal VOC format (simplified, without annotations)."""
        # Create directory structure
        images_dir = os.path.join(output_path, "JPEGImages")
        os.makedirs(images_dir, exist_ok=True)
        
        annotations_dir = os.path.join(output_path, "Annotations")
        os.makedirs(annotations_dir, exist_ok=True)
        
        imagesets_dir = os.path.join(output_path, "ImageSets", "Main")
        os.makedirs(imagesets_dir, exist_ok=True)
        
        # Process all image files
        source_path = dataset['path']
        image_ids = []
        
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    file_path = os.path.join(root, file)
                    
                    # Generate ID from filename (without extension)
                    image_id = os.path.splitext(file)[0]
                    image_ids.append(image_id)
                    
                    # Copy image to JPEGImages directory
                    target_path = os.path.join(images_dir, f"{image_id}.jpg")
                    
                    # Convert to JPG if needed
                    if file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg'):
                        shutil.copy2(file_path, target_path)
                    else:
                        try:
                            from PIL import Image
                            img = Image.open(file_path)
                            rgb_img = img.convert('RGB')
                            rgb_img.save(target_path)
                        except Exception as e:
                            print(f"Error converting {file_path}: {str(e)}")
                            continue
                    
                    # Create empty XML annotation
                    xml_path = os.path.join(annotations_dir, f"{image_id}.xml")
                    
                    # Get image dimensions
                    try:
                        from PIL import Image
                        with Image.open(file_path) as img:
                            width, height = img.size
                    except:
                        width, height = 0, 0
                    
                    # Create basic XML structure
                    xml_content = f"""
    <annotation>
        <folder>JPEGImages</folder>
        <filename>{image_id}.jpg</filename>
        <path>{target_path}</path>
        <source>
            <database>Unknown</database>
        </source>
        <size>
            <width>{width}</width>
            <height>{height}</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
    </annotation>
    """
                    with open(xml_path, 'w') as f:
                        f.write(xml_content)
        
        # Create ImageSets files
        import random
        random.shuffle(image_ids)
        
        # Create train/val/test splits (70/15/15 split)
        train_size = int(len(image_ids) * 0.7)
        val_size = int(len(image_ids) * 0.15)
        
        train_ids = image_ids[:train_size]
        val_ids = image_ids[train_size:train_size+val_size]
        test_ids = image_ids[train_size+val_size:]
        
        # Write split files
        with open(os.path.join(imagesets_dir, "train.txt"), 'w') as f:
            f.write("\n".join(train_ids))
            
        with open(os.path.join(imagesets_dir, "val.txt"), 'w') as f:
            f.write("\n".join(val_ids))
            
        with open(os.path.join(imagesets_dir, "test.txt"), 'w') as f:
            f.write("\n".join(test_ids))
            
        with open(os.path.join(imagesets_dir, "trainval.txt"), 'w') as f:
            f.write("\n".join(train_ids + val_ids))