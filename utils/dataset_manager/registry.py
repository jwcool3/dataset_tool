"""
Dataset Registry for Dataset Preparation Tool
Handles database operations for tracking datasets.
"""

import os
import json
import datetime
import sqlite3
import shutil

class DatasetRegistry:
    """Manages the dataset catalog and metadata storage."""
    
    def __init__(self, app):
        """
        Initialize the dataset registry.
        
        Args:
            app: The main application
        """
        self.app = app
        self.db_path = os.path.join(app.config_dir, "dataset_registry.db")
        self._initialize_database()
        
    def _initialize_database(self):
        """Create or connect to the SQLite database and initialize tables."""
        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create datasets table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
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
        ''')
        
        # Create tags table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
        ''')
        
        # Create dataset_tags mapping table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS dataset_tags (
            dataset_id INTEGER,
            tag_id INTEGER,
            PRIMARY KEY(dataset_id, tag_id),
            FOREIGN KEY(dataset_id) REFERENCES datasets(id),
            FOREIGN KEY(tag_id) REFERENCES tags(id)
        )
        ''')
        
        # Commit changes and close
        conn.commit()
        conn.close()
    
    def add_dataset(self, name, path, description="", category="", parent_id=None, attributes=None):
        """
        Add a new dataset to the registry.
        
        Args:
            name: Dataset name
            path: Path to dataset directory
            description: Optional description
            category: Optional category (e.g., "training", "validation")
            parent_id: ID of parent dataset (if this is a derived dataset)
            attributes: Additional attributes as a dictionary
            
        Returns:
            int: ID of the newly created dataset
        """
        # Normalize the path
        path = os.path.abspath(path)
        
        # Calculate file counts
        file_count, image_count, video_count = self._count_files(path)
        
        # Prepare dates
        now = datetime.datetime.now().isoformat()
        
        # Convert attributes to JSON
        attrs_json = json.dumps(attributes or {})
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert the dataset
        cursor.execute('''
        INSERT INTO datasets 
        (name, path, description, created_date, modified_date, category, 
         file_count, image_count, video_count, parent_id, attributes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, path, description, now, now, category, 
              file_count, image_count, video_count, parent_id, attrs_json))
        
        # Get the ID of the newly inserted dataset
        dataset_id = cursor.lastrowid
        
        # Commit changes and close
        conn.commit()
        conn.close()
        
        return dataset_id
    
    def update_dataset(self, dataset_id, **kwargs):
        """
        Update an existing dataset.
        
        Args:
            dataset_id: ID of the dataset to update
            **kwargs: Fields to update (name, path, description, etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if dataset exists
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return False
        
        # Get current values to update only what's necessary
        updates = {}
        for key, value in kwargs.items():
            if key in dataset and dataset[key] != value:
                updates[key] = value
        
        if not updates:
            return True  # Nothing to update
        
        # Add modified_date
        updates['modified_date'] = datetime.datetime.now().isoformat()
        
        # Special handling for attributes
        if 'attributes' in updates:
            updates['attributes'] = json.dumps(updates['attributes'])
        
        # Generate SQL update statement
        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        values = list(updates.values())
        values.append(dataset_id)
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update the dataset
        try:
            cursor.execute(f'''
            UPDATE datasets
            SET {set_clause}
            WHERE id = ?
            ''', values)
            conn.commit()
            success = True
        except Exception as e:
            print(f"Error updating dataset: {str(e)}")
            conn.rollback()
            success = False
        
        conn.close()
        return success
    
    def get_dataset(self, dataset_id):
        """
        Get a dataset by ID.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            dict: Dataset information or None if not found
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        # Query the dataset
        cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        row = cursor.fetchone()
        
        # Close connection
        conn.close()
        
        if not row:
            return None
            
        # Convert to dict and parse attributes
        dataset = dict(row)
        if dataset.get('attributes'):
            try:
                dataset['attributes'] = json.loads(dataset['attributes'])
            except:
                dataset['attributes'] = {}
                
        return dataset
    
    def get_datasets(self, parent_id=None, category=None, search_term=None):
        """
        Get datasets matching the specified criteria.
        
        Args:
            parent_id: Filter by parent dataset ID
            category: Filter by category
            search_term: Search in name and description
            
        Returns:
            list: List of matching datasets
        """
        # Build query conditions
        conditions = []
        params = []
        
        if parent_id is not None:
            conditions.append("parent_id = ?")
            params.append(parent_id)
        
        if category:
            conditions.append("category = ?")
            params.append(category)
        
        if search_term:
            conditions.append("(name LIKE ? OR description LIKE ?)")
            params.extend([f"%{search_term}%", f"%{search_term}%"])
        
        # Construct query
        query = "SELECT * FROM datasets"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Execute query
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Close connection
        conn.close()
        
        # Convert rows to dicts and parse attributes
        datasets = []
        for row in rows:
            dataset = dict(row)
            if dataset.get('attributes'):
                try:
                    dataset['attributes'] = json.loads(dataset['attributes'])
                except:
                    dataset['attributes'] = {}
            datasets.append(dataset)
            
        return datasets
    
    def delete_dataset(self, dataset_id, delete_files=False):
        """
        Delete a dataset from the registry.
        
        Args:
            dataset_id: ID of the dataset to delete
            delete_files: Whether to also delete the dataset files
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get dataset info first
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return False
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Begin transaction
            conn.execute("BEGIN")
            
            # Delete tags associations
            cursor.execute("DELETE FROM dataset_tags WHERE dataset_id = ?", (dataset_id,))
            
            # Delete the dataset
            cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            
            # Commit transaction
            conn.commit()
            
            # Delete files if requested
            if delete_files and os.path.exists(dataset['path']):
                shutil.rmtree(dataset['path'])
                
            success = True
        except Exception as e:
            print(f"Error deleting dataset: {str(e)}")
            conn.rollback()
            success = False
            
        conn.close()
        return success
    
    def add_tag(self, dataset_id, tag_name):
        """
        Add a tag to a dataset.
        
        Args:
            dataset_id: Dataset ID
            tag_name: Tag name
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not tag_name.strip():
            return False
            
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Begin transaction
            conn.execute("BEGIN")
            
            # Get or create tag
            cursor.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
            row = cursor.fetchone()
            
            if row:
                tag_id = row[0]
            else:
                cursor.execute("INSERT INTO tags (name) VALUES (?)", (tag_name,))
                tag_id = cursor.lastrowid
            
            # Add association (ignore if already exists)
            cursor.execute("""
            INSERT OR IGNORE INTO dataset_tags (dataset_id, tag_id)
            VALUES (?, ?)
            """, (dataset_id, tag_id))
            
            # Commit transaction
            conn.commit()
            success = True
        except Exception as e:
            print(f"Error adding tag: {str(e)}")
            conn.rollback()
            success = False
            
        conn.close()
        return success
    
    def remove_tag(self, dataset_id, tag_name):
        """
        Remove a tag from a dataset.
        
        Args:
            dataset_id: Dataset ID
            tag_name: Tag name
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get tag ID
            cursor.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
            row = cursor.fetchone()
            if not row:
                return False
                
            tag_id = row[0]
            
            # Remove association
            cursor.execute("""
            DELETE FROM dataset_tags
            WHERE dataset_id = ? AND tag_id = ?
            """, (dataset_id, tag_id))
            
            conn.commit()
            success = True
        except Exception as e:
            print(f"Error removing tag: {str(e)}")
            conn.rollback()
            success = False
            
        conn.close()
        return success
    
    def get_tags(self, dataset_id):
        """
        Get all tags for a dataset.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            list: List of tag names
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Query tags
        cursor.execute("""
        SELECT t.name
        FROM tags t
        JOIN dataset_tags dt ON t.id = dt.tag_id
        WHERE dt.dataset_id = ?
        ORDER BY t.name
        """, (dataset_id,))
        
        tags = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return tags
    
    def get_all_tags(self):
        """
        Get all available tags.
        
        Returns:
            list: List of tag names
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Query all tags
        cursor.execute("SELECT name FROM tags ORDER BY name")
        tags = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return tags
    
    def _count_files(self, path):
        """
        Count files in a directory, categorizing by type.
        
        Args:
            path: Directory path
            
        Returns:
            tuple: (total_count, image_count, video_count)
        """
        if not os.path.isdir(path):
            return 0, 0, 0
            
        total_count = 0
        image_count = 0
        video_count = 0
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
        
        for root, _, files in os.walk(path):
            for file in files:
                total_count += 1
                ext = os.path.splitext(file)[1].lower()
                
                if ext in image_extensions:
                    image_count += 1
                elif ext in video_extensions:
                    video_count += 1
        
        return total_count, image_count, video_count
    
    def record_processing_step(self, dataset_id, operation, params=None):
        """
        Record a processing step in the dataset's history.
        
        Args:
            dataset_id: Dataset ID
            operation: Name of the operation
            params: Parameters used in the operation
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get current dataset
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return False
        
        # Create history entry
        history_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "operation": operation,
            "params": params or {}
        }
        
        # Parse existing history
        history = []
        if dataset.get('processing_history'):
            try:
                history = json.loads(dataset['processing_history'])
            except:
                history = []
        
        # Add new entry
        history.append(history_entry)
        
        # Update dataset
        return self.update_dataset(
            dataset_id, 
            processing_history=json.dumps(history)
        )