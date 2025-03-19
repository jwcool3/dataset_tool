"""
Dataset Manager Module for Dataset Preparation Tool
Core components for managing multiple datasets and their versions.
"""

from utils.dataset_manager.registry import DatasetRegistry
from utils.dataset_manager.explorer import DatasetExplorer
from utils.dataset_manager.operations import DatasetOperations
from utils.dataset_manager.analyzer import DatasetAnalyzer
from utils.dataset_manager.integration import add_dataset_manager_to_app, initialize_dataset_manager

# Main tab component
from ui.tabs.dataset_manager_tab import DatasetManagerTab

__all__ = [
    'DatasetRegistry',
    'DatasetExplorer',
    'DatasetOperations',
    'DatasetAnalyzer',
    'DatasetManagerTab',
    'add_dataset_manager_to_app',
    'initialize_dataset_manager'
]