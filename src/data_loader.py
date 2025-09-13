# src/data_loader.py
"""
Data loading and preprocessing module for Clinical Prediction System
Handles JSON data loading, text extraction, and label generation
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from .config import TREATMENT_TYPES, RADIOLOGY_TYPES, DATA_VALIDATION

logger = logging.getLogger(__name__)


class ClinicalDataLoader:
    """
    Handles loading and preprocessing of clinical data from JSON files
    """
    
    def __init__(self, data_path: str):
        """
        Initialize data loader
        
        Args:
            data_path: Path to data directory or JSON file
        """
        self.data_path = Path(data_path)
        self.treatment_types = TREATMENT_TYPES
        self.radiology_types = RADIOLOGY_TYPES
        self.validation_config = DATA_VALIDATION
    
    def load_json_data(self) -> List[Dict]:
        """
        Load JSON data from file(s)
        
        Returns:
            List of clinical data dictionaries
        """
        all_data = []
        
        if self.data_path.is_dir():
            # Load all JSON files from directory
            json_files = list(self.data_path.glob("*.json"))
            if not json_files:
                logger.error(f"No JSON files found in directory: {self.data_path}")
                return []
            
            for file_path in json_files:
                data = self._load_single_file(file_path)
                if data:
                    all_data.extend(data if isinstance(data, list) else [data])
            
            logger.info(f"Loaded {len(all_data)} records from {len(json_files)} files")
            
        elif self.data_path.is_file():
            # Load single file
            data = self._load_single_file(self.data_path)
            if data:
                all_data = data if isinstance(data, list) else [data]
            
        else:
            logger.error(f"Data path does not exist: {self.data_path}")
            return []
        
        # Validate loaded data
        return self._validate_data(all_data)
    
    def _load_single_file(self, file_path: Path) -> Optional[List[Dict]]:
        """
        Load data from a single JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded data or None if error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Successfully loaded file: {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            return None
    
    def _validate_data(self, data: List[Dict]) -> List[Dict]:
        """
        Validate loaded data structure
        
        Args:
            data: List of data records
            
        Returns:
            Validated data records
        """
        valid_data = []
        
        for i, record in enumerate(data):
            if self._is_valid_record(record):
                valid_data.append(record)
            else:
                logger.warning(f"Invalid record at index {i}, skipping")
        
        logger.info(f"Validated {len(valid_data)} out of {len(data)} records")
        return valid_data
    
    def _is_valid_record(self, record: Dict) -> bool:
        """
        Check if a record has the required structure
        
        Args:
            record: Single data record
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        for field in self.validation_config['required_fields']:
            if field not in record:
                return False
        
        return True
    
    def extract_features_and_labels(self, data: List[Dict]) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Extract text features and labels from clinical data
        
        Args:
            data: List of clinical data records
            
        Returns:
            Tuple of (texts, treatment_labels, radiology_labels)
        """
        texts = []
        treatment_labels = []
        radiology_labels = []
        
        for record in data:
            # Extract text features
            text = self._extract_text(record)
            texts.append(text)
            
            # Extract treatment labels
            treatment_label = self._extract_treatment_labels(record)
            treatment_labels.append(treatment_label)
            
            # Extract radiology labels
            radiology_label = self._extract_radiology_labels(record)
            radiology_labels.append(radiology_label)
        
        # Convert to numpy arrays
        treatment_labels = np.array(treatment_labels)
        radiology_labels = np.array(radiology_labels)
        
        logger.info(f"Extracted features from {len(texts)} records")
        logger.info(f"Treatment labels shape: {treatment_labels.shape}")
        logger.info(f"Radiology labels shape: {radiology_labels.shape}")
        
        return texts, treatment_labels, radiology_labels
    
    def _extract_text(self, record: Dict) -> str:
        """
        Extract all text information from a clinical record
        
        Args:
            record: Single clinical data record
            
        Returns:
            Combined text string
        """
        text_parts = []
        
        # Extract text from Study section
        if 'Study' in record:
            study = record['Study']
            for key, value in study.items():
                if isinstance(value, str) and value.strip():
                    text_parts.append(f"{key}: {value}")
        
        # Extract from other text fields
        for field in self.validation_config['text_fields']:
            if field in record and record[field]:
                text_parts.append(str(record[field]))
        
        # Return combined text or default
        if not text_parts:
            return "No clinical information available"
        
        return " ".join(text_parts)
    
    def _extract_treatment_labels(self, record: Dict) -> List[int]:
        """
        Extract treatment type labels (multi-label binary encoding)
        
        Args:
            record: Single clinical data record
            
        Returns:
            Binary label vector for treatments
        """
        labels = [0] * len(self.treatment_types)
        
        # Get all text content for label extraction
        text = self._extract_text(record).upper()
        
        # Match treatment types based on keywords
        for idx, (treatment, keywords) in enumerate(self.treatment_types.items()):
            for keyword in keywords:
                if keyword.upper() in text:
                    labels[idx] = 1
                    break
        
        # Set default treatment if none matched
        if sum(labels) == 0:
            # Set a default treatment (e.g., middle index)
            default_idx = len(self.treatment_types) // 2
            labels[default_idx] = 1
        
        return labels
    
    def _extract_radiology_labels(self, record: Dict) -> List[int]:
        """
        Extract radiology examination labels
        
        Args:
            record: Single clinical data record
            
        Returns:
            Binary label vector for radiology examinations
        """
        labels = [0] * len(self.radiology_types)
        
        if 'Study' in record:
            study = record['Study']
            for idx, rad_type in enumerate(self.radiology_types):
                # Check if radiology examination exists and has content
                if (rad_type in study and 
                    study[rad_type] and 
                    str(study[rad_type]).strip()):
                    labels[idx] = 1
        
        return labels
    
    def get_label_names(self) -> Tuple[List[str], List[str]]:
        """
        Get label names for treatments and radiology
        
        Returns:
            Tuple of (treatment_names, radiology_names)
        """
        treatment_names = list(self.treatment_types.keys())
        radiology_names = list(self.radiology_types)
        return treatment_names, radiology_names