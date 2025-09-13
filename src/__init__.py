# src/__init__.py
"""
Clinical Prediction System Package
A machine learning system for predicting lumbar spine treatment types and radiology examinations
"""

__version__ = "1.0.0"
__author__ = "Clinical ML Team"
__description__ = "Lumbar Spine Treatment Prediction System"

# Import main classes and functions for easy access
from .clinical_prediction_system import ClinicalPredictionSystem, main, run_prediction_only, create_demo_data
from .data_loader import ClinicalDataLoader
from .feature_engineering import ClinicalFeatureEngineer
from .models import ClinicalPredictor
from .utils import setup_logging, create_sample_data, validate_data_structure

# Define what gets imported with "from src import *"
__all__ = [
    'ClinicalPredictionSystem',
    'ClinicalDataLoader', 
    'ClinicalFeatureEngineer',
    'ClinicalPredictor',
    'main',
    'run_prediction_only',
    'create_demo_data',
    'setup_logging',
    'create_sample_data',
    'validate_data_structure'
]

# Package metadata
PACKAGE_INFO = {
    'name': 'clinical-prediction-system',
    'version': __version__,
    'description': __description__,
    'author': __author__,
    'python_requires': '>=3.8',
    'keywords': ['machine learning', 'clinical prediction', 'lumbar spine', 'medical AI']
}