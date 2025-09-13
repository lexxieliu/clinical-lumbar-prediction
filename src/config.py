# src/config.py
"""
Configuration file for Clinical Prediction System
All system parameters and constants are defined here
"""

from pathlib import Path
import logging

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Treatment types configuration
TREATMENT_TYPES = {
    'PEN': ['PEN', 'PELAN'],
    'OLM': ['OLM'],
    '3DPC': ['3DPC'],
    'consultation': ['채진'],
    'follow_up': ['f/u', 'F/U'],
    'MED': ['MED'],
    'TPI': ['TPI'],
    'UBL': ['UBL'],
    'PELD': ['PELD'],
    'PVP': ['PVP'],
    'ESI': ['ESI'],
    'ILF': ['ILF', 'ILIF'],
    'LIF': ['LIF', 'LFLIF', 'LLIF', 'OLIF', 'TLIF', 'PLIF'],
    'RB': ['RB'],
    'laminectomy': ['lami', 'laminectomy'],
    'MRI': ['MR', 'MRI'],
    'caudal': ['caudal']
}

# Radiology types configuration
RADIOLOGY_TYPES = ['X-RAY', 'CT', 'MRI', 'MYELO']

# Feature engineering parameters
FEATURE_PARAMS = {
    'tfidf_max_features': 500,
    'tfidf_ngram_range': (1, 2),
    'tfidf_min_df': 2,
    'tfidf_max_df': 0.8,
    'feature_selection_k': 100
}

# Model parameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 50,
        'max_depth': 8,
        'min_samples_split': 5,
        'random_state': 42,
        'n_jobs': -1
    },
    'test_size': 0.2,
    'random_state': 42
}

# Logging configuration
LOG_CONFIG = {
    'level': logging.INFO,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOGS_DIR / 'clinical_prediction.log'
}

# File names
MODEL_FILES = {
    'treatment_model': MODELS_DIR / 'treatment_model.joblib',
    'radiology_model': MODELS_DIR / 'radiology_model.joblib',
    'feature_engineer': MODELS_DIR / 'feature_engineer.joblib',
    'vectorizer': MODELS_DIR / 'tfidf_vectorizer.joblib'
}

# Data validation parameters
DATA_VALIDATION = {
    'min_samples': 5,
    'required_fields': ['Study'],
    'text_fields': ['clinical_notes', 'notes', 'description']
}