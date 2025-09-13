# src/utils.py
"""
Utility functions for Clinical Prediction System
Contains helper functions, logging setup, and data generation utilities
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .config import LOG_CONFIG, PROJECT_ROOT, DATA_DIR


def setup_logging(log_level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    """
    Setup logging configuration for the application
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional log file path
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_file.parent.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=LOG_CONFIG['format'],
        handlers=[
            logging.StreamHandler(),  # Console handler
            logging.FileHandler(log_file or LOG_CONFIG['file'], encoding='utf-8')
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def create_sample_data(output_path: Optional[Path] = None, n_samples: int = 20) -> Path:
    """
    Create sample clinical data for testing and demonstration
    
    Args:
        output_path: Path to save sample data
        n_samples: Number of sample records to generate
        
    Returns:
        Path to created sample data file
    """
    if output_path is None:
        output_path = DATA_DIR / "raw" / "sample_clinical_data.json"
    
    # Sample clinical scenarios
    clinical_scenarios = [
        {
            "condition": "disc herniation L4-L5",
            "symptoms": "lower back pain with right leg radiation",
            "treatments": ["PEN", "MED", "f/u"],
            "radiology": ["MRI", "X-RAY"]
        },
        {
            "condition": "spinal stenosis L3-L4-L5",
            "symptoms": "neurogenic claudication and bilateral leg weakness",
            "treatments": ["PELD", "ESI", "MED"],
            "radiology": ["MRI", "CT", "MYELO"]
        },
        {
            "condition": "compression fracture L1",
            "symptoms": "acute back pain after fall",
            "treatments": ["PVP", "MED", "f/u"],
            "radiology": ["X-RAY", "CT", "MRI"]
        },
        {
            "condition": "degenerative disc disease L5-S1",
            "symptoms": "chronic lower back pain",
            "treatments": ["TPI", "MED", "채진"],
            "radiology": ["MRI", "X-RAY"]
        },
        {
            "condition": "spondylolisthesis L4-L5",
            "symptoms": "back pain with standing and walking",
            "treatments": ["LIF", "lami", "f/u"],
            "radiology": ["X-RAY", "MRI", "CT"]
        }
    ]
    
    sample_data = []
    
    for i in range(n_samples):
        # Select random scenario
        scenario = clinical_scenarios[i % len(clinical_scenarios)]
        
        # Create patient record
        record = {
            "patient_id": f"P{i+1:03d}",
            "Study": {},
            "clinical_notes": f"Patient presents with {scenario['condition']}. "
                            f"Symptoms include {scenario['symptoms']}. "
                            f"Clinical examination and imaging studies performed."
        }
        
        # Add radiology findings
        for rad_type in ["X-RAY", "CT", "MRI", "MYELO"]:
            if rad_type in scenario['radiology']:
                if rad_type == "X-RAY":
                    record["Study"][rad_type] = f"Lumbar spine shows {scenario['condition']}"
                elif rad_type == "MRI":
                    record["Study"][rad_type] = f"MRI lumbar spine demonstrates {scenario['condition']} with associated findings"
                elif rad_type == "CT":
                    record["Study"][rad_type] = f"CT lumbar spine reveals {scenario['condition']}"
                elif rad_type == "MYELO":
                    record["Study"][rad_type] = f"Myelogram shows {scenario['condition']} with compression"
            else:
                record["Study"][rad_type] = ""
        
        # Add treatment information to clinical notes
        treatments_str = ", ".join(scenario['treatments'])
        record["clinical_notes"] += f" Recommended treatments: {treatments_str}."
        
        sample_data.append(record)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save sample data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Created {n_samples} sample records at {output_path}")
    return output_path


def validate_data_structure(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate the structure of clinical data
    
    Args:
        data: List of clinical records
        
    Returns:
        Validation report dictionary
    """
    report = {
        'total_records': len(data),
        'valid_records': 0,
        'missing_study': 0,
        'missing_clinical_notes': 0,
        'empty_records': 0,
        'radiology_coverage': {rad_type: 0 for rad_type in ['X-RAY', 'CT', 'MRI', 'MYELO']},
        'issues': []
    }
    
    for i, record in enumerate(data):
        is_valid = True
        
        # Check if record is empty
        if not record:
            report['empty_records'] += 1
            report['issues'].append(f"Record {i}: Empty record")
            continue
        
        # Check for Study section
        if 'Study' not in record:
            report['missing_study'] += 1
            report['issues'].append(f"Record {i}: Missing 'Study' section")
            is_valid = False
        else:
            # Check radiology coverage
            for rad_type in report['radiology_coverage']:
                if (rad_type in record['Study'] and 
                    record['Study'][rad_type] and 
                    str(record['Study'][rad_type]).strip()):
                    report['radiology_coverage'][rad_type] += 1
        
        # Check for clinical notes
        has_notes = any(
            field in record and record[field] and str(record[field]).strip()
            for field in ['clinical_notes', 'notes', 'description']
        )
        if not has_notes:
            report['missing_clinical_notes'] += 1
            report['issues'].append(f"Record {i}: Missing clinical notes")
            is_valid = False
        
        if is_valid:
            report['valid_records'] += 1
    
    return report


def print_data_summary(data: List[Dict[str, Any]], validation_report: Optional[Dict] = None) -> None:
    """
    Print a summary of the clinical data
    
    Args:
        data: List of clinical records
        validation_report: Optional validation report
    """
    print("=== Clinical Data Summary ===")
    print(f"Total records: {len(data)}")
    
    if validation_report:
        print(f"Valid records: {validation_report['valid_records']}")
        print(f"Records with missing Study: {validation_report['missing_study']}")
        print(f"Records with missing clinical notes: {validation_report['missing_clinical_notes']}")
        
        print("\nRadiology examination coverage:")
        for rad_type, count in validation_report['radiology_coverage'].items():
            percentage = (count / len(data)) * 100 if data else 0
            print(f"  {rad_type}: {count} records ({percentage:.1f}%)")
        
        if validation_report['issues']:
            print(f"\nFound {len(validation_report['issues'])} issues:")
            for issue in validation_report['issues'][:5]:  # Show first 5 issues
                print(f"  - {issue}")
            if len(validation_report['issues']) > 5:
                print(f"  ... and {len(validation_report['issues']) - 5} more issues")


def format_prediction_results(interpretations: List[Dict[str, Any]], 
                            max_samples: int = 5) -> str:
    """
    Format prediction results for display
    
    Args:
        interpretations: List of prediction interpretations
        max_samples: Maximum number of samples to display
        
    Returns:
        Formatted string with prediction results
    """
    output = ["=== Prediction Results ===\n"]
    
    for i, result in enumerate(interpretations[:max_samples]):
        output.append(f"Sample {i + 1}:")
        output.append(f"  Text: {result['text_snippet']}")
        output.append(f"  Predicted Treatments: {', '.join(result['predicted_treatments'])}")
        output.append(f"  Predicted Radiology: {', '.join(result['predicted_radiology'])}")
        output.append("")
    
    if len(interpretations) > max_samples:
        output.append(f"... and {len(interpretations) - max_samples} more samples\n")
    
    return "\n".join(output)


def calculate_label_statistics(labels: np.ndarray, label_names: List[str]) -> Dict[str, Any]:
    """
    Calculate statistics for multi-label data
    
    Args:
        labels: Binary label matrix
        label_names: Names of labels
        
    Returns:
        Dictionary with label statistics
    """
    stats = {
        'total_samples': len(labels),
        'total_labels': len(label_names),
        'label_frequencies': {},
        'label_percentages': {},
        'avg_labels_per_sample': np.mean(labels.sum(axis=1)),
        'samples_with_no_labels': np.sum(labels.sum(axis=1) == 0),
        'most_common_combinations': []
    }
    
    # Calculate per-label statistics
    for i, label_name in enumerate(label_names):
        freq = np.sum(labels[:, i])
        stats['label_frequencies'][label_name] = int(freq)
        stats['label_percentages'][label_name] = (freq / len(labels)) * 100
    
    return stats


def export_results_to_json(results: Dict[str, Any], 
                         output_path: Optional[Path] = None) -> Path:
    """
    Export evaluation results to JSON file
    
    Args:
        results: Results dictionary to export
        output_path: Optional output file path
        
    Returns:
        Path to exported file
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = PROJECT_ROOT / f"results_export_{timestamp}.json"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    # Convert and save results
    serializable_results = convert_numpy(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Exported results to {output_path}")
    return output_path