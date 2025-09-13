#!/usr/bin/env python3
# scripts/train_model.py
"""
Training script for Clinical Prediction System
This script handles model training with command line arguments
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ClinicalPredictionSystem, setup_logging
from src.utils import export_results_to_json


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Train Clinical Prediction System models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data-path', 
        type=str,
        default='data/raw',
        help='Path to training data directory or JSON file'
    )
    
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create sample data if no data exists'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data to use for testing'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save trained models'
    )
    
    parser.add_argument(
        '--export-results',
        type=str,
        help='Path to export training results (JSON format)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        help='Number of sample data records to create (if --create-sample is used)'
    )
    
    return parser.parse_args()


def main():
    """
    Main training function
    """
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    import logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(log_level=log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Clinical Prediction System training...")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Initialize system
        system = ClinicalPredictionSystem(data_path=args.data_path)
        
        # Load data
        if not system.load_data(create_sample=args.create_sample):
            logger.error("Failed to load training data")
            return 1
        
        # Create additional sample data if requested
        if args.create_sample and args.num_samples:
            from src.utils import create_sample_data
            sample_path = Path(args.data_path) / "additional_sample_data.json"
            create_sample_data(sample_path, n_samples=args.num_samples)
            logger.info(f"Created additional {args.num_samples} sample records")
        
        # Train models
        training_results = system.train_models(
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # Save models unless disabled
        if not args.no_save:
            system.save_models()
            logger.info("Models saved successfully")
        
        # Generate sample predictions
        sample_predictions = system.predict_samples(num_samples=5)
        logger.info("Generated sample predictions")
        
        # Export results if requested
        if args.export_results:
            results = {
                'training_results': training_results,
                'sample_predictions': sample_predictions,
                'model_info': system.get_model_summary()
            }
            
            export_path = Path(args.export_results)
            export_results_to_json(results, export_path)
            logger.info(f"Results exported to {export_path}")
        
        # Display summary
        eval_results = training_results['evaluation_results']
        treatment_acc = eval_results['treatment_metrics']['accuracy']
        radiology_acc = eval_results['radiology_metrics']['accuracy']
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Treatment Prediction Accuracy: {treatment_acc:.4f}")
        print(f"Radiology Prediction Accuracy: {radiology_acc:.4f}")
        print(f"Total Training Samples: {training_results['data_split']['train_size']}")
        print(f"Total Test Samples: {training_results['data_split']['test_size']}")
        print("="*50)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())