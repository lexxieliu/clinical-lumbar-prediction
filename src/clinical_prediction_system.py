# src/clinical_prediction_system.py
"""
Lumbar Spine Treatment Prediction System - Main Application
Refactored version using modular architecture for better maintainability
"""

import logging
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from .config import MODEL_PARAMS, DATA_DIR
from .data_loader import ClinicalDataLoader
from .models import ClinicalPredictor
from .utils import (
    setup_logging, 
    create_sample_data, 
    validate_data_structure,
    print_data_summary,
    format_prediction_results,
    calculate_label_statistics
)

logger = logging.getLogger(__name__)


class ClinicalPredictionSystem:
    """
    Main application class for the Clinical Prediction System
    Orchestrates data loading, training, and prediction workflows
    """
    
    def __init__(self, data_path: str = None, enable_translation: bool = None):
        """
        Initialize the clinical prediction system
        
        Args:
            data_path: Path to clinical data directory or file
            enable_translation: Whether to enable Korean to English translation
        """
        self.data_path = Path(data_path) if data_path else DATA_DIR / "raw"
        self.data_loader = None
        self.predictor = None
        self.training_data = None
        self.enable_translation = enable_translation
        
        # Setup logging
        setup_logging()
        logger.info("Clinical Prediction System initialized")
    
    def load_data(self, create_sample: bool = False) -> bool:
        """
        Load clinical data from configured path
        
        Args:
            create_sample: Whether to create sample data if none exists
            
        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            # Check if data exists
            if not self.data_path.exists() or not list(self.data_path.glob("*.json")):
                if create_sample:
                    logger.info("No data found, creating sample data...")
                    create_sample_data(self.data_path / "sample_clinical_data.json")
                else:
                    logger.error(f"No data found at {self.data_path}")
                    return False
            
            # Initialize data loader
            self.data_loader = ClinicalDataLoader(str(self.data_path), enable_translation=self.enable_translation)
            
            # Load data
            raw_data = self.data_loader.load_json_data()
            if not raw_data:
                logger.error("Failed to load data")
                return False
            
            # Validate data
            validation_report = validate_data_structure(raw_data)
            print_data_summary(raw_data, validation_report)
            
            # Extract features and labels
            texts, treatment_labels, radiology_labels = self.data_loader.extract_features_and_labels(raw_data)
            
            # Get label names
            treatment_names, radiology_names = self.data_loader.get_label_names()
            
            # Store training data
            self.training_data = {
                'texts': texts,
                'treatment_labels': treatment_labels,
                'radiology_labels': radiology_labels,
                'treatment_names': treatment_names,
                'radiology_names': radiology_names,
                'raw_data': raw_data
            }
            
            logger.info(f"Successfully loaded {len(texts)} clinical records")
            self._print_data_statistics()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def train_models(self, test_size: float = None, random_state: int = None) -> dict:
        """
        Train treatment and radiology prediction models
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with training results
        """
        if not self.training_data:
            raise ValueError("No training data available. Please load data first.")
        
        logger.info("Starting model training...")
        
        # Get parameters
        test_size = test_size or MODEL_PARAMS['test_size']
        random_state = random_state or MODEL_PARAMS['random_state']
        
        # Split data if we have enough samples
        texts = self.training_data['texts']
        treatment_labels = self.training_data['treatment_labels']
        radiology_labels = self.training_data['radiology_labels']
        
        if len(texts) > 10:
            X_train, X_test, y_treat_train, y_treat_test, y_rad_train, y_rad_test = train_test_split(
                texts, treatment_labels, radiology_labels,
                test_size=test_size, 
                random_state=random_state
            )
        else:
            logger.warning("Small dataset detected, using all data for training and testing")
            X_train = X_test = texts
            y_treat_train = y_treat_test = treatment_labels
            y_rad_train = y_rad_test = radiology_labels
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Initialize and train predictor
        self.predictor = ClinicalPredictor()
        
        training_results = self.predictor.train(
            texts=X_train,
            treatment_labels=y_treat_train,
            radiology_labels=y_rad_train,
            treatment_names=self.training_data['treatment_names'],
            radiology_names=self.training_data['radiology_names']
        )
        
        # Evaluate on test set
        evaluation_results = self.predictor.evaluate(X_test, y_treat_test, y_rad_test)
        
        # Combine results
        results = {
            'training_info': training_results,
            'evaluation_results': evaluation_results,
            'data_split': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'test_ratio': test_size
            }
        }
        
        logger.info("Model training completed successfully!")
        return results
    
    def predict_samples(self, texts: list = None, num_samples: int = 3) -> list:
        """
        Make predictions on sample texts
        
        Args:
            texts: List of texts to predict (uses test data if None)
            num_samples: Number of samples to predict
            
        Returns:
            List of prediction interpretations
        """
        if not self.predictor or not self.predictor.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Use provided texts or sample from training data
        if texts is None:
            if not self.training_data:
                raise ValueError("No training data available for sampling")
            
            sample_texts = self.training_data['texts'][:num_samples]
        else:
            sample_texts = texts[:num_samples]
        
        # Make predictions
        treatment_pred, radiology_pred = self.predictor.predict(sample_texts)
        
        # Get interpretations
        interpretations = self.predictor.interpret_predictions(
            sample_texts, treatment_pred, radiology_pred
        )
        
        return interpretations
    
    def save_models(self) -> None:
        """
        Save trained models to disk
        """
        if not self.predictor:
            raise ValueError("No trained models to save")
        
        self.predictor.save_models()
        logger.info("Models saved successfully")
    
    def load_models(self) -> bool:
        """
        Load previously trained models from disk
        
        Returns:
            True if models loaded successfully, False otherwise
        """
        try:
            self.predictor = ClinicalPredictor.load_models()
            logger.info("Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def run_complete_pipeline(self, save_models: bool = True) -> dict:
        """
        Run the complete machine learning pipeline
        
        Args:
            save_models: Whether to save trained models
            
        Returns:
            Dictionary with complete pipeline results
        """
        logger.info("=== Starting Complete Clinical Prediction Pipeline ===")
        
        # Step 1: Load data
        if not self.load_data(create_sample=True):
            raise RuntimeError("Failed to load data")
        
        # Step 2: Train models
        training_results = self.train_models()
        
        # Step 3: Generate sample predictions
        sample_predictions = self.predict_samples()
        
        # Step 4: Save models if requested
        if save_models:
            self.save_models()
        
        # Step 5: Display results
        self._display_results(training_results, sample_predictions)
        
        # Compile final results
        pipeline_results = {
            'training_results': training_results,
            'sample_predictions': sample_predictions,
            'model_info': self.predictor.get_model_info() if self.predictor else {},
            'data_info': {
                'total_samples': len(self.training_data['texts']),
                'treatment_labels': len(self.training_data['treatment_names']),
                'radiology_labels': len(self.training_data['radiology_names'])
            }
        }
        
        logger.info("=== Pipeline Execution Completed Successfully ===")
        return pipeline_results
    
    def _print_data_statistics(self) -> None:
        """
        Print detailed statistics about the loaded data
        """
        if not self.training_data:
            return
        
        logger.info("=== Data Statistics ===")
        
        # Calculate label statistics
        treatment_stats = calculate_label_statistics(
            self.training_data['treatment_labels'],
            self.training_data['treatment_names']
        )
        
        radiology_stats = calculate_label_statistics(
            self.training_data['radiology_labels'],
            self.training_data['radiology_names']
        )
        
        # Print treatment statistics
        logger.info("Treatment Type Statistics:")
        for name, freq in treatment_stats['label_frequencies'].items():
            pct = treatment_stats['label_percentages'][name]
            logger.info(f"  {name}: {freq} samples ({pct:.1f}%)")
        
        # Print radiology statistics
        logger.info("Radiology Examination Statistics:")
        for name, freq in radiology_stats['label_frequencies'].items():
            pct = radiology_stats['label_percentages'][name]
            logger.info(f"  {name}: {freq} samples ({pct:.1f}%)")
        
        # Print average labels per sample
        logger.info(f"Average treatments per sample: {treatment_stats['avg_labels_per_sample']:.2f}")
        logger.info(f"Average radiology exams per sample: {radiology_stats['avg_labels_per_sample']:.2f}")
    
    def _display_results(self, training_results: dict, sample_predictions: list) -> None:
        """
        Display training and prediction results
        
        Args:
            training_results: Results from model training
            sample_predictions: Sample prediction results
        """
        logger.info("=== Model Performance Results ===")
        
        # Display evaluation metrics
        eval_results = training_results['evaluation_results']
        treatment_acc = eval_results['treatment_metrics']['accuracy']
        radiology_acc = eval_results['radiology_metrics']['accuracy']
        
        logger.info(f"Treatment Prediction Accuracy: {treatment_acc:.4f}")
        logger.info(f"Radiology Prediction Accuracy: {radiology_acc:.4f}")
        
        # Display sample predictions
        logger.info("=== Sample Predictions ===")
        formatted_predictions = format_prediction_results(sample_predictions)
        print(formatted_predictions)
    
    def get_model_summary(self) -> dict:
        """
        Get a comprehensive summary of the trained model
        
        Returns:
            Dictionary with model summary information
        """
        if not self.predictor:
            return {'error': 'No trained model available'}
        
        summary = {
            'model_info': self.predictor.get_model_info(),
            'feature_importance': self.predictor.get_feature_importance(),
            'data_summary': {}
        }
        
        if self.training_data:
            summary['data_summary'] = {
                'total_samples': len(self.training_data['texts']),
                'treatment_types': self.training_data['treatment_names'],
                'radiology_types': self.training_data['radiology_names'],
                'average_text_length': np.mean([len(text) for text in self.training_data['texts']])
            }
        
        return summary
    
    def get_translation_info(self) -> dict:
        """
        Get translation system information
        
        Returns:
            Dictionary with translation information
        """
        if not self.data_loader:
            return {"translation_enabled": False, "error": "Data loader not initialized"}
        
        return self.data_loader.get_translation_stats()


def main():
    """
    Main function to run the clinical prediction system
    """
    try:
        # Initialize system
        system = ClinicalPredictionSystem()
        
        # Run complete pipeline
        results = system.run_complete_pipeline(save_models=True)
        
        # Optional: Export results
        #from .utils import export_results_to_json
        #export_results_to_json(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


def run_prediction_only(data_path: str = None, model_texts: list = None):
    """
    Run prediction using pre-trained models
    
    Args:
        data_path: Path to data for prediction
        model_texts: List of texts to predict on
    """
    try:
        system = ClinicalPredictionSystem(data_path)
        
        # Load pre-trained models
        if not system.load_models():
            raise RuntimeError("Failed to load pre-trained models")
        
        # Make predictions
        if model_texts:
            predictions = system.predict_samples(texts=model_texts)
        else:
            # Load data for prediction
            if not system.load_data():
                raise RuntimeError("Failed to load prediction data")
            predictions = system.predict_samples()
        
        # Display results
        formatted_results = format_prediction_results(predictions)
        print(formatted_results)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise


def create_demo_data():
    """
    Create demonstration data for testing the system
    """
    try:
        logger.info("Creating demonstration data...")
        sample_path = create_sample_data(n_samples=25)
        logger.info(f"Demo data created at: {sample_path}")
        return sample_path
        
    except Exception as e:
        logger.error(f"Error creating demo data: {e}")
        raise


if __name__ == "__main__":
    # Setup logging for direct script execution
    setup_logging()
    
    # Run main application
    main()