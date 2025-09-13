# src/models.py
"""
Machine learning models for Clinical Prediction System
Contains training, prediction, and evaluation functionality
"""

import logging
import numpy as np
import joblib
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split

from .config import MODEL_PARAMS, MODEL_FILES
from .feature_engineering import ClinicalFeatureEngineer

logger = logging.getLogger(__name__)


class ClinicalPredictor:
    """
    Multi-output clinical prediction system for treatment types and radiology examinations
    """
    
    def __init__(self):
        """Initialize the clinical predictor"""
        self.model_params = MODEL_PARAMS
        self.model_files = MODEL_FILES
        
        # Initialize models
        self.treatment_model = None
        self.radiology_model = None
        self.feature_engineer = None
        
        # Training metadata
        self.is_trained = False
        self.training_info = {}
        self.label_names = {'treatment': [], 'radiology': []}
    
    def train(self, 
              texts: List[str], 
              treatment_labels: np.ndarray, 
              radiology_labels: np.ndarray,
              treatment_names: Optional[List[str]] = None,
              radiology_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train both treatment and radiology prediction models
        
        Args:
            texts: List of clinical text documents
            treatment_labels: Binary matrix for treatment labels
            radiology_labels: Binary matrix for radiology labels
            treatment_names: Names of treatment types
            radiology_names: Names of radiology examinations
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training...")
        
        # Store label names
        if treatment_names:
            self.label_names['treatment'] = treatment_names
        if radiology_names:
            self.label_names['radiology'] = radiology_names
        
        # Initialize feature engineer
        self.feature_engineer = ClinicalFeatureEngineer()
        
        # Extract and transform features
        logger.info("Extracting features...")
        features = self.feature_engineer.fit_transform(texts, treatment_labels)
        
        # Train treatment prediction model
        logger.info("Training treatment prediction model...")
        self.treatment_model = self._create_model()
        self.treatment_model.fit(features, treatment_labels)
        
        # Train radiology prediction model
        logger.info("Training radiology prediction model...")
        self.radiology_model = self._create_model()
        self.radiology_model.fit(features, radiology_labels)
        
        # Store training information
        self.training_info = {
            'n_samples': len(texts),
            'n_features': features.shape[1],
            'n_treatment_labels': treatment_labels.shape[1],
            'n_radiology_labels': radiology_labels.shape[1],
            'feature_params': self.feature_engineer.get_pipeline_info()
        }
        
        self.is_trained = True
        logger.info("Model training completed successfully!")
        
        return self.training_info
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict treatment types and radiology examinations for new texts
        
        Args:
            texts: List of clinical text documents
            
        Returns:
            Tuple of (treatment_predictions, radiology_predictions)
            
        Raises:
            ValueError: If models are not trained
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Transform features
        features = self.feature_engineer.transform(texts)
        
        # Make predictions
        treatment_pred = self.treatment_model.predict(features)
        radiology_pred = self.radiology_model.predict(features)
        
        return treatment_pred, radiology_pred
    
    def predict_proba(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict probabilities for treatment types and radiology examinations
        
        Args:
            texts: List of clinical text documents
            
        Returns:
            Tuple of (treatment_probabilities, radiology_probabilities)
            
        Raises:
            ValueError: If models are not trained
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Transform features
        features = self.feature_engineer.transform(texts)
        
        # Get prediction probabilities
        treatment_proba = self._get_model_probabilities(self.treatment_model, features)
        radiology_proba = self._get_model_probabilities(self.radiology_model, features)
        
        return treatment_proba, radiology_proba
    
    def evaluate(self, 
                 texts: List[str], 
                 treatment_labels: np.ndarray, 
                 radiology_labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance on test data
        
        Args:
            texts: List of clinical text documents
            treatment_labels: True treatment labels
            radiology_labels: True radiology labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before evaluation")
        
        # Make predictions
        treatment_pred, radiology_pred = self.predict(texts)
        
        # Calculate metrics
        evaluation_results = {
            'treatment_metrics': self._calculate_metrics(
                treatment_labels, treatment_pred, 'Treatment'
            ),
            'radiology_metrics': self._calculate_metrics(
                radiology_labels, radiology_pred, 'Radiology'
            )
        }
        
        logger.info("=== Model Evaluation Results ===")
        logger.info(f"Treatment Accuracy: {evaluation_results['treatment_metrics']['accuracy']:.4f}")
        logger.info(f"Radiology Accuracy: {evaluation_results['radiology_metrics']['accuracy']:.4f}")
        
        return evaluation_results
    
    def _create_model(self) -> MultiOutputClassifier:
        """
        Create a new multi-output classifier
        
        Returns:
            Configured multi-output classifier
        """
        rf_params = self.model_params['random_forest']
        base_classifier = RandomForestClassifier(**rf_params)
        return MultiOutputClassifier(base_classifier)
    
    def _get_model_probabilities(self, model: MultiOutputClassifier, features: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities from multi-output model
        
        Args:
            model: Trained multi-output classifier
            features: Feature matrix
            
        Returns:
            Probability matrix
        """
        # Get probabilities for each output
        probabilities = []
        for i, estimator in enumerate(model.estimators_):
            if hasattr(estimator, 'predict_proba'):
                # Get probabilities for positive class
                proba = estimator.predict_proba(features)[:, 1]
                probabilities.append(proba)
            else:
                # Fallback to decision function if available
                decision = estimator.decision_function(features)
                probabilities.append(decision)
        
        return np.column_stack(probabilities)
    
    def _calculate_metrics(self, 
                          y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          task_name: str) -> Dict[str, Any]:
        """
        Calculate evaluation metrics for multi-label classification
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            task_name: Name of the prediction task
            
        Returns:
            Dictionary with calculated metrics
        """
        # Calculate overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate per-label metrics
        n_labels = y_true.shape[1]
        per_label_accuracy = []
        
        for i in range(n_labels):
            label_accuracy = accuracy_score(y_true[:, i], y_pred[:, i])
            per_label_accuracy.append(label_accuracy)
        
        # Generate classification report
        try:
            # For multi-label classification, we need to handle the report carefully
            class_report = classification_report(
                y_true, y_pred, 
                output_dict=True, 
                zero_division=0
            )
        except Exception as e:
            logger.warning(f"Could not generate classification report: {e}")
            class_report = {}
        
        metrics = {
            'accuracy': accuracy,
            'per_label_accuracy': per_label_accuracy,
            'mean_per_label_accuracy': np.mean(per_label_accuracy),
            'classification_report': class_report
        }
        
        return metrics
    
    def save_models(self, base_path: Optional[Path] = None) -> None:
        """
        Save trained models to disk
        
        Args:
            base_path: Base directory for saving models
        """
        if not self.is_trained:
            logger.warning("Models are not trained, cannot save")
            return
        
        try:
            # Save treatment model
            treatment_path = self.model_files['treatment_model']
            joblib.dump(self.treatment_model, treatment_path)
            logger.info(f"Saved treatment model to {treatment_path}")
            
            # Save radiology model
            radiology_path = self.model_files['radiology_model']
            joblib.dump(self.radiology_model, radiology_path)
            logger.info(f"Saved radiology model to {radiology_path}")
            
            # Save feature engineer
            self.feature_engineer.save_pipeline(base_path)
            
            # Save predictor metadata
            metadata = {
                'training_info': self.training_info,
                'label_names': self.label_names,
                'is_trained': self.is_trained
            }
            
            metadata_path = self.model_files['treatment_model'].parent / 'predictor_metadata.joblib'
            joblib.dump(metadata, metadata_path)
            logger.info(f"Saved predictor metadata to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    @classmethod
    def load_models(cls, base_path: Optional[Path] = None) -> 'ClinicalPredictor':
        """
        Load trained models from disk
        
        Args:
            base_path: Base directory containing saved models
            
        Returns:
            Loaded predictor instance
        """
        try:
            predictor = cls()
            
            # Load models
            treatment_path = MODEL_FILES['treatment_model']
            radiology_path = MODEL_FILES['radiology_model']
            
            predictor.treatment_model = joblib.load(treatment_path)
            predictor.radiology_model = joblib.load(radiology_path)
            logger.info("Loaded treatment and radiology models")
            
            # Load feature engineer
            predictor.feature_engineer = ClinicalFeatureEngineer.load_pipeline(base_path)
            
            # Load metadata
            metadata_path = treatment_path.parent / 'predictor_metadata.joblib'
            if metadata_path.exists():
                metadata = joblib.load(metadata_path)
                predictor.training_info = metadata.get('training_info', {})
                predictor.label_names = metadata.get('label_names', {'treatment': [], 'radiology': []})
                predictor.is_trained = metadata.get('is_trained', True)
            else:
                logger.warning("No metadata file found, using defaults")
                predictor.is_trained = True
            
            logger.info("Successfully loaded complete predictor")
            return predictor
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained models
        
        Returns:
            Dictionary with model information
        """
        info = {
            'is_trained': self.is_trained,
            'model_params': self.model_params,
            'training_info': self.training_info,
            'label_names': self.label_names
        }
        
        if self.feature_engineer:
            info['feature_info'] = self.feature_engineer.get_pipeline_info()
        
        return info
    
    def interpret_predictions(self, 
                             texts: List[str], 
                             treatment_pred: np.ndarray, 
                             radiology_pred: np.ndarray) -> List[Dict[str, Any]]:
        """
        Interpret predictions in a human-readable format
        
        Args:
            texts: Original text documents
            treatment_pred: Treatment predictions
            radiology_pred: Radiology predictions
            
        Returns:
            List of interpretation dictionaries
        """
        interpretations = []
        
        for i, text in enumerate(texts):
            # Get predicted treatments
            predicted_treatments = [
                self.label_names['treatment'][j] 
                for j, pred in enumerate(treatment_pred[i]) 
                if pred == 1 and j < len(self.label_names['treatment'])
            ]
            
            # Get predicted radiology examinations
            predicted_radiology = [
                self.label_names['radiology'][j] 
                for j, pred in enumerate(radiology_pred[i]) 
                if pred == 1 and j < len(self.label_names['radiology'])
            ]
            
            interpretation = {
                'text_snippet': text[:200] + "..." if len(text) > 200 else text,
                'predicted_treatments': predicted_treatments if predicted_treatments else ['None'],
                'predicted_radiology': predicted_radiology if predicted_radiology else ['None'],
                'n_treatments': len(predicted_treatments),
                'n_radiology': len(predicted_radiology)
            }
            
            interpretations.append(interpretation)
        
        return interpretations


class ModelTrainer:
    """
    Utility class for training clinical prediction models with cross-validation
    """
    
    def __init__(self, predictor: ClinicalPredictor):
        """
        Initialize model trainer
        
        Args:
            predictor: ClinicalPredictor instance
        """
        self.predictor = predictor
        self.model_params = MODEL_PARAMS
    
    def train_with_validation(self, 
                             texts: List[str],
                             treatment_labels: np.ndarray,
                             radiology_labels: np.ndarray,
                             treatment_names: Optional[List[str]] = None,
                             radiology_names: Optional[List[str]] = None,
                             test_size: Optional[float] = None) -> Dict[str, Any]:
        """
        Train models with train/validation split
        
        Args:
            texts: Clinical text documents
            treatment_labels: Treatment labels
            radiology_labels: Radiology labels
            treatment_names: Names of treatment types
            radiology_names: Names of radiology examinations
            test_size: Proportion of data for testing
            
        Returns:
            Training and validation results
        """
        if test_size is None:
            test_size = self.model_params['test_size']
        
        # Split data
        if len(texts) > 10:  # Only split if we have enough data
            (X_train, X_test, 
             y_treatment_train, y_treatment_test,
             y_radiology_train, y_radiology_test) = train_test_split(
                texts, treatment_labels, radiology_labels,
                test_size=test_size,
                random_state=self.model_params['random_state']
            )
        else:
            logger.warning("Dataset too small for train/test split, using all data for training")
            X_train = X_test = texts
            y_treatment_train = y_treatment_test = treatment_labels
            y_radiology_train = y_radiology_test = radiology_labels
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Train models
        training_info = self.predictor.train(
            X_train, y_treatment_train, y_radiology_train,
            treatment_names, radiology_names
        )
        
        # Evaluate on test set
        evaluation_results = self.predictor.evaluate(
            X_test, y_treatment_test, y_radiology_test
        )
        
        return {
            'training_info': training_info,
            'evaluation_results': evaluation_results,
            'data_split': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'test_proportion': test_size
            }
        }