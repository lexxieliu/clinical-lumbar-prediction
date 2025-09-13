# src/feature_engineering.py
"""
Feature engineering module for Clinical Prediction System
Handles text preprocessing, TF-IDF vectorization, and feature selection
"""

import logging
import numpy as np
import joblib
from typing import List, Optional, Dict, Any
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_PARAMS, MODEL_FILES

logger = logging.getLogger(__name__)


class ClinicalFeatureEngineer:
    """
    Handles feature extraction and selection for clinical text data
    """
    
    def __init__(self):
        """Initialize feature engineering components"""
        self.feature_params = FEATURE_PARAMS
        self.model_files = MODEL_FILES
        
        # Initialize components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.feature_params['tfidf_max_features'],
            stop_words='english',
            ngram_range=self.feature_params['tfidf_ngram_range'],
            min_df=self.feature_params['tfidf_min_df'],
            max_df=self.feature_params['tfidf_max_df'],
            lowercase=True,
            strip_accents='unicode'
        )
        
        self.feature_selector = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit_transform(self, texts: List[str], labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit feature engineering pipeline and transform texts
        
        Args:
            texts: List of clinical text documents
            labels: Optional labels for supervised feature selection
            
        Returns:
            Transformed feature matrix
        """
        logger.info("Starting feature engineering pipeline fitting...")
        
        # Preprocess texts
        processed_texts = self._preprocess_texts(texts)
        
        # Fit and transform TF-IDF
        logger.info("Fitting TF-IDF vectorizer...")
        tfidf_features = self.tfidf_vectorizer.fit_transform(processed_texts).toarray()
        logger.info(f"TF-IDF feature matrix shape: {tfidf_features.shape}")
        
        # Feature selection if labels are provided
        if labels is not None:
            features = self._fit_feature_selection(tfidf_features, labels)
        else:
            features = tfidf_features
        
        # Scale features
        logger.info("Scaling features...")
        features = self.scaler.fit_transform(features)
        
        self.is_fitted = True
        logger.info(f"Feature engineering completed. Final shape: {features.shape}")
        
        return features
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform new texts using fitted pipeline
        
        Args:
            texts: List of clinical text documents
            
        Returns:
            Transformed feature matrix
            
        Raises:
            ValueError: If pipeline is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transforming new data")
        
        # Preprocess texts
        processed_texts = self._preprocess_texts(texts)
        
        # Transform with TF-IDF
        tfidf_features = self.tfidf_vectorizer.transform(processed_texts).toarray()
        
        # Apply feature selection if fitted
        if self.feature_selector is not None:
            tfidf_features = self.feature_selector.transform(tfidf_features)
        
        # Scale features
        features = self.scaler.transform(tfidf_features)
        
        return features
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess clinical texts
        
        Args:
            texts: Raw text documents
            
        Returns:
            Preprocessed text documents
        """
        processed_texts = []
        
        for text in texts:
            if not text or not text.strip():
                processed_texts.append("empty clinical record")
            else:
                # Basic text cleaning
                cleaned_text = str(text).strip()
                # Remove excessive whitespace
                cleaned_text = ' '.join(cleaned_text.split())
                processed_texts.append(cleaned_text)
        
        logger.debug(f"Preprocessed {len(processed_texts)} text documents")
        return processed_texts
    
    def _fit_feature_selection(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Fit feature selection using statistical tests
        
        Args:
            features: Feature matrix
            labels: Target labels for supervised selection
            
        Returns:
            Selected features
        """
        logger.info("Performing feature selection...")
        
        # For multi-label problems, use label sums for feature selection
        if labels.ndim > 1:
            # Sum across labels to get single target for feature selection
            target_for_selection = labels.sum(axis=1)
        else:
            target_for_selection = labels
        
        # Determine number of features to select
        k_value = min(
            self.feature_params['feature_selection_k'],
            features.shape[1] // 2,
            features.shape[1]
        )
        
        # Initialize and fit feature selector
        self.feature_selector = SelectKBest(
            score_func=f_classif, 
            k=k_value
        )
        
        selected_features = self.feature_selector.fit_transform(features, target_for_selection)
        
        logger.info(f"Selected {selected_features.shape[1]} features out of {features.shape[1]}")
        return selected_features
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores if available
        
        Returns:
            Feature importance scores or None
        """
        if self.feature_selector is not None and hasattr(self.feature_selector, 'scores_'):
            return self.feature_selector.scores_
        return None
    
    def get_selected_feature_names(self) -> Optional[List[str]]:
        """
        Get names of selected features
        
        Returns:
            List of selected feature names or None
        """
        if (self.feature_selector is not None and 
            hasattr(self.tfidf_vectorizer, 'get_feature_names_out')):
            
            # Get all feature names
            all_feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get selected feature indices
            selected_indices = self.feature_selector.get_support(indices=True)
            
            # Return selected feature names
            return [all_feature_names[i] for i in selected_indices]
        
        return None
    
    def save_pipeline(self, base_path: Optional[Path] = None) -> None:
        """
        Save fitted feature engineering pipeline
        
        Args:
            base_path: Base directory for saving models
        """
        if not self.is_fitted:
            logger.warning("Pipeline is not fitted, cannot save")
            return
        
        try:
            # Save TF-IDF vectorizer
            vectorizer_path = self.model_files['vectorizer']
            joblib.dump(self.tfidf_vectorizer, vectorizer_path)
            logger.info(f"Saved TF-IDF vectorizer to {vectorizer_path}")
            
            # Save feature engineer object
            engineer_path = self.model_files['feature_engineer']
            joblib.dump(self, engineer_path)
            logger.info(f"Saved feature engineer to {engineer_path}")
            
        except Exception as e:
            logger.error(f"Error saving feature engineering pipeline: {e}")
    
    @classmethod
    def load_pipeline(cls, base_path: Optional[Path] = None) -> 'ClinicalFeatureEngineer':
        """
        Load saved feature engineering pipeline
        
        Args:
            base_path: Base directory containing saved models
            
        Returns:
            Loaded feature engineer instance
        """
        try:
            engineer_path = MODEL_FILES['feature_engineer']
            engineer = joblib.load(engineer_path)
            logger.info(f"Loaded feature engineer from {engineer_path}")
            return engineer
        except Exception as e:
            logger.error(f"Error loading feature engineering pipeline: {e}")
            raise
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted pipeline
        
        Returns:
            Dictionary with pipeline information
        """
        info = {
            'is_fitted': self.is_fitted,
            'tfidf_params': self.tfidf_vectorizer.get_params(),
            'feature_selection_enabled': self.feature_selector is not None,
        }
        
        if self.is_fitted:
            info.update({
                'vocab_size': len(self.tfidf_vectorizer.vocabulary_),
                'selected_features': (self.feature_selector.k
                                    if self.feature_selector else None),
            })
        
        return info