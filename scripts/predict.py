#!/usr/bin/env python3
# scripts/predict.py
"""
Prediction script for Clinical Prediction System
This script loads trained models and makes predictions on new data
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ClinicalPredictionSystem, ClinicalPredictor, setup_logging
from src.utils import format_prediction_results, export_results_to_json


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Make predictions using trained Clinical Prediction System models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input text file or JSON file with clinical data'
    )
    
    parser.add_argument(
        '--text',
        type=str,
        help='Single text to predict on (alternative to --input)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/raw',
        help='Path to data directory (used if no --input or --text provided)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file to save predictions (JSON format)'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of samples to predict (when using data-path)'
    )
    
    parser.add_argument(
        '--probabilities',
        action='store_true',
        help='Include prediction probabilities in output'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['console', 'json', 'detailed'],
        default='console',
        help='Output format'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    return parser.parse_args()


def load_input_data(input_path: str):
    """
    Load input data from file
    
    Args:
        input_path: Path to input file
        
    Returns:
        List of texts to predict
    """
    input_file = Path(input_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if input_file.suffix.lower() == '.json':
        # Load JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Extract texts from list of records
            texts = []
            for record in data:
                if isinstance(record, dict):
                    # Try to extract text from various fields
                    text_parts = []
                    if 'clinical_notes' in record:
                        text_parts.append(record['clinical_notes'])
                    if 'Study' in record:
                        for key, value in record['Study'].items():
                            if value and str(value).strip():
                                text_parts.append(f"{key}: {value}")
                    texts.append(" ".join(text_parts) if text_parts else str(record))
                else:
                    texts.append(str(record))
            return texts
        else:
            return [str(data)]
    
    else:
        # Load as text file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Split by lines or use as single text
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        return lines if len(lines) > 1 else [content]


def format_output(predictions, probabilities=None, format_type='console'):
    """
    Format prediction output
    
    Args:
        predictions: List of prediction interpretations
        probabilities: Optional probabilities
        format_type: Output format type
        
    Returns:
        Formatted output string or dictionary
    """
    if format_type == 'json':
        output_data = {'predictions': predictions}
        if probabilities:
            output_data['probabilities'] = probabilities
        return json.dumps(output_data, indent=2, ensure_ascii=False)
    
    elif format_type == 'detailed':
        output = []
        output.append("=" * 80)
        output.append("CLINICAL PREDICTION RESULTS")
        output.append("=" * 80)
        
        for i, pred in enumerate(predictions):
            output.append(f"\nPrediction {i + 1}:")
            output.append("-" * 40)
            output.append(f"Text Preview: {pred['text_preview']}")
            output.append(f"Predicted Treatments: {', '.join(pred['predicted_treatments'])}")
            output.append(f"Predicted Radiology: {', '.join(pred['predicted_radiology'])}")
            
            if probabilities:
                output.append("Treatment Probabilities:")
                for j, prob in enumerate(probabilities[0][i]):
                    output.append(f"  Label {j}: {prob:.3f}")
                output.append("Radiology Probabilities:")
                for j, prob in enumerate(probabilities[1][i]):
                    output.append(f"  Label {j}: {prob:.3f}")
        
        output.append("\n" + "=" * 80)
        return "\n".join(output)
    
    else:  # console format
        return format_prediction_results(predictions)


def main():
    """
    Main prediction function
    """
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    import logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(log_level=log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Clinical Prediction System prediction...")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load trained models
        predictor = ClinicalPredictor.load_models()
        logger.info("Models loaded successfully")
        
        # Determine input texts
        texts_to_predict = []
        
        if args.text:
            # Single text provided
            texts_to_predict = [args.text]
            logger.info("Using provided text for prediction")
            
        elif args.input:
            # Load from input file
            texts_to_predict = load_input_data(args.input)
            logger.info(f"Loaded {len(texts_to_predict)} texts from {args.input}")
            
        else:
            # Use sample data
            system = ClinicalPredictionSystem(data_path=args.data_path)
            if not system.load_data():
                raise RuntimeError("Failed to load sample data for prediction")
            
            texts_to_predict = system.training_data['texts'][:args.num_samples]
            logger.info(f"Using {len(texts_to_predict)} sample texts for prediction")
        
        if not texts_to_predict:
            raise ValueError("No texts provided for prediction")
        
        # Make predictions
        treatment_pred, radiology_pred = predictor.predict(texts_to_predict)
        logger.info("Predictions completed")
        
        # Get probabilities if requested
        probabilities = None
        if args.probabilities:
            treatment_proba, radiology_proba = predictor.predict_proba(texts_to_predict)
            probabilities = (treatment_proba, radiology_proba)
            logger.info("Probabilities calculated")
        
        # Interpret predictions
        interpretations = predictor.interpret_predictions(
            texts_to_predict, treatment_pred, radiology_pred
        )
        
        # Format output
        formatted_output = format_output(
            interpretations, 
            probabilities, 
            args.format
        )
        
        # Save output if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if args.format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_output)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_output)
            
            logger.info(f"Results saved to {output_path}")
        
        # Display results
        print(formatted_output)
        
        return 0
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())