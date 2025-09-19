#!/usr/bin/env python3
"""
Translation script for Clinical Prediction System
Demonstrates Korean to English translation functionality
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.translation import ClinicalTranslator, create_translator_from_config
from src.data_loader import ClinicalDataLoader
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def translate_sample_data():
    """Translate sample clinical data to demonstrate functionality"""
    
    # Sample Korean clinical text
    sample_korean_texts = [
        "오래전부터",
        "세수하기 힘들다.숙였다 펴기..",
        "지속적으로 아프다",
        "P)Rt L4, caudal ...무릎...OS refer..24-03-05 최건 PEN free :요추 3-4-5 사이(Rt),요추 5-천추1사이(Lt)",
        "12/5새벽",
        "23.10.6 다침"
    ]
    
    print("=== Korean to English Translation Demo ===\n")
    
    # Initialize translator
    try:
        translator = create_translator_from_config()
        
        if not translator.is_available():
            print("❌ No translation providers available.")
            print("Please set up API keys in environment variables or translation_config.json")
            print("\nRequired environment variables:")
            print("- OPENAI_API_KEY")
            print("- GOOGLE_API_KEY (optional)")
            print("- DEEPL_API_KEY (optional)")
            return
        
        print("✅ Translation service initialized successfully\n")
        
        # Translate sample texts
        for i, korean_text in enumerate(sample_korean_texts, 1):
            print(f"Sample {i}:")
            print(f"  Korean: {korean_text}")
            
            translated = translator.translate_text(korean_text)
            print(f"  English: {translated}")
            print()
        
        # Show cache statistics
        cache_stats = translator.get_cache_stats()
        print(f"Translation cache: {cache_stats['cache_size']} entries")
        
    except Exception as e:
        logger.error(f"Translation demo failed: {e}")
        print(f"❌ Error: {e}")


def translate_clinical_file(input_file: str, output_file: str = None):
    """Translate a clinical data file"""
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    if output_file is None:
        output_file = input_path.parent / f"{input_path.stem}_translated{input_path.suffix}"
    
    print(f"Translating clinical data from {input_file} to {output_file}")
    
    try:
        # Initialize data loader with translation enabled
        data_loader = ClinicalDataLoader(str(input_path.parent), enable_translation=True)
        
        # Load and translate data
        translated_data = data_loader.load_json_data()
        
        # Save translated data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Translation completed. Output saved to: {output_file}")
        
        # Show translation statistics
        stats = data_loader.get_translation_stats()
        print(f"\nTranslation Statistics:")
        print(f"  Translation enabled: {stats.get('translation_enabled', False)}")
        print(f"  Translator available: {stats.get('translator_available', False)}")
        
        if 'cache_stats' in stats:
            cache_stats = stats['cache_stats']
            print(f"  Cache size: {cache_stats.get('cache_size', 0)}")
        
    except Exception as e:
        logger.error(f"File translation failed: {e}")
        print(f"❌ Error: {e}")


def main():
    """Main function"""
    setup_logging()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python translate_data.py demo                    # Run translation demo")
        print("  python translate_data.py translate <input_file> [output_file]  # Translate file")
        print()
        print("Examples:")
        print("  python translate_data.py demo")
        print("  python translate_data.py translate data/raw/70000769-Lumbar.json")
        print("  python translate_data.py translate data/raw/70000769-Lumbar.json data/processed/70000769_translated.json")
        return
    
    command = sys.argv[1].lower()
    
    if command == "demo":
        translate_sample_data()
    
    elif command == "translate":
        if len(sys.argv) < 3:
            print("❌ Please provide input file path")
            return
        
        input_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        
        translate_clinical_file(input_file, output_file)
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: demo, translate")


if __name__ == "__main__":
    main()
