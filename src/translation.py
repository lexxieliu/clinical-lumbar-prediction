# src/translation.py
"""
Translation module for Clinical Prediction System
Supports multiple LLM APIs for Korean to English translation
"""

import logging
import re
import time
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
import requests
import json

logger = logging.getLogger(__name__)


class TranslationProvider(ABC):
    """Abstract base class for translation providers"""
    
    @abstractmethod
    def translate(self, text: str, source_lang: str = "ko", target_lang: str = "en") -> str:
        """Translate text from source to target language"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the translation provider is available"""
        pass


class OpenAITranslator(TranslationProvider):
    """OpenAI GPT-based translator"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def translate(self, text: str, source_lang: str = "ko", target_lang: str = "en") -> str:
        """Translate text using OpenAI GPT"""
        if not text.strip():
            return text
        
        # Check if text contains Korean characters
        if not self._contains_korean(text):
            return text
        
        try:
            prompt = self._create_translation_prompt(text, source_lang, target_lang)
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional medical translator specializing in Korean to English translation. Translate medical terms accurately while preserving the original meaning and context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.1
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            translated_text = result["choices"][0]["message"]["content"].strip()
            
            logger.debug(f"Translated: '{text}' -> '{translated_text}'")
            return translated_text
            
        except Exception as e:
            logger.error(f"OpenAI translation failed: {e}")
            return text  # Return original text if translation fails
    
    def _create_translation_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """Create a translation prompt for the LLM"""
        return f"""Translate the following Korean medical text to English. Preserve medical terminology and context:

Korean: {text}

English translation:"""
    
    def _contains_korean(self, text: str) -> bool:
        """Check if text contains Korean characters"""
        korean_pattern = re.compile(r'[가-힣]')
        return bool(korean_pattern.search(text))
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available"""
        try:
            # Simple test request
            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
            }
            response = requests.post(self.base_url, headers=self.headers, json=test_payload, timeout=5)
            return response.status_code == 200
        except:
            return False


class GoogleTranslator(TranslationProvider):
    """Google Translate API translator"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://translation.googleapis.com/language/translate/v2"
    
    def translate(self, text: str, source_lang: str = "ko", target_lang: str = "en") -> str:
        """Translate text using Google Translate API"""
        if not text.strip():
            return text
        
        if not self._contains_korean(text):
            return text
        
        try:
            params = {
                "key": self.api_key,
                "q": text,
                "source": source_lang,
                "target": target_lang,
                "format": "text"
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            translated_text = result["data"]["translations"][0]["translatedText"]
            
            logger.debug(f"Translated: '{text}' -> '{translated_text}'")
            return translated_text
            
        except Exception as e:
            logger.error(f"Google translation failed: {e}")
            return text
    
    def _contains_korean(self, text: str) -> bool:
        """Check if text contains Korean characters"""
        korean_pattern = re.compile(r'[가-힣]')
        return bool(korean_pattern.search(text))
    
    def is_available(self) -> bool:
        """Check if Google Translate API is available"""
        try:
            test_params = {
                "key": self.api_key,
                "q": "test",
                "source": "en",
                "target": "ko"
            }
            response = requests.get(self.base_url, params=test_params, timeout=5)
            return response.status_code == 200
        except:
            return False


class ReplicateTranslator(TranslationProvider):
    """Replicate API translator using custom model"""
    
    def __init__(self, api_key: str, model_version: str = "db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf"):
        self.api_key = api_key
        self.model_version = model_version
        self.base_url = "https://api.replicate.com/v1/predictions"
    
    def translate(self, text: str, source_lang: str = "ko", target_lang: str = "en") -> str:
        """Translate text using Replicate API"""
        if not text.strip():
            return text
        
        if not self._contains_korean(text):
            return text
        
        try:
            # Create translation prompt
            prompt = f"Translate the following Korean medical text to English. Preserve medical terminology and context:\n\nKorean: {text}\n\nEnglish translation:"
            
            body = json.dumps({
                "version": self.model_version,
                "input": {
                    "prompt": prompt
                }
            })
            
            headers = {
                "authorization": f"Token {self.api_key}",
                "content-type": "application/json"
            }
            
            # Submit prediction
            response = requests.post(self.base_url, data=body, headers=headers, timeout=30)
            response.raise_for_status()
            
            prediction_data = response.json()
            get_url = prediction_data['urls']['get']
            
            # Wait for completion and get result
            time.sleep(10)  # Wait for processing
            result_response = requests.post(get_url, headers=headers, timeout=30)
            result_response.raise_for_status()
            
            result_data = result_response.json()
            translated_text = result_data["output"][0] if isinstance(result_data["output"], list) else result_data["output"]
            
            logger.debug(f"Translated: '{text}' -> '{translated_text}'")
            return translated_text
            
        except Exception as e:
            logger.error(f"Replicate translation failed: {e}")
            return text
    
    def _contains_korean(self, text: str) -> bool:
        """Check if text contains Korean characters"""
        korean_pattern = re.compile(r'[가-힣]')
        return bool(korean_pattern.search(text))
    
    def is_available(self) -> bool:
        """Check if Replicate API is available"""
        try:
            test_body = json.dumps({
                "version": self.model_version,
                "input": {
                    "prompt": "test"
                }
            })
            test_headers = {
                "authorization": f"Token {self.api_key}",
                "content-type": "application/json"
            }
            response = requests.post(self.base_url, data=test_body, headers=test_headers, timeout=5)
            return response.status_code == 200
        except:
            return False


class DeepLTranslator(TranslationProvider):
    """DeepL API translator"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-free.deepl.com/v2/translate"
    
    def translate(self, text: str, source_lang: str = "ko", target_lang: str = "en") -> str:
        """Translate text using DeepL API"""
        if not text.strip():
            return text
        
        if not self._contains_korean(text):
            return text
        
        try:
            # DeepL uses different language codes
            source_code = "KO" if source_lang == "ko" else source_lang.upper()
            target_code = "EN" if target_lang == "en" else target_lang.upper()
            
            data = {
                "auth_key": self.api_key,
                "text": text,
                "source_lang": source_code,
                "target_lang": target_code
            }
            
            response = requests.post(self.base_url, data=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            translated_text = result["translations"][0]["text"]
            
            logger.debug(f"Translated: '{text}' -> '{translated_text}'")
            return translated_text
            
        except Exception as e:
            logger.error(f"DeepL translation failed: {e}")
            return text
    
    def _contains_korean(self, text: str) -> bool:
        """Check if text contains Korean characters"""
        korean_pattern = re.compile(r'[가-힣]')
        return bool(korean_pattern.search(text))
    
    def is_available(self) -> bool:
        """Check if DeepL API is available"""
        try:
            test_data = {
                "auth_key": self.api_key,
                "text": "test",
                "source_lang": "EN",
                "target_lang": "KO"
            }
            response = requests.post(self.base_url, data=test_data, timeout=5)
            return response.status_code == 200
        except:
            return False


class ClinicalTranslator:
    """
    Main translation class for clinical data
    Supports multiple translation providers with fallback options
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize translator with configuration
        
        Args:
            config: Translation configuration dictionary
        """
        self.config = config
        self.providers = []
        self._initialize_providers()
        self.translation_cache = {}
        self.cache_enabled = config.get('cache_enabled', True)
        self.max_cache_size = config.get('max_cache_size', 1000)
    
    def _initialize_providers(self):
        """Initialize available translation providers"""
        # OpenAI
        if self.config.get('openai_api_key'):
            openai_provider = OpenAITranslator(
                api_key=self.config['openai_api_key'],
                model=self.config.get('openai_model', 'gpt-3.5-turbo')
            )
            if openai_provider.is_available():
                self.providers.append(openai_provider)
                logger.info("OpenAI translator initialized")
        
        # Google Translate
        if self.config.get('google_api_key'):
            google_provider = GoogleTranslator(api_key=self.config['google_api_key'])
            if google_provider.is_available():
                self.providers.append(google_provider)
                logger.info("Google translator initialized")
        
        # DeepL
        if self.config.get('deepl_api_key'):
            deepl_provider = DeepLTranslator(api_key=self.config['deepl_api_key'])
            if deepl_provider.is_available():
                self.providers.append(deepl_provider)
                logger.info("DeepL translator initialized")
        
        # Replicate
        if self.config.get('replicate_api_key'):
            replicate_provider = ReplicateTranslator(
                api_key=self.config['replicate_api_key'],
                model_version=self.config.get('replicate_model_version', 'db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf')
            )
            if replicate_provider.is_available():
                self.providers.append(replicate_provider)
                logger.info("Replicate translator initialized")
        
        if not self.providers:
            logger.warning("No translation providers available")
    
    def translate_text(self, text: str, source_lang: str = "ko", target_lang: str = "en") -> str:
        """
        Translate text using available providers
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text or original text if translation fails
        """
        if not text.strip() or not self.providers:
            return text
        
        # Check cache first
        cache_key = f"{text}_{source_lang}_{target_lang}"
        if self.cache_enabled and cache_key in self.translation_cache:
            logger.debug(f"Using cached translation for: {text[:50]}...")
            return self.translation_cache[cache_key]
        
        # Try each provider in order
        for provider in self.providers:
            try:
                translated = provider.translate(text, source_lang, target_lang)
                
                # Cache the result
                if self.cache_enabled:
                    self._add_to_cache(cache_key, translated)
                
                return translated
                
            except Exception as e:
                logger.warning(f"Translation failed with {provider.__class__.__name__}: {e}")
                continue
        
        logger.error("All translation providers failed")
        return text
    
    def translate_clinical_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate Korean text in a clinical record
        
        Args:
            record: Clinical data record
            
        Returns:
            Record with translated text fields
        """
        translated_record = record.copy()
        
        # Fields that commonly contain Korean text
        korean_fields = [
            "When the pain start",
            "When the pain became severe", 
            "etc",
            "Plan",
            "clinical_notes",
            "notes",
            "description"
        ]
        
        # Translate text in Study section
        if "Study" in record:
            translated_study = {}
            for key, value in record["Study"].items():
                if isinstance(value, str) and value.strip():
                    translated_study[key] = self.translate_text(value)
                else:
                    translated_study[key] = value
            translated_record["Study"] = translated_study
        
        # Translate other text fields
        for field in korean_fields:
            if field in record and isinstance(record[field], str) and record[field].strip():
                translated_record[field] = self.translate_text(record[field])
        
        # Recursively translate nested dictionaries
        translated_record = self._translate_nested_dict(translated_record)
        
        return translated_record
    
    def _translate_nested_dict(self, data: Any) -> Any:
        """Recursively translate Korean text in nested dictionaries"""
        if isinstance(data, dict):
            translated = {}
            for key, value in data.items():
                if isinstance(value, str) and value.strip() and self._contains_korean(value):
                    translated[key] = self.translate_text(value)
                elif isinstance(value, (dict, list)):
                    translated[key] = self._translate_nested_dict(value)
                else:
                    translated[key] = value
            return translated
        elif isinstance(data, list):
            return [self._translate_nested_dict(item) for item in data]
        else:
            return data
    
    def _contains_korean(self, text: str) -> bool:
        """Check if text contains Korean characters"""
        korean_pattern = re.compile(r'[가-힣]')
        return bool(korean_pattern.search(text))
    
    def _add_to_cache(self, key: str, value: str):
        """Add translation to cache with size limit"""
        if len(self.translation_cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.translation_cache))
            del self.translation_cache[oldest_key]
        
        self.translation_cache[key] = value
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get translation cache statistics"""
        return {
            "cache_size": len(self.translation_cache),
            "max_cache_size": self.max_cache_size,
            "cache_enabled": self.cache_enabled
        }
    
    def clear_cache(self):
        """Clear translation cache"""
        self.translation_cache.clear()
        logger.info("Translation cache cleared")
    
    def is_available(self) -> bool:
        """Check if any translation provider is available"""
        return len(self.providers) > 0


def create_translator_from_config(config_path: Optional[str] = None) -> ClinicalTranslator:
    """
    Create a ClinicalTranslator instance from configuration file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured ClinicalTranslator instance
    """
    import os
    from pathlib import Path
    
    # Default configuration
    default_config = {
        'cache_enabled': True,
        'max_cache_size': 1000,
        'openai_model': 'gpt-3.5-turbo'
    }
    
    # Load from environment variables if no config file
    if not config_path:
        config = default_config.copy()
        
        # Try to get API keys from environment variables
        if os.getenv('OPENAI_API_KEY'):
            config['openai_api_key'] = os.getenv('OPENAI_API_KEY')
        
        if os.getenv('REPLICATE_API_KEY'):
            config['replicate_api_key'] = os.getenv('REPLICATE_API_KEY')
        
        if os.getenv('REPLICATE_MODEL_VERSION'):
            config['replicate_model_version'] = os.getenv('REPLICATE_MODEL_VERSION')
        
        if os.getenv('GOOGLE_API_KEY'):
            config['google_api_key'] = os.getenv('GOOGLE_API_KEY')
        
        if os.getenv('DEEPL_API_KEY'):
            config['deepl_api_key'] = os.getenv('DEEPL_API_KEY')
        
        return ClinicalTranslator(config)
    
    # Load from config file
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
        
        config = {**default_config, **file_config}
        return ClinicalTranslator(config)
    else:
        logger.warning(f"Config file not found: {config_path}")
        return ClinicalTranslator(default_config)
