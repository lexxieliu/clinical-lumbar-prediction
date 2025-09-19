# Clinical Lumbar Spine Treatment Prediction System

A machine learning-based clinical decision support system for predicting treatment plans and imaging requirements for patients with lumbar spine disorders.

## 🎯 Project Overview

This system analyzes clinical text data to automatically predict:
- **Treatment types**: 17 different treatment options (PEN, OLM, 3DPC, ESI, PELD, etc.)
- **Imaging studies**: 4 types of radiology examinations (X-RAY, CT, MRI, MYELO)

## ✨ Key Features

- 📊 **Multi-label classification**: Predict multiple treatments and imaging types simultaneously  
- 🔤 **Text feature extraction**: TF-IDF vectorization of clinical notes  
- 🎛️ **Feature selection**: Automatic selection of most relevant features (SelectKBest)
- 📈 **Model evaluation**: Detailed performance metrics and validation
- 🏥 **Real-world performance**: 95.7% treatment accuracy, 85.0% radiology accuracy
- 🌐 **Korean-English Translation**: LLM-powered translation of Korean clinical text using multiple providers (OpenAI, Replicate, Google, DeepL)
- 💾 **Translation Caching**: Intelligent caching system to reduce API costs and improve performance

## 🛠️ Tech Stack

- **Python 3.9+**
- **Scikit-learn**: Machine learning framework
- **Pandas & NumPy**: Data processing and numerical computation
- **Random Forest**: Multi-output classification algorithm
- **TF-IDF**: Text feature extraction
- **Joblib**: Model persistence
- **OpenAI GPT**: Korean-English translation
- **Replicate API**: Custom model integration
- **Google Translate API**: Alternative translation provider
- **DeepL API**: High-quality translation service

## 📁 Project Structure

```
clinical-prediction-system/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment config
├── .gitignore                   # Git ignore rules
├── src/                         # Source code
│   ├── __init__.py
│   ├── clinical_prediction_system.py  # Main application
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── feature_engineering.py  # Feature extraction pipeline
│   ├── models.py               # ML models and training
│   ├── translation.py          # Korean-English translation module
│   ├── utils.py                # Utility functions
│   └── config.py               # Configuration settings
├── scripts/                    # Executable scripts
│   ├── train_model.py          # Training script
│   ├── predict.py              # Prediction script
│   └── translate_data.py       # Translation demonstration script
├── data/                       # Data directory (excluded from git)
├── models/                     # Saved models (excluded from git)
├── logs/                       # Log files (excluded from git)
├── notebooks/                  # Jupyter notebooks
├── tests/                      # Test files
└── examples/                   # Example usage
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Using Conda (recommended)
conda env create -f environment.yml
conda activate clinical-prediction

# Or using pip
pip install -r requirements.txt
```

### 2. Directory Setup

```bash
mkdir -p data/raw data/processed models logs
```

### 3. Translation Setup (Optional)

To enable Korean to English translation, set up API keys:

**Option A: Environment Variables**
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export REPLICATE_API_KEY="your_replicate_api_key_here"  # Your existing key
export REPLICATE_MODEL_VERSION="db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf"
export GOOGLE_API_KEY="your_google_api_key_here"  # Optional
export DEEPL_API_KEY="your_deepl_api_key_here"    # Optional
```

**Option B: Configuration File**
```bash
# Copy and edit the configuration template
cp translation_config.json my_translation_config.json
# Edit my_translation_config.json with your API keys
```

### 4. Running the System

**Option A: Complete Pipeline with Sample Data**
```bash
python -m src.clinical_prediction_system
```

**Option B: Training with Custom Data**
```bash
# Place your JSON data files in data/raw/
python scripts/train_model.py --data-path data/raw
```

**Option C: Making Predictions**
```bash
python scripts/predict.py --data-path data/raw --num-samples 5
```

**Option D: Translation Demo**
```bash
# Test translation functionality
python scripts/translate_data.py demo

# Translate a specific file
python scripts/translate_data.py translate data/raw/70000769-Lumbar.json
```

## 📊 Performance Metrics

Based on 700 clinical records:
- **Treatment Prediction**: 95.7% accuracy
- **Radiology Prediction**: 85.0% accuracy
- **Data Split**: 560 training, 140 testing samples
- **Feature Reduction**: 500 → 100 selected features

## 📋 Data Format

The system expects JSON files with the following structure:

```json
[
  {
    "patient_id": "P001",
    "Study": {
      "X-RAY": "imaging findings...",
      "CT": "ct scan results...",
      "MRI": "mri findings...",
      "MYELO": ""
    },
    "clinical_notes": "patient clinical information and symptoms..."
  }
]
```

## 🔧 Configuration

Key parameters can be modified in `src/config.py`:

- **Model Parameters**: Random Forest settings
- **Feature Engineering**: TF-IDF and selection parameters  
- **Treatment Types**: Configurable treatment categories
- **File Paths**: Data and model directories
- **Translation Settings**: LLM API configuration and caching options

## 🌐 Translation Features

The system now includes advanced Korean to English translation capabilities:

### Supported Translation Providers
- **OpenAI GPT**: High-quality contextual translation using GPT models
- **Replicate API**: Custom model integration with your existing API setup
- **Google Translate**: Reliable translation with good medical terminology support
- **DeepL**: Premium translation service with excellent accuracy

### Translation Features
- **Automatic Detection**: Identifies Korean text in clinical records
- **Intelligent Caching**: Reduces API costs by caching translations
- **Fallback Support**: Multiple providers ensure high availability
- **Medical Context**: Specialized prompts for medical terminology
- **Batch Processing**: Efficient translation of large datasets

### Translation Configuration
```python
# Enable/disable translation
system = ClinicalPredictionSystem(enable_translation=True)

# Check translation status
translation_info = system.get_translation_info()
print(f"Translation enabled: {translation_info['translation_enabled']}")
```

## 📝 Usage Examples

### Training a Model
```python
from src import ClinicalPredictionSystem

system = ClinicalPredictionSystem(data_path="path/to/data")
results = system.run_complete_pipeline()
```

### Making Predictions
```python
from src import ClinicalPredictor

predictor = ClinicalPredictor.load_models()
treatment_pred, radiology_pred = predictor.predict(["clinical text..."])
```

## 🧪 Testing

```bash
python -m pytest tests/
```

## ⚠️ Data Privacy Notice

This system is designed to work with clinical data, which contains patients' privacy. 
The data in this project isn't and will not be uploaded 

Never commit actual patient data to version control.

## 📞 Contact

xliu174@illinois.edu

---

**Note**: This system is for research and educational purposes. Always validate predictions with clinical expertise before making medical decisions.
