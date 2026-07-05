# Depression Detection Using Multimodal Fusion

## Overview

This project implements a comprehensive depression detection system that combines multiple modalities (text, audio/speech, and facial expressions) using deep learning techniques. The system uses multimodal fusion to achieve higher accuracy in detecting depressive symptoms from diverse data sources.

## 📊 Project Structure

```
depression-detection-multimodal/
├── data/
│   ├── raw/
│   │   ├── audio/              # Raw audio files
│   │   ├── video/              # Raw video files
│   │   └── transcripts/         # Transcribed text
│   ├── processed/
│   │   ├── text_features/      # Extracted text features
│   │   ├── audio_features/     # Extracted audio features
│   │   └── facial_features/    # Extracted facial features
│   └── datasets/
│       ├── DAIC-WOZ/           # DAIC-WOZ dataset
│       ├── D-Vlog/             # D-Vlog dataset
│       └── E-DAIC/             # Extended DAIC dataset
├── models/
│   ├── text_encoder/           # Text feature extraction
│   ├── audio_encoder/          # Audio feature extraction
│   ├── facial_encoder/         # Facial feature extraction
│   ├── fusion_module/          # Multimodal fusion
│   └── classifier/             # Depression classifier
├── src/
│   ├── data_processing/
│   │   ├── text_processor.py
│   │   ├── audio_processor.py
│   │   └── facial_processor.py
│   ├── feature_extraction/
│   │   ├── text_features.py
│   │   ├── audio_features.py
│   │   └── facial_features.py
│   ├── models/
│   │   ├── text_encoder.py
│   │   ├── audio_encoder.py
│   │   ├── facial_encoder.py
│   │   ├── fusion_strategies.py
│   │   └── depression_classifier.py
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── callbacks.py
│   └── inference/
│       ├── predictor.py
│       ├── preprocessing.py
│       └── postprocessing.py
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_evaluation.ipynb
├── configs/
│   ├── config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
├── tests/
│   ├── test_text_processor.py
│   ├── test_audio_processor.py
│   └── test_models.py
├── requirements.txt
├── setup.py
└── main.py
```

## 🎯 Key Features

### 1. **Multimodal Data Processing**
- **Text Processing**: NLP preprocessing, tokenization, sentiment analysis
- **Audio Processing**: Speech feature extraction (MFCCs, spectrograms, prosody)
- **Facial Processing**: Facial action units, expression recognition, gaze patterns

### 2. **Feature Extraction**
- **Text Features**: 
  - Linguistic features (vocabulary, syntax complexity)
  - Semantic features (word embeddings, topic modeling)
  - Sentiment features (polarity, subjectivity)
  - Psycholinguistic features (using LIWC categories)

- **Audio Features**:
  - MFCCs (Mel-Frequency Cepstral Coefficients)
  - Spectral features (energy, zero-crossing rate)
  - Prosodic features (pitch, duration, intensity)
  - Voice quality features (jitter, shimmer)

- **Facial Features**:
  - Facial Action Units (AUs)
  - Facial expressions (emotion classification)
  - Eye gaze patterns
  - Head pose estimation

### 3. **Deep Learning Models**
- Individual encoders for each modality
- Multi-layer fusion strategies (early, mid, late fusion)
- Attention mechanisms for modality weighting
- Transformer-based architectures

### 4. **Training & Evaluation**
- Cross-validation strategies
- Class imbalance handling
- Multiple evaluation metrics
- Interpretability analysis

## 🔧 Technologies & Libraries

### Core Libraries
```
torch>=1.9.0              # Deep learning framework
torchvision>=0.10.0       # Computer vision
torchaudio>=0.9.0         # Audio processing
transformers>=4.0.0       # Pre-trained models
scipy>=1.7.0              # Scientific computing
scikit-learn>=0.24.0      # Machine learning utilities
pandas>=1.3.0             # Data manipulation
numpy>=1.21.0             # Numerical computing
```

### Specialized Libraries
```
librosa>=0.9.0            # Audio analysis
opencv-python>=4.5.0      # Computer vision
mediapy>=0.4.0            # Media processing
openpyxl>=3.6.0           # Excel support
matplotlib>=3.4.0         # Visualization
seaborn>=0.11.0           # Statistical visualization
```

### Text Processing
```
nltk>=3.6.0               # Natural language toolkit
spacy>=3.0.0              # NLP pipeline
gensim>=4.0.0             # Topic modeling
sentencepiece>=0.1.96     # Tokenization
```

### Audio Processing
```
pyaudio>=0.2.11           # Audio I/O
wavfile>=0.1.0            # WAV file handling
soundfile>=0.10.0         # Sound file I/O
```

### Facial Recognition
```
dlib>=19.20.0             # Face detection & landmarks
face-recognition>=1.3.0   # Face recognition
opencv-contrib-python     # Additional CV tools
```

## 📈 Model Architecture

### Multimodal Fusion Framework

```
┌──────────────────────────────────────────────────────────────────────┐
│                   Input Data                             │
│          (Text, Audio, Video/Facial)                    │
└──────────────┬──────────────────────────────┬──────────────┬─────────┘
     │                      │                   │
     ▼                      ▼                   ▼
┌──────────────────────┐    ┌──────────────────────┐   ┌──────────────────────┐
│ Text Encoder │    │Audio Encoder │   │Facial Encoder│
│   (BERT)     │    │  (ResNet)    │   │  (ResNet)    │
└──────────────┬──────┘    └──────────────┬──────┘   └──────────────┬──────┘
       │                   │                   │
       ▼                   ▼                   ▼
   ┌────────────────────────────────────────────────────────────────┐
   │      Feature Extraction Layer      │
   │  (Dimensionality Reduction)        │
   └──────────────────────┬─────────────────────────────────────┘
                │
       ┌────────┴────────┬───────────────┬────────────────┐
       │ Early Fusion    │               │                │
       │ Mid Fusion      │               │                │
       │ Late Fusion     │               │                │
       └────────┬────────┴───────────────┴────────────────┘
                │
       ┌────────┴────────────────┐
       │  Attention      │
       │  Mechanism      │
       └────────┬────────────────┘
                │
       ┌────────┴────────────────┬────────────┐
       │ Fully Connected   │
       │ Classification   │
       └────────┬────────────────┘
                │
                ▼
       [Depression Score]
       [Risk Level]
       [Confidence]
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- ffmpeg (for media processing)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/depression-detection-multimodal.git
cd depression-detection-multimodal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python scripts/download_models.py
```

### Quick Start

```python
from src.inference.predictor import DepressionPredictor

# Initialize predictor
predictor = DepressionPredictor(model_path='models/checkpoint.pth')

# Predict from video
result = predictor.predict_from_video('path/to/video.mp4')
print(f"Depression Risk: {result['risk_level']}")
print(f"Score: {result['depression_score']:.3f}")
print(f"Confidence: {result['confidence']:.3f}")
```

## 📊 Datasets

### DAIC-WOZ (Depression and Anxiety Inventory Collection)
- **Size**: 1,189 interviews
- **Duration**: ~10 hours per interview
- **Labels**: PHQ-9 scores (0-27)
- **Modalities**: Audio, Video, Transcripts
- **Access**: Requires registration

### D-Vlog (Vlogs for Depression Detection)
- **Size**: 1,000+ vlogs
- **Duration**: Variable
- **Labels**: Binary (depressed/non-depressed)
- **Modalities**: Video, Audio
- **Public**: Available on request

### E-DAIC (Extended DAIC)
- **Size**: Extended version of DAIC-WOZ
- **Labels**: PHQ-9 scores
- **Modalities**: Audio, Video, Text
- **Access**: Limited

## 🏋️ Training

```bash
# Train the model
python src/training/train.py --config configs/training_config.yaml

# Evaluate on test set
python src/training/evaluate.py --model models/checkpoint.pth

# Inference on new data
python main.py --video path/to/video.mp4 --output results.json
```

## 📈 Performance Metrics

- **Accuracy**: Classification accuracy
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate
- **AUC-ROC**: Area under ROC curve
- **F1-Score**: Harmonic mean of precision and recall
- **MAE**: Mean absolute error (for depression scores)

## 🔍 Fusion Strategies

### Early Fusion
- Concatenate features from all modalities
- Feed to shared network
- **Pros**: Information interaction early
- **Cons**: Ignores modality-specific patterns

### Mid Fusion
- Extract modality-specific representations
- Fuse at intermediate layers
- Apply attention mechanisms
- **Pros**: Balance between early and late
- **Cons**: Higher complexity

### Late Fusion
- Train separate classifiers per modality
- Combine predictions (voting, weighted average)
- **Pros**: Modality independence
- **Cons**: Less feature interaction

## 🧠 Feature Extraction Details

### Text Features
```python
from src.feature_extraction.text_features import TextFeatureExtractor

extractor = TextFeatureExtractor()
features = extractor.extract(text)
# Returns: linguistic, semantic, sentiment, psycholinguistic features
```

### Audio Features
```python
from src.feature_extraction.audio_features import AudioFeatureExtractor

extractor = AudioFeatureExtractor()
features = extractor.extract_from_file('audio.wav')
# Returns: MFCCs, spectral, prosodic, voice quality features
```

### Facial Features
```python
from src.feature_extraction.facial_features import FacialFeatureExtractor

extractor = FacialFeatureExtractor()
features = extractor.extract_from_video('video.mp4')
# Returns: Action Units, expressions, gaze, head pose
```

## 🎯 Use Cases

1. **Clinical Screening**: Initial depression risk assessment
2. **Remote Monitoring**: Track depression symptoms over time
3. **Mental Health Apps**: Integrated mental health solutions
4. **Research**: Depression detection algorithm development
5. **Telehealth**: Support for remote therapy sessions

## ⚠️ Ethical Considerations

- **Privacy**: Secure handling of sensitive data
- **Bias**: Regular bias detection and mitigation
- **Fairness**: Equitable performance across demographics
- **Transparency**: Clear explanation of predictions
- **Not Diagnostic**: Model outputs are screening tools, not medical diagnoses

## 📚 References

### Key Papers
1. "Multimodal Machine Learning: A Survey and Taxonomy" - Baltrušaitis et al., 2017
2. "Detecting Depression in Twitter Posts" - De Choudhury et al., 2013
3. "Detecting Bipolar Disorder in Social Media" - Sap et al., 2018
4. "Depression Detection from Speech" - Nasir et al., 2016
5. "Facial Expression Analysis for Depression Detection" - Kaya et al., 2017

### Datasets
- DAIC-WOZ: https://dcapswoz.ict.usc.edu/
- D-Vlog: Contact authors
- E-DAIC: Limited access

## 🔗 Related Repositories

- [DepMamba](https://github.com/Jiaxin-Ye/DepMamba)
- [HiQuE](https://github.com/Juho-Jung/HiQuE)
- [CAF-Mamba](https://github.com/zbw-zhou/CAF-Mamba)

## 📝 Citation

If you use this project, please cite:

```bibtex
@software{depression_detection_2024,
  title={Depression Detection Using Multimodal Fusion},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/depression-detection-multimodal}
}
```

## 📧 Contact

For questions, issues, or collaborations, please open an issue on GitHub.

## 📄 License

MIT License - See LICENSE file for details.

## ⚖️ Disclaimer

This model is for research and screening purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult with qualified healthcare professionals for mental health concerns.

---

**Note**: This project addresses a sensitive topic. All development follows ethical guidelines and data privacy regulations (HIPAA, GDPR).
