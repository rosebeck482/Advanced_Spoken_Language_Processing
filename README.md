# Advanced Spoken Language Processing - Spring 2025

This repository contains coursework for Advanced Spoken Language Processing, focusing on speech analysis, dialogue processing, and emotion recognition using machine learning techniques.


## Project Overview

| Project | Topic | Key Technologies | Best Performance |
|------------|-------|------------------|------------------|
| [**Speech Feature Extraction**](./Speech%20Feature%20Extraction/) | Acoustic Feature Analysis | Praat-Parselmouth, Statistical Analysis | 12 acoustic features across 7 emotions |
| [**Dialogue Act Recognition**](./Dialogue%20Act%20Recognition/) | Multimodal Conversation Analysis | CNN-Transformer, LinearSVC, LightGBM | 68% macro F1-score (text-only) |
| [**Speech Emotion Recognition**](./Speech%20Emotion%20Recognition/) | Emotion Classification from Speech | OpenSMILE IS09, SVM, Random Forest | ~26% accuracy (15 emotion classes) |

---

## Speech Feature Extraction: Acoustic Analysis for Emotion Recognition

**Objective**: Systematic extraction and analysis of acoustic features from emotional speech recordings to understand how emotions manifest in speech characteristics.

### Technical Approach:
- **Feature Set**: 12 comprehensive acoustic features extracted using Praat algorithms
  - **Pitch**: Min, Max, Mean, Standard Deviation (75-600 Hz range)
  - **Intensity**: Min, Max, Mean, Standard Deviation (dB measurements)
  - **Voice Quality**: Jitter, Shimmer, Harmonics-to-Noise Ratio (HNR)
  - **Temporal**: Speaking Rate (words per second calculation)

### Dataset & Methodology:
- **7 emotion categories**: Happy, Angry, Sad, Afraid, Surprised, Disgusted, Neutral
- **Comparative analysis**: Personal recordings vs MSP Podcast corpus samples
- **Standardized protocol**: Same transcript ("My mama lives in Memphis") for all recordings
- **Technical implementation**: Praat-Parselmouth library with specified parameter settings

### Key Insights:
- Demonstrates systematic differences in acoustic characteristics across emotions
- Provides baseline understanding of emotion-speech relationships
- Establishes feature extraction pipeline for emotion recognition tasks

### Deliverables:
- `features_extract.py` - Automated feature extraction pipeline
- `my_features.csv` / `msp_features.csv` - Comparative feature datasets
- `bonus.Manipulation` - Praat manipulation demonstration

---

## Dialogue Act Recognition: Multimodal Conversation Understanding

**Objective**: Develop an intelligent system to automatically classify conversational utterances into functional dialogue acts, enabling machines to understand the communicative intent behind spoken language.

### Technical Innovation:
- **Multimodal Architecture**: Combines both textual and acoustic information
  - **Text Features** (3,841 dimensions): LIWC psycholinguistic features, TF-IDF n-grams, sentence embeddings
  - **Speech Features** (18 dimensions): Pitch, intensity, voice quality, and speaking rate measures
- **Advanced ML Pipeline**: Three-model ensemble approach with sophisticated stacking

### Dataset & Challenge:
- **Switchboard Corpus**: Large-scale conversational telephone speech dataset
- **10 dialogue act categories**: Statements, questions, backchannels, agreements, etc.
- **Class imbalance**: Ranging from 23,540 test samples to handle diverse conversation patterns
- **Real-world complexity**: Natural, spontaneous conversation analysis

### Model Architecture:
1. **Speech-Only Model**: CNN-Transformer with contextual utterance windows
2. **Text-Only Model**: LinearSVC + LightGBM ensemble with probability stacking  
3. **Multimodal Model**: Meta-learner combining predictions from specialized models

### Performance Insights:
- **Text dominance**: Text-only model achieves 68% macro F1-score (best performance)
- **Speech limitations**: Acoustic features alone reach 27% macro F1-score
- **Multimodal potential**: Combined approach shows 67% performance with room for improvement
- **Class-specific patterns**: High performance on distinctive acts (non-verbal: F1=1.00), challenges with similar acts (yes-answers vs backchannels)

### Key Contributions:
- Comprehensive feature engineering pipeline combining linguistic and acoustic modalities
- Systematic evaluation of unimodal vs multimodal approaches
- Detailed error analysis revealing confusion patterns between similar dialogue acts
- Production-ready classification system with practical conversation understanding capabilities

### Deliverables:
- `features.ipynb` - Comprehensive feature extraction and exploratory data analysis
- `classification.ipynb` - Complete model training, evaluation, and prediction pipeline
- `results/` - Model predictions for speech-only, text-only, and multimodal approaches

---

## Speech Emotion Recognition: Advanced Acoustic-Based Emotion Classification

**Objective**: Develop a robust machine learning system capable of recognizing complex emotional states from speech signals, addressing the challenging problem of fine-grained emotion classification in real-world scenarios.

### Technical Sophistication:
- **Dual-Pipeline Architecture**: 
  - **Custom Feature Analysis**: 12 Praat-based features with speaker-specific Z-score normalization
  - **Industrial-Grade Features**: 382 OpenSMILE IS09 features from INTERSPEECH 2009 Emotion Challenge
- **Advanced Preprocessing**: Speaker-specific normalization to handle individual vocal characteristics
- **Intelligent Feature Selection**: Random Forest-based reduction from 382 to 128 most informative features

### Dataset Complexity:
- **Emotional Granularity**: 15 distinct emotion classes including anxiety, boredom, cold/hot anger, contempt, despair, disgust, elation, happiness, interest, neutral, panic, pride, sadness, shame
- **Speaker Diversity**: 7 different speakers (cc, cl, gg, jg, mf, mk, mm) providing 2,324 total samples
- **Methodological Rigor**: Leave-one-speaker-out cross-validation for unbiased evaluation

### Machine Learning Innovation:
- **SVM Classification**: RBF kernel with optimized hyperparameters (C=10, gamma='scale')
- **Class Balancing**: Inverse frequency weighting to handle emotion class imbalance
- **Robust Evaluation**: Comprehensive per-speaker and aggregated performance metrics

### Performance Analysis:
- **Overall Performance**: ~26% accuracy and ~26% weighted F1-score across 15 emotion classes
- **High-Performance Emotions**: Hot anger (F1=0.621) and happiness (F1=0.429) show distinctive acoustic signatures
- **Classification Challenges**: Acoustically similar emotions (elation/happiness, disgust/contempt) present ongoing research challenges
- **Baseline Significance**: Performance demonstrates feasibility while highlighting complexity of fine-grained emotion recognition

### Research Contributions:
- **Comprehensive Feature Analysis**: Systematic comparison of raw vs normalized acoustic features across emotion classes
- **Speaker Independence**: Rigorous evaluation methodology ensuring generalizability across speakers
- **Error Pattern Analysis**: Detailed confusion analysis revealing acoustic similarities between emotion categories
- **Industrial Relevance**: Implementation using established feature sets (IS09) for reproducibility and comparison

### Deliverables:
- `feature_extraction.ipynb` - Comprehensive acoustic feature analysis with visualization
- `classification.ipynb` - Complete SVM-based emotion recognition pipeline
- `data/` - Structured training, validation, and test datasets

---

## Technical Stack & Research Impact

### Core Technologies
- **Audio Processing & Analysis**:
  - **Praat-Parselmouth**: Robust acoustic feature extraction with precise parameter control
  - **OpenSMILE**: Industry-standard feature extraction (IS09 Emotion Challenge configuration)
  
- **Machine Learning & Deep Learning**:
  - **scikit-learn**: Classical ML algorithms (SVM, Random Forest, LinearSVC) with comprehensive evaluation
  - **PyTorch**: Deep learning frameworks for CNN-Transformer architectures
  - **LightGBM**: Gradient boosting for high-dimensional feature processing
  - **Optuna**: Automated hyperparameter optimization for model tuning

- **Natural Language Processing**:
  - **LIWC**: Psycholinguistic feature analysis for dialogue understanding
  - **TF-IDF**: N-gram based lexical pattern recognition
  - **Sentence Transformers**: Contextual embeddings for semantic understanding

- **Data Science & Visualization**:
  - **pandas/numpy**: Efficient data manipulation and numerical computing
  - **matplotlib/seaborn**: Statistical visualization and result presentation
  - **scipy**: Statistical testing and signal processing

### Research Methodology
- **Cross-Validation**: Leave-one-speaker-out methodology ensuring speaker independence
- **Statistical Analysis**: Mann-Whitney U tests, Cohen's d effect sizes for hypothesis testing  
- **Feature Engineering**: Systematic dimensionality reduction and normalization techniques
- **Error Analysis**: Confusion matrix analysis and class-specific performance evaluation

### Practical Applications
This work provides foundational techniques applicable to:
- **Human-Computer Interaction**: Emotion-aware interfaces and conversational agents
- **Healthcare**: Speech-based mental health monitoring and assessment
- **Customer Service**: Automated sentiment analysis and conversation understanding
- **Education**: Adaptive learning systems responsive to student emotional states
- **Entertainment**: Emotion-responsive gaming and interactive media systems

### Development Environment
- **Platform**: Google Colab for scalable computation and GPU acceleration
- **Language**: Python 3.x with comprehensive scientific computing ecosystem
- **Version Control**: Structured project organization with clear documentation and reproducible results
