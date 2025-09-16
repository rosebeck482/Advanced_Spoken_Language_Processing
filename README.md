# Advanced Spoken Language Processing - Spring 2025

This repository contains coursework for Advanced Spoken Language Processing, focusing on speech analysis, dialogue processing, and emotion recognition using machine learning techniques.

## Project Overview

| Assignment | Topic | Key Technologies | Best Performance |
|------------|-------|------------------|------------------|
| **HW1** | Speech Feature Extraction | Praat-Parselmouth, Python | 12 acoustic features across 7 emotions |
| **HW2** | Dialogue Act Recognition | Multimodal ML, Transformers, SVM | 68% macro F1-score |
| **HW3** | Speech Emotion Recognition | OpenSMILE IS09, SVM | 26.1% accuracy, 25.7% F1 (15 emotions) |

---

## HW1: Speech Analysis - Audio Feature Extraction for Emotion Recognition

**Objective**: Analyze emotional speech patterns through acoustic feature extraction

### Key Features:
- **12 acoustic features**: Pitch statistics, intensity measures, voice quality metrics, speaking rate
- **7 emotion categories**: Happy, Angry, Sad, Afraid, Surprised, Disgusted, Neutral
- **Comparative analysis**: Personal recordings vs MSP Podcast corpus
- **Technology**: Praat-Parselmouth library for acoustic analysis

### Main Files:
- `features_extract.py` - Core extraction script
- `my_features.csv` / `msp_features.csv` - Output feature datasets

---

## HW2: Dialogue Act Recognition

**Objective**: Classify conversational utterances into dialogue acts using multimodal features

### Key Features:
- **10 dialogue act categories** from Switchboard corpus
- **Multimodal approach**: Text + speech features
- **Advanced models**: CNN-Transformer, LinearSVC, LightGBM ensemble
- **Feature sets**: LIWC psycholinguistic features, TF-IDF, sentence embeddings

### Performance Results:
- **Text-only model**: 68% macro F1-score (best performer)
- **Speech-only model**: 27% macro F1-score  
- **Multimodal model**: 67% macro F1-score

### Main Files:
- `features.ipynb` - Feature extraction and analysis
- `classification.ipynb` - Model training and evaluation

---

## HW3: Speech Emotion Recognition

**Objective**: Recognize emotional states from speech using acoustic features and machine learning

### Key Features:
- **15 emotion classes**: Including anxiety, boredom, anger variants, happiness, sadness
- **382 IS09 features**: Reduced to 128 using Random Forest feature selection
- **Speaker normalization**: Z-score normalization for individual vocal characteristics
- **Cross-validation**: Leave-one-speaker-out methodology

### Performance Results:
- **Overall accuracy**: 26.1% 
- **Overall weighted F1-score**: 25.7%
- **Best emotions**: Hot anger (F1=0.621), Happy (F1=0.429)
- **Dataset**: 2,324 samples from 7 speakers

### Main Files:
- `feature_extraction.ipynb` - Acoustic feature analysis
- `classification.ipynb` - SVM classification experiments

---

## Technologies Used

- **Audio Processing**: Praat-Parselmouth, OpenSMILE
- **Machine Learning**: scikit-learn, PyTorch, LightGBM
- **Feature Engineering**: LIWC, TF-IDF, Sentence Transformers
- **Languages**: Python, Jupyter Notebooks
