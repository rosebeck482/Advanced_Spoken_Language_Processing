# Speech Emotion Recognition Project

## Overview

This project implements a **speech emotion recognition system** that analyzes emotional speech patterns using acoustic features. The system consists of two main components:

1. **Feature Extraction & Analysis** (`feature_extraction.ipynb`) - Extracts pitch and intensity features from speech segments using Praat-Parselmouth, applies speaker-specific normalization, and visualizes emotion-specific acoustic patterns across 15 emotion classes.

2. **Classification Experiments** (`classification.ipynb`) - Uses OpenSMILE's IS09 feature set with SVM classification and leave-one-speaker-out cross-validation to predict emotional states from speech recordings.

## Requirements

```
numpy>=1.7.0
pandas
matplotlib
seaborn
scikit-learn
torch
lightgbm
nltk
sentence-transformers
praat-parselmouth>=0.4.5
tqdm
optuna
scipy
```

## File Structure
```
HW3/
├── feature_extraction.ipynb       # Task 1: Feature Analysis - extracts and visualizes 6 pitch/intensity features
├── classification.ipynb           # Task 2: Classification Experiments - OpenSMILE features and SVM classification
├── data/                         # Dataset directory
│   ├── train.csv                 # Training dataset
│   ├── test.csv                  # Test dataset
│   └── valid.csv                 # Validation dataset
├── HW3-Instructions.pdf          # Assignment instructions and requirements
├── Report.pdf                    # Report documenting findings and analysis
└── README.md                     # This file
```

## Running the Code

### 1. Mount Google Drive
- Ensure `.wav` files are in: `/content/drive/MyDrive/hw3_speech_files/`

### 2. Clone and build OpenSMILE
```bash
!git clone https://github.com/audeering/opensmile.git
%cd opensmile
!bash build.sh
```

### 3. Install Praat-Parselmouth
```bash
pip install praat-parselmouth==0.4.5
```


## Feature Extraction & Analysis (`feature_extraction.ipynb`)

This notebook performs **acoustic feature extraction and analysis** on emotional speech data.

### Data Processing
- Processes **2,324 WAV audio files** containing emotional speech samples
- Analyzes data from **7 speakers** (cc, cl, gg, jg, mf, mk, mm) across **15 emotion classes**
- **Emotions**: anxiety, boredom, cold-anger, contempt, despair, disgust, elation, happy, hot-anger, interest, neutral, panic, pride, sadness, shame
- Uses only **left channel (channel 1)** for analysis as specified

### Feature Extraction
The notebook extracts **six acoustic features** from each speech segment:

#### Pitch Features
- **Minimum pitch** (Hz)
- **Maximum pitch** (Hz) 
- **Mean pitch** (Hz)
- **Praat settings**: pitch range 75-600 Hz, autocorrelation method

#### Intensity Features
- **Minimum intensity** (dB)
- **Maximum intensity** (dB)
- **Mean intensity** (dB)
- **Praat settings**: pitch floor 75 Hz, time step 0.0

### Normalization Method
Implements **Z-score normalization** per speaker to account for individual vocal characteristics:
1. Extract raw pitch/intensity values for all speech segments from each speaker
2. Calculate overall mean (μ) and standard deviation (σ) per speaker across all their segments
3. Normalize each value: `Z = (value - μ_speaker) / σ_speaker`
4. Calculate min, max, mean features from normalized arrays
5. Excludes zero and NaN values from calculations

### Technical Implementation
- Uses **Praat-Parselmouth** library for robust acoustic analysis
- Extracts features using `parselmouth.praat.call()` with specified Praat parameters
- Stores speaker-level statistics for normalization
- Generates both **raw and normalized feature sets**

### Visualization & Analysis
- Creates **12 plots total**: 6 features × 2 conditions (raw vs normalized)
- **Bar plots with error bars** showing emotion-level means and standard deviations
- **X-axis**: 15 emotion classes, **Y-axis**: feature values
- Compares patterns before and after speaker normalization
- Enables identification of emotion-specific acoustic patterns

### Output Files
- **Speaker statistics**: `/content/drive/MyDrive/hw3_outputs/speaker_statistics.csv`
- **Feature data** available for downstream classification tasks

### Dependencies
- `praat-parselmouth>=0.4.5` for acoustic feature extraction
- `pandas`, `numpy` for data processing
- `matplotlib`, `seaborn` for visualization
- `tqdm` for progress tracking


## Classification Experiments (`classification.ipynb`)

This notebook implements a **speech emotion recognition system**.

### Overview
Speech emotion recognition system classifying **15 different emotional states** from audio recordings using the **Interspeech 2009 Emotion Challenge (IS09)** feature set with **leave-one-speaker-out cross-validation**.

### Data Processing Pipeline
- Processes **2,324 audio samples** from **7 speakers** expressing **15 emotions**
- Extracts **384 acoustic features** using OpenSMILE's `IS09_emotion.conf` configuration
- Implements **speaker-specific Z-score normalization** to account for individual vocal characteristics
- Uses the command: `./build/progsrc/smilextract/SMILExtract -C config_path -I input_path -O output_path`

### Feature Engineering

#### 1. OpenSMILE Feature Extraction
- Uses `IS09_emotion.conf` to extract **384+ emotional features** from INTERSPEECH 2009 Emotion Challenge
- Saves per-file CSVs, then merges into: `/content/extracted/is09_all_speakers.csv`
   
#### 2. Data Preprocessing
- Removes constant/near-constant features: `frameTime`, `F0_sma_min`, `F0_sma_minPos`
- Drops columns with zeros across all 2,324 samples to ensure numerical stability
- Applies speaker-specific Z-score normalization: `Z = (value - speaker_mean) / speaker_std`
- Reduces dimensionality from **382 to 128 features** using Random Forest feature selection

#### 3. Label Construction
- Parses emotion and speaker information from filenames using regex
- Saves labels to: `/content/drive/MyDrive/hw3_outputs/labels.csv`

### Classification Approach
- **Model**: Support Vector Machine (SVM) with RBF kernel (`C=10`, `gamma='scale'`)
- **Feature Selection**: Random Forest with 100 estimators (importance threshold filtering)
- **Cross-Validation**: Leave-one-speaker-out cross-validation (**7 experiments total**)
- **Class Balancing**: Inverse class frequency weighting to handle emotion class imbalance
- **Evaluation**: sklearn classification reports for each speaker + aggregated metrics

### Performance Results
- **Overall aggregated accuracy**: 26.1%
- **Overall weighted F1-score**: 25.7% 
- **Best performing speaker (mm)**: 30.5% accuracy, 30.3% F1-score
- **Strong performance** on high-arousal emotions: hot anger (F1=0.621), happy (F1=0.429)
- **Challenges** with acoustically similar emotions (elation/happy confusion, disgust/contempt overlap)

### Error Analysis Findings
- **Hot anger, happy, and anxiety** achieved highest classification success due to distinctive acoustic signatures
- **Elation, disgust, and despair** proved most challenging due to acoustic similarity with other emotions
- **Class imbalance issues** particularly affected neutral emotion classification (only 9 samples for speaker mm)
- **Confusion patterns** show difficulty distinguishing between emotions with similar arousal levels

### Model Justification
- **Random Forest feature selector** captures complex interactions between acoustic parameters
- **SVM with RBF kernel** creates non-linear decision boundaries for overlapping emotion categories  
- **Minimal hyperparameter tuning** to avoid overfitting (`C=10`, `gamma='scale'`)
- **Class weighting** addresses inherent imbalance in emotional speech data

### Technical Requirements
- **OpenSMILE toolkit** for IS09 feature extraction
- **scikit-learn** for classification, cross-validation, and evaluation
- **pandas, numpy** for data processing and manipulation
- **seaborn, matplotlib** for confusion matrix visualization

### Output Files
- **Feature matrix**: `/content/extracted/is09_all_speakers.csv`
- **Labels**: `/content/drive/MyDrive/hw3_outputs/labels.csv`
- **Individual speaker classification reports** (7 total)
- **Aggregated performance metrics** across all experiments

