# Dialogue Act Recognition

## Overview

This project implements a multimodal machine learning system for automatic dialogue act recognition (DAR) using both text and speech features. The system classifies conversational utterances into 10 different dialogue act categories such as statements, questions, backchannels, and agreements. 

The project consists of two main components:
1. **Feature extraction** (`features.ipynb`) - from text transcripts and audio files to create comprehensive linguistic and acoustic representations
2. **Classification models** (`classification.ipynb`) - including speech-only, text-only, and combined multimodal approaches to predict dialogue acts

The implementation demonstrates the effectiveness of different feature types and model architectures, with detailed analysis of class-level performance and confusion patterns between similar dialogue acts.

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

## Directory Structure

```
Dialogue Act Recognition/
├── data/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── results/
│   ├── test_sb4446_multi.csv
│   ├── test_sb4446_speech.csv
│   └── test_sb4446_text.csv
├── features.ipynb
├── classification.ipynb
├── HW2_Instructions.pdf
├── HW2_responses.pdf
└── README.md
```

## File Description

- `features.ipynb` - Feature extraction and analysis script
- `classification.ipynb` - Model training and evaluation script
- `data/` - Raw dataset files
- `results/` - Model prediction outputs
- `text_features_{train,valid,test}.csv` - Text-based feature files (not included - too large)
- `{train,valid,test}_speech_features.csv` - Speech-based feature files (not included - too large)

## Running the Code

1. **First run `features.ipynb`** to extract features from the raw data
   - Raw files are located in the `data/` directory (`train.csv`, `test.csv`, `valid.csv`)
   - This will generate `text_features_*.csv` and `*_speech_features.csv` files
   - Make sure WAV files are in `/content/drive/MyDrive/Colab Notebooks/wav/`

2. **Then run `classification.ipynb`** to train models and generate predictions
   - This will load the feature files and train three models:
     - Speech-only model (CNN + Transformer)
     - Text-only model (LinearSVC + LightGBM ensemble)
     - Combined multimodal model (stacked ensemble)
   - Predictions will be saved as `test_sb4446_*.csv` files in the `results/` directory

## Feature Extraction and Analysis (`features.ipynb`)

This section describes the feature extraction and exploratory data analysis implemented in the `features.ipynb` notebook using the Switchboard corpus.

### Dataset Overview
- **Training set**: 10,140 utterances with 79 features
- **Validation set**: 19,156 utterances with 79 features  
- **Test set**: 23,540 utterances with 79 features

Each utterance includes dialogue ID, speaker information, transcript text, dialogue act tags, timing information, and 69 pre-existing LIWC features.

### Dialogue Act Categories (10 types)
- `%`: Abandoned/incomplete utterances
- `aa`: Agree/accept responses  
- `b`: Acknowledge/backchannel responses
- `ba`: Appreciation/assessment responses
- `fc`: Conventional closing phrases
- `ny`: Yes answers
- `qy`: Yes-no questions
- `sd`: Statement-declaration
- `sv`: Statement-opinion  
- `x`: Non-verbal sounds (silence, laughter, etc.)

### 1. Text-Based Features (3,841 dimensions)

#### Feature Set Components:
- **LIWC features**: 69 pre-existing psycholinguistic features (pronouns, emotions, temporal references)
- **Basic metrics**: 4 features - `word_count`, `character_count`, punctuation presence (`has_question_mark`, `has_exclamation_mark`)
- **Sentence embeddings**: 768-dimensional vectors using `sentence-transformers/all-mpnet-base-v2`
- **TF-IDF n-grams**: 3,000 features (1-3 grams) capturing lexical patterns

#### Rationale:
- **LIWC features** provide comprehensive linguistic indicators including structural features (pronouns like "I" and "we"), emotional markers ("anger," "sad," "posemo," "negemo"), and temporal references ("focuspast," "focuspresent," "focusfuture")
- **Word and character counts** capture significant length variations across dialogue acts (e.g., opinion statements average 12 words vs acknowledgments averaging 2 words)
- **Punctuation features** provide clear indicators (question marks exclusive to yes-no questions, exclamation marks primarily in appreciations)
- **Sentence transformer embeddings** capture contextual meaning essential for dialogue act classification
- **TF-IDF n-grams** emphasize class-specific phrases (e.g., "do you," "have you" for questions; "oh," "wow," "good" for appreciations)

### 2. Speech-Based Features (18 dimensions)

#### Feature Set Components:
- **Pitch statistics**: `Pitch_Min`, `Pitch_Max`, `Pitch_Mean`, `Pitch_SD` (75-600 Hz range)
- **Intensity measures**: `Intensity_Min`, `Intensity_Max`, `Intensity_Mean`, `Intensity_SD` (measured in dB)
- **Voice quality metrics**: 
  - `Jitter`: Pitch perturbation (voice stability)
  - `Shimmer`: Amplitude perturbation  
  - `HNR`: Harmonics-to-Noise Ratio (voice quality indicator)
- **Speaking_Rate**: Syllables per second using intensity variation-based syllable nuclei detection (1.0-8.0 range)

#### Rationale:
- **Pitch features** capture prosodic variations characterizing different dialogue acts (rising pitch for questions, flatter patterns for statements)
- **Intensity variations** capture emphasis and emotional engagement in dialogue acts
- **Jitter and shimmer** detect voice quality characteristics like breathiness and tension
- **HNR** distinguishes clear, harmonic speech from noisier productions
- **Speaking rate** differentiates between slower, thoughtful statements and faster, more assertive utterances

#### Technical Implementation:
- **Audio processing**: Multiprocessing for efficient extraction (~43 minutes for 74,111 training utterances)
- **Library**: Parselmouth (Praat-based) for robust acoustic analysis
- **Duration handling**: Minimum 64ms segments enforced for reliable analysis
- **Speaking rate**: Adapted syllable nuclei detection using intensity variations
- **Output**: `[split]_speech_features.csv` files with 18 acoustic dimensions

### Exploratory Data Analysis

#### Most Frequent Words by Dialogue Act
- **Backchannels (b, aa, ny)**: "yeah", "huh", "right", "okay"  
- **Questions (qy)**: "do you", "have you", interrogative patterns
- **Statements (sd, sv)**: "know", "think", "like", "really"
- **Closings (fc)**: "bye", "talking", "good"
- **Appreciations (ba)**: "oh", "wow", "good", "great"

#### Length Distribution Patterns
- **Informative acts**: Opinion statements (`sv`) average 11.55 words, declarations (`sd`) average 10.99 words
- **Acknowledgments**: Backchannels (`b`) average 1.07 words, agreements (`aa`) average 1.65 words  
- **Length-based classification**: 89.6% accuracy using 5-word threshold distinguishing informative vs. acknowledgment acts

## Feature Analysis

### Text Feature Analysis

#### 1. Length-based Hypothesis
**Hypothesis**: Informative acts ('sv', 'sd') contain more words than acknowledgment acts ('b', 'aa')

**Results**:
- Informative acts averaged ~11 words vs acknowledgment acts averaged ~1 word
- **Statistical significance**: Mann-Whitney U test (U = 336,073,025.5, p < 0.001, d = 1.58)
- **Classification accuracy** using 5-word threshold: 89.6%

#### 2. LIWC Feature Hypothesis
**Hypothesis**: Backchannel acts show more "informal" and "assent" tokens, while informative acts show more function words and verbs

**Results**:
- Backchannels averaged 0.78 for informal and 0.71 for assent vs ~0.04 and ~0.00 for informative acts
- Informative acts averaged 0.62 function words and 0.23 verbs vs ~0.07 and 0.03 for backchannels
- **All differences highly significant** (p < 0.001) with large effect sizes (d > 1.6)

### Speech Feature Analysis

#### 1. Pitch Hypothesis
**Hypothesis**: Yes-no questions ('qy') have higher pitch than statements ('sd', 'sv')

**Results**:
- Mean pitch significantly higher in 'qy' (~180 Hz) vs statements (~166 Hz)
- **Statistical significance**: U = 18,837,546, p < 0.001, d = 0.26

#### 2. Intensity Hypothesis
**Hypothesis**: Appreciation acts ('ba') are louder than ordinary backchannels ('b', 'aa')

**Results**:
- 'ba' peaks at ~66 dB vs 64 dB for ordinary backchannels
- **Statistical significance**: U = 8,075,216, p < 0.001, d = 0.28

#### 3. Speaking Rate Hypothesis
**Hypothesis**: Content-heavy acts ('sv', 'sd') spoken more slowly than backchannel acts ('b', 'aa')

**Results**:
- Informative statements averaged 3.7 syllables/second vs backchannels at 1.17 syllables/second
- **Statistical significance**: U = 907,432,394, p < 0.001, d = 1.67

#### 4. Additional Acoustic Findings
- **Closings (fc) vs Statements (sd)**: Closings show higher pitch (206.2 Hz vs 166.8 Hz, p < 0.001)
- **Average speaking rate**: ~6.7 syllables/second across all dataset splits
- **Processing efficiency**: Parallel extraction handles variable-length utterances with robust error handling

## Classification Models (`classification.ipynb`)

This section describes the comprehensive multi-modal machine learning approach implemented in the `classification.ipynb` notebook using the Switchboard Dialog Act (SwDA) corpus for predicting the 10 most frequent dialogue acts.

### Data Processing Pipeline
- **Feature loading**: Pre-extracted speech and text features from CSV files
- **Missing value handling**: Mean imputation for speech features
- **Normalization**: Standard scaling for feature distribution normalization
- **Input features**:
  - Speech: Acoustic features (pitch, intensity, jitter, shimmer, HNR, speaking rate)
  - Text: LIWC psycholinguistic features, TF-IDF vectors (3,000 dims), sentence embeddings (768 dims)

### Technical Stack
- **Deep Learning**: PyTorch (CNN-Transformer for speech modeling)
- **Traditional ML**: scikit-learn (LinearSVC, LogisticRegression), LightGBM
- **Optimization**: Optuna hyperparameter tuning (30 trials)
- **Data Processing**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn for confusion matrices and analysis

### Model Architectures

#### 1. Model 1: Speech-Only Classifier (CNN + Transformer + SoftMax)
**Architecture**:
- **1D CNN**: Processes speech features and extracts acoustic patterns
- **2-layer Transformer**: Encoder layers with multiple attention heads capturing contextual relationships
- **Temporal modeling**: Contextual windows of 6 consecutive utterances for sequence understanding
- **Components**: Learnable positional encodings and classification head (layer normalization + linear layer)

**Hyperparameters** (optimized using Optuna - 30 trials):
- Hidden dimension: 128
- Attention heads: 4
- Learning rate: ~0.00075
- Training: 25 epochs with early stopping

**Performance**:
- Macro F1 score: ~27% (computed via leave-one-out evaluation)
- Accuracy: ~67% (varies by experiment)

#### 2. Model 2: Text-Only Classifier (Stacked Ensemble)
**Architecture**:
- **LinearSVC**: Processes TF-IDF features (3,000 dims) with CalibratedClassifierCV for probability estimates
- **LightGBM**: Handles sentence embeddings (768-dimensional) and LIWC psycholinguistic features
- **Ensemble strategy**: 50/50 probability weighting between models for final prediction

**Configuration**:
- LinearSVC: Balanced class weights for class imbalance handling
- LightGBM: 400 estimators, learning rate 0.05
- Final prediction: Class with highest combined probability

**Performance**:
- Macro F1 score: ~68% (best performing model)
- Accuracy: ~86% (varies by experiment)

#### 3. Model 3: Multi-Modal Stacking Classifier
**Architecture**:
- **Base models**: CNN-Transformer (speech), LinearSVC (TF-IDF), LightGBM (embeddings + LIWC)
- **Meta-learner**: Logistic Regression combining predictions from all base models
- **Training strategy**: 5-fold cross-validation for unbiased out-of-fold predictions to prevent overfitting

**Performance**:
- Macro F1 score: ~67% (competitive multimodal performance)
- Demonstrates model diversity through stacking approach

### Evaluation & Analysis

#### Comprehensive Assessment
- **Classification reports**: Per-class precision, recall, and F1 scores for detailed performance analysis
- **Confusion matrix visualizations**: Raw counts, row-normalized, and column-normalized matrices
- **Error analysis**: Identification of most confused class pairs and challenging dialogue acts
- **Test predictions**: Generated for all three model configurations

#### Best Performing Model
The **text-only model (Model 2)** achieved the best performance (~68% macro F1) due to:
- Rich linguistic information from LIWC psycholinguistic features, sentence embeddings, and TF-IDF vectors
- Effective ensemble approach combining LinearSVC (lexical patterns) and LightGBM (semantic + psychological features)
- Superior handling of syntax, word choice, and contextual meaning

#### Model Performance Comparison
- **Speech model (Model 1)**: Limited by less diverse acoustic features and missing value imputation noise
- **Multi-modal model (Model 3)**: Shows promise but requires careful calibration; potential dilution of strong text signals
- **Key insight**: Text features significantly outperform speech features alone for dialogue act classification

### Class-Level Performance Analysis

#### Easiest to Predict Classes
1. **'x' (non-verbal)**: Perfect performance (F1 = 1.00)
   - Distinctive linguistic features
   - Largest dataset representation (7,174 samples)
   - Highly consistent and unique patterns

2. **'b' (backchannel)**: Strong performance (F1 = 0.85)
   - High recall (0.94), good precision (0.78)
   - Substantial representation (2,409 samples)
   - Distinctive short affirming phrase patterns

3. **'sd' (statement-non-opinion)**: Strong performance (F1 = 0.85)
   - High recall (0.89), good precision (0.80)
   - Large sample size (5,122 instances)
   - Clear syntactic structures

#### Most Difficult to Predict Classes
1. **'ny' (yes-answers)**: Lowest performance (F1 = 0.12)
   - Extremely low recall (0.06), moderate precision (0.76)
   - Small representation (206 samples)
   - 71.36% misclassified as 'b' (backchannel)

2. **'aa' (agree/accept)**: Poor performance (F1 = 0.43)
   - Low recall (0.34), moderate precision (0.58)
   - 53.61% misclassified as 'b' (backchannel)

3. **'sv' (statement-opinion)**: Moderate performance (F1 = 0.59)
   - Recall (0.52), precision (0.67)
   - 43% misclassified as 'sd' (statement-non-opinion)

#### Easily Confused Class Pairs
1. **'sv' ↔ 'sd'**: 759 instances (43% of 'sv') confused
   - Similar syntactic structures
   - Requires deeper semantic understanding for opinion vs. fact distinction

2. **'aa' ↔ 'b'**: 356 instances (53.61% of 'aa') confused
   - Similar short affirmative responses
   - Distinction depends on conversational context rather than lexical form

3. **'ny' ↔ 'b'**: 147 instances (71.36% of 'ny') confused
   - Both involve short responses with similar structures

4. **'qy' ↔ 'sd'**: 101 instances (29.53% of 'qy') confused
   - Questions can appear similar to statements without prosodic markers

### Potential Improvements

#### Feature Engineering
- Specialized features targeting problematic class distinctions
- Conversational context inclusion (previous/subsequent utterances)
- Syntactic dependency analysis for question/statement differentiation

#### Model Architecture
- Specialized binary classifiers for commonly confused pairs
- Adjusted ensemble weights emphasizing better-performing models on difficult classes
- Sequence models for longer-range dependency capture

#### Data Handling
- Class weighting or oversampling for underrepresented classes
- Better handling of class imbalance, especially for 'ny' class

### File Outputs
The notebook generates three prediction files in the `results/` directory:
- `test_sb4446_speech.csv`: Speech-only model (Model 1) predictions
- `test_sb4446_text.csv`: Text-only model (Model 2) predictions  
- `test_sb4446_multi.csv`: Multi-modal stacked model (Model 3) predictions
