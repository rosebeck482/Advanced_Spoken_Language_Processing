# Speech Analysis - Audio Feature Extraction for Emotion Recognition

This project extracts acoustic features from emotional speech recordings to analyze speech characteristics across different emotions. The script processes audio files and generates feature vectors for comparative analysis between personal recordings and MSP Podcast corpus samples.

## Project Overview

The `features_extract.py` script analyzes speech recordings and extracts 12 key acoustic features that are commonly used in emotion recognition:

### Extracted Features
1. **Pitch Features**: Min, Max, Mean, Standard Deviation
2. **Intensity Features**: Min, Max, Mean, Standard Deviation  
3. **Voice Quality**: Jitter, Shimmer, Harmonics-to-Noise Ratio (HNR)
4. **Temporal**: Speaking Rate (words per second)

### Emotions Analyzed
The script processes audio files for 7 basic emotions (matching  Podcast corpus format):
- Happy
- Angry  
- Sad
- Afraid
- Surprised
- Disgusted
- Neutral

## Dependencies

### Method 1: Using requirements.txt
```bash
pip install -r requirements.txt
```

### Method 2: Manual installation
```bash
pip install praat-parselmouth==0.4.5
```


## File Structure

```
HW1/
├── features_extract.py          # main extraction script
├── requirements.txt             # python dependencies
├── recordings/
│   ├── my_recordings/          # My emotion recordings (reading "My mama lives in Memphis")
│   │   ├── Happy.wav
│   │   ├── Angry.wav
│   │   ├── Sad.wav
│   │   ├── Afraid.wav
│   │   ├── Surprised.wav
│   │   ├── Disgusted.wav
│   │   └── Neutral.wav
│   └── msp_recordings/         # MSP Podcast corpus samples
│       ├── Happy.wav
│       ├── Angry.wav
│       ├── Sad.wav
│       ├── Afraid.wav
│       ├── Surprised.wav
│       ├── Disgusted.wav
│       └── Neutral.wav
├── my_features.csv             # output features from personal recordings
├── msp_features.csv            # output features from MSP Podcast samples
├── bonus.Manipulation          # bonus: Praat manipulation file
└── README.md                   # this file
```

## Usage

### Prerequisites
1. Ensure you have Python 3.x installed
2. Install dependencies: `pip install -r requirements.txt`
3. Place the audio files (.wav format) in the appropriate directories

### Running the Script
```bash
python features_extract.py
```

The script will:
1. Process all .wav files in `./recordings/my_recordings/`
2. Process all .wav files in `./recordings/msp_recordings/`
3. Generate `my_features.csv` with features from personal recordings
4. Generate `msp_features.csv` with features from msp samples

### Output Format
The generated CSV files follow the provided template format with the following columns:
- **Speech File**: emotion label extracted from filename (Happy, Angry, Sad, Afraid, Surprised, Disgusted, Neutral)
- **Min/Max/Mean/Sd Pitch**: pitch statistics in Hz (fundamental frequency measures)
- **Min/Max/Mean/Sd Intensity**: intensity statistics in dB (loudness measures)
- **Speaking Rate**: words per second (calculated as 5 words / audio duration)
- **Jitter**: pitch period variability measure (voice quality indicator)
- **Shimmer**: amplitude variability measure (voice quality indicator)
- **HNR**: harmonics-to-noise ratio in dB (voice quality measure)

## Technical Details

### Audio Processing
Feature extraction follows homework specifications using Praat algorithms via parselmouth.praat.call():
- **Pitch extraction**: Floor=75Hz, Ceiling=600Hz (avoids autocorrelation)
- **Intensity extraction**: Pitch floor=100Hz, uses 'energy' averaging method for mean intensity
- **Jitter**: Local jitter only, period floor=0.0001s, ceiling=0.02s, max period factor=1.3
- **Shimmer**: Local shimmer only, period floor=0.0001s, ceiling=0.02s, max period factor=1.3, max amplitude factor=1.6
- **HNR**: Harmonicity (cc) extraction with time step=0.01, min pitch=75Hz, silence threshold=0.1, periods per window=1.0

### Speaking Rate Calculation
- **Method**: #words/duration approximation as specified in homework
- **Transcript**: "My mama lives in Memphis." (5 words)
- **Calculation**: word_count / audio_duration = 5 / duration (words per second)
- **Implementation**: All recordings use the same sentence for consistent comparison

