import parselmouth
from parselmouth.praat import call
import glob
import os
import csv
from typing import Dict, List, Optional

# extraction parameters
PITCH_FLOOR: float = 75.0
PITCH_CEILING: float = 600.0
INTENSITY_PITCH_FLOOR: float = 100.0
JITTER_PERIOD_FLOOR: float = 0.0001
JITTER_PERIOD_CEILING: float = 0.02
JITTER_MAX_PERIOD_FACTOR: float = 1.3
SHIMMER_PERIOD_FLOOR: float = 0.0001
SHIMMER_PERIOD_CEILING: float = 0.02
SHIMMER_MAX_PERIOD_FACTOR: float = 1.3
SHIMMER_MAX_AMPLITUDE_FACTOR: float = 1.6
HNR_TIME_STEP: float = 0.01
HNR_MIN_PITCH: float = 75.0
HNR_SILENCE_THRESHOLD: float = 0.1
HNR_PERIODS_PER_WINDOW: float = 1.0

def extract(sound_path: str, output_file: str, is_my_speech: bool = True) -> None:
    """extract acoustic features from emotional speech recordings.
    
    args:
        sound_path: path to directory containing .wav files or single .wav file
        output_file: path for output csv file
        is_my_speech: flag to indicate if processing personal recordings
    """
    results: Dict[str, Dict[str, float]] = {}
    
    # validate input path
    if not os.path.exists(sound_path):
        print(f"error: path does not exist: {sound_path}")
        return
    
    # get list of wav files to process
    if os.path.isdir(sound_path):
        sound_files: List[str] = glob.glob(os.path.join(sound_path, "*.wav"))
        if not sound_files:
            print(f"error: no .wav files found in {sound_path}")
            return
    else:
        sound_files = [sound_path]
    
    for sound_file in sound_files:
        try:
            # emotion name extracted from file name
            emotion: str = os.path.basename(sound_file).split('.')[0]
            
            # load sound file
            sound = parselmouth.Sound(sound_file)
            
            # extract features using praat algorithms
            feature_dict: Dict[str, float] = {}
            
            # extract pitch features (fundamental frequency)
            pitch = call(sound, "To Pitch", 0.0, PITCH_FLOOR, PITCH_CEILING)
            feature_dict["Min Pitch"] = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
            feature_dict["Max Pitch"] = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
            feature_dict["Mean Pitch"] = call(pitch, "Get mean", 0, 0, "Hertz")
            feature_dict["Sd Pitch"] = call(pitch, "Get standard deviation", 0, 0, "Hertz")
            
            # extract intensity features (loudness)
            intensity = call(sound, "To Intensity", INTENSITY_PITCH_FLOOR, 0, "yes")
            feature_dict["Min Intensity"] = call(intensity, "Get minimum", 0, 0, "Parabolic")
            feature_dict["Max Intensity"] = call(intensity, "Get maximum", 0, 0, "Parabolic")
            feature_dict["Mean Intensity"] = call(intensity, "Get mean", 0, 0, "energy")
            feature_dict["Sd Intensity"] = call(intensity, "Get standard deviation", 0, 0)
            
            # extract voice quality features (jitter, shimmer)
            pitch_for_quality = call(sound, "To Pitch", 0.0, PITCH_FLOOR, PITCH_CEILING)
            point_process = call([sound, pitch_for_quality], "To PointProcess (cc)")
            
            # jitter - pitch period variability
            feature_dict["Jitter"] = call(point_process, "Get jitter (local)", 0, 0, 
                                        JITTER_PERIOD_FLOOR, JITTER_PERIOD_CEILING, JITTER_MAX_PERIOD_FACTOR)
            
            # shimmer - amplitude variability
            feature_dict["Shimmer"] = call([sound, point_process], "Get shimmer (local)", 0, 0, 
                                         SHIMMER_PERIOD_FLOOR, SHIMMER_PERIOD_CEILING, 
                                         SHIMMER_MAX_PERIOD_FACTOR, SHIMMER_MAX_AMPLITUDE_FACTOR)
            
            # hnr - harmonics-to-noise ratio
            harmonicity = call(sound, "To Harmonicity (cc)", HNR_TIME_STEP, HNR_MIN_PITCH, 
                             HNR_SILENCE_THRESHOLD, HNR_PERIODS_PER_WINDOW)
            feature_dict["HNR"] = call(harmonicity, "Get mean", 0, 0)
            
            # speaking rate calculation
            transcript: str = "My mama lives in Memphis."
            word_count: int = len(transcript.split())
            duration: float = call(sound, "Get total duration")
            feature_dict["Speaking Rate"] = word_count / duration
            
            # store results
            results[emotion] = feature_dict
            print(f"processed {emotion}")
            
        except FileNotFoundError:
            print(f"error: file not found: {sound_file}")
        except Exception as e:
            print(f"error processing {sound_file}: {str(e)}")
    
    if not results:
        print("no results generated. check input files and paths.")
        return
    
    # csv headers
    fieldnames: List[str] = ['Speech File', 'Min Pitch', 'Max Pitch', 'Mean Pitch', 'Sd Pitch', 
                            'Min Intensity', 'Max Intensity', 'Mean Intensity', 'Sd Intensity', 
                            'Speaking Rate', 'Jitter', 'Shimmer', 'HNR']
    
    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for emotion, features in results.items():
                row: Dict[str, any] = {"Speech File": emotion}
                row.update(features)
                writer.writerow(row)
        
        print(f"features saved to {output_file}")
    except IOError as e:
        print(f"error writing to file {output_file}: {str(e)}")

def main() -> None:
    """main function to process both personal and msp recordings."""
    try:
        # process personal recordings
        my_recording_path: str = "./recordings/my_recordings"
        print("processing personal recordings...")
        extract(my_recording_path, "my_features.csv", is_my_speech=True)
        
        # process msp podcast corpus samples
        msp_recording_path: str = "./recordings/msp_recordings"
        print("processing msp recordings...")
        extract(msp_recording_path, "msp_features.csv", is_my_speech=False)
        
        print("feature extraction completed successfully.")
        
    except Exception as e:
        print(f"unexpected error in main: {str(e)}")

if __name__ == "__main__":
    main()