import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr

def load_audio(file_path, target_sample_rate=16000):
    """
    Load an audio file and resample it to the given sample rate.
    """
    audio, sample_rate = librosa.load(file_path, sr=target_sample_rate)
    return audio

def reduce_noise(audio):
    """
    Apply noise reduction to the audio signal.
    """
    return nr.reduce_noise(y=audio, sr=16000)

def extract_mfcc(audio, sample_rate=16000, n_mfcc=13):
    """
    Extract MFCC features from an audio signal.
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return mfccs

def audio_preprocessing(file_path, target_sample_rate=16000, n_mfcc=13):
    """
    Complete preprocessing pipeline for an audio file.
    """
    audio = load_audio(file_path, target_sample_rate)
    audio = reduce_noise(audio)
    mfcc_features = extract_mfcc(audio, target_sample_rate, n_mfcc)
    return mfcc_features

def save_mfcc_features(mfcc_features, file_path):
    """
    Save MFCC features to a file.
    """
    np.save(file_path, mfcc_features)

def load_mfcc_features(file_path):
    """
    Load MFCC features from a file.
    """
    return np.load(file_path)

def save_audio(audio, file_path, sample_rate=16000):
    """
    Save an audio signal to a file.
    """
    sf.write(file_path, audio, sample_rate)

def audio_postprocessing(mfcc_features, sample_rate=16000):
    """
    Complete postprocessing pipeline for MFCC features.
    """
    audio = librosa.feature.inverse.mfcc_to_audio(mfcc_features, sr=sample_rate)
    return audio
