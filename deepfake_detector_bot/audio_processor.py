import os
import numpy as np
import librosa
import soundfile as sf
import tempfile
import warnings
warnings.filterwarnings('ignore')

from config import SAMPLE_RATE, N_MFCC

class AudioFeatureExtractor:
    
    @staticmethod
    def load_audio(file_path, target_sr=SAMPLE_RATE):
        try:
            y, sr = librosa.load(file_path, sr=target_sr, mono=True)
            return y, sr
        except Exception as e:
            print(f"Ошибка загрузки аудио {file_path}: {e}")
            return None, None
    
    @staticmethod
    def preprocess_audio(audio, sr=SAMPLE_RATE):
        try:
            audio = librosa.util.normalize(audio)
            
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
            
            if len(audio_trimmed) < sr * 1:
                audio_trimmed = np.tile(audio_trimmed, 3)[:sr * 3]
            
            return audio_trimmed
        except Exception as e:
            print(f"Ошибка предобработки аудио: {e}")
            return audio
    
    @staticmethod
    def extract_mfcc_features(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
        try:
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=sr, 
                n_mfcc=n_mfcc,
                n_fft=2048,
                hop_length=512
            )
            
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
            features = (features - np.mean(features)) / (np.std(features) + 1e-10)
            target_length = 130
            
            if features.shape[1] > target_length:
                features = features[:, :target_length]
            else:
                padding = target_length - features.shape[1]
                features = np.pad(features, ((0, 0), (0, padding)), mode='constant')
            
            return features.flatten()
            
        except Exception as e:
            print(f"Ошибка извлечения MFCC: {e}")
            return None
    
    @staticmethod
    def extract_spectral_features(audio, sr=SAMPLE_RATE):
        try:
            features = []

            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features.append(np.mean(spectral_centroid))
            features.append(np.std(spectral_centroid))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            features.append(np.mean(spectral_bandwidth))
            features.append(np.std(spectral_bandwidth))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
            features.append(np.mean(zero_crossing_rate))
            features.append(np.std(zero_crossing_rate))
            
            rms = librosa.feature.rms(y=audio)[0]
            features.append(np.mean(rms))
            features.append(np.std(rms))
            
            return np.array(features)
            
        except Exception as e:
            print(f"Ошибка извлечения спектральных признаков: {e}")
            return None
    
    @staticmethod
    def extract_all_features(file_path, use_mfcc=True, use_spectral=False):
        audio, sr = AudioFeatureExtractor.load_audio(file_path)
        if audio is None:
            return None
        
        audio = AudioFeatureExtractor.preprocess_audio(audio, sr)
        
        all_features = []
        
        if use_mfcc:
            mfcc_features = AudioFeatureExtractor.extract_mfcc_features(audio, sr)
            if mfcc_features is not None:
                all_features.append(mfcc_features)
        
        if use_spectral:
            spectral_features = AudioFeatureExtractor.extract_spectral_features(audio, sr)
            if spectral_features is not None:
                all_features.append(spectral_features)
        
        if not all_features:
            return None
        
        return np.concatenate(all_features)
    
    @staticmethod
    def extract_features_for_model(file_path):
        return AudioFeatureExtractor.extract_all_features(
            file_path, 
            use_mfcc=True, 
            use_spectral=False
        )


class AudioConverter:
    
    @staticmethod
    def convert_to_wav(input_path, output_path=None, target_sr=SAMPLE_RATE):
        try:
            if output_path is None:
                output_path = tempfile.mktemp(suffix='.wav')
            
            audio, sr = AudioFeatureExtractor.load_audio(input_path, target_sr)
            if audio is None:
                return None
            
            sf.write(output_path, audio, target_sr, subtype='PCM_16')
            return output_path
            
        except Exception as e:
            print(f"Ошибка конвертации в WAV: {e}")
            return None
    
    @staticmethod
    def get_audio_info(file_path):
        try:
            audio, sr = AudioFeatureExtractor.load_audio(file_path)
            if audio is None:
                return None
            
            duration = len(audio) / sr
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'samples': len(audio),
                'max_amplitude': np.max(np.abs(audio))
            }
            
        except Exception as e:
            print(f"Ошибка получения информации: {e}")
            return None