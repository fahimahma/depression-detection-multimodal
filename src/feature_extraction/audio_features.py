import numpy as np
import librosa
import soundfile as sf
from scipy import signal, stats
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    """
    Extracts comprehensive audio features for depression detection.
    Features include MFCCs, spectral, prosodic, and voice quality features.
    """
    
    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
    
    def extract_from_file(self, audio_path: str) -> Dict[str, float]:
        """
        Extract audio features from file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing all extracted features
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        return self.extract_from_array(y, sr)
    
    def extract_from_array(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract audio features from numpy array.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        # MFCC features
        features.update(self._extract_mfcc_features(y, sr))
        
        # Spectral features
        features.update(self._extract_spectral_features(y, sr))
        
        # Prosodic features
        features.update(self._extract_prosodic_features(y, sr))
        
        # Voice quality features
        features.update(self._extract_voice_quality_features(y, sr))
        
        # Energy features
        features.update(self._extract_energy_features(y, sr))
        
        return features
    
    def _extract_mfcc_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract MFCC features."""
        features = {}
        
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        
        # Statistics over time
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            features[f'mfcc_{i}_min'] = float(np.min(mfccs[i]))
            features[f'mfcc_{i}_max'] = float(np.max(mfccs[i]))
        
        # Delta MFCCs (first derivative)
        mfcc_delta = librosa.feature.delta(mfccs)
        for i in range(min(3, self.n_mfcc)):
            features[f'mfcc_delta_{i}_mean'] = float(np.mean(mfcc_delta[i]))
            features[f'mfcc_delta_{i}_std'] = float(np.std(mfcc_delta[i]))
        
        return features
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract spectral features."""
        features = {}
        
        # Short-time Fourier transform
        D = librosa.stft(y)
        S = np.abs(D) ** 2
        
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spec_centroid_mean'] = float(np.mean(centroid))
        features['spec_centroid_std'] = float(np.std(centroid))
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spec_rolloff_mean'] = float(np.mean(rolloff))
        features['spec_rolloff_std'] = float(np.std(rolloff))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # Spectral flatness
        spec_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spec_flatness_mean'] = float(np.mean(spec_flatness))
        features['spec_flatness_std'] = float(np.std(spec_flatness))
        
        # Spectral contrast
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(spec_contrast.shape[0]):
            features[f'spec_contrast_{i}_mean'] = float(np.mean(spec_contrast[i]))
            features[f'spec_contrast_{i}_std'] = float(np.std(spec_contrast[i]))
        
        return features
    
    def _extract_prosodic_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract prosodic features (pitch, duration, intensity)."""
        features = {}
        
        # Fundamental frequency (using autocorrelation)
        f0 = self._estimate_fundamental_frequency(y, sr)
        features['f0_mean'] = float(np.mean(f0[f0 > 0])) if np.any(f0 > 0) else 0.0
        features['f0_std'] = float(np.std(f0[f0 > 0])) if np.any(f0 > 0) else 0.0
        features['f0_min'] = float(np.min(f0[f0 > 0])) if np.any(f0 > 0) else 0.0
        features['f0_max'] = float(np.max(f0[f0 > 0])) if np.any(f0 > 0) else 0.0
        
        # Energy/Intensity
        energy = librosa.feature.rmse(y=y)[0]
        features['energy_mean'] = float(np.mean(energy))
        features['energy_std'] = float(np.std(energy))
        features['energy_min'] = float(np.min(energy))
        features['energy_max'] = float(np.max(energy))
        
        # Voiced/Unvoiced ratio
        voiced_frames = np.sum(f0 > 0)
        features['voiced_ratio'] = float(voiced_frames / len(f0)) if len(f0) > 0 else 0.0
        
        # Speech rate estimation
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        feature_delta = np.mean(np.abs(np.diff(mfcc, axis=1)))
        features['speech_rate'] = float(feature_delta)
        
        return features
    
    def _extract_voice_quality_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract voice quality features (jitter, shimmer, etc.)."""
        features = {}
        
        # Frame-based analysis
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Jitter (frequency perturbation)
        f0 = self._estimate_fundamental_frequency(y, sr)
        voiced = f0 > 0
        if np.sum(voiced) > 1:
            f0_voiced = f0[voiced]
            jitter = np.mean(np.abs(np.diff(f0_voiced))) / np.mean(f0_voiced) if np.mean(f0_voiced) > 0 else 0.0
            features['jitter'] = float(jitter)
        else:
            features['jitter'] = 0.0
        
        # Shimmer (amplitude perturbation)
        S = np.abs(librosa.stft(y))
        magnitude = np.sqrt(np.sum(S ** 2, axis=0))
        if len(magnitude) > 1:
            shimmer = np.mean(np.abs(np.diff(magnitude))) / np.mean(magnitude) if np.mean(magnitude) > 0 else 0.0
            features['shimmer'] = float(shimmer)
        else:
            features['shimmer'] = 0.0
        
        # Harmonic-to-noise ratio (HNR)
        hnr = self._estimate_hnr(y, sr)
        features['hnr'] = float(hnr)
        
        return features
    
    def _extract_energy_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract energy-related features."""
        features = {}
        
        # Total energy
        features['total_energy'] = float(np.sum(y ** 2))
        
        # RMS Energy
        rms = librosa.feature.rmse(y=y)[0]
        features['rms_energy_mean'] = float(np.mean(rms))
        features['rms_energy_std'] = float(np.std(rms))
        
        # Energy entropy
        energy_entropy = -np.sum((rms ** 2) * np.log(rms + 1e-10))
        features['energy_entropy'] = float(energy_entropy)
        
        return features
    
    @staticmethod
    def _estimate_fundamental_frequency(y: np.ndarray, sr: int, 
                                       f_min: float = 50, 
                                       f_max: float = 400) -> np.ndarray:
        """Estimate fundamental frequency using autocorrelation."""
        # Use PYIN algorithm via librosa
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=f_min, fmax=f_max, sr=sr)
        return f0
    
    @staticmethod
    def _estimate_hnr(y: np.ndarray, sr: int) -> float:
        """Estimate Harmonic-to-Noise Ratio."""
        # Simple HNR estimation using spectral analysis
        D = librosa.stft(y)
        S = np.abs(D) ** 2
        
        # Harmonic part (low frequency)
        harmonic = np.sum(S[:10, :])  # Low frequency bins
        noise = np.sum(S[10:, :])      # Higher frequency bins
        
        hnr = 10 * np.log10((harmonic + 1e-10) / (noise + 1e-10))
        return float(np.mean(hnr))
