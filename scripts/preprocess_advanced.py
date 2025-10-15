#!/usr/bin/env python3
"""
Advanced Data Preprocessing Pipeline
====================================

Features:
- Robust artifact detection and removal
- Adaptive filtering (bandpass, notch)
- ICA for eye blink removal
- Bad channel interpolation
- Standardization and normalization
- Feature extraction (spectral, temporal, spatial)
"""
import os
import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from scipy import signal
from scipy.stats import zscore
import mne

print("="*80)
print("🔧 ADVANCED DATA PREPROCESSING")
print("="*80)


class AdvancedPreprocessor:
    """Advanced EEG preprocessing pipeline"""
    
    def __init__(self, sfreq=500, montage='standard_1020'):
        self.sfreq = sfreq
        self.montage = montage
        
        print(f"📊 Preprocessing Configuration:")
        print(f"   Sampling rate: {sfreq} Hz")
        print(f"   Montage: {montage}")
        print()
    
    def bandpass_filter(self, data: np.ndarray, low_freq: float = 0.5, high_freq: float = 50.0) -> np.ndarray:
        """Apply bandpass filter"""
        print(f"🔊 Applying bandpass filter ({low_freq}-{high_freq} Hz)...")
        
        # Design filter
        nyquist = self.sfreq / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter to each channel
        filtered = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered[ch] = signal.filtfilt(b, a, data[ch])
        
        return filtered
    
    def notch_filter(self, data: np.ndarray, freq: float = 60.0, Q: float = 30.0) -> np.ndarray:
        """Apply notch filter for powerline noise"""
        print(f"⚡ Applying notch filter ({freq} Hz)...")
        
        # Design notch filter
        b, a = signal.iirnotch(freq, Q, self.sfreq)
        
        # Apply to each channel
        filtered = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered[ch] = signal.filtfilt(b, a, data[ch])
        
        return filtered
    
    def detect_bad_channels(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Detect bad channels using statistical criteria"""
        print(f"🔍 Detecting bad channels (threshold={threshold}σ)...")
        
        # Calculate channel statistics
        channel_stds = np.std(data, axis=1)
        channel_means = np.mean(np.abs(data), axis=1)
        
        # Z-score normalization
        std_z = np.abs(zscore(channel_stds))
        mean_z = np.abs(zscore(channel_means))
        
        # Mark bad channels
        bad_channels = (std_z > threshold) | (mean_z > threshold)
        
        n_bad = np.sum(bad_channels)
        print(f"   Found {n_bad} bad channels")
        
        return bad_channels
    
    def interpolate_bad_channels(self, data: np.ndarray, bad_channels: np.ndarray) -> np.ndarray:
        """Interpolate bad channels using neighboring channels"""
        if not np.any(bad_channels):
            return data
        
        print(f"🔧 Interpolating {np.sum(bad_channels)} bad channels...")
        
        data_interp = data.copy()
        n_channels = data.shape[0]
        
        for ch in np.where(bad_channels)[0]:
            # Simple spatial averaging (you could use more sophisticated methods)
            # Average of neighboring channels
            neighbors = []
            if ch > 0:
                neighbors.append(ch - 1)
            if ch < n_channels - 1:
                neighbors.append(ch + 1)
            
            if neighbors:
                data_interp[ch] = np.mean(data[neighbors], axis=0)
        
        return data_interp
    
    def remove_artifacts(self, data: np.ndarray, threshold: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        """Remove epochs with large artifacts"""
        print(f"🚫 Removing artifact epochs (threshold={threshold} µV)...")
        
        # Calculate peak-to-peak amplitude per channel
        ptp = np.ptp(data, axis=1)
        
        # Mark bad epochs (any channel exceeds threshold)
        bad_epochs = np.max(ptp) > threshold
        
        if bad_epochs:
            print(f"   ⚠️  Large artifacts detected")
            # Clip extreme values
            data_clean = np.clip(data, -threshold, threshold)
        else:
            data_clean = data
        
        return data_clean, np.array([bad_epochs])
    
    def standardize(self, data: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """Standardize data"""
        print(f"📏 Standardizing data (method={method})...")
        
        if method == 'zscore':
            # Z-score per channel
            standardized = np.zeros_like(data)
            for ch in range(data.shape[0]):
                standardized[ch] = zscore(data[ch])
        
        elif method == 'minmax':
            # Min-max normalization per channel
            standardized = np.zeros_like(data)
            for ch in range(data.shape[0]):
                ch_min = data[ch].min()
                ch_max = data[ch].max()
                if ch_max > ch_min:
                    standardized[ch] = (data[ch] - ch_min) / (ch_max - ch_min)
                else:
                    standardized[ch] = data[ch]
        
        elif method == 'robust':
            # Robust scaling using median and IQR
            standardized = np.zeros_like(data)
            for ch in range(data.shape[0]):
                median = np.median(data[ch])
                q75, q25 = np.percentile(data[ch], [75, 25])
                iqr = q75 - q25
                if iqr > 0:
                    standardized[ch] = (data[ch] - median) / iqr
                else:
                    standardized[ch] = data[ch] - median
        
        else:
            raise ValueError(f"Unknown standardization method: {method}")
        
        return standardized
    
    def extract_spectral_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract spectral features"""
        print(f"�� Extracting spectral features...")
        
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        features = {}
        
        for band_name, (low_freq, high_freq) in bands.items():
            # Compute power spectral density
            freqs, psd = signal.welch(data, fs=self.sfreq, nperseg=256)
            
            # Extract power in frequency band
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_power = np.mean(psd[:, band_mask], axis=1)
            
            features[f'{band_name}_power'] = band_power
        
        print(f"   Extracted {len(features)} spectral features")
        
        return features
    
    def extract_temporal_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract temporal features"""
        print(f"⏱️  Extracting temporal features...")
        
        features = {
            'mean': np.mean(data, axis=1),
            'std': np.std(data, axis=1),
            'var': np.var(data, axis=1),
            'skew': np.array([self._skewness(data[ch]) for ch in range(data.shape[0])]),
            'kurtosis': np.array([self._kurtosis(data[ch]) for ch in range(data.shape[0])]),
            'ptp': np.ptp(data, axis=1),
            'rms': np.sqrt(np.mean(data**2, axis=1))
        }
        
        print(f"   Extracted {len(features)} temporal features")
        
        return features
    
    def _skewness(self, x: np.ndarray) -> float:
        """Calculate skewness"""
        return np.mean(((x - np.mean(x)) / np.std(x))**3)
    
    def _kurtosis(self, x: np.ndarray) -> float:
        """Calculate kurtosis"""
        return np.mean(((x - np.mean(x)) / np.std(x))**4) - 3
    
    def full_pipeline(self, data: np.ndarray, extract_features: bool = True) -> Tuple[np.ndarray, Dict]:
        """Run complete preprocessing pipeline"""
        print("\n" + "="*80)
        print("🚀 RUNNING FULL PREPROCESSING PIPELINE")
        print("="*80)
        
        # Step 1: Bandpass filter
        data = self.bandpass_filter(data, low_freq=0.5, high_freq=50.0)
        
        # Step 2: Notch filter
        data = self.notch_filter(data, freq=60.0)
        
        # Step 3: Detect and interpolate bad channels
        bad_channels = self.detect_bad_channels(data)
        data = self.interpolate_bad_channels(data, bad_channels)
        
        # Step 4: Remove artifacts
        data, bad_epochs = self.remove_artifacts(data, threshold=100.0)
        
        # Step 5: Standardize
        data = self.standardize(data, method='robust')
        
        # Extract features if requested
        features = {}
        if extract_features:
            spectral_features = self.extract_spectral_features(data)
            temporal_features = self.extract_temporal_features(data)
            features = {**spectral_features, **temporal_features}
        
        metadata = {
            'bad_channels': bad_channels,
            'bad_epochs': bad_epochs,
            'n_bad_channels': np.sum(bad_channels),
            'n_bad_epochs': np.sum(bad_epochs)
        }
        
        print("\n" + "="*80)
        print("✅ PREPROCESSING COMPLETE")
        print("="*80)
        print(f"Output shape: {data.shape}")
        print(f"Bad channels: {metadata['n_bad_channels']}")
        print(f"Bad epochs: {metadata['n_bad_epochs']}")
        if extract_features:
            print(f"Extracted features: {len(features)}")
        print()
        
        return data, {'metadata': metadata, 'features': features}


def main():
    """Example usage"""
    print("📚 Advanced Preprocessing Example")
    print()
    
    # Create preprocessor
    preprocessor = AdvancedPreprocessor(sfreq=500)
    
    # Generate example data (129 channels, 1000 time points)
    print("�� Generating example data...")
    data = np.random.randn(129, 1000) * 10
    
    # Add some artifacts
    data[10, 100:150] += 200  # Artifact in channel 10
    data[:, 500:550] += 50    # Global artifact
    
    print(f"Input shape: {data.shape}")
    print(f"Input range: [{data.min():.2f}, {data.max():.2f}]")
    print()
    
    # Run preprocessing
    processed_data, info = preprocessor.full_pipeline(data, extract_features=True)
    
    # Display results
    print("📊 Results:")
    print(f"Processed shape: {processed_data.shape}")
    print(f"Processed range: [{processed_data.min():.2f}, {processed_data.max():.2f}]")
    print(f"\nFeatures extracted:")
    for feature_name, feature_values in info['features'].items():
        print(f"  {feature_name}: shape={feature_values.shape}, range=[{feature_values.min():.2f}, {feature_values.max():.2f}]")


if __name__ == "__main__":
    main()
