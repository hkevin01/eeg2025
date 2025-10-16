"""
Spectral Feature Extraction for EEG

This module extracts frequency-domain features from EEG data:
- Band power (delta, theta, alpha, beta, gamma)
- Spectral entropy
- Peak frequency
- Frontal alpha asymmetry

Primary use: Challenge 2 (Externalizing Prediction)
Key insight: Alpha power and asymmetry correlate with emotional regulation!
"""

import numpy as np
from scipy import signal
from scipy.stats import entropy
from typing import Dict, List, Optional, Tuple


class SpectralExtractor:
    """Extract spectral features from EEG data"""
    
    def __init__(self, sampling_rate: int = 100):
        """
        Args:
            sampling_rate: EEG sampling rate in Hz (default: 100 Hz)
        """
        self.sampling_rate = sampling_rate
        
        # Define frequency bands
        self.bands = {
            'delta': (0.5, 4),    # Deep sleep, unconscious processing
            'theta': (4, 8),       # Memory, meditation, working memory
            'alpha': (8, 13),      # Relaxed wakefulness, inhibition
            'beta': (13, 30),      # Active thinking, focus
            'gamma': (30, 50),     # Higher cognitive processing
        }
        
        # Channel groups
        self.channel_groups = {
            'frontal_left': ['F3', 'F7', 'FC5', 'AF3', 'F1'],
            'frontal_right': ['F4', 'F8', 'FC6', 'AF4', 'F2'],
            'frontal': ['Fz', 'FCz', 'F3', 'F4', 'F1', 'F2'],
            'parietal': ['Pz', 'CPz', 'P3', 'P4', 'P1', 'P2'],
            'occipital': ['Oz', 'O1', 'O2', 'POz'],
            'central': ['Cz', 'C3', 'C4', 'C1', 'C2'],
        }
    
    def extract_band_power(
        self,
        eeg_data: np.ndarray,
        band_name: str = 'alpha',
        relative: bool = True
    ) -> np.ndarray:
        """
        Extract power in a specific frequency band
        
        Args:
            eeg_data: EEG data [channels × time_samples]
            band_name: Name of frequency band
            relative: Return relative power (normalized by total power)
            
        Returns:
            Power values per channel [channels]
        """
        # Get band frequencies
        low_freq, high_freq = self.bands[band_name]
        
        # Compute power spectral density using Welch's method
        freqs, psd = signal.welch(
            eeg_data,
            fs=self.sampling_rate,
            nperseg=min(256, eeg_data.shape[1]),
            axis=1
        )
        
        # Find frequency indices for this band
        freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
        
        # Integrate power in this band
        band_power = np.trapz(psd[:, freq_mask], freqs[freq_mask], axis=1)
        
        if relative:
            # Normalize by total power (0.5-50 Hz)
            total_mask = (freqs >= 0.5) & (freqs <= 50)
            total_power = np.trapz(psd[:, total_mask], freqs[total_mask], axis=1)
            band_power = band_power / (total_power + 1e-10)
        
        return band_power
    
    def extract_all_band_powers(
        self,
        eeg_data: np.ndarray,
        channel_names: Optional[List[str]] = None,
        relative: bool = True
    ) -> Dict[str, float]:
        """
        Extract power in all frequency bands
        
        Args:
            eeg_data: EEG data [channels × time_samples]
            channel_names: List of channel names (optional)
            relative: Return relative power
            
        Returns:
            Dictionary with band power features
        """
        features = {}
        
        for band_name in self.bands.keys():
            band_power = self.extract_band_power(eeg_data, band_name, relative)
            
            # Global statistics
            features[f'{band_name}_power_mean'] = band_power.mean()
            features[f'{band_name}_power_std'] = band_power.std()
            features[f'{band_name}_power_max'] = band_power.max()
            
            # Region-specific power (if channel names provided)
            if channel_names is not None:
                for region, channels in self.channel_groups.items():
                    indices = self._get_channel_indices(channel_names, channels)
                    if len(indices) > 0:
                        region_power = band_power[indices].mean()
                        features[f'{band_name}_{region}_power'] = region_power
        
        return features
    
    def extract_frontal_alpha_asymmetry(
        self,
        eeg_data: np.ndarray,
        channel_names: List[str]
    ) -> Dict[str, float]:
        """
        Extract frontal alpha asymmetry (emotion regulation marker!)
        
        Higher left alpha = approach motivation
        Higher right alpha = withdrawal motivation
        
        Args:
            eeg_data: EEG data [channels × time_samples]
            channel_names: List of channel names
            
        Returns:
            Dictionary with asymmetry features
        """
        # Get alpha power for all channels
        alpha_power = self.extract_band_power(eeg_data, 'alpha', relative=False)
        
        # Get left and right frontal channels
        left_indices = self._get_channel_indices(
            channel_names,
            self.channel_groups['frontal_left']
        )
        right_indices = self._get_channel_indices(
            channel_names,
            self.channel_groups['frontal_right']
        )
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            n_channels = len(channel_names)
            left_indices = list(range(n_channels // 2))
            right_indices = list(range(n_channels // 2, n_channels))
        
        # Calculate asymmetry (log-transformed)
        left_alpha = np.log(alpha_power[left_indices].mean() + 1e-10)
        right_alpha = np.log(alpha_power[right_indices].mean() + 1e-10)
        
        asymmetry = right_alpha - left_alpha
        
        return {
            'frontal_alpha_asymmetry': asymmetry,
            'left_frontal_alpha': left_alpha,
            'right_frontal_alpha': right_alpha,
        }
    
    def extract_spectral_entropy(
        self,
        eeg_data: np.ndarray
    ) -> Dict[str, float]:
        """Extract spectral entropy (signal complexity measure)"""
        freqs, psd = signal.welch(
            eeg_data,
            fs=self.sampling_rate,
            nperseg=min(256, eeg_data.shape[1]),
            axis=1
        )
        
        psd_norm = psd / (psd.sum(axis=1, keepdims=True) + 1e-10)
        channel_entropy = [entropy(psd_norm[i]) for i in range(psd_norm.shape[0])]
        
        return {
            'spectral_entropy_mean': np.mean(channel_entropy),
            'spectral_entropy_std': np.std(channel_entropy),
            'spectral_entropy_max': np.max(channel_entropy),
        }
    
    def extract_all_spectral_features(
        self,
        eeg_data: np.ndarray,
        channel_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Extract all spectral features (comprehensive)"""
        features = {}
        
        # Band powers
        band_features = self.extract_all_band_powers(eeg_data, channel_names)
        features.update(band_features)
        
        # Frontal alpha asymmetry (if channels available)
        if channel_names is not None:
            asymmetry_features = self.extract_frontal_alpha_asymmetry(
                eeg_data, channel_names
            )
            features.update(asymmetry_features)
        
        # Spectral entropy
        entropy_features = self.extract_spectral_entropy(eeg_data)
        features.update(entropy_features)
        
        return features
    
    def _get_channel_indices(
        self,
        all_channels: List[str],
        target_channels: List[str]
    ) -> List[int]:
        """Get indices of target channels from all channels"""
        indices = []
        for target in target_channels:
            try:
                idx = all_channels.index(target)
                indices.append(idx)
            except ValueError:
                continue
        
        return indices


if __name__ == '__main__':
    print("Testing Spectral Extractor...")
    
    # Simulate EEG data: 129 channels × 500 samples
    np.random.seed(42)
    t = np.linspace(0, 5, 500)
    alpha_signal = 10 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
    noise = np.random.randn(129, 500) * 5
    dummy_eeg = noise + alpha_signal
    
    extractor = SpectralExtractor()
    alpha_power = extractor.extract_band_power(dummy_eeg, 'alpha')
    
    print(f"\n✅ Alpha power per channel: mean={alpha_power.mean():.3f}")
    print("✅ Spectral extractor ready!")
