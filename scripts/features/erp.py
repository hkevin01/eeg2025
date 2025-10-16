"""
Event-Related Potential (ERP) Feature Extraction

This module extracts ERP components from EEG data, with a focus on:
- P300: Decision-making and response time marker (300-600ms)
- N2: Conflict detection (200-350ms)
- P1/N1: Early visual processing (100-200ms)

Primary use: Challenge 1 (Response Time Prediction)
Key insight: P300 latency correlates with reaction time!
"""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, List, Optional


class ERPExtractor:
    """Extract ERP components from EEG data"""
    
    def __init__(self, sampling_rate: int = 100):
        """
        Args:
            sampling_rate: EEG sampling rate in Hz (default: 100 Hz for competition data)
        """
        self.sampling_rate = sampling_rate
        
        # Define ERP components (in samples @ 100Hz)
        # Component: (start_ms, end_ms, peak_polarity)
        self.components = {
            'P1': (80, 150, 'positive'),    # Early visual processing
            'N1': (120, 200, 'negative'),   # Attention allocation
            'N2': (200, 350, 'negative'),   # Conflict detection
            'P300': (300, 600, 'positive'), # Decision-making (KEY for RT!)
            'CNV': (-200, 0, 'negative'),   # Motor preparation
        }
        
        # Channel groups for different components
        self.channel_groups = {
            'visual': ['Oz', 'O1', 'O2', 'PO3', 'PO4', 'PO7', 'PO8'],
            'frontal': ['Fz', 'FCz', 'F3', 'F4', 'F1', 'F2'],
            'central': ['Cz', 'C3', 'C4', 'C1', 'C2'],
            'parietal': ['Pz', 'CPz', 'P3', 'P4', 'P1', 'P2'],
        }
    
    def extract_p300(
        self, 
        eeg_data: np.ndarray,
        channel_names: Optional[List[str]] = None,
        baseline_correct: bool = True
    ) -> Dict[str, float]:
        """
        Extract P300 component features (PRIMARY feature for Challenge 1!)
        
        Args:
            eeg_data: EEG data [channels × time_samples]
            channel_names: List of channel names (optional, for channel selection)
            baseline_correct: Apply baseline correction
            
        Returns:
            Dictionary with P300 features:
            - peak_latency: Time of peak (ms) - CORRELATES WITH RT!
            - peak_amplitude: Amplitude at peak (μV)
            - mean_amplitude: Mean amplitude in P300 window (μV)
            - area_under_curve: Total area in P300 window
            - onset_latency: First significant deflection
        """
        # Apply baseline correction if requested
        if baseline_correct:
            baseline_window = slice(0, 20)  # First 200ms
            baseline = eeg_data[:, baseline_window].mean(axis=1, keepdims=True)
            eeg_data = eeg_data - baseline
        
        # Get parietal channels (P300 is maximal here)
        if channel_names is not None:
            parietal_indices = self._get_channel_indices(
                channel_names, 
                self.channel_groups['parietal']
            )
        else:
            # Default: use middle 1/3 of channels (approximation)
            n_channels = eeg_data.shape[0]
            parietal_indices = slice(n_channels // 3, 2 * n_channels // 3)
        
        # Average over parietal channels
        parietal_avg = eeg_data[parietal_indices].mean(axis=0)
        
        # P300 window: 300-600ms @ 100Hz = samples 30-60
        p300_start = int(300 * self.sampling_rate / 1000)
        p300_end = int(600 * self.sampling_rate / 1000)
        p300_window = parietal_avg[p300_start:p300_end]
        
        # Extract features
        peak_idx = p300_window.argmax()
        peak_latency = 300 + (peak_idx * 1000 / self.sampling_rate)  # Convert to ms
        peak_amplitude = p300_window[peak_idx]
        mean_amplitude = p300_window.mean()
        area_under_curve = np.trapz(p300_window)
        
        # Find onset (first point > 25% of peak)
        threshold = peak_amplitude * 0.25
        onset_indices = np.where(p300_window > threshold)[0]
        if len(onset_indices) > 0:
            onset_latency = 300 + (onset_indices[0] * 1000 / self.sampling_rate)
        else:
            onset_latency = peak_latency  # Fallback
        
        return {
            'p300_peak_latency': peak_latency,
            'p300_peak_amplitude': peak_amplitude,
            'p300_mean_amplitude': mean_amplitude,
            'p300_area_under_curve': area_under_curve,
            'p300_onset_latency': onset_latency,
            'p300_rise_time': peak_latency - onset_latency,
        }
    
    def extract_all_components(
        self,
        eeg_data: np.ndarray,
        channel_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Extract all ERP components
        
        Args:
            eeg_data: EEG data [channels × time_samples]
            channel_names: List of channel names
            
        Returns:
            Dictionary with features from all components
        """
        features = {}
        
        # Baseline correction
        baseline = eeg_data[:, :20].mean(axis=1, keepdims=True)
        eeg_corrected = eeg_data - baseline
        
        # Extract P300 (most important!)
        p300_features = self.extract_p300(eeg_data, channel_names, baseline_correct=False)
        features.update(p300_features)
        
        # Extract N2 (conflict detection)
        n2_features = self._extract_component(
            eeg_corrected, 
            'N2', 
            channel_group='frontal',
            channel_names=channel_names
        )
        features.update(n2_features)
        
        # Extract P1/N1 (early visual)
        p1_features = self._extract_component(
            eeg_corrected,
            'P1',
            channel_group='visual',
            channel_names=channel_names
        )
        features.update(p1_features)
        
        return features
    
    def _extract_component(
        self,
        eeg_data: np.ndarray,
        component_name: str,
        channel_group: str,
        channel_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Extract a specific ERP component"""
        
        start_ms, end_ms, polarity = self.components[component_name]
        
        # Convert to samples
        start_sample = int(start_ms * self.sampling_rate / 1000)
        end_sample = int(end_ms * self.sampling_rate / 1000)
        
        # Handle negative time (pre-stimulus)
        if start_sample < 0:
            start_sample = max(0, eeg_data.shape[1] + start_sample)
        if end_sample <= 0:
            end_sample = eeg_data.shape[1] + end_sample
        
        # Get relevant channels
        if channel_names is not None:
            channel_indices = self._get_channel_indices(
                channel_names,
                self.channel_groups[channel_group]
            )
        else:
            # Use all channels
            channel_indices = slice(None)
        
        # Average over channels
        channel_avg = eeg_data[channel_indices].mean(axis=0)
        
        # Extract window
        component_window = channel_avg[start_sample:end_sample]
        
        # Find peak (max for positive, min for negative)
        if polarity == 'positive':
            peak_idx = component_window.argmax()
            peak_amplitude = component_window[peak_idx]
        else:
            peak_idx = component_window.argmin()
            peak_amplitude = component_window[peak_idx]
        
        # Calculate latency
        peak_latency = start_ms + (peak_idx * 1000 / self.sampling_rate)
        
        return {
            f'{component_name.lower()}_peak_latency': peak_latency,
            f'{component_name.lower()}_peak_amplitude': peak_amplitude,
            f'{component_name.lower()}_mean_amplitude': component_window.mean(),
        }
    
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
                continue  # Channel not found, skip
        
        if len(indices) == 0:
            # Fallback: use all channels
            return list(range(len(all_channels)))
        
        return indices


def extract_p300_features_batch(
    eeg_batch: np.ndarray,
    channel_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Convenience function to extract P300 features from a batch
    
    Args:
        eeg_batch: Batch of EEG data [batch_size × channels × time_samples]
        channel_names: List of channel names
        
    Returns:
        Feature array [batch_size × num_features]
    """
    extractor = ERPExtractor()
    
    features_list = []
    for i in range(eeg_batch.shape[0]):
        features = extractor.extract_p300(eeg_batch[i], channel_names)
        # Convert to array in consistent order
        feature_vector = [
            features['p300_peak_latency'],
            features['p300_peak_amplitude'],
            features['p300_mean_amplitude'],
            features['p300_area_under_curve'],
            features['p300_rise_time'],
        ]
        features_list.append(feature_vector)
    
    return np.array(features_list)


if __name__ == '__main__':
    # Test the extractor
    print("Testing ERP Extractor...")
    
    # Simulate EEG data: 129 channels × 500 samples (5 seconds @ 100Hz)
    dummy_eeg = np.random.randn(129, 500) * 10  # Random data with 10μV std
    
    # Add a synthetic P300 (for testing)
    p300_channels = slice(50, 70)  # Simulate parietal channels
    p300_time = slice(35, 55)  # 350-550ms
    dummy_eeg[p300_channels, p300_time] += 15  # Add positive deflection
    
    extractor = ERPExtractor()
    features = extractor.extract_p300(dummy_eeg)
    
    print("\n✅ P300 Features Extracted:")
    for key, value in features.items():
        print(f"  {key}: {value:.2f}")
    
    print("\n✅ ERP extractor ready!")
    print("Expected P300 latency for this dummy data: ~400-450ms")
    print(f"Actual extracted latency: {features['p300_peak_latency']:.2f}ms")
