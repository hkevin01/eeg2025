"""
Neuroscience-inspired feature extraction for CCD task.

Extracts established ERP components and spectral features that correlate
with response time in detection tasks.

Anti-overfitting measures:
- Features are theory-driven (not data-mined)
- Normalized and clipped to prevent outliers
- Simple, interpretable features only
"""

import numpy as np
import torch
from scipy import signal
from typing import Dict, List, Tuple, Optional


# Standard EEG channel names (128-channel BioSemi layout + 1 reference)
STANDARD_CHANNEL_NAMES = [
    'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
    'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20',
    'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30',
    'A31', 'A32', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',
    'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19',
    'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29',
    'B30', 'B31', 'B32', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
    'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18',
    'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28',
    'C29', 'C30', 'C31', 'C32', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7',
    'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17',
    'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27',
    'D28', 'D29', 'D30', 'D31', 'D32', 'EXG1'
]

# Channel groups for feature extraction
CHANNEL_GROUPS = {
    'parietal': ['B19', 'B20', 'B21', 'B28', 'B29'],  # Pz, P3, P4 region
    'motor': ['A1', 'A32', 'B5', 'B32', 'C17'],  # C3, Cz, C4 region
    'frontal': ['A15', 'A16', 'A17', 'B1', 'D32'],  # Fz, F3, F4 region
    'occipital': ['C17', 'C18', 'C19', 'D7', 'D8'],  # O1, Oz, O2 region
}


def get_channel_indices(
    channel_names: List[str],
    target_channels: List[str],
    fallback_to_nearby: bool = True
) -> np.ndarray:
    """
    Get indices of target channels from channel list.
    
    Args:
        channel_names: List of available channel names
        target_channels: List of desired channel names
        fallback_to_nearby: If True, use nearby channels if exact match not found
        
    Returns:
        Array of channel indices
    """
    indices = []
    for target in target_channels:
        if target in channel_names:
            indices.append(channel_names.index(target))
        elif fallback_to_nearby:
            # Try to find similar channel (e.g., 'B19' if 'Pz' not found)
            for i, ch in enumerate(channel_names):
                if ch in CHANNEL_GROUPS.get(target.lower(), []):
                    indices.append(i)
                    break
    
    return np.array(indices) if indices else np.array([0])  # Fallback to first channel


def extract_p300_features(
    eeg_signal: np.ndarray,
    sfreq: float = 100.0,
    channel_indices: Optional[np.ndarray] = None,
    stimulus_time: float = 0.0
) -> Dict[str, float]:
    """
    Extract P300 amplitude and latency from parietal channels.
    
    P300 is a positive deflection ~300ms post-stimulus over parietal regions,
    associated with stimulus detection and attention allocation.
    
    Args:
        eeg_signal: (n_channels, n_samples) EEG data
        sfreq: Sampling frequency in Hz
        channel_indices: Indices of parietal channels (if None, use middle channels)
        stimulus_time: Time of stimulus in seconds (default 0.0)
        
    Returns:
        Dictionary with p300_amplitude and p300_latency
    """
    try:
        # Use parietal channels or fallback to middle channels
        if channel_indices is None or len(channel_indices) == 0:
            n_channels = eeg_signal.shape[0]
            channel_indices = np.array([n_channels // 2, n_channels // 2 + 1])
        
        # Average across parietal channels
        parietal_signal = eeg_signal[channel_indices, :].mean(axis=0)
        
        # P300 window: 250-450ms post-stimulus
        stimulus_idx = int(stimulus_time * sfreq)
        p300_start = stimulus_idx + int(0.25 * sfreq)  # +250ms
        p300_end = stimulus_idx + int(0.45 * sfreq)    # +450ms
        
        # Ensure valid window
        p300_start = max(0, p300_start)
        p300_end = min(len(parietal_signal), p300_end)
        
        if p300_end <= p300_start:
            return {'p300_amplitude': 0.0, 'p300_latency': 300.0}
        
        # Find peak in window
        p300_segment = parietal_signal[p300_start:p300_end]
        peak_idx = p300_segment.argmax()
        
        p300_amplitude = float(p300_segment[peak_idx])
        p300_latency = (p300_start + peak_idx) / sfreq * 1000  # Convert to ms
        
        # Normalize and clip to prevent outliers
        p300_amplitude = np.clip(p300_amplitude, -50, 50)
        
        return {
            'p300_amplitude': float(p300_amplitude),
            'p300_latency': float(p300_latency)
        }
        
    except Exception as e:
        # Robust fallback
        return {'p300_amplitude': 0.0, 'p300_latency': 300.0}


def extract_motor_preparation(
    eeg_signal: np.ndarray,
    sfreq: float = 100.0,
    channel_indices: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Extract motor preparation features from central (motor cortex) channels.
    
    Readiness potential (Bereitschaftspotential) is a negative slow wave that
    builds up before voluntary movement. Steeper slope predicts faster RT.
    
    Args:
        eeg_signal: (n_channels, n_samples) EEG data
        sfreq: Sampling frequency in Hz
        channel_indices: Indices of motor channels
        
    Returns:
        Dictionary with motor_slope and motor_amplitude
    """
    try:
        # Use motor channels or fallback
        if channel_indices is None or len(channel_indices) == 0:
            n_channels = eeg_signal.shape[0]
            channel_indices = np.array([n_channels // 3, 2 * n_channels // 3])
        
        # Average across motor channels
        motor_signal = eeg_signal[channel_indices, :].mean(axis=0)
        
        # Pre-response window (last 500ms of trial)
        pre_response_samples = int(0.5 * sfreq)  # 500ms
        pre_response = motor_signal[-pre_response_samples:]
        
        # Calculate slope (more negative = faster preparation = shorter RT)
        time_axis = np.arange(len(pre_response))
        if len(time_axis) < 2:
            return {'motor_slope': 0.0, 'motor_amplitude': 0.0}
            
        slope, _ = np.polyfit(time_axis, pre_response, 1)
        
        # Peak amplitude (most negative point in pre-response window)
        motor_amplitude = float(pre_response.min())
        
        # Normalize and clip
        slope = np.clip(slope, -1.0, 1.0)
        motor_amplitude = np.clip(motor_amplitude, -50, 50)
        
        return {
            'motor_slope': float(slope),
            'motor_amplitude': float(motor_amplitude)
        }
        
    except Exception as e:
        return {'motor_slope': 0.0, 'motor_amplitude': 0.0}


def extract_n200_features(
    eeg_signal: np.ndarray,
    sfreq: float = 100.0,
    channel_indices: Optional[np.ndarray] = None,
    stimulus_time: float = 0.0
) -> Dict[str, float]:
    """
    Extract N200 amplitude from frontal channels.
    
    N200 is a negative deflection ~200ms post-stimulus over frontal regions,
    associated with conflict detection and inhibitory control.
    
    Args:
        eeg_signal: (n_channels, n_samples) EEG data
        sfreq: Sampling frequency in Hz
        channel_indices: Indices of frontal channels
        stimulus_time: Time of stimulus in seconds
        
    Returns:
        Dictionary with n200_amplitude
    """
    try:
        # Use frontal channels or fallback
        if channel_indices is None or len(channel_indices) == 0:
            n_channels = eeg_signal.shape[0]
            channel_indices = np.array([0, 1, 2])  # Front channels
        
        # Average across frontal channels
        frontal_signal = eeg_signal[channel_indices, :].mean(axis=0)
        
        # N200 window: 150-250ms post-stimulus
        stimulus_idx = int(stimulus_time * sfreq)
        n200_start = stimulus_idx + int(0.15 * sfreq)  # +150ms
        n200_end = stimulus_idx + int(0.25 * sfreq)    # +250ms
        
        # Ensure valid window
        n200_start = max(0, n200_start)
        n200_end = min(len(frontal_signal), n200_end)
        
        if n200_end <= n200_start:
            return {'n200_amplitude': 0.0}
        
        # Find peak (most negative) in window
        n200_segment = frontal_signal[n200_start:n200_end]
        n200_amplitude = float(n200_segment.min())
        
        # Normalize and clip
        n200_amplitude = np.clip(n200_amplitude, -50, 50)
        
        return {'n200_amplitude': float(n200_amplitude)}
        
    except Exception as e:
        return {'n200_amplitude': 0.0}


def extract_alpha_suppression(
    eeg_signal: np.ndarray,
    sfreq: float = 100.0,
    channel_indices: Optional[np.ndarray] = None,
    stimulus_time: float = 0.0
) -> Dict[str, float]:
    """
    Calculate alpha power suppression as marker of visual attention.
    
    Alpha (8-12Hz) decreases over occipital cortex when visual attention
    is engaged. More suppression = better attention = faster RT.
    
    Args:
        eeg_signal: (n_channels, n_samples) EEG data
        sfreq: Sampling frequency in Hz
        channel_indices: Indices of occipital channels
        stimulus_time: Time of stimulus in seconds
        
    Returns:
        Dictionary with alpha_suppression ratio
    """
    try:
        # Use occipital channels or fallback
        if channel_indices is None or len(channel_indices) == 0:
            n_channels = eeg_signal.shape[0]
            channel_indices = np.array(range(n_channels - 3, n_channels))  # Back channels
        
        # Average across occipital channels
        occipital_signal = eeg_signal[channel_indices, :].mean(axis=0)
        
        # Baseline: pre-stimulus (200ms before stimulus)
        stimulus_idx = int(stimulus_time * sfreq)
        baseline_start = max(0, stimulus_idx - int(0.2 * sfreq))
        baseline_signal = occipital_signal[baseline_start:stimulus_idx]
        
        # Task period: post-stimulus (200-600ms after stimulus)
        task_start = stimulus_idx + int(0.2 * sfreq)
        task_end = stimulus_idx + int(0.6 * sfreq)
        task_signal = occipital_signal[task_start:min(task_end, len(occipital_signal))]
        
        # Require minimum signal length
        if len(baseline_signal) < 10 or len(task_signal) < 10:
            return {'alpha_suppression': 1.0}
        
        # Calculate alpha power (8-12 Hz) using Welch's method
        nperseg = min(len(baseline_signal), 20)
        freqs_baseline, psd_baseline = signal.welch(
            baseline_signal, sfreq, nperseg=nperseg
        )
        
        nperseg = min(len(task_signal), 40)
        freqs_task, psd_task = signal.welch(
            task_signal, sfreq, nperseg=nperseg
        )
        
        # Extract alpha band (8-12 Hz)
        alpha_idx_baseline = np.where((freqs_baseline >= 8) & (freqs_baseline <= 12))[0]
        alpha_idx_task = np.where((freqs_task >= 8) & (freqs_task <= 12))[0]
        
        if len(alpha_idx_baseline) == 0 or len(alpha_idx_task) == 0:
            return {'alpha_suppression': 1.0}
        
        alpha_power_baseline = psd_baseline[alpha_idx_baseline].mean()
        alpha_power_task = psd_task[alpha_idx_task].mean()
        
        # Suppression ratio (> 1 means suppression occurred)
        alpha_suppression = alpha_power_baseline / (alpha_power_task + 1e-10)
        
        # Clip to reasonable range
        alpha_suppression = np.clip(alpha_suppression, 0.1, 10.0)
        
        return {'alpha_suppression': float(alpha_suppression)}
        
    except Exception as e:
        return {'alpha_suppression': 1.0}


def extract_all_neuro_features(
    eeg_signal: np.ndarray,
    sfreq: float = 100.0,
    channel_groups: Optional[Dict[str, np.ndarray]] = None,
    stimulus_time: float = 0.0,
    normalize: bool = True
) -> np.ndarray:
    """
    Extract all neuroscience features in one call.
    
    Returns features in fixed order:
    [p300_amp, p300_lat, motor_slope, motor_amp, n200_amp, alpha_supp]
    
    Args:
        eeg_signal: (n_channels, n_samples) EEG data
        sfreq: Sampling frequency
        channel_groups: Dict mapping group names to channel indices
        stimulus_time: Time of stimulus in seconds
        normalize: Whether to z-score normalize features
        
    Returns:
        Feature vector of shape (6,)
    """
    if channel_groups is None:
        channel_groups = {}
    
    # Extract features
    p300 = extract_p300_features(
        eeg_signal, sfreq,
        channel_groups.get('parietal'),
        stimulus_time
    )
    
    motor = extract_motor_preparation(
        eeg_signal, sfreq,
        channel_groups.get('motor')
    )
    
    n200 = extract_n200_features(
        eeg_signal, sfreq,
        channel_groups.get('frontal'),
        stimulus_time
    )
    
    alpha = extract_alpha_suppression(
        eeg_signal, sfreq,
        channel_groups.get('occipital'),
        stimulus_time
    )
    
    # Combine into feature vector
    features = np.array([
        p300['p300_amplitude'],
        p300['p300_latency'],
        motor['motor_slope'],
        motor['motor_amplitude'],
        n200['n200_amplitude'],
        alpha['alpha_suppression'],
    ], dtype=np.float32)
    
    # Normalize if requested
    if normalize:
        # Z-score normalize (assuming reasonable ranges from literature)
        # These are approximate means/stds from ERP literature
        means = np.array([5.0, 350.0, 0.0, -10.0, -5.0, 1.5])
        stds = np.array([10.0, 50.0, 0.3, 10.0, 10.0, 1.0])
        features = (features - means) / (stds + 1e-8)
        
        # Clip to prevent extreme outliers
        features = np.clip(features, -3.0, 3.0)
    
    return features


# For backwards compatibility
def get_channel_mapping(channel_names: List[str]) -> Dict[str, np.ndarray]:
    """
    Get channel group indices for standard channel groups.
    
    Args:
        channel_names: List of available channel names
        
    Returns:
        Dictionary mapping group names to channel indices
    """
    channel_groups = {}
    
    for group_name, group_channels in CHANNEL_GROUPS.items():
        indices = get_channel_indices(channel_names, group_channels)
        channel_groups[group_name] = indices
    
    return channel_groups
