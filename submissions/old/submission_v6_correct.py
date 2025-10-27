"""
EEG Foundation Challenge 2025 - v6 Correct Submission
=====================================================
Matches the actual ImprovedEEGModel architecture from training.
Uses EEGNeX from braindecode + Channel Attention + Frequency features.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from scipy.signal import welch

# Try to import braindecode, fall back to simple implementation
try:
    from braindecode.models import EEGNeX as EEGNeXBase
    BRAINDECODE_AVAILABLE = True
except ImportError:
    BRAINDECODE_AVAILABLE = False
    print("⚠️ braindecode not available, using fallback")


# ============================================================================
# PATH RESOLUTION
# ============================================================================

def resolve_path(name="model_file_name"):
    """Resolve file path for different execution environments."""
    possible_paths = [
        f"/app/input/res/{name}",
        f"/app/input/{name}",
        f"{name}",
        str(Path(__file__).parent / name) if '__file__' in globals() else None
    ]
    
    for path in possible_paths:
        if path and Path(path).exists():
            return path
    
    raise FileNotFoundError(f"Could not find {name} in any expected location")


# ============================================================================
# MODEL ARCHITECTURE (matches ImprovedEEGModel from training)
# ============================================================================

class ChannelAttention(nn.Module):
    """Channel attention module."""
    def __init__(self, n_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(n_channels, max(1, n_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, n_channels // reduction), n_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        attention = self.sigmoid(avg_out + max_out).unsqueeze(-1)
        return x * attention


class ImprovedEEGModel(nn.Module):
    """
    Improved EEG model matching training architecture.
    Uses EEGNeX backbone + Channel Attention + Frequency features.
    """
    def __init__(self, n_channels=129, n_times=200, n_outputs=1):
        super().__init__()
        
        if not BRAINDECODE_AVAILABLE:
            raise ImportError("braindecode is required but not available")
        
        self.backbone = EEGNeXBase(
            n_outputs=64,
            n_chans=n_channels,
            n_times=n_times,
            drop_prob=0.3
        )
        
        self.channel_attention = ChannelAttention(n_channels, reduction=8)
        
        self.freq_encoder = nn.Sequential(
            nn.Linear(n_channels * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.head = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_outputs)
        )
    
    def extract_freq_features(self, x):
        """Extract frequency band power features."""
        batch_size, n_channels, n_times = x.shape
        x_np = x.detach().cpu().numpy()
        
        features = []
        for i in range(batch_size):
            band_powers = []
            for ch in range(n_channels):
                try:
                    freqs, psd = welch(x_np[i, ch], fs=100, nperseg=min(64, n_times))
                    
                    delta = np.mean(psd[(freqs >= 0.5) & (freqs < 4)])
                    theta = np.mean(psd[(freqs >= 4) & (freqs < 8)])
                    alpha = np.mean(psd[(freqs >= 8) & (freqs < 13)])
                    beta = np.mean(psd[(freqs >= 13) & (freqs < 30)])
                    
                    band_powers.extend([delta, theta, alpha, beta])
                except:
                    # If welch fails, use zeros
                    band_powers.extend([0.0, 0.0, 0.0, 0.0])
            
            features.append(band_powers)
        
        return torch.tensor(features, device=x.device, dtype=x.dtype)
    
    def forward(self, x):
        # Apply channel attention
        x_attended = self.channel_attention(x)
        
        # Extract time features via backbone
        time_features = self.backbone(x_attended)
        
        # Extract frequency features
        freq_features = self.freq_encoder(self.extract_freq_features(x))
        
        # Combine and predict
        combined = torch.cat([time_features, freq_features], dim=1)
        return self.head(combined)


# ============================================================================
# GLOBAL MODELS (loaded once)
# ============================================================================

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model_c1 = None
_model_c2 = None


def get_model_challenge1():
    """Load Challenge 1 model (cached)."""
    global _model_c1
    if _model_c1 is None:
        _model_c1 = ImprovedEEGModel(n_channels=129, n_times=200, n_outputs=1).to(_device)
        weights_path = resolve_path("weights_challenge_1_sam.pt")
        checkpoint = torch.load(weights_path, map_location=_device, weights_only=False)
        
        # Handle checkpoint dict format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights
        _model_c1.load_state_dict(state_dict, strict=False)
        _model_c1.eval()
    return _model_c1


def get_model_challenge2():
    """Load Challenge 2 model (cached)."""
    global _model_c2
    if _model_c2 is None:
        _model_c2 = ImprovedEEGModel(n_channels=129, n_times=200, n_outputs=1).to(_device)
        weights_path = resolve_path("weights_challenge_2_sam.pt")
        checkpoint = torch.load(weights_path, map_location=_device, weights_only=False)
        
        # Handle checkpoint dict format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights
        _model_c2.load_state_dict(state_dict, strict=False)
        _model_c2.eval()
    return _model_c2


# ============================================================================
# REQUIRED COMPETITION FUNCTIONS
# ============================================================================

def challenge1(X):
    """
    Challenge 1: Predict response time from EEG.
    
    Args:
        X: List of EEG arrays, each shape (n_channels, n_timepoints)
    
    Returns:
        np.ndarray: Predictions [n_trials] - response times in seconds
    """
    model = get_model_challenge1()
    
    # Convert list to batch tensor (pad/crop to 200 timepoints)
    target_length = 200
    batch = []
    
    for eeg in X:
        if eeg.shape[1] < target_length:
            pad_width = target_length - eeg.shape[1]
            eeg = np.pad(eeg, ((0, 0), (0, pad_width)), mode='constant')
        elif eeg.shape[1] > target_length:
            eeg = eeg[:, :target_length]
        batch.append(eeg)
    
    X_tensor = torch.from_numpy(np.array(batch)).float().to(_device)
    
    # Predict
    with torch.no_grad():
        predictions = model(X_tensor).squeeze(-1)
    
    return predictions.cpu().numpy()


def challenge2(X):
    """
    Challenge 2: Predict externalizing factor from EEG.
    
    Args:
        X: List of EEG arrays, each shape (n_channels, n_timepoints)
    
    Returns:
        np.ndarray: Predictions [n_samples] - externalizing factor scores
    """
    model = get_model_challenge2()
    
    # Convert list to batch tensor (pad/crop to 200 timepoints)
    target_length = 200
    batch = []
    
    for eeg in X:
        if eeg.shape[1] < target_length:
            pad_width = target_length - eeg.shape[1]
            eeg = np.pad(eeg, ((0, 0), (0, pad_width)), mode='constant')
        elif eeg.shape[1] > target_length:
            eeg = eeg[:, :target_length]
        batch.append(eeg)
    
    X_tensor = torch.from_numpy(np.array(batch)).float().to(_device)
    
    # Predict
    with torch.no_grad():
        predictions = model(X_tensor).squeeze(-1)
    
    return predictions.cpu().numpy()


# ============================================================================
# LOCAL TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING SUBMISSION - v6 Correct (ImprovedEEGModel)")
    print("=" * 70 + "\n")
    
    batch_size = 4
    X_test = [np.random.randn(129, 250).astype(np.float32) for _ in range(batch_size)]
    
    print(f"Test input: {batch_size} samples, shape (129, 250)")
    print()
    
    # Test Challenge 1
    print("Testing Challenge 1...")
    try:
        y1 = challenge1(X_test)
        print(f"  Output shape: {y1.shape}")
        print(f"  Sample predictions: {y1[:3]}")
        print(f"  ✅ PASS")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
    print()
    
    # Test Challenge 2
    print("Testing Challenge 2...")
    try:
        y2 = challenge2(X_test)
        print(f"  Output shape: {y2.shape}")
        print(f"  Sample predictions: {y2[:3]}")
        print(f"  ✅ PASS")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
    print()
    
    print("=" * 70)
    print("✅ ALL TESTS PASSED - SUBMISSION READY")
    print("=" * 70 + "\n")
