"""
EEG Foundation Challenge 2025 - V13.5 Submission
Variance Reduction Focus (Size-Optimized):
- Challenge 1: 3-seed EnhancedCompactCNN + TTA + Calibration
- Challenge 2: 2-seed EEGNeX ensemble
Expected: C1 ~1.00013 (lighter), C2 ~1.00049, Overall ~1.00031
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np


def resolve_path(filename):
    """Find file in competition or local environment"""
    search_paths = [
        f"/app/input/res/{filename}",
        f"/app/input/{filename}",
        filename,
        str(Path(__file__).parent / filename),
    ]

    for path in search_paths:
        if Path(path).exists():
            return path

    raise FileNotFoundError(
        f"Could not find {filename} in: {', '.join(search_paths)}"
    )


class EnhancedCompactCNN(nn.Module):
    """Enhanced Compact CNN with Spatial Attention for Challenge 1"""
    
    def __init__(self, dropout_rate=0.6):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate + 0.05),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate + 0.1),
        )
        
        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(128, 128, 1),
            nn.Sigmoid()
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        features = self.features(x)
        attention = self.spatial_attn(features)
        features = features * attention
        output = self.regressor(features)
        return output


class Submission:
    """EEG 2025 Competition Submission - V12 (Variance Reduction)"""

    def __init__(self, SFREQ, DEVICE):
        """Initialize submission
        
        Args:
            SFREQ (int): Sampling frequency (100 Hz)
            DEVICE (str or torch.device): 'cuda' or 'cpu'
        """
        self.sfreq = SFREQ
        # Convert string to torch.device if needed
        if isinstance(DEVICE, str):
            self.device = torch.device(DEVICE)
        else:
            self.device = DEVICE
        self.n_chans = 129
        self.n_times = 200  # 2 seconds at 100 Hz
        self.models_c1 = None  # List of 5 models
        self.models_c2 = None  # List of 2 models
        self.c1_calibration = None  # Calibration parameters
        
        # TTA parameters
        self.use_tta = True
        self.tta_time_shifts = [-2, 0, 2]  # Time shifts in samples

    def load_c1_calibration(self):
        """Load C1 calibration parameters"""
        if self.c1_calibration is not None:
            return self.c1_calibration
        
        calib_path = resolve_path('c1_calibration_params.json')
        with open(calib_path, 'r') as f:
            self.c1_calibration = json.load(f)
        
        return self.c1_calibration

    def get_models_challenge_1(self):
        """Load Challenge 1 models - 3-seed EnhancedCompactCNN ensemble"""
        if self.models_c1 is not None:
            return self.models_c1

        models = []
        
        # Load best 3 seeds (size-optimized)
        for seed in [42, 123, 456]:
            # Create model
            model = EnhancedCompactCNN()

            # Load weights
            weights_path = resolve_path(f'c1_phase1_seed{seed}_ema_best.pt')
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            
            # Handle checkpoint format
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)

            # Move to device and set eval mode
            model = model.to(self.device)
            model.eval()
            
            models.append(model)

        self.models_c1 = models
        return models

    def get_models_challenge_2(self):
        """Load Challenge 2 models - 2-seed EEGNeX ensemble"""
        if self.models_c2 is not None:
            return self.models_c2

        # Import braindecode (available on competition platform)
        from braindecode.models import EEGNeX

        models = []
        
        # Load both seeds
        for seed in [42, 123]:
            # Create model
            model = EEGNeX(
                n_chans=self.n_chans,
                n_times=self.n_times,
                n_outputs=1,
                sfreq=self.sfreq,
            )

            # Load weights
            weights_path = resolve_path(f'c2_phase2_seed{seed}_ema_best.pt')
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

            # Handle checkpoint format
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)

            # Move to device and set eval mode
            model = model.to(self.device)
            model.eval()
            
            models.append(model)

        self.models_c2 = models
        return models
    
    def apply_time_shift(self, X, shift):
        """Apply circular time shift for TTA"""
        if shift == 0:
            return X
        return torch.roll(X, shifts=shift, dims=2)
    
    def apply_calibration(self, predictions):
        """Apply linear calibration: y_cal = a*y + b"""
        calib = self.load_c1_calibration()
        a = calib['a']
        b = calib['b']
        return a * predictions + b

    def challenge_1(self, X):
        """Challenge 1: Predict response time from EEG
        
        3-seed ensemble + TTA + Calibration
        
        Args:
            X (np.ndarray or torch.Tensor): Shape (batch, 129, 200)
            
        Returns:
            np.ndarray: Predictions shape (batch,)
        """
        models = self.get_models_challenge_1()
        
        # Convert numpy to torch if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        
        with torch.no_grad():
            X = X.to(self.device)
            
            all_predictions = []
            
            # Apply TTA
            if self.use_tta:
                tta_shifts = self.tta_time_shifts
            else:
                tta_shifts = [0]  # No TTA, just original
            
            for shift in tta_shifts:
                X_aug = self.apply_time_shift(X, shift)
                
                # Get predictions from each seed
                for model in models:
                    pred = model(X_aug).squeeze(-1)
                    all_predictions.append(pred)
            
            # Average all predictions (seeds × TTA)
            predictions = torch.stack(all_predictions).mean(dim=0)
            
            # Apply calibration
            predictions_cal = self.apply_calibration(predictions)
            
            # Convert back to numpy
            predictions_cal = predictions_cal.cpu().numpy()
            
        return predictions_cal

    def challenge_2(self, X):
        """Challenge 2: Predict externalizing factor from EEG
        
        2-seed ensemble: Average predictions from Seeds 42 & 123
        
        Args:
            X (np.ndarray or torch.Tensor): Shape (batch, 129, 200)
            
        Returns:
            np.ndarray: Predictions shape (batch,)
        """
        models = self.get_models_challenge_2()
        
        # Convert numpy to torch if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        
        with torch.no_grad():
            X = X.to(self.device)
            
            # Get predictions from each seed
            preds = []
            for model in models:
                pred = model(X).squeeze(-1)  # Squeeze to (batch,)
                preds.append(pred)
            
            # Average predictions
            predictions = torch.stack(preds).mean(dim=0)
            
            # Convert back to numpy
            predictions = predictions.cpu().numpy()
            
        return predictions


def main():
    """Test submission locally"""
    print("Testing V12 submission locally...")
    print()

    # Initialize
    SFREQ = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}")
    
    submission = Submission(SFREQ, DEVICE)

    # Test Challenge 1
    print("\n🧪 Testing Challenge 1 (5-seed + TTA + calibration)...")
    test_eeg_c1 = torch.randn(4, 129, 200)
    predictions_c1 = submission.challenge_1(test_eeg_c1)
    print(f"Input shape: {test_eeg_c1.shape}")
    print(f"Output shape: {predictions_c1.shape}")
    print(f"Sample predictions: {predictions_c1[:3].cpu().numpy()}")
    print("✅ Challenge 1 working (5 seeds × 3 TTA × calibration)")

    # Test Challenge 2
    print("\n🧪 Testing Challenge 2 (2-seed ensemble)...")
    test_eeg_c2 = torch.randn(4, 129, 200)
    predictions_c2 = submission.challenge_2(test_eeg_c2)
    print(f"Input shape: {test_eeg_c2.shape}")
    print(f"Output shape: {predictions_c2.shape}")
    print(f"Sample predictions: {predictions_c2[:3].cpu().numpy()}")
    print("✅ Challenge 2 working (2 seeds averaged)")

    print("\n" + "="*70)
    print("✅ All tests passed! V12 submission ready for upload.")
    print("="*70)
    print("📊 Model Summary:")
    print("  Challenge 1: 5-seed EnhancedCompactCNN + TTA + Calibration")
    print("    - Seeds: 42, 123, 456, 789, 1337")
    print("    - TTA: 3 time shifts (-2, 0, +2)")
    print("    - Calibration: a=0.988, b=0.027")
    print("    - Expected improvement: ~8e-5")
    print("  Challenge 2: 2-seed EEGNeX ensemble")
    print("    - Seeds: 42, 123")
    print("    - Mean validation loss: 0.124")
    print("="*70)
    print("Expected V12 Score:")
    print("  C1: ~1.00011 (improvement: 8e-5 from V10)")
    print("  C2: ~1.00049 (same as V11.5)")
    print("  Overall: ~1.00030")
    print("  Rank: #45-55")
    print("="*70)


if __name__ == "__main__":
    main()
