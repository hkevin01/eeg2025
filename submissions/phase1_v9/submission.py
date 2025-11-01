"""
EEG Foundation Challenge 2025 - Quick Fix Submission
Combines:
- Challenge 1: Oct 16 CompactCNN (score 1.0015)
- Challenge 2: Oct 24 EEGNeX (score 1.0087)
Expected overall: ~1.005
"""

import torch
import torch.nn as nn
from pathlib import Path


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


class CompactResponseTimeCNN(nn.Module):
    """Compact CNN for response time prediction (Challenge 1)
    
    Oct 16 model - Score: 1.0015
    Architecture: 75K parameters
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: 129 channels x 200 timepoints -> 32x100
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Conv2: 32x100 -> 64x50
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),

            # Conv3: 64x50 -> 128x25
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            # Global pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        features = self.features(x)
        output = self.regressor(features)
        return output


class Submission:
    """EEG 2025 Competition Submission - Quick Fix Version"""

    def __init__(self, SFREQ, DEVICE):
        """Initialize submission
        
        Args:
            SFREQ (int): Sampling frequency (100 Hz)
            DEVICE (str or torch.device): 'cuda' or 'cpu'
        """
        self.sfreq = SFREQ
        # FORCE CPU for stability - GPU causes memory errors on platform
        # V9 submission failed twice with GPU memory violations
        # CPU version works perfectly in all tests
        self.device = torch.device('cpu')
        self.n_chans = 129
        self.n_times = 200  # 2 seconds at 100 Hz
        self.model_c1 = None
        self.model_c2 = None

    def get_model_challenge_1(self):
        """Load Challenge 1 model - CompactCNN (Oct 16, score 1.0015)"""
        if self.model_c1 is not None:
            return self.model_c1

        # Create model
        model = CompactResponseTimeCNN()

        # Load weights
        weights_path = resolve_path('weights_challenge_1.pt')
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
        model.load_state_dict(state_dict)

        # Move to device and set eval mode
        model = model.to(self.device)
        model.eval()

        self.model_c1 = model
        return model

    def get_model_challenge_2(self):
        """Load Challenge 2 model - EEGNeX (Oct 24, score 1.0087)"""
        if self.model_c2 is not None:
            return self.model_c2

        # Import braindecode (available on competition platform)
        from braindecode.models import EEGNeX

        # Create model
        model = EEGNeX(
            n_chans=self.n_chans,
            n_times=self.n_times,
            n_outputs=1,
            sfreq=self.sfreq,
        )

        # Load weights
        weights_path = resolve_path('weights_challenge_2.pt')
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

        self.model_c2 = model
        return model

    def challenge_1(self, X):
        """Challenge 1: Predict response time from EEG
        
        Args:
            X (torch.Tensor): Shape (batch, 129, 200)
            
        Returns:
            torch.Tensor: Predictions shape (batch,)
        """
        model = self.get_model_challenge_1()
        
        with torch.no_grad():
            X = X.to(self.device)
            predictions = model(X).squeeze(-1)
            
        return predictions

    def challenge_2(self, X):
        """Challenge 2: Predict externalizing factor from EEG
        
        Args:
            X (torch.Tensor): Shape (batch, 129, 200)
            
        Returns:
            torch.Tensor: Predictions shape (batch,)
        """
        model = self.get_model_challenge_2()
        
        with torch.no_grad():
            X = X.to(self.device)
            predictions = model(X).squeeze(-1)
            
        return predictions


def main():
    """Test submission locally"""
    print("Testing submission locally...")
    print()

    # Initialize
    SFREQ = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    submission = Submission(SFREQ, DEVICE)

    # Test Challenge 1
    print("🧪 Testing Challenge 1...")
    test_eeg_c1 = torch.randn(4, 129, 200)
    predictions_c1 = submission.challenge_1(test_eeg_c1)
    print(f"Input shape: {test_eeg_c1.shape}")
    print(f"Output shape: {predictions_c1.shape}")
    print(f"Sample predictions: {predictions_c1[:3].cpu().numpy()}")
    print("✅ Challenge 1 working")
    print()

    # Test Challenge 2
    print("🧪 Testing Challenge 2...")
    test_eeg_c2 = torch.randn(4, 129, 200)
    predictions_c2 = submission.challenge_2(test_eeg_c2)
    print(f"Input shape: {test_eeg_c2.shape}")
    print(f"Output shape: {predictions_c2.shape}")
    print(f"Sample predictions: {predictions_c2[:3].cpu().numpy()}")
    print("✅ Challenge 2 working")
    print()

    print("="*70)
    print("✅ All tests passed! Submission ready for upload.")
    print("="*70)


if __name__ == "__main__":
    main()
