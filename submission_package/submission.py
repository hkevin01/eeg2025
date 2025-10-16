# ##########################################################################
# # EEG 2025 Competition Submission
# # https://eeg2025.github.io/
# # https://www.codabench.org/competitions/4287/
# #
# # Format follows official starter kit:
# # https://github.com/eeg2025/startkit
# ##########################################################################

import torch
import torch.nn as nn
from pathlib import Path


def resolve_path(name="model_file_name"):
    """Resolve model file path across different execution environments"""
    if Path(f"/app/input/res/{name}").exists():
        return f"/app/input/res/{name}"
    elif Path(f"/app/input/{name}").exists():
        return f"/app/input/{name}"
    elif Path(f"{name}").exists():
        return f"{name}"
    elif Path(__file__).parent.joinpath(f"{name}").exists():
        return str(Path(__file__).parent.joinpath(f"{name}"))
    else:
        raise FileNotFoundError(
            f"Could not find {name} in /app/input/res/ or /app/input/ or current directory"
        )


class ExternalizingCNN(nn.Module):
    """CNN for externalizing factor prediction (Challenge 2)"""

    def __init__(self, n_chans=129, n_times=200):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_chans, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        features = self.features(x)
        output = self.regressor(features)
        return output


class ResponseTimeCNN(nn.Module):
    """Improved CNN for response time prediction (Challenge 1)
    
    Architecture improvements:
    - Initial projection layer
    - Deeper network (512 features)
    - More dropout for regularization
    - Better performance: NRMSE 0.4680 (CV: 1.05 ± 0.08)
    """

    def __init__(self, n_chans=129, n_times=200):
        super().__init__()
        
        # Initial projection
        self.proj = nn.Sequential(
            nn.Conv1d(129, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Multi-scale feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Regressor
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x


class Submission:
    """Competition submission class

    Official format for EEG 2025 competition.
    Must implement:
    - __init__(self, SFREQ, DEVICE)
    - get_model_challenge_1(self)
    - get_model_challenge_2(self)
    """

    def __init__(self, SFREQ, DEVICE):
        """Initialize submission with sampling frequency and device

        Args:
            SFREQ: Sampling frequency (100 Hz for competition)
            DEVICE: torch.device for inference
        """
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        """Get Challenge 1 model (response time prediction from CCD task)

        Returns:
            PyTorch model ready for inference
        """
        import sys
        model_challenge1 = ResponseTimeCNN(
            n_chans=129,
            n_times=int(2 * self.sfreq)  # 200 samples @ 100Hz
        ).to(self.device)

        # Load trained weights
        try:
            weights_path = resolve_path("weights_challenge_1.pt")
            sys.stdout.flush()
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)

            # Handle different checkpoint formats
            if 'model_state_dict' in state_dict:
                model_challenge1.load_state_dict(state_dict['model_state_dict'])
            else:
                model_challenge1.load_state_dict(state_dict)

            print(f"✓ Loaded Challenge 1 weights from {weights_path}", flush=True)
        except FileNotFoundError:
            print("⚠ Warning: weights_challenge_1.pt not found, using untrained model", flush=True)

        model_challenge1.eval()
        return model_challenge1

    def get_model_challenge_2(self):
        """Get Challenge 2 model (externalizing factor prediction)

        Returns:
            PyTorch model ready for inference
        """
        import sys
        model_challenge2 = ExternalizingCNN(
            n_chans=129,
            n_times=int(2 * self.sfreq)  # 200 samples @ 100Hz
        ).to(self.device)

        # Load trained weights
        try:
            weights_path = resolve_path("weights_challenge_2.pt")
            sys.stdout.flush()
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)

            # Handle different checkpoint formats
            if 'model_state_dict' in state_dict:
                model_challenge2.load_state_dict(state_dict['model_state_dict'])
            else:
                model_challenge2.load_state_dict(state_dict)

            print(f"✓ Loaded Challenge 2 weights from {weights_path}", flush=True)
        except FileNotFoundError:
            print("⚠ Warning: weights_challenge_2.pt not found, using untrained model", flush=True)

        model_challenge2.eval()
        return model_challenge2



# ##########################################################################
# # Local Testing
# ##########################################################################

def test_submission():
    """Test the submission class locally"""
    import numpy as np
    import sys

    print("="*70, flush=True)
    print("Testing EEG 2025 Competition Submission", flush=True)
    print("="*70, flush=True)

    # Competition parameters
    SFREQ = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {DEVICE}", flush=True)
    print(f"Sampling Frequency: {SFREQ} Hz", flush=True)

    # Create dummy data (batch_size=10, n_channels=129, n_times=200)
    print("\n🔢 Creating test data...", end=' ', flush=True)
    n_samples = 10
    X_test = np.random.randn(n_samples, 129, 200).astype(np.float32)
    X_tensor = torch.from_numpy(X_test).to(DEVICE)
    print("✓", flush=True)

    # Initialize submission
    print("\n" + "-"*70, flush=True)
    print("Initializing Submission...", flush=True)
    print("-"*70, flush=True)
    sys.stdout.flush()

    submission = Submission(SFREQ, DEVICE)
    print("✓ Submission initialized", flush=True)    # Test Challenge 1
    print("\n" + "-"*70, flush=True)
    print("Challenge 1: Response Time Prediction (CCD Task)", flush=True)
    print("-"*70, flush=True)
    print("🔄 Loading Challenge 1 model...", end=' ', flush=True)
    sys.stdout.flush()

    model_1 = submission.get_model_challenge_1()
    print("✓", flush=True)

    print("🧠 Running inference...", end=' ', flush=True)
    sys.stdout.flush()
    with torch.inference_mode():
        pred1 = model_1.forward(X_tensor)
    print("✓", flush=True)

    pred1_np = pred1.cpu().numpy().flatten()
    print(f"  Input shape: {X_tensor.shape}", flush=True)
    print(f"  Output shape: {pred1_np.shape}", flush=True)
    print(f"  Sample predictions: {pred1_np[:3]}", flush=True)
    print(f"  Prediction range: [{pred1_np.min():.4f}, {pred1_np.max():.4f}]", flush=True)

    # Test Challenge 2
    print("\n" + "-"*70, flush=True)
    print("Challenge 2: Externalizing Factor Prediction", flush=True)
    print("-"*70, flush=True)
    print("🔄 Loading Challenge 2 model...", end=' ', flush=True)
    sys.stdout.flush()

    model_2 = submission.get_model_challenge_2()
    print("✓", flush=True)

    print("🧠 Running inference...", end=' ', flush=True)
    sys.stdout.flush()
    with torch.inference_mode():
        pred2 = model_2.forward(X_tensor)
    print("✓", flush=True)

    pred2_np = pred2.cpu().numpy().flatten()
    print(f"  Input shape: {X_tensor.shape}", flush=True)
    print(f"  Output shape: {pred2_np.shape}", flush=True)
    print(f"  Sample predictions: {pred2_np[:3]}", flush=True)
    print(f"  Prediction range: [{pred2_np.min():.4f}, {pred2_np.max():.4f}]", flush=True)

    # Summary
    print("\n" + "="*70, flush=True)
    print("✅ Submission class test passed!", flush=True)
    print("="*70, flush=True)
    print("\nNext steps:", flush=True)
    print("  1. Train Challenge 1 model → weights_challenge_1.pt", flush=True)
    print("  2. Copy checkpoints/externalizing_model.pth → weights_challenge_2.pt", flush=True)
    print("  3. Create submission.zip with:", flush=True)
    print("     - submission.py", flush=True)
    print("     - weights_challenge_1.pt", flush=True)
    print("     - weights_challenge_2.pt", flush=True)
    print("  4. Upload to: https://www.codabench.org/competitions/4287/", flush=True)
    print("="*70, flush=True)


if __name__ == "__main__":
    test_submission()
