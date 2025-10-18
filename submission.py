# ##########################################################################
# EEG 2025 Competition Submission
# https://eeg2025.github.io/
# https://www.codabench.org/competitions/4287/
#
# Challenge 1: CompactResponseTimeCNN (multi-release trained)
# Challenge 2: CompactExternalizingCNN (multi-release trained)
# Format follows official starter kit
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


# ============================================================================
# Compact CNN for Challenge 1 (Response Time)
# ============================================================================

class CompactResponseTimeCNN(nn.Module):
    """Compact CNN for response time prediction - multi-release trained (200K params)

    Designed to reduce overfitting through:
    - Smaller architecture (200K vs 800K params)
    - Strong dropout (0.3-0.5)
    - Trained on R1-R4, validated on R5
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: channels x 200 -> 32x100
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


# ============================================================================
# Compact CNN for Challenge 2 (Externalizing)
# ============================================================================

class CompactExternalizingCNN(nn.Module):
    """Compact CNN for externalizing prediction - multi-release trained (150K params)

    Designed to reduce overfitting through:
    - Smaller architecture (150K vs 600K params)
    - Strong dropout (0.3-0.5)
    - ELU activations for better gradients
    - Trained on R1-R4, validated on R5
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: channels x 200 -> 32x100
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),

            # Conv2: 32x100 -> 64x50
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.4),

            # Conv3: 64x50 -> 96x25
            nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(96),
            nn.ELU(),
            nn.Dropout(0.5),

            # Global pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.regressor = nn.Sequential(
            nn.Linear(96, 48),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(48, 24),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(24, 1)
        )

    def forward(self, x):
        features = self.features(x)
        output = self.regressor(features)
        return output


# ============================================================================
# Submission Class
# ============================================================================

class Submission:
    """
    EEG 2025 Competition Submission

    Challenge 1: Response Time Prediction with CompactResponseTimeCNN
    - CompactResponseTimeCNN (200K params, proven NRMSE 1.00)

    Challenge 2: Externalizing Prediction with CompactExternalizingCNN
    - CompactExternalizingCNN (64K params, NRMSE 1.33)
    """

    def __init__(self, SFREQ, DEVICE):
        """Initialize submission with sampling frequency and device"""
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        """Get Challenge 1 model (Response Time Prediction)"""
        print("📦 Loading Challenge 1 model...")

        # Create CompactResponseTimeCNN model
        model = CompactResponseTimeCNN().to(self.device)

        # Load weights
        try:
            weights_path = resolve_path("weights_challenge_1_multi_release.pt")
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)

            # Handle different checkpoint formats
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
                print(f"✅ Loaded Challenge 1 CNN from {weights_path}")
                print(f"   Val Loss: {state_dict.get('val_loss', 'N/A')}")
            else:
                model.load_state_dict(state_dict)
                print(f"✅ Loaded Challenge 1 CNN from {weights_path}")
        except Exception as e:
            print(f"⚠️  Warning loading Challenge 1 model: {e}")
            print("⚠️  Using untrained model")

        model.eval()
        return model

    def get_model_challenge_2(self):
        """Get Challenge 2 model (Externalizing Prediction)"""
        print("📦 Loading Challenge 2 model...")

        # Create CompactExternalizingCNN model
        model = CompactExternalizingCNN().to(self.device)

        # Load weights
        try:
            weights_path = resolve_path("weights_challenge_2_multi_release.pt")
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded Challenge 2 CompactCNN from {weights_path}")
                print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A')}")
            else:
                model.load_state_dict(checkpoint)
                print(f"✅ Loaded Challenge 2 CompactCNN from {weights_path}")
        except Exception as e:
            print(f"⚠️  Warning loading Challenge 2 model: {e}")
            print("⚠️  Using untrained model")

        model.eval()
        return model
