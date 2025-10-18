# ##########################################################################
# EEG 2025 Competition Submission
# https://eeg2025.github.io/
# https://www.codabench.org/competitions/9975/
#
# Challenge 1: TCN trained on competition data (R1-R4)
# Challenge 2: Compact CNN multi-release trained
# Format follows official starter kit
# ##########################################################################

import numpy as np
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
# TCN Components for Challenge 1
# ============================================================================

class TemporalBlock(nn.Module):
    """Temporal block with dilated causal convolutions"""
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN_EEG(nn.Module):
    """Temporal Convolutional Network for EEG (Challenge 1)"""
    def __init__(self, num_channels=129, num_outputs=1, num_filters=48,
                 kernel_size=7, dropout=0.3, num_levels=5):
        super().__init__()
        layers = []
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_channels if i == 0 else num_filters
            layers.append(TemporalBlock(in_channels, num_filters, kernel_size,
                                       dilation, dropout))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_filters, num_outputs)

    def forward(self, x):
        out = self.network(x)
        out = out.mean(dim=2)
        return self.fc(out)


# ============================================================================
# Compact CNN for Challenge 2
# ============================================================================

class CompactExternalizingCNN(nn.Module):
    """Compact CNN for externalizing prediction (Challenge 2)"""
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(129, 32, kernel_size=7, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 96, kernel_size=3, stride=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.regressor = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(24, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)


# ============================================================================
# Submission Class (Competition API)
# ============================================================================

class Submission:
    """
    EEG 2025 Competition Submission

    Challenge 1: Response Time Prediction with TCN
    - TCN_EEG (196K params, Val Loss 0.0102)

    Challenge 2: Externalizing Prediction with CompactCNN
    - CompactExternalizingCNN (64K params, Val NRMSE 0.2917)
    """

    def __init__(self, SFREQ, DEVICE):
        """Initialize submission with sampling frequency and device"""
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        """Get Challenge 1 model (Response Time Prediction with TCN)"""
        print("📦 Loading Challenge 1 model...")

        # Create TCN model
        model = TCN_EEG(
            num_channels=129,
            num_outputs=1,
            num_filters=48,
            kernel_size=7,
            dropout=0.3,
            num_levels=5
        ).to(self.device)

        # Load weights
        try:
            weights_path = resolve_path("challenge1_tcn_competition_best.pth")
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded Challenge 1 TCN model from {weights_path}")
                print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A')}")
            else:
                model.load_state_dict(checkpoint)
                print(f"✅ Loaded Challenge 1 TCN model from {weights_path}")
        except Exception as e:
            print(f"⚠️  Warning loading Challenge 1 model: {e}")
            print("⚠️  Using untrained model")

        model.eval()
        return model

    def get_model_challenge_2(self):
        """Get Challenge 2 model (Externalizing Prediction with CompactCNN)"""
        print("📦 Loading Challenge 2 model...")

        # Create CompactCNN model
        model = CompactExternalizingCNN().to(self.device)

        # Load weights - try multiple filenames
        weights_loaded = False

        # Try primary filename first
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
            weights_loaded = True
        except Exception as e:
            print(f"⚠️  Warning loading from weights_challenge_2_multi_release.pt: {e}")

        # Try alternative filename
        if not weights_loaded:
            try:
                weights_path = resolve_path("challenge2_tcn_competition_best.pth")
                checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✅ Loaded Challenge 2 model from {weights_path}")
                else:
                    model.load_state_dict(checkpoint)
                    print(f"✅ Loaded Challenge 2 model from {weights_path}")
                weights_loaded = True
            except Exception as e2:
                print(f"⚠️  Warning loading from challenge2_tcn_competition_best.pth: {e2}")

        if not weights_loaded:
            print("⚠️  No weights found, using untrained model")

        model.eval()
        return model
