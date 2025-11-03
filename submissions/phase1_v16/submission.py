"""
EEG Foundation Challenge 2025 - V16 Submission
Enhancements:
- Challenge 1: 5-seed ensemble + TTA
- Challenge 2: Best from V14 (1.00066)
Expected C1: 0.999-1.000 (improved from 1.00019)
"""

import torch
import torch.nn as nn
import numpy as np
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
    """Compact CNN for response time prediction (Challenge 1)"""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
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
    """EEG 2025 Competition Submission - V16"""

    def __init__(self, SFREQ, DEVICE):
        """Initialize submission"""
        self.sfreq = SFREQ
        if isinstance(DEVICE, str):
            self.device = torch.device(DEVICE)
        else:
            self.device = DEVICE
        self.n_chans = 129
        self.n_times = 200
        self.models_c1 = None
        self.model_c2 = None
        self.use_tta = True
        self.seeds = [42, 123, 456, 789, 1337]

    def get_models_challenge_1(self):
        """Load Challenge 1 ensemble"""
        if self.models_c1 is not None:
            return self.models_c1

        models = []
        for seed in self.seeds:
            model = CompactResponseTimeCNN()
            weights_path = resolve_path(f'weights_challenge_1_seed{seed}.pt')
            state_dict = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            models.append(model)

        self.models_c1 = models
        return models

    def get_model_challenge_2(self):
        """Load Challenge 2 model"""
        if self.model_c2 is not None:
            return self.model_c2

        from braindecode.models import EEGNeX

        model = EEGNeX(
            n_chans=self.n_chans,
            n_times=self.n_times,
            n_outputs=1,
            sfreq=self.sfreq,
        )

        weights_path = resolve_path('weights_challenge_2.pt')
        checkpoint = torch.load(weights_path, map_location=self.device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        self.model_c2 = model
        return model

    def apply_tta_augmentations(self, X):
        """Apply TTA augmentations"""
        augmentations = [X]  # Original

        # Time shifts
        for shift in [-3, 3]:
            augmentations.append(torch.roll(X, shifts=shift, dims=2))

        # Amplitude scaling
        for scale in [0.97, 1.03]:
            augmentations.append(X * scale)

        # Small noise
        noise = torch.randn_like(X) * 0.01
        augmentations.append(X + noise)

        return augmentations

    def challenge_1(self, X):
        """Challenge 1: Response time prediction with ensemble + TTA"""
        models = self.get_models_challenge_1()

        with torch.no_grad():
            X = X.to(self.device)

            if self.use_tta:
                # TTA augmentations
                X_variants = self.apply_tta_augmentations(X)

                # Ensemble predictions for each augmentation
                all_predictions = []
                for X_aug in X_variants:
                    for model in models:
                        pred = model(X_aug).squeeze(-1)
                        all_predictions.append(pred)

                # Average all predictions
                predictions = torch.stack(all_predictions).mean(dim=0)
            else:
                # Just ensemble (no TTA)
                predictions = []
                for model in models:
                    pred = model(X).squeeze(-1)
                    predictions.append(pred)
                predictions = torch.stack(predictions).mean(dim=0)

        return predictions

    def challenge_2(self, X):
        """Challenge 2: Externalizing factor prediction"""
        model = self.get_model_challenge_2()

        with torch.no_grad():
            X = X.to(self.device)
            predictions = model(X).squeeze(-1)

        return predictions


def main():
    """Test submission locally"""
    print("Testing V16 submission locally...")
    print()

    # Initialize
    SFREQ = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {DEVICE}")
    print(f"Seeds: [42, 123, 456, 789, 1337]")
    print(f"TTA: True")
    print()

    submission = Submission(SFREQ, DEVICE)

    # Test with dummy data
    X = torch.randn(2, 129, 200)

    print("Testing Challenge 1...")
    pred_c1 = submission.challenge_1(X)
    print(f"  Output shape: {pred_c1.shape}")
    print(f"  Predictions: {pred_c1.cpu().numpy()}")

    print("\nTesting Challenge 2...")
    pred_c2 = submission.challenge_2(X)
    print(f"  Output shape: {pred_c2.shape}")
    print(f"  Predictions: {pred_c2.cpu().numpy()}")

    print("\n✅ Submission test complete!")


if __name__ == '__main__':
    main()
